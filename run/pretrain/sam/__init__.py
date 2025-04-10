from dataclasses import dataclass, field
from typing import Any, List
from PIL import Image
from mos.models.sam.modeling_sam.sam_model import (
    SamModel,
)
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

from mos.models.sam.modeling_sam.embedding.typing import (
    GrayImageTensor,
    SparseEmbeddingsTensor,
    SparseEmbeddingsTensor,
    DenseEmbeddingsTensor,
    ImageEmbeddingTensor,
    SegmentTensor,
)
from mos.models.sam.modeling_sam.embedding import BatchPrompt, PointBatchPrompt, Prompt
from run.pretrain.sam.sam_dataset_compat import SAMDatasetCompat
from run.pretrain.sam.token_text import prepare_token_embedding_to_device
from .model_factory import ModelFactory
from .batch_soft_dice import BatchSoftIoU, DiceScore

from .sam_dataset import SAMDataset, SAMDatasetItem, get_compose_datset_for_train, get_compose_datset_for_validate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


@dataclass
class ModelArguments(object):
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    init_model: bool = field(
        default=False,
        metadata={"help": ("是否重新初始化模型, (从原始的SAM迁移)")},
    )

    start_epoch: int = field(
        default=1,
        metadata={"help": ("开始训练的epoch")},
    )

    finetune_vit: bool = field(
        default=False,
        metadata={"help": ("是否微调vit")},
    )
    patch_img_embedding: float = field(
        default=0.0,
        metadata={"help": ("随机对 image embedding进行patch mask的比例, 0.0~1.0")},
    )
    ds_device_image_cache_count: int = field(
        default=8000,
        metadata={"help": ("dataset缓存多少个图片到显存中")},
    )
    description: str = field(
        default="",
        metadata={"help": ("训练的描述")},
    )


import logging

logger = logging.getLogger(__name__)


def init_logger(train_arg: TrainingArguments):
    import transformers, sys

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if train_arg.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = train_arg.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def get_args(args: List[str] | None) -> tuple[ModelArguments, TrainingArguments]:
    import os, sys

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(args) == 2 and args[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # type: Tuple[ModelArguments, DataTrainingArguments, CustomTrainingArguments]
        model_args, training_args = parser.parse_args_into_dataclasses(args=args)
    return (model_args, training_args)


def train(args: List[str] | None):
    model_args, training_args = get_args(args)

    run_name = training_args.run_name
    device = training_args.device

    prepare_token_embedding_to_device(device)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log_writer = SummaryWriter(log_dir=f"{training_args.output_dir}/logs")
    log_writer.add_text("description", model_args.description, 0)

    if model_args.init_model:
        model_factory = ModelFactory("cpu", run_name)
        model_factory.init_model_from_pretrain()

    model_factory = ModelFactory(device, run_name)

    model: SamModel = model_factory.load_model()

    new_dataset = SAMDatasetCompat(
        base_path=".cache/dataset/sam-dataset",
        device=device,
    )
    new_dataset.filter_cmri_dataset()

    BATCH_SIZE = training_args.train_batch_size
    MICRO_BATCH_SIZE = 200 / BATCH_SIZE
    USE_AMP = True

    # make sure we only compute gradients for mask decoder
    if (not model_args.finetune_vit) and model_args.patch_img_embedding <= 0.0:
        # import loralib as lora
        # lora.mark_only_lora_as_trainable(model.prompt.embed_image.vision_encoder)

        print("make sure we only compute gradients for mask decoder, freeze prompt.embed_image.vision_encoder.vit")
        for name, param in model.named_parameters():
            if name.startswith("prompt.embed_image.vision_encoder.vit"):
                print("freeze:", name)
                param.requires_grad_(False)

    # calc_seg_loss = monai.losses.DiceCELoss(
    # sigmoid=True, squared_pred=True, reduction='mean'
    # )
    calc_seg_loss = torch.nn.MSELoss()
    calc_iou = BatchSoftIoU()
    calc_iou_loss = torch.nn.MSELoss()
    calc_dice_score = DiceScore()

    # num_epochs = 3000
    num_epochs = int(training_args.num_train_epochs)

    scaler = GradScaler()

    optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-5)
    current_micro_batch = 0
    for epoch in range(model_args.start_epoch, num_epochs + 1):
        epoch_losses = []
        iou_scores = []
        dice_losses = []
        model.train()
        for batch in tqdm(new_dataset.shuffer_train_id(training_args.train_batch_size)):
            image, labeled_segment, prompt = new_dataset.batch_get_train(batch)
            sparse_embeddings = prompt.encode(model.prompt)

            # forward pass, keys:
            # input_points, input_labels, input_boxes, input_masks, image_embeddings
            image_embeddings = model.prompt.embed_image.vision_encoder(image)
            dense_embeddings: DenseEmbeddingsTensor = model.prompt.embed_mask(None)
            last_image = image

            with autocast(enabled=USE_AMP):
                pred_iou_scores, pred_masks = model(
                    image_embeddings,
                    sparse_embeddings,
                    dense_embeddings,
                    multimask_output=False,
                )

                first_pred_masks = pred_masks[:, 0, 0, :, :]
                # first_pred_masks = torch.sigmoid(first_pred_masks)
                first_pred_iou_scores = pred_iou_scores[:, 0, 0]

                # print("first_pred_masks:", first_pred_masks.shape, "labeled_segment:", labeled_segment.shape)
                seg_loss: torch.Tensor = calc_seg_loss(first_pred_masks, labeled_segment)

                with torch.no_grad():
                    target_iou: torch.Tensor = calc_iou(first_pred_masks, labeled_segment)
                    iou_scores.append(target_iou.mean().item())

                    dice_loss = calc_dice_score(first_pred_masks, labeled_segment)
                    dice_losses.append(dice_loss.item())

                iou_loss = calc_iou_loss(target_iou, first_pred_iou_scores)

                loss = seg_loss + iou_loss
                epoch_losses.append(loss.item())
                loss = loss / MICRO_BATCH_SIZE

            current_micro_batch += 1
            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if current_micro_batch < MICRO_BATCH_SIZE:
                continue
            current_micro_batch = 0

            if USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        print(
            f"epoch: {epoch}, mean loss: {mean(epoch_losses)}, mean iou: {mean(iou_scores)}, dice: {mean(dice_losses)}"
        )
        log_writer.add_image("train/image", last_image[0, :, :, :], epoch, dataformats="CHW")
        log_writer.add_image("train/segment_pred", first_pred_masks[0, :, :], epoch, dataformats="HW")

        segment_label = last_image[0, :, :, :].repeat(3, 1, 1)
        segment_label[1, :, :] += labeled_segment[0, :, :]
        segment_label = segment_label.clamp(0, 1)
        if isinstance(prompt, PointBatchPrompt):
            x, y = prompt.point_embeddings[0][0].point
            segment_label[:, y, x] = 0
            segment_label[0, y, x] = 1
        log_writer.add_image("train/segment_label", segment_label, epoch, dataformats="CHW")

        log_writer.add_scalar("train/loss", mean(epoch_losses), epoch)
        log_writer.add_scalar("train/iou", mean(iou_scores), epoch)
        log_writer.add_scalar("train/dice", mean(dice_losses), epoch)

        if epoch % 30 == 0:
            model_factory.save_model(model, f"{training_args.output_dir}/{epoch:04}")

        print("evaluating...")
        validate_dice = []
        model.eval()
        with torch.no_grad():
            for batch in new_dataset.shuffer_valid_id(training_args.eval_batch_size):
                image, labeled_segment, prompt = new_dataset.batch_get_valid(batch)
                sparse_embeddings = prompt.encode(model.prompt)

                # forward pass, keys:
                # input_points, input_labels, input_boxes, input_masks, image_embeddings
                image_embeddings = model.prompt.embed_image.vision_encoder(image)
                dense_embeddings: DenseEmbeddingsTensor = model.prompt.embed_mask(None)
                last_image = image

                pred_iou_scores, pred_masks = model(
                    image_embeddings=image_embeddings,
                    sparse_embeddings=sparse_embeddings,
                    dense_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                first_pred_masks = pred_masks[:, 0, 0, :, :]
                dice_loss = calc_dice_score(first_pred_masks, labeled_segment).item()
                validate_dice.append(dice_loss)
            log_writer.add_image("valid/image", last_image[0, :, :], epoch, dataformats="CHW")

            segment_label = last_image[0, :, :, :].repeat(3, 1, 1)
            segment_label[1, :, :] += labeled_segment[0, :, :]
            segment_label = segment_label.clamp(0, 1)
            if isinstance(prompt, PointBatchPrompt):
                x, y = prompt.point_embeddings[0][0].point
                segment_label[:, y, x] = 0
                segment_label[0, y, x] = 1
            log_writer.add_image("valid/segment_label", segment_label, epoch, dataformats="CHW")

            log_writer.add_image("valid/segment_pred", first_pred_masks[0, :, :], epoch, dataformats="HW")
            print(f"validate epoch-{epoch:04} dice: {mean(validate_dice)}, count:{len(validate_dice)}")
            log_writer.add_scalar("valid/dice", mean(validate_dice), epoch)

    model_factory.save_model(model, f"{training_args.output_dir}/{epoch:04}")
    log_writer.close()


def run(args: List[str] | None):
    # torch.multiprocessing.set_start_method("spawn")
    print("args:", args)

    train(args)


# 训练时间
# AMP:  52s, 2.9GB
# FP32: 65s, 3.9GB
# 速度提升 20%, 内存减少 25%
