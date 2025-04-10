from dataclasses import dataclass, field
import os
from typing import Any, List, Optional, Sequence, Tuple, Union
from mos.datasets.common.dataset_utils import locate_dataset_base_url
import mos.models.mmae.model_mmae as model_mmae
from mos.models.mmae.model_mmae import MMaeVit
import torch
import random
from torch import nn, Tensor
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
from mos.utils.files import relative_symlink_file

from mos.utils.model_utils import use_cudnn
from .mmae_dataset_compat import MmaeDatasetCompat
from torch.optim import Optimizer


class CustomDataParallel(torch.nn.DataParallel):
    def __init__(
        self,
        module: Any,
        device_ids: Sequence[int | torch.device] | None = None,
        output_device: int | torch.device | None = None,
        dim: int = 0,
    ) -> None:
        super().__init__(module, device_ids, output_device, dim)

    def scatter(
        self,
        inputs: Tuple[Any, ...],
        kwargs: Optional[dict[str, Any]],
        device_ids: Sequence[Union[int, torch.device]],
    ) -> Any:
        all_device_param_list = []
        for device_index in device_ids:
            param_list = []
            for input in inputs:
                param_list.append(input[device_index])
            all_device_param_list.append(param_list)
        return all_device_param_list, ([kwargs for _ in device_ids])


global_last_result = (None, None, None)


class MmaeTrainModel(nn.Module):
    def __init__(
        self,
        model: MMaeVit,
        dataset: MmaeDatasetCompat,
        use_contrast_loss: bool = True,
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.use_contrast_loss = use_contrast_loss

    def forward(self, device: int, ids: torch.Tensor, is_train: bool = True):
        (
            source_types,
            source_images,
            target_types,
            target_images,
            contrast_matrix,
        ) = (
            self.dataset.batch_get_train_device(device, ids)
            if is_train
            else self.dataset.batch_get_valid_device(device, ids)
        )

        # latent: cls_token + patch_token
        if self.use_contrast_loss:
            pred, reconstruction_loss, _mask, contrast_loss = self.model.forward_contrast_loss(
                source_types, source_images, target_types, target_images, contrast_matrix
            )
            loss = reconstruction_loss + contrast_loss
        else:
            (pred, loss, _mask) = self.model(source_types, source_images, target_types, target_images)

        if device == 0:
            global global_last_result
            global_last_result = (source_images[0], target_images[0], pred[0].detach())

        return (loss,)


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

    note: str = field(
        default="",
        metadata={"help": ("训练备注")},
    )

    finetune_index: int = field(
        default=-1,
        metadata={"help": ("加载微并调数据集")},
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": ("冻结vit参数")},
    )
    use_contrast_loss: bool = field(
        default=False,
        metadata={"help": ("是否使用对比学习loss")},
    )


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


class ModelFactory:
    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments) -> None:
        self.model_args = model_args
        self.training_args = training_args
        self.latest_model_path = f"{training_args.output_dir}/latest"

    def create_model(self) -> MMaeVit:
        model = model_mmae.sam_mae()

        if self.model_args.init_model:
            pass

        model = model.to(self.training_args.device)
        return model

    def load_optimizer(self, optimizer: Optimizer, optimizer_path: str | None = None):
        if optimizer is None:
            return
        if optimizer_path is None or optimizer_path == "":
            optimizer_path = f"{self.latest_model_path}/optimizer.pt"
        if not os.path.exists(optimizer_path):
            return
        print(f"loading optimizer state: {optimizer_path}")
        optimizer.load_state_dict(torch.load(optimizer_path))

    def load_model(self, model_path: str | None = None) -> MMaeVit:
        model = self.create_model()

        if model_path is None or model_path == "":
            model_path = f"{self.latest_model_path}/model.pt"

        if not os.path.exists(model_path):
            print(f"model not found: {model_path}")
            return model

        print(f"loading model: {model_path}")
        param = torch.load(model_path, map_location=self.training_args.device)
        model.load_state_dict(param, strict=False)
        return model

    def save_model(self, model: MMaeVit, optimizer: Optimizer, path, link_latest: bool = True):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), f"{path}/model.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), f"{path}/optimizer.pt")
        if link_latest:
            relative_symlink_file(path, self.latest_model_path)

    def freeze_vit(self, model: MMaeVit, is_freeze=True):
        model.vit.requires_grad_(not is_freeze)
        model.vit.image_types.requires_grad_(True)

    def init_model_from_mae(self, model: MMaeVit, mae_state_dict: dict[str, Any]):
        bs, n = model.decoder.image_types.cls_embedding.shape[:2]
        decoder_cls_embedding: Tensor = mae_state_dict["decoder.mask_token"]
        decoder_cls_embedding = decoder_cls_embedding.repeat(bs, n, 1)

        bs, n = model.vit.image_types.shape[:2]
        vit_cls_embedding: Tensor = mae_state_dict["vit.cls_token"]
        vit_cls_embedding = vit_cls_embedding.repeat(bs, n, 1)

        decoder_pos_embed = mae_state_dict["decoder.decoder_pos_embed"]  # shape不同
        bs, _n, h = decoder_pos_embed.shape
        decoder_pos_embed = torch.cat([torch.zeros(bs, 1, h), decoder_pos_embed], dim=1)
        mae_state_dict["decoder.decoder_pos_embed"] = decoder_pos_embed

        mae_state_dict["decoder.image_types.cls_embedding"] = decoder_cls_embedding
        mae_state_dict["vit.image_types"] = vit_cls_embedding

        model.load_state_dict(mae_state_dict, strict=False)


def train(args: List[str] | None):
    model_args, training_args = get_args(args)

    run_name = training_args.run_name
    device = training_args.device

    model_factory = ModelFactory(model_args, training_args)

    model_path = None
    if model_args.finetune_index >= 0:
        model_path = f"{training_args.output_dir}/finetune-{model_args.finetune_index}/model.pt"
        if not os.path.exists(model_path):
            model_path = None

    model = model_factory.load_model(model_path)

    if model_args.init_model:
        mae_model_state = dict(
            torch.load(
                f"{training_args.output_dir}/mae-model.pt",
                map_location="cpu",
            )
        )
        model_factory.init_model_from_mae(model, mae_model_state)
        model_factory.save_model(model, None, f"{training_args.output_dir}/epoch-0000")
        print("model initialize completed!")
        return

    dataset = MmaeDatasetCompat(load_finetune_index=model_args.finetune_index)

    train_model = MmaeTrainModel(model, dataset, model_args.use_contrast_loss)
    dp_model = CustomDataParallel(train_model).cuda()
    device_list = range(torch.cuda.device_count())
    train_phase = [True] * torch.cuda.device_count()
    valid_phase = [False] * torch.cuda.device_count()

    scaler = GradScaler()
    optimizer = AdamW(dp_model.parameters(), lr=training_args.learning_rate, eps=1e-5)

    optimizer_path = None
    if model_args.finetune_index >= 0:
        optimizer_path = f"{training_args.output_dir}/finetune-{model_args.finetune_index}/optimizer.pt"
        if not os.path.exists(optimizer_path):
            optimizer_path = None
    model_factory.load_optimizer(optimizer, optimizer_path)

    if model_args.freeze_vit:
        model_factory.freeze_vit(model, True)
        if model_args.note == "":
            model_args.note = "freeze vit parameter"

    log_writer = SummaryWriter(log_dir=f"{training_args.output_dir}/logs")

    if model_args.finetune_index >= 0:
        log_writer.add_text(
            f"finetune/{model_args.finetune_index}/description",
            f"""对全部数据集进行微调, 加载微调数据集cmri-{model_args.finetune_index}. """,
            global_step=model_args.start_epoch,
        )
        log_writer.add_text(f"finetune/{model_args.finetune_index}/cmd_args", str(args), model_args.start_epoch)
    else:
        log_writer.add_text(
            "description",
            f"""对全部数据集进行训练目, 使用多模态数据集进行MAE训练
        """,
        )
        log_writer.add_text("cmd_args", str(args), model_args.start_epoch)

    if model_args.note is not None and model_args.note != "":
        log_writer.add_text("note", model_args.note, model_args.start_epoch)
        if model_args.finetune_index >= 0:
            log_writer.add_text(f"finetune/{model_args.finetune_index}/note", model_args.note, model_args.start_epoch)

    num_epochs = int(training_args.num_train_epochs)
    for epoch in range(model_args.start_epoch, num_epochs + 1):
        epoch_losses = []
        dp_model.train()
        mini_batch_id = 0
        MINI_BATCH_SIZE = training_args.gradient_accumulation_steps
        for batch in tqdm(dataset.shuffer_train_id(training_args.train_batch_size)):
            with autocast():
                (loss,) = dp_model(device_list, batch, train_phase)
                loss = loss.mean()
                if loss.isnan():
                    raise "loss is nan"
                scaler.scale(loss).backward()
                mini_batch_id += 1
                if mini_batch_id % MINI_BATCH_SIZE == 0:
                    mini_batch_id = 0
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            epoch_losses.append(loss.item())

        if mini_batch_id % MINI_BATCH_SIZE != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if model_args.finetune_index < 0:
            print(f"epoch: {epoch:04}, loss: {mean(epoch_losses)}")
        else:
            print(f"finetune epoch: {epoch:04}, loss: {mean(epoch_losses)}")

        if model_args.finetune_index >= 0:
            log_writer.add_text(f"finetune/{model_args.finetune_index}", f"train loss: {mean(epoch_losses)}", epoch)
            model_factory.save_model(
                model, optimizer, f"{training_args.output_dir}/finetune-{model_args.finetune_index}", link_latest=False
            )
        else:
            if epoch % 10 == 0:
                model_factory.save_model(model, optimizer, f"{training_args.output_dir}/epoch-{epoch:04}")
            else:
                model_factory.save_model(model, optimizer, f"{training_args.output_dir}/temp-{epoch%10}")

        source_image, target_image, pred_image = global_last_result
        log_writer.add_image("train/image_source", source_image, epoch, dataformats="CHW")
        log_writer.add_image("train/image_target", target_image, epoch, dataformats="CHW")
        with torch.no_grad():
            pred_image = model.vit.unpatchify(pred_image.unsqueeze(0))
        log_writer.add_image("train/image_pred", pred_image[0], epoch, dataformats="CHW")
        log_writer.add_scalar("train/loss", mean(epoch_losses), epoch)

        dp_model.eval()
        epoch_losses = []
        for batch in dataset.shuffer_valid_id(training_args.eval_batch_size):
            with autocast(), torch.no_grad():
                (loss,) = dp_model(device_list, batch, valid_phase)
                loss = loss.mean()
            epoch_losses.append(loss.item())

        source_image, target_image, pred_image = global_last_result
        print(f"epoch: {epoch:04}, valid loss: {mean(epoch_losses)}")
        log_writer.add_image("valid/image_source", source_image, epoch, dataformats="CHW")
        log_writer.add_image("valid/image_target", target_image, epoch, dataformats="CHW")
        with torch.no_grad():
            pred_image = model.vit.unpatchify(pred_image.unsqueeze(0))
        log_writer.add_image("valid/image_pred", pred_image[0], epoch, dataformats="CHW")
        log_writer.add_scalar("valid/loss", mean(epoch_losses), epoch)


def run(args: List[str] | None):
    # torch.multiprocessing.set_start_method("spawn")
    print("args:", args)

    use_cudnn()
    train(args)
