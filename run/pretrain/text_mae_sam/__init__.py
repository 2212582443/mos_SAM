import os, torch
from dataclasses import dataclass, field
from statistics import mean
from typing import List

from torch.cuda.amp import autocast, GradScaler
from torch import Tensor
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments

from mos.losses.hausdorff import avg_hausdorff_distance, hausdorff_distance

from .batch_soft_dice import DiceScore
from .model.sam_model_simple import SamModelSimple
from .model_factory import ModelFactory

from .sam_dataset_compat import SAMDatasetCompat
from .sam_dataset_compat_simple import SAMDatasetCompatSimple


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

    simple_dataset: int = field(
        default=0,
        metadata={"help": ("使用简化的数据集的程度, 0-不简化, 1-不使用组合token, 2-只使用cmri数据集")},
    )

    valid_dataset_id: int = field(
        default=0,
        metadata={"help": ("交叉验证集的数据集id")},
    )

    save_interval: int = field(
        default=10,
        metadata={"help": ("保存模型的间隔")},
    )
    dataset: str = field(
        default="all-label",
        metadata={"help": ("数据集名称")},
    )
    note: str = field(
        default="",
        metadata={"help": ("训练备注")},
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": ("冻结vit参数")},
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


def train(args: List[str] | None):
    model_args, training_args = get_args(args)

    run_name = training_args.run_name
    device = training_args.device

    os.makedirs(training_args.output_dir, exist_ok=True)

    log_writer = SummaryWriter(log_dir=f"{training_args.output_dir}/logs")
    simple_dataset_desc = ["使用组合的token作为训练数据", "只使用简单的tonken作为训练数据", "只使用cmri数据集+简单的tonken作为训练数据"]
    log_writer.add_text(
        "description",
        f"""text+mae+sam 模型, 从预训练的MAE模型微调
1. {'冻结MAE模型' if model_args.freeze_vit else '打开MAE模型训练'};
2. 数据使用全部的数据集{model_args.dataset}, 仅使用其中带label部分, {simple_dataset_desc[model_args.simple_dataset]};
3. 使用提示文本作为输入, 文本根据label组合随机生成;
4. CMRI-{model_args.valid_dataset_id}作为验证集;
5. 验证集只显示最差的30个样本;
""",
        0,
    )
    log_writer.add_text("cmd_args", str(args), model_args.start_epoch)
    if model_args.note is not None and model_args.note != "":
        log_writer.add_text("note", model_args.note, model_args.start_epoch)

    model_factory = ModelFactory(device, run_name)

    model: SamModelSimple = model_factory.load_model(f"{training_args.output_dir}/latest")

    if model_args.init_model:
        mae_model_state = dict(
            torch.load(
                f"{training_args.output_dir}/mae-model.pt",
                map_location=device,
            )
        )
        model_factory.init_model_from_mae(model, mae_model_state)
        model_factory.save_model(model, None, f"{training_args.output_dir}/0000")
        print("model initialize completed!")
        return

    if model_args.simple_dataset == 0:
        new_dataset = SAMDatasetCompat(
            base_path=f".cache/dataset/text-mae-sam-dataset/dataset-{model_args.dataset}-{model_args.valid_dataset_id}.pt",
            device=device,
        )
    else:
        new_dataset = SAMDatasetCompatSimple(
            base_path=f".cache/dataset/text-mae-sam-dataset/dataset-{model_args.dataset}-{model_args.valid_dataset_id}.pt",
            device=device,
            only_cmri_dataset=model_args.simple_dataset == 2,
        )

    if model_args.freeze_vit:
        model_factory.freeze_mae_encoder(model, True)

    calc_seg_loss = torch.nn.MSELoss()
    calc_dice_score = DiceScore()

    scaler = GradScaler()

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, eps=1e-5)
    model_factory.load_optimizer(optimizer)

    # 使用cuda graph
    # 1. prepare
    batch = new_dataset.shuffer_train_id(training_args.train_batch_size)[0]
    static_image, static_labeled_segment, static_token = new_dataset.batch_get_train(batch)
    static_first_pred_masks, static_dice_loss, static_loss = None, None, None
    # 2. warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                static_first_pred_masks = model.forward_image(static_image, static_token)
                static_first_pred_masks = static_first_pred_masks[:, 0, 0, :, :]
                static_loss = calc_seg_loss(static_first_pred_masks, static_labeled_segment)
            with torch.no_grad():
                static_dice_loss = calc_dice_score(static_first_pred_masks, static_labeled_segment)
            scaler.scale(static_loss).backward()
            scaler.step(optimizer)
            scaler.update()
    torch.cuda.current_stream().wait_stream(s)
    # 3. capture
    g = torch.cuda.CUDAGraph()
    # Sets grads to None before capture, so backward() will create
    # .grad attributes with allocations from the graph's private pool
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        with autocast(enabled=True):
            static_first_pred_masks = model.forward_image(static_image, static_token)
            static_first_pred_masks = static_first_pred_masks[:, 0, 0, :, :]
            static_loss = calc_seg_loss(static_first_pred_masks, static_labeled_segment)
        with torch.no_grad():
            static_dice_loss = calc_dice_score(static_first_pred_masks, static_labeled_segment)
        scaler.scale(static_loss).backward()
        # don't capture scaler.step(optimizer) or scaler.update()
        # scaler.step(optimizer)
        # scaler.update()

    for epoch in range(int(model_args.start_epoch), int(training_args.num_train_epochs) + 1):
        epoch_losses = []
        dice_losses = []
        model.train()
        for batch in tqdm(new_dataset.shuffer_train_id(training_args.train_batch_size)):
            image, labeled_segment, token = new_dataset.batch_get_train(batch)

            if True:  # 4. use graph
                static_image.copy_(image)
                static_labeled_segment.copy_(labeled_segment)
                static_token.copy_(token)
                # replay() includes forward, backward, and step.
                # You don't even need to call optimizer.zero_grad() between iterations
                # because the captured backward refills static .grad tensors in place.
                g.replay()
                # Params have been updated. static_image, static_labeled_segment, static_token, and .grad
                # attributes hold values from computing on this iteration's data.
                # Runs scaler.step and scaler.update eagerly
                scaler.step(optimizer)
                scaler.update()
                # print(static_first_pred_masks.sum(), static_loss.sum(), static_dice_loss.sum())
                epoch_losses.append(static_loss.item())
                dice_losses.append(static_dice_loss.item())

                first_pred_masks = static_first_pred_masks
            else:
                with autocast(enabled=True):
                    pred_masks = model.forward_image(image, token)
                    # (bs, h, w)
                    first_pred_masks = pred_masks[:, 0, 0, :, :]

                    seg_loss: torch.Tensor = calc_seg_loss(first_pred_masks, labeled_segment)

                    with torch.no_grad():
                        dice_loss = calc_dice_score(first_pred_masks, labeled_segment)
                        dice_losses.append(dice_loss.item())

                    loss = seg_loss
                    epoch_losses.append(loss.item())
                    loss = loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        print(f"epoch: {epoch}, mean loss: {mean(epoch_losses)}, dice: {mean(dice_losses)}")
        h = image.shape[2]
        image = image[:10, 0, :, :].permute(1, 0, 2).reshape(h, -1)
        labeled_segment = labeled_segment[:10, :, :].permute(1, 0, 2).reshape(h, -1)
        first_pred_masks = first_pred_masks[:10, :, :].permute(1, 0, 2).reshape(h, -1)

        segment = image.repeat(3, 1, 1)
        segment[0] += labeled_segment
        segment = segment.clamp(0, 1)

        log_writer.add_image("train/image", image, epoch, dataformats="HW")
        log_writer.add_image("train/segment_label", segment, epoch, dataformats="CHW")
        log_writer.add_image("train/segment_pred", first_pred_masks, epoch, dataformats="HW")

        log_writer.add_scalar("train/loss", mean(epoch_losses), epoch)
        log_writer.add_scalar("train/dice", mean(dice_losses), epoch)

        if epoch % model_args.save_interval == 0:
            model_factory.save_model(model, optimizer, f"{training_args.output_dir}/{epoch:04}")
        else:
            model_factory.save_model(model, optimizer, f"{training_args.output_dir}/temp-{epoch % 10}")

        print("evaluating...")
        dice_losses = []
        avg_hausdorff_dist_losses = []
        max_hausdorff_dist_losses = []
        model.eval()
        KEEP_WORST_COUNT = 30
        last_image_list: list[tuple[Tensor, Tensor, Tensor]] = []
        last_dice_list = []
        with torch.no_grad(), autocast(enabled=True):
            for batch in new_dataset.shuffer_valid_id(training_args.eval_batch_size):
                image, labeled_segment, token = new_dataset.batch_get_valid(batch)

                pred_masks = model.forward_image(image, token)

                # (bs, h, w)
                first_pred_masks = pred_masks[:, 0, 0, :, :]

                dice_loss = calc_dice_score(first_pred_masks, labeled_segment, reduce=False)
                dice_losses.extend(dice_loss.tolist())
                avg_hausdorff_dist, max_hausdorff_dist = hausdorff_distance(
                    first_pred_masks, labeled_segment, reduce=False
                )
                avg_hausdorff_dist_losses.extend(avg_hausdorff_dist.tolist())
                max_hausdorff_dist_losses.extend(max_hausdorff_dist.tolist())

                # 求最差的KEEP_WORST_COUNT个
                dice_loss, dice_loss_index = torch.topk(dice_loss, KEEP_WORST_COUNT, largest=False)
                last_image_list.append(
                    (
                        image[dice_loss_index],
                        labeled_segment[dice_loss_index],
                        first_pred_masks[dice_loss_index],
                    )
                )
                last_dice_list.append(dice_loss)

                if len(last_image_list) > 1:  # 列表只保留最差的KEEP_WORST_COUNT个
                    last_image_list = (
                        torch.cat([v[0] for v in last_image_list], dim=0),
                        torch.cat([v[1] for v in last_image_list], dim=0),
                        torch.cat([v[2] for v in last_image_list], dim=0),
                    )

                    last_dice_list = torch.cat(last_dice_list, dim=0)
                    last_dice_list, dice_loss_index = torch.topk(last_dice_list, KEEP_WORST_COUNT, largest=False)

                    last_image_list = [
                        (
                            last_image_list[0][dice_loss_index],
                            last_image_list[1][dice_loss_index],
                            last_image_list[2][dice_loss_index],
                        )
                    ]

        # 只选dice取最差的KEEP_WORST_COUNT个
        image, labeled_segment, first_pred_masks = last_image_list[0]

        h, w = image.shape[2:]
        image = image.reshape(3, 10, h, w).permute(0, 2, 1, 3).reshape(3 * h, -1)
        labeled_segment = labeled_segment.reshape(3, 10, h, w).permute(0, 2, 1, 3).reshape(3 * h, -1)
        first_pred_masks = first_pred_masks.reshape(3, 10, h, w).permute(0, 2, 1, 3).reshape(3 * h, -1)

        segment = image.repeat(3, 1, 1)
        segment[0] += labeled_segment
        segment = segment.clamp(0, 1)

        log_writer.add_image("valid/image", image, epoch, dataformats="HW")
        log_writer.add_image("valid/segment_label", segment, epoch, dataformats="CHW")
        log_writer.add_image("valid/segment_pred", first_pred_masks, epoch, dataformats="HW")

        # 计算dice的均值和方差
        dice_std, dice_mean = torch.std_mean(torch.tensor(dice_losses))
        log_writer.add_scalar("valid/dice", dice_mean, epoch)
        log_writer.add_scalar("valid/dice_std", dice_std, epoch)

        # 计算hausdorff(均值)的均值和方差
        hausdorff_std, hausdorff_mean = torch.std_mean(torch.tensor(avg_hausdorff_dist_losses))
        log_writer.add_scalar("valid/haudsorff", hausdorff_mean, epoch)
        log_writer.add_scalar("valid/haudsorff_std", hausdorff_std, epoch)
        # 计算hausdorff(最大值)的均值和方差
        hausdorffmax_std, hausdorffmax_mean = torch.std_mean(torch.tensor(max_hausdorff_dist_losses))
        log_writer.add_scalar("valid/haudsorffmax/mean", hausdorffmax_mean, epoch)
        log_writer.add_scalar("valid/haudsorffmax/std", hausdorffmax_std, epoch)

        print(
            f"validate epoch-{epoch:04} dice: {dice_mean.item():.04}/{dice_std.item():.04}, haudsorff: {hausdorff_mean.item():.04}/{hausdorff_std.item():.04}, haudsorff max: {hausdorffmax_mean.item():.04}/{hausdorff_std.item():.04} count:{len(dice_losses)}"
        )

    model_factory.save_model(model, optimizer, f"{training_args.output_dir}/{epoch:04}")
    log_writer.close()


def run(args: List[str] | None):
    # torch.multiprocessing.set_start_method("spawn")
    print("args:", args)

    train(args)
