from dataclasses import dataclass, field
import os
from typing import List
from mos.datasets.common.dataset_utils import locate_dataset_base_url
import mos.models.mae.model_mae as model_mae
from mos.models.mae.model_mae import MaeVit
import torch
import random
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW, Optimizer
from tqdm import tqdm
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
from mos.utils.files import relative_symlink_file

from mos.utils.model_utils import use_cudnn
from .mae_dataset_compat import MaeDatasetCompat
from typing import Any, List, Optional, Sequence, Tuple, Union
from torch import nn, Tensor


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


global_last_result = (None, None)


def calc_reconstruction_loss(latent1: Tensor, latent2: Tensor) -> Tensor:
    latent1 = latent1 / latent1.norm(dim=-1, keepdim=True)
    latent2 = latent2 / latent2.norm(dim=-1, keepdim=True)

    similarity = latent1 @ latent2.T

    bs = latent1.shape[0]
    scale = torch.ones((bs, bs), dtype=torch.float, device=latent1.device)
    # instance norm为F范数, dim为768, 大致cos相似度极端情况下缩放值为 sqrt(768) = 27 才能归一化到1
    scale_diag = torch.eye(bs, dtype=torch.float, device=latent1.device)
    target_similarity = scale_diag

    scale += scale_diag
    similarity *= scale

    contrast_loss = (similarity - target_similarity) ** 2
    contrast_loss = torch.mean(contrast_loss, 1)

    return contrast_loss


class MaeTrainModel(nn.Module):
    def __init__(
        self,
        model: MaeVit,
        dataset: MaeDatasetCompat,
        use_contrast_loss: bool = True,
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.use_contrast_loss = use_contrast_loss

    def forward(self, device: int, ids: torch.Tensor, is_train: bool = True):
        if is_train:
            images = self.dataset.batch_get_train_device(device, ids)
        else:
            images = self.dataset.batch_get_valid_device(device, ids)

        # with autocast(enabled=False):
        pred1, reconstruction_loss1, _mask1, latent1 = self.model(images)

        loss = reconstruction_loss1
        if self.use_contrast_loss:
            _pred2, reconstruction_loss2, _mask2, latent2 = self.model(images)
            latent1, latent2 = latent1[:, 0], latent2[:, 0]
            loss = (reconstruction_loss1 + reconstruction_loss2) / 2
        else:
            latent1 = latent1[:, 0]
            latent2 = None

        if device == 0:
            global global_last_result
            global_last_result = (images[0], pred1[0].detach())

        return (loss, latent1, latent2)


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

    def create_model(self) -> MaeVit:
        model = model_mae.sam_mae()

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

    def load_model(self, model_path: str | None = None) -> MaeVit:
        model = self.create_model()

        if model_path is None or model_path == "":
            model_path = f"{self.latest_model_path}/model.pt"

        if not os.path.exists(model_path):
            print(f"model not found: {model_path}")
            return model

        print(f"loading model: {model_path}")
        param = torch.load(model_path, map_location=self.training_args.device)
        model.load_state_dict(param)
        return model

    def save_model(self, model, optimizer: Optimizer, path, link_latest: bool = True):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), f"{path}/model.pt")
        torch.save(optimizer.state_dict(), f"{path}/optimizer.pt")
        if link_latest:
            relative_symlink_file(path, self.latest_model_path)


def train(args: List[str] | None):
    model_args, training_args = get_args(args)

    run_name = training_args.run_name
    device = training_args.device

    dataset = MaeDatasetCompat(load_finetune_index=model_args.finetune_index)

    model_factory = ModelFactory(model_args, training_args)

    model_path = None
    if model_args.finetune_index >= 0:
        model_path = f"{training_args.output_dir}/finetune-{model_args.finetune_index}/model.pt"
        if not os.path.exists(model_path):
            model_path = None

    model = model_factory.load_model(model_path)

    train_model = MaeTrainModel(model, dataset, model_args.use_contrast_loss)
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
            f"""对全部数据集进行训练目, 进行MAE训练, 358358 train images, 30527 valid images. 其中CMRI和VarDA数据集作为验证集使用
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
        reconstruction_losses = []
        contrast_losses = []
        dp_model.train()
        for batch in tqdm(dataset.shuffer_train_id(training_args.train_batch_size)):
            with autocast():
                (reconstruction_loss, latent1, latent2) = dp_model(device_list, batch, train_phase)

                # 计算对比学习的特征loss
                if model_args.use_contrast_loss:
                    contrast_loss = calc_reconstruction_loss(latent1, latent2)
                else:
                    contrast_loss = None

                reconstruction_loss = reconstruction_loss.mean()

                loss = reconstruction_loss

                if contrast_loss is not None:
                    contrast_loss = contrast_loss.mean()
                    loss += contrast_loss

                if loss.isnan():
                    raise "loss is nan"
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_losses.append(loss.item())
            reconstruction_losses.append(reconstruction_loss.item())
            if contrast_loss is not None:
                contrast_losses.append(contrast_loss.item())
        if model_args.finetune_index < 0:
            print(f"epoch: {epoch:04}, loss: {mean(epoch_losses)}, reconstruction_loss: {mean(reconstruction_losses)}")
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

        source_image, pred_image = global_last_result
        log_writer.add_image("train/image_source", source_image, epoch, dataformats="CHW")
        with torch.no_grad():
            pred_image = model.vit.unpatchify(pred_image.unsqueeze(0))
        log_writer.add_image("train/image_pred", pred_image[0], epoch, dataformats="CHW")
        log_writer.add_scalar("train/loss", mean(epoch_losses), epoch)
        log_writer.add_scalar("train/loss/reconstruction", mean(reconstruction_losses), epoch)
        if len(contrast_losses) > 0:
            log_writer.add_scalar("train/loss/contrast", mean(contrast_losses), epoch)

        dp_model.eval()
        epoch_losses = []
        reconstruction_losses = []
        contrast_losses = []
        for batch in dataset.shuffer_valid_id(training_args.eval_batch_size):
            with autocast(), torch.no_grad():
                (reconstruction_loss, latent1, latent2) = dp_model(device_list, batch, valid_phase)

                # 计算对比学习的特征loss
                if model_args.use_contrast_loss:
                    contrast_loss = calc_reconstruction_loss(latent1, latent2)
                else:
                    contrast_loss = None

                # print("hello, world, ", func(fun2("yes, yess")))
                reconstruction_loss = reconstruction_loss.mean()

                loss = reconstruction_loss

            if contrast_loss is not None:
                contrast_loss = contrast_loss.mean()
                loss += contrast_loss
                contrast_losses.append(contrast_loss.item())

            epoch_losses.append(loss.item())
            reconstruction_losses.append(reconstruction_loss.item())

        source_image, pred_image = global_last_result
        print(f"epoch: {epoch:04}, valid loss: {mean(epoch_losses)}")
        log_writer.add_image("valid/image_source", source_image, epoch, dataformats="CHW")
        with torch.no_grad():
            pred_image = model.vit.unpatchify(pred_image.unsqueeze(0))
        log_writer.add_image("valid/image_pred", pred_image[0], epoch, dataformats="CHW")
        log_writer.add_scalar("valid/loss", mean(epoch_losses), epoch)
        log_writer.add_scalar("valid/loss/reconstruction", mean(reconstruction_losses), epoch)
        if len(contrast_losses) > 0:
            log_writer.add_scalar("valid/loss/contrast", mean(contrast_losses), epoch)


def run(args: List[str] | None):
    # torch.multiprocessing.set_start_method("spawn")
    print("args:", args)

    use_cudnn()
    train(args)
