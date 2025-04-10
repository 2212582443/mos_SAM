import os, glob

import torch
from torch.optim import Optimizer
from transformers import TrainingArguments

from mos.utils.files import relative_symlink_file
from run.baseline.model_arguments import ModelArguments

from .models import *
from .model_wrapper import (
    CompressDbWrapper2d,
    ModelWrapper2d,
    ModelWrapper3d,
    SamWrapper2d,
    SamWrapper3d,
    TxtSamWrapper2d,
    TxtSamWrapper3d,
    EfficientSamWrapper3d,
    EfficientSamWrapper2d,
    SamusWrapper2d,
    SamusWrapper3d,
    SammedWrapper3d,
    SammedCNNWrapper3d,
)

MODEL_LIST = {
    "segnet2d": lambda factory: ModelWrapper2d(
        SegNet2d(1, factory.model_args.label_count),
        label_count=factory.model_args.label_count,
    ),
    "segnet3d": lambda factory: ModelWrapper3d(SegNet3d(1, factory.model_args.label_count)),
    "unet2d": lambda factory: ModelWrapper2d(
        UNet2d(1, factory.model_args.label_count),
        label_count=factory.model_args.label_count,
    ),
    "unet3d": lambda factory: ModelWrapper3d(UNet3d(1, factory.model_args.label_count)),
    "unetr3d": lambda factory: ModelWrapper3d(
        UNetr3d(
            img_shape=factory.model_args.get_dataset_crop_size(),
            output_dim=factory.model_args.label_count,
        )
    ),
    "unetr2d": lambda factory: ModelWrapper2d(
        UNetr2d(
            img_shape=factory.model_args.get_dataset_crop_size(),
            output_dim=factory.model_args.label_count,
        ),
        label_count=factory.model_args.label_count,
    ),
    "resunet2d": lambda factory: ModelWrapper2d(
        ResUnet2d(out_channels=factory.model_args.label_count),
        label_count=factory.model_args.label_count,
    ),
    "resunet3d": lambda factory: ModelWrapper3d(ResUnet3d(out_channels=factory.model_args.label_count)),
    "resunet++2d": lambda factory: ModelWrapper2d(
        ResUnetPlusPlus2d(out_channels=factory.model_args.label_count),
        label_count=factory.model_args.label_count,
    ),
    "resunet++3d": lambda factory: ModelWrapper3d(ResUnetPlusPlus3d(out_channels=factory.model_args.label_count)),
    "nnformer3d": lambda factory: ModelWrapper3d(
        nnFormer3d(
            crop_size=factory.model_args.get_dataset_crop_size(),
            num_classes=factory.model_args.label_count,
        )
    ),
    "txtsam3d": lambda factory: TxtSamWrapper3d(
        factory.training_args.device, model_args=factory.model_args, args=factory.other_args
    ),
    "trainds2d": lambda factory: CompressDbWrapper2d(
        factory.training_args.device, factory.model_args.label_count, args=factory.other_args
    ),
    "txtsam2d": lambda factory: TxtSamWrapper2d(
        factory.training_args.device, model_args=factory.model_args, args=factory.other_args
    ),
    "sam3d": lambda factory: SamWrapper3d(factory.training_args.device, factory.model_args),
    "sam2d": lambda factory: SamWrapper2d(factory.training_args.device, factory.model_args),
    "effsamti3d": lambda factory: EfficientSamWrapper3d(
        factory.training_args.device, build_efficient_sam_vitt(), factory.model_args
    ),
    "effsamti2d": lambda factory: EfficientSamWrapper2d(
        factory.training_args.device, build_efficient_sam_vitt(), factory.model_args
    ),
    "effsams3d": lambda factory: EfficientSamWrapper3d(
        factory.training_args.device, build_efficient_sam_vits(), factory.model_args
    ),
    "effsams2d": lambda factory: EfficientSamWrapper2d(
        factory.training_args.device, build_efficient_sam_vits(), factory.model_args
    ),
    "samus3d": lambda factory: SamusWrapper3d(factory.training_args.device, factory.model_args),
    "samus2d": lambda factory: SamusWrapper2d(factory.training_args.device, factory.model_args),
    "sammed3d": lambda factory: SammedWrapper3d(factory.training_args.device, factory.model_args, factory.other_args),
    "sammedcnn3d": lambda factory: SammedCNNWrapper3d(factory.training_args.device, factory.model_args),
}


class ModelFactory:
    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments, other_args: list[str]) -> None:
        self.model_args = model_args
        self.training_args = training_args
        self.train_dataset = model_args.train_dataset
        self.latest_model_path = f"{self.get_output_dir()}/latest"
        self.other_args = other_args

        os.makedirs(self.get_output_dir(), exist_ok=True)

    def get_output_dir(self):
        aux = self.model_args.aux_dataset
        if aux is None or len(aux) == 0:
            aux = "none"

        baseline = "baseline" if self.train_dataset == "cmri712-0" else f"baseline-{self.train_dataset}"

        dir_postfix = self.model_args.dir_postfix
        if dir_postfix is None or len(dir_postfix) == 0:
            dir_postfix = ""

        output_dir = self.training_args.output_dir.format(baseline=baseline)

        if self.model_args.tag is not None and len(self.model_args.tag) > 0:
            tag = f"[{self.model_args.tag}]"
        else:
            tag = ""

        return f"{output_dir}{tag}-{'2d' if self.model_args.is2d() else '3d'}-{aux}{dir_postfix}"

    def create_model(self):
        model = f'{self.model_args.model}{"2d" if self.model_args.is2d() else "3d"}'
        model = MODEL_LIST[model]
        model = model(self)

        model = model.to(self.training_args.device)
        return model

    def load_optimizer(self, optimizer: Optimizer, optimizer_path: str | None = None, pretrain_model_path: str = None):
        if optimizer is None:
            return

        if optimizer_path is None or optimizer_path == "":
            optimizer_path = self.latest_model_path

        optimizer_path = f"{optimizer_path}/optimizer.pt"

        if not os.path.exists(optimizer_path) and pretrain_model_path is not None and len(pretrain_model_path) > 0:
            optimizer_path = f"{pretrain_model_path}/optimizer.pt"
            if not os.path.exists(optimizer_path):
                raise f"optimizer {optimizer_path} NOT found!"
            else:
                print(f"load from pretrain: {optimizer_path}")

        if not os.path.exists(optimizer_path):
            return

        print(f"loading optimizer state: {optimizer_path}")
        optimizer.load_state_dict(torch.load(optimizer_path))

    def load_model(
        self,
        model_path: str | None = None,
        pretrain_model_path: str = None,
        strict: bool = True,
    ):
        model_wrapper = self.create_model()

        if model_path is None or model_path == "":
            model_path = f"{self.latest_model_path}/model.pt"
            if not os.path.exists(model_path):  # 如果latest不存在，那么就找最新的epoch
                epoches_models = glob.glob(f"epoch-*/model.pt", root_dir=self.get_output_dir())
                epoches_models.sort(reverse=True)
                if len(epoches_models) > 0:
                    model_path = f"{self.get_output_dir()}/{epoches_models[0]}"
        else:
            model_path = f"{model_path}/model.pt"

        # 如果没开始训练，尝试加载预训练模型
        if not os.path.exists(model_path) and pretrain_model_path is not None and len(pretrain_model_path) > 0:
            model_path = f"{pretrain_model_path}/model.pt"
            if not os.path.exists(model_path):
                raise f"model not found: {model_path}"
            else:
                print(f"load from pretrain:{model_path}")

        if os.path.exists(model_path):
            print(f"loading model: {model_path}")
            param = torch.load(model_path, map_location=self.training_args.device)
            model_wrapper.model.load_state_dict(param, strict)
        else:
            print(f"model not found: {model_path}")

        if hasattr(model_wrapper, "on_model_loaded"):
            model_wrapper.on_model_loaded()

        return model_wrapper

    def save_model(self, model_wrapper, optimizer: Optimizer, path, link_latest: bool = True):
        if not path.startswith("/"):
            path = f"{self.get_output_dir()}/{path}"

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(model_wrapper.model.state_dict(), f"{path}/model.pt")
        torch.save(optimizer.state_dict(), f"{path}/optimizer.pt")
        if link_latest:
            relative_symlink_file(path, self.latest_model_path)

    def save_epoch_model(self, epoch, model_wrapper, optimizer: Optimizer):
        if epoch % 10 == 0:
            self.save_model(model_wrapper, optimizer, f"epoch-{epoch:04}")
        else:
            temp_path = f"{self.get_output_dir()}/temp-{epoch%10}"
            # check if temp path is symlink
            if os.path.islink(temp_path):
                os.remove(temp_path)
            self.save_model(model_wrapper, optimizer, f"temp-{epoch%10}")

    def persist_model(self, epoch: int):
        if epoch % 10 == 0:
            return
        target_path = f"{self.get_output_dir()}/epoch-{epoch:04}"
        if os.path.exists(target_path):
            return
        src_path = f"{self.get_output_dir()}/temp-{epoch%10}"
        if not os.path.exists(src_path):
            print("temp model not found:", src_path)
            return
        os.rename(src_path, target_path)
        relative_symlink_file(target_path, src_path)

    def save_epoch_metric(self, epoch, stage, metrics: dict[str, list[float]]):
        model_path = f"epoch-{epoch:04}" if epoch % 10 == 0 else f"temp-{epoch%10}"
        model_path = f"{self.get_output_dir()}/{model_path}/{stage}-metrics.pt"

        metrics = {k: torch.cat(v, 0) if k.startswith("raw_") else torch.tensor(v) for k, v in metrics.items()}
        torch.save(metrics, model_path)

    def get_start_epoch_number(self) -> int:  # 返回接下来训练的开始epoch
        out_dir = self.get_output_dir()
        last_tmp = os.readlink(f"{out_dir}/latest") if os.path.exists(f"{out_dir}/latest") else "temp-0"
        if last_tmp.startswith("temp-"):
            last_tmp = int(last_tmp[5:])
        else:
            last_tmp = 0
        epoches = glob.glob("epoch-*", root_dir=out_dir)
        epoches = list(map(lambda x: int(x[6:]), epoches))
        start_epoch = max(epoches) if len(epoches) > 0 else 0
        start_epoch = (start_epoch // 10) * 10
        start_epoch += last_tmp
        if os.path.exists(f"{out_dir}/latest/test-metrics.pt"):
            start_epoch += 1
        if start_epoch < 1:
            start_epoch = 1

        return start_epoch
