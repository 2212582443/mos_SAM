import os, glob
from dataclasses import dataclass, field
from typing import List

import torch
from torch.optim import Optimizer
from transformers import HfArgumentParser, TrainingArguments
from enum import Enum

from mos.utils.files import relative_symlink_file


class ReEvalType(Enum):
    NoReEval = ""  # 不重新计算指标（默认）
    Complete = "complete"  # 如果不存在，那么补全
    Normal = "normal"  # 如果已经存在更新，那么不重复更新
    Force = "force"  # 强制全部更新一遍

    @staticmethod
    def from_str(label):
        if isinstance(label, ReEvalType):
            return label
        if label is None or label == "" or label in ("none", "None", "NoReEval"):
            return ReEvalType.NoReEval
        elif label in ("complete", "Complete"):
            return ReEvalType.Complete
        elif label in ("normal", "Normal"):
            return ReEvalType.Normal
        elif label in ("force", "Force"):
            return ReEvalType.Force
        else:
            print(f"label [{label}] NOT support!")
            raise label


def expand_id_list(id_str: str) -> list[int]:
    """
    Args:
        id_str: 1,2,6-15,17
    """
    if id_str is None or id_str == "":
        return []
    return sum(
        (
            (list(range(*[int(b) + c for c, b in enumerate(a.split("-"))])) if "-" in a else [int(a)])
            for a in id_str.split(",")
        ),
        [],
    )


@dataclass
class ModelArguments(object):
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    train_dataset: str = field(
        default="cmri712-0",
        metadata={"help": ("训练使用的数据集")},
    )

    start_epoch: int = field(
        default=1,
        metadata={"help": ("开始训练的epoch")},
    )
    increase_epoch: str = field(
        default=None,
        metadata={
            "help": (
                "不管当前是哪个epoch，只增量训练指定的epoch个数,正数表示增加，负数表示重新训练最后几个，fix表示训练到最近整10的epoch"
            )
        },
    )
    # ===== debug config =====
    sample_count: int = field(
        default=0,
        metadata={"help": ("训练样本采样的数量，用于debug")},
    )
    log_every_n_epochs: int = field(
        default=10,
        metadata={"help": ("多少epoch输出一次日志到tensorboard")},
    )

    # ===== learning rate =====
    lr_scheduler_parameter: str = field(
        default="100",
        metadata={"help": ("学习率调度器")},
    )
    lr_scheduler_gamma: float = field(
        default=0.1,
        metadata={"help": ("学习率调度器gamma")},
    )

    note: str = field(
        default="",
        metadata={"help": ("训练备注")},
    )

    dir_postfix: str = field(
        default=None,
        metadata={"help": ("保存的默认路径后缀")},
    )
    tag: str = field(
        default=None,
        metadata={"help": ("tag")},
    )

    # ===== dataset =====
    dataset_train_labels: str = field(
        default="1",
        metadata={"help": ("数据集使用的label id")},
    )
    dataset_eval_labels: str = field(
        default="1",
        metadata={"help": ("验证数据集使用的label id")},
    )
    dataset_device: str = field(
        default=None,
        metadata={"help": ("数据集存放的位置，默认和模型一起")},
    )
    dataset_image_size: str = field(
        default="214,214",
        metadata={"help": ("图像大小")},
    )
    dataset_crop_size: str = field(
        default="128,128",
        metadata={"help": ("train裁剪大小")},
    )
    dataset_spacing: str = field(
        default="10,1.5,1.5",
        metadata={"help": ("数据集的分辨率,d,h,w方向")},
    )
    dataset_crop_deep_maxscale: int = field(
        default=1,
        metadata={"help": ("3d情况下，deep维度裁剪最大间隔大小")},
    )
    dataset_valid_crop_size: str = field(
        default="16,128,128",
        metadata={"help": ("valid/test裁剪大小")},
    )
    dataset_valid_crop_deep_maxscale: int = field(
        default=1,
        metadata={"help": ("3d情况下，valid/test的deep维度裁剪最大间隔大小")},
    )
    dataset_train_all: bool = field(
        default=False,
        metadata={"help": ("是否把数据集的所有数据都用于训练")},
    )
    image_enhance_level: int = field(
        default=1,
        metadata={"help": ("图像增强等级")},
    )
    main_dataset_only: bool = field(
        default=False,
        metadata={"help": ("训练时只使用主数据集，用于训练末尾，加速收敛")},
    )

    # ===== aux =====
    aux_dataset: str = field(
        default=None,
        metadata={"help": ("是否使用aux数据集")},
    )
    aux_dataset_only: bool = field(
        default=False,
        metadata={"help": ("训练时只使用辅助数据，用于预训练，加速交叉验证的速度")},
    )
    aux_dataset_labels: str = field(
        default="2,3,4",
        metadata={"help": ("aux数据集使用的label id")},
    )

    # ===== zeroshot =====
    zeroshot_dataset: str = field(
        default="",
        metadata={"help": ("是否使用zeroshot数据集")},
    )
    zeroshot_dataset_labels: str = field(
        default="2,3,4",
        metadata={"help": ("aux数据集使用的label id")},
    )

    # ===== pseudo =====
    pseudo: bool = field(
        default=False,
        metadata={"help": ("是否生成伪标签")},
    )
    pseudo_dataset: str = field(
        default="pseudo",
        metadata={"help": ("伪标签数据集")},
    )
    pseudo_dataset_key: str = field(
        default="crop3d",
        metadata={"help": ("伪标签数据集的key")},
    )
    pseudo_output_labels: str = field(
        default="1,2,3,4",
        metadata={"help": ("伪标签输出的label id")},
    )
    pseudo_output: str = field(
        default=".checkpoint/pseudo",
        metadata={"help": ("伪标签输出路径")},
    )

    # ===== re eval =====
    re_eval: ReEvalType = field(
        default=ReEvalType.NoReEval,
        metadata={"help": ("重新对所有指标进行计算")},
    )

    # ===== plot test =====
    run_plot_test: bool = field(
        default=False,
        metadata={"help": ("是否重新绘制测试结果")},
    )

    # ===== model =====
    label_count: int = field(default=5, metadata={"help": ("模型label的数量，包括背景")})

    model: str = field(
        default="segnet",
        metadata={"help": ("模型名称")},
    )
    model_path: str = field(
        default=None,
        metadata={"help": ("模型所在的路径，用于从指定的断电恢复")},
    )
    pretrain_model_path: str = field(
        default=None,
        metadata={"help": ("模型所在的路径，用于从预训练中加载模型")},
    )
    load_optimizer: bool = field(
        default=True,
        metadata={"help": ("是否加载optimizer")},
    )
    load_model_strict: bool = field(
        default=True,
        metadata={"help": ("是否严格加载模型")},
    )

    init: bool = field(
        default=False,
        metadata={"help": ("是否从预训练模型初始化")},
    )

    loop: int = field(
        default=1,
        metadata={"help": ("训练循环的次数")},
    )

    use_profile: bool = field(
        default=False,
        metadata={
            "help": "是否进行性能分析,以查找代码的性能瓶颈",
        },
    )

    def get_dataset_train_labels(self) -> list[int]:
        return expand_id_list(self.dataset_train_labels)

    def get_dataset_eval_labels(self) -> list[int]:
        return expand_id_list(self.dataset_eval_labels)

    def get_dataset_crop_size(self) -> list[int]:
        if self.dataset_crop_size is None:
            return []
        labels = self.dataset_crop_size.split(",")
        labels = map(lambda x: int(x), labels)
        return list(labels)

    def get_dataset_spacing(self) -> list[float]:
        if self.dataset_spacing is None:
            return []
        labels = self.dataset_spacing.split(",")
        labels = map(lambda x: float(x), labels)
        return list(labels)

    def get_dataset_valid_crop_size(self) -> list[int]:
        if self.dataset_valid_crop_size is None:
            return []
        labels = self.dataset_valid_crop_size.split(",")
        labels = map(lambda x: int(x), labels)
        return list(labels)

    def is2d(self) -> bool:
        return len(self.get_dataset_crop_size()) < 3

    def get_dataset_image_size(self) -> list[int]:
        if self.dataset_image_size is None:
            return []
        labels = self.dataset_image_size.split(",")
        labels = map(lambda x: int(x), labels)
        return list(labels)

    def get_aux_dataset(self) -> set[str]:
        if self.aux_dataset is None:
            return set()
        return set(self.aux_dataset.split(","))

    def get_aux_dataset_labels(self) -> list[int]:
        return expand_id_list(self.aux_dataset_labels)

    def get_pseudo_output_labels(self) -> list[int]:
        return expand_id_list(self.pseudo_output_labels)

    def get_zeroshot_dataset_labels(self) -> list[int]:
        return expand_id_list(self.zeroshot_dataset_labels)

    def get_training_labels(self) -> list[list[int]]:
        return [
            None,  # None
            self.get_dataset_train_labels(),  # DATASET.CMRI
            self.get_aux_dataset_labels(),  # DATASET.AUX
            self.get_zeroshot_dataset_labels(),  # DATASET.ZERO_SHOT
            list(range(1, self.label_count)),  # Dataset.PRUSUDO
        ]

    def get_eval_labels(self) -> list[list[int]]:
        return [
            None,  # None
            self.get_dataset_eval_labels(),  # DATASET.CMRI
            self.get_aux_dataset_labels(),  # DATASET.AUX
            self.get_zeroshot_dataset_labels(),  # DATASET.ZERO_SHOT
            list(range(1, self.label_count)),  # Dataset.PSESUDO
        ]

    def get_multi_step_lr_scheduler_parameter(self) -> list[int]:
        if self.lr_scheduler_parameter is None:
            return []
        return list(map(lambda x: int(x), self.lr_scheduler_parameter.split(",")))


def get_args(args: List[str] | None) -> tuple[ModelArguments, TrainingArguments, list[str]]:
    import os
    import sys

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(args) == 2 and args[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # type: Tuple[ModelArguments, DataTrainingArguments, CustomTrainingArguments]
        model_args, training_args, others = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)
    return (model_args, training_args, others)
