import argparse
from dataclasses import dataclass
from typing import List

from .shape import ImageShape


@dataclass
class DefaultAppArgs(object):
    """ 默认的app参数

    属性:
        seed: 训练时的随机种子
        model: 模型路径
        dim: 数据的维度
        shape: 图片分辨率
        device: 模型的运行环境, -1: cpu, >=0 cuda index
        web: 开启web页面
        port: web页面端口
        enhances: 数据增强
        init_only: 是否只初始, 不实际执行command
        flags: 其他标志(通用, 具体含义视情况而定)
        command: 命令
    """
    # 训练时的随机种子
    seed: int = 121033910097
    # mode path
    model: str = "./.checkpoint"
    # 数据的维度
    dim: int = 2
    # 图片分辨率
    shape: ImageShape = None
    # 模型的运行环境, -1: cpu, >=0 cuda index
    device: int = -1
    # 开启web页面
    web: bool = False
    # web页面端口
    port: int = 3000
    # 数据增强
    enhances: List[str] = None
    # 是否只初始, 不实际执行command
    init_only: bool = False
    # 其他标志(通用, 具体含义视情况而定)
    flags: List[str] = None
    command: str = "run"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def from_args(args: List[str] | None = None):
        parser = argparse.ArgumentParser(args)
        parser.add_argument(
            "--seed", type=int, default=DefaultAppArgs.seed, help="random seed"
        )
        parser.add_argument(
            "--model", type=str, default=DefaultAppArgs.model, help="model path"
        )
        parser.add_argument(
            "--dim", type=int, default=DefaultAppArgs.dim, help="data dim"
        )
        parser.add_argument(
            "--shape", type=ImageShape, default=DefaultAppArgs.shape, help="image shape"
        )
        parser.add_argument(
            "--device", type=int, default=DefaultAppArgs.device, help="device index"
        )
        parser.add_argument(
            "--web", type=bool, default=DefaultAppArgs.web, help="enable web"
        )
        parser.add_argument(
            "--port", type=int, default=DefaultAppArgs.port, help="web port"
        )
        parser.add_argument(
            "--enhances",
            type=List[str],
            default=DefaultAppArgs.enhances,
            help="enhances",
        )
        parser.add_argument(
            "--init_only", type=bool, default=DefaultAppArgs.init_only, help="init only"
        )
        parser.add_argument(
            "--flags", type=List[str], default=DefaultAppArgs.flags, help="flags"
        )
        parser.add_argument(
            "--command", type=str, default=DefaultAppArgs.command, help="command"
        )
        args = parser.parse_args()
        arg_dic = vars(args)
        return DefaultAppArgs(**arg_dic)
