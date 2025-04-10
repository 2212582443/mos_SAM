""" 配准两张图片

    ```bash
        bins/run run.regist.regist_image -m ./data/regist/moving.dcm -f ./data/regist/fixed.dcm -o ./data/regist/out -d 2
    ```
"""

from dataclasses import dataclass
import argparse
import os
from typing import List

ANTSPATH = os.path.abspath("./bins/linux/ants/bin")
os.environ['ANTSPATH'] = ANTSPATH


def run(args: List[str] | None = None):
    args = AppArgs.from_args(args)
    cmd = f"{ANTSPATH}/antsRegistrationSyN.sh -d {args.dim} -f {args.moving} -m {args.fixed} -o {args.out_dir}",
    print("run cmd:", cmd)
    os.system(cmd)


@dataclass
class AppArgs(object):
    """ 工具命令行配置

    属性:
        moving: 待配准图像
        fixed: 基准图像
        out_dir: 输出目录
        dim: 图像维度, 2d或者3d
    """
    moving: str
    fixed: str
    out_dir: str
    dim: int = 2

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def from_args(args: List[str] | None = None):
        parser = argparse.ArgumentParser(args)
        parser.add_argument(
            "--moving", "-m", type=str, required=True, help="moving image path"
        )
        parser.add_argument(
            "--fixed", "-f", type=str, required=True,  help="fixed image path"
        )
        parser.add_argument(
            "--out_dir", "-o", type=str, required=True, help="output dir"
        )
        parser.add_argument(
            "--dim", "-d", type=int, default=2, help="image dim"
        )
        args = parser.parse_args()
        arg_dic = vars(args)
        return AppArgs(**arg_dic)
