""" 配准整个目录下的图片, 以fixed为基准, 配准结果保存到out_dir

    ```bash
        bins/run run.regist.regist_dir -m ./data/regist/moving -f ./data/regist/fixed.dcm -o ./data/regist/out -d 2
    ```
"""

from dataclasses import dataclass
import argparse
import os
import multiprocessing as mp
from typing import List, TypeAlias
from mos.utils.files import load_files, FileInfo

ANTSPATH = os.path.abspath("./bins/linux/ants/bin")
os.environ['ANTSPATH'] = ANTSPATH

# task type: (fixed, moving, output, dim)
_TASK_TYPE: TypeAlias = tuple[str, str, str, str]


def run(args: List[str] | None = None):
    args = AppArgs.from_args(args)
    files_list: List[FileInfo] = load_files(args.moving_dir, args.ext)
    # 生成要执行的任务列表
    task_list: List[_TASK_TYPE] = []  # (fixed, moving, output, dim)
    for i in range(len(files_list) - 1):
        file_info = files_list[i]

        relate_dir = file_info.root.replace(args.moving_dir, "").strip("/")
        target_dir = f"{args.out_dir}/{relate_dir}"

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        task_list.append((
            args.fixed,
            file_info.file_abs_path,
            f"{target_dir}/{file_info.name}",
            args.dim
        ))

    # 并行执行所有任务
    with mp.Pool(mp.cpu_count()) as p:
        p.map(_run_regist_syn_task, task_list)


@dataclass
class AppArgs(object):
    """ 工具命令行配置

    属性:
        moving_dir: 待配准图像的目录
        fixed: 基准图像
        out_dir: 输出目录
        ext: 待配准文件后缀
        dim: 图像维度, 2d或者3d
    """
    #  待配准图像的目录
    moving_dir: str
    # 基准图像
    fixed: str
    # 输出目录
    out_dir: str
    # 待配准文件后缀
    ext: str = ".dcm"
    #  图像维度, 2d或者3d
    dim: int = 2

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def from_args(args: List[str] | None = None):
        parser = argparse.ArgumentParser(args)
        parser.add_argument(
            "--moving_dir", "-m", type=str, required=True, help="moving image path"
        )
        parser.add_argument(
            "--fixed", "-f", type=str, required=True,  help="fixed image"
        )
        parser.add_argument(
            "--out_dir", "-o", type=str, required=True, help="output dir"
        )
        parser.add_argument(
            "--ext", "-e", type=str, default=".dcm", help="file ext"
        )
        parser.add_argument(
            "--dim", "-d", type=int, default=2, help="image dim"
        )
        args = parser.parse_args()
        arg_dic = vars(args)
        return AppArgs(**arg_dic)


def _run_regist_syn_task(task_info: _TASK_TYPE):
    (fixed, moving, output, dim) = task_info

    cmd = f"{ANTSPATH}/antsRegistrationSyN.sh -d {dim} -f {fixed} -m {moving} -o {output}"
    print("run cmd:", cmd)

    os.system(cmd)
