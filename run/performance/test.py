import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class AppArgs(object):
    """ 测试显卡性能

    属性:
    """
    #  test device
    device: str

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def from_args(args: List[str] | None = None):
        parser = argparse.ArgumentParser(args)
        parser.add_argument(
            "--device", "-d", type=str, required=True, help="device"
        )
        args = parser.parse_args()
        arg_dic = vars(args)
        return AppArgs(**arg_dic)


def run(args: List[str] | None):
    """ Test cuda performance

    ```bash
        bins/run run.test --seed 9998
    ```
    """
    app_args = AppArgs.from_args(args)
    print("Test, args:", app_args)
