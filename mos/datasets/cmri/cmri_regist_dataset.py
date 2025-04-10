
"""心脏数据集

属性:
    base_dir: 数据集根目录

"""
from typing import TypeVar

from ..common.mydataset.my_regist_dataset import MyRegistDataset
from . import labels


TCmriRegistDataset = TypeVar("TCmriRegistDataset", bound="CmriRegistDataset")


class CmriRegistDataset(MyRegistDataset):
    def __init__(self, base_dir: str = 'cmri', **args):
        super().__init__(base_dir, args)
        self.dataset_id = 1
        self.labels = labels
