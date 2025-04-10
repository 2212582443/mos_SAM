
"""心脏数据集

属性:
    base_dir: 数据集根目录

"""
from ..common.mydataset.my_dataset import MyDataset
from .import labels


class CmriDataset(MyDataset):
    def __init__(self, base_dir: str = 'cmri', **args,):
        super().__init__(base_dir, **args)
        self.dataset_id = 1
        self.labels = labels
