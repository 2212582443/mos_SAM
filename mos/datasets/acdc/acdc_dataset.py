
"""心脏数据集

属性:
    base_dir: 数据集根目录

"""


from mos.datasets.common.mydataset.my_dataset import MyDataset
from . import labels

class AcdcDataset(MyDataset):

    def __init__(self, base_dir: str = 'acdc', **args):
        super().__init__(base_dir, **args)
        self.dataset_id = 2
        self.labels = labels
