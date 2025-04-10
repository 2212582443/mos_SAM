from ..common.mydataset.my_dataset2d import MyDataset2d
from . import labels


class MnmsDataset2d(MyDataset2d):
    """ 用于分割模型的CMRI数据库(2d)
    属性:
        base_dir: 数据集根目录
    """

    def __init__(self, base_dir: str = 'mnms', **args):
        super().__init__(base_dir, **args)
        self.dataset_id = 3
        self.labels = labels
