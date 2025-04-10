
"""心脏数据集

属性:
    base_dir: 数据集根目录

"""
from typing import List, Set, Tuple, TypeVar
import random

from .my_dataset import MyDataset, MyFileInfo

TMyRegistDataset = TypeVar("TMyRegistDataset", bound="MyRegistDataset")


class MyRegistDataset(MyDataset):
    """心脏数据集(用于3d配准)
    """
    # true: 只在用一个用户内部进行数据配准
    pair_inner_user: bool = True
    # 配对的文件列表
    pairs: List[Tuple[MyFileInfo, MyFileInfo]] = []

    def __init__(self, base_dir: str, ext: str = '.npz', files: List[MyFileInfo] | None = None):
        super(MyRegistDataset, self).__init__(base_dir, ext, files)

    def __getitem__(self, index: int) -> Tuple[MyFileInfo, MyFileInfo]:
        return self.pairs[index]

    def __len__(self) -> int:
        return len(self.files)

    def split_dataset_by_uids(self, uid: str | Set[str]) -> TMyRegistDataset:
        """根据uid分割数据集
        """
        (true_list, false_list) = self._split_dataset_by_uids_inner(uid)

        self.files = false_list

        # 需要重新洗牌
        self.shuffer()

        return MyRegistDataset(self.base_dir, self.ext, true_list)

    def shuffer(self):
        """两两组合一对图像"""
        groups = {}
        if self.pair_inner_user:
            for file in self.files:
                if file.uid not in groups:
                    groups[file.uid] = []
                groups[file.uid].append(file)
        else:
            groups[0] = self.files

        pairs = []
        for _uid, group in groups:
            len = len(group)
            if len < 2:
                continue
            selected = list(range(len))
            random.shuffle(selected)
            for v in len:
                # 避免自己和自己配对
                if v == selected[v]:
                    selected.remove(v)
                    selected.push(v)
                a = group[v]
                b = group[selected[v]]
                pairs.push((a, b))
        random.shuffle(pairs)

        self.pairs = pairs
