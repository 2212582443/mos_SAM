"""心脏数据集

属性:
    base_dir: 数据集根目录

"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set, Tuple, TypeAlias, TypeVar
import torch
from torch import Tensor
from torch.utils.data import Dataset
from functools import cache, lru_cache
import hashlib

import json

from ..dataset_utils import locate_dataset_base_url
from ....utils.ops import bi_partition
from ....utils.tensors import load_tensor_file

TMyFileInfo = TypeVar("TMyFileInfo", bound="MyFileInfo")
TMyDataset = TypeVar("TMyDataset", bound="MyDataset")


@dataclass
class MyFileInfo(object):
    """心脏数据集文件信息(3d)

    属性:
        uid: 文件uid
        group: 文件所属组
        phase: 文件所属心动周期
        diastole: 是否舒张期
        systole: 是否收缩期
        segment: 是否包含分割信息
        shape: 文件数据形状
        meta: 文件元数据
    """

    base_dir: str
    file_name: str
    uid: str
    group: str
    phase: int
    diastole: bool
    systole: bool
    segment: bool
    shape: Tuple[int, int, int]
    meta: Dict[str, Any]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def from_meta(meta: Dict[str, Any]) -> TMyFileInfo:
        file_name = meta["file_name"]
        uid = meta["uid"]
        group = meta["diagnosis"]
        phase = meta["seq_index"]
        diastole = meta["diastole"]
        systole = meta["systole"]
        segment_key = meta["segment_key"]
        shape = meta["shape"]
        return MyFileInfo(
            file_name=file_name,
            uid=uid,
            group=group,
            phase=phase,
            diastole=diastole,
            systole=systole,
            segment=segment_key is not None,
            shape=shape,
            meta=meta,
        )


class ItemTransformerContext(object):
    def __init__(self, db: Dataset, index: int, file_info: Any) -> None:
        self.db = db
        self.index = index
        self.file_info = file_info


DatasetItemTransformer: TypeAlias = Callable[[ItemTransformerContext, Any], Any]


class MyDataset(Dataset):
    base_dir: str = "cmri"
    meta: Dict[str, Any] = {}
    files: List[MyFileInfo]
    item_transformer: DatasetItemTransformer | None = None
    dataset_id: int = 0

    def __init__(
        self,
        base_dir: str,
        files: List[MyFileInfo] | None = None,
        item_transormer: DatasetItemTransformer | None = None,
        file_filter: Callable[[MyFileInfo], bool] | None = None,
        on_file_loaded: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__()
        self.base_dir = locate_dataset_base_url(base_dir)
        self.item_transformer = item_transormer

        with open(f"{self.base_dir}/dataset.json") as f:
            self.meta = json.load(f)
        if files is None:
            files = []
            for meta in self.meta["files"]:
                file_info = MyFileInfo.from_meta(meta)
                file_info.base_dir = self.base_dir
                if file_filter is None or file_filter(file_info):
                    files.append(file_info)

        if files is None or len(files) == 0:
            raise ValueError(f"dataset is empty: {self.base_dir}")

        if on_file_loaded is None:
            self._cache_load_file = cache(load_tensor_file)
        else:
            self._cache_load_file = cache(lambda file: on_file_loaded(load_tensor_file(file)))

        self.files = files

    def set_dataset_id(self, id: str):
        id = hashlib.sha256(str(2**128 - 1).encode(id))
        # todo, set dataset id with hash
        pass

    def set_item_transform(self, transformer: DatasetItemTransformer | None = None) -> TMyDataset:
        self.item_transformer = transformer
        return self

    def get_item_info(self, index: int) -> MyFileInfo:
        return self.files[index]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """{
            'image': (ch, d, h, w),
        }
        """
        file: MyFileInfo = self.get_item_info(index)
        # [ch, d, h, w]
        image = self._get_file_tensor_key(file.file_name, "image")
        result = {"image": image}
        if self.item_transformer is not None:
            context = ItemTransformerContext(self, index, file_info=file)
            result = self.item_transformer(context, result)
        return result

    def __len__(self) -> int:
        return len(self.files)

    def uids(self) -> Set[str]:
        """返回所有文件的uid"""
        return set(map(lambda x: x.uid, self.files))

    def split_dataset_by_uids(self, uid: str | Set[str]) -> Any:
        """根据uid分割数据集"""
        (true_list, false_list) = self._split_dataset_by_uids_inner(uid)

        self.files = false_list

        # 需要重新洗牌
        self.shuffer()

        return MyDataset(self.base_dir, self.ext, true_list)

    def _split_dataset_by_uids_inner(self, uid: str | Set[str]) -> Tuple[List[MyFileInfo], List[MyFileInfo]]:
        """根据uid分割数据集

        return: (true_list, false_list)
        """
        if type(uid) is str or type(uid) is int:
            uid = set([uid])

        (true_list, false_list) = bi_partition(lambda x: x.uid in uid, self.files)

        # todo, split meta

        return (true_list, false_list)

    def _get_file_tensors(self, file_name: str) -> Dict[str, torch.Tensor]:
        file_name = f"{self.base_dir}/{file_name}"
        return self._cache_load_file(file_name)

    def _get_file_tensor_key(self, file_name: str, key: str, default_key: str = "") -> torch.Tensor | None:
        tensor = self._get_file_tensors(file_name)
        key_tensor = tensor.get(key, None)
        if tensor is None:
            key_tensor = tensor.get(default_key, None)
        # 文件没有channel, 补上
        key_tensor = key_tensor.unsqueeze(0)
        # dataset的tensor格式[ch, d, h, w], 不包含bs
        return key_tensor
