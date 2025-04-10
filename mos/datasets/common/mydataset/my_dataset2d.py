from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Dict, List, Set, Tuple, TypeVar

from torch import Tensor

from mos.datasets.common.mydataset.my_dataset import MyFileInfo
from .my_dataset import ItemTransformerContext, MyDataset, MyFileInfo, TMyDataset
import numpy as np

TMyDataset2d = TypeVar(name="TMyDataset2d", bound="MyDataset2d")


@dataclass
class MyFileInfoFlettern(object):
    """心脏数据集文件信息(2d展开)"""

    file_info: MyFileInfo
    slice_index: int  # 0-based
    box: Tuple[int, int, int, int] | None  # [x1, y1, x2, y2]

    def __init__(self, file_info: MyFileInfo, slice_index: int, box: Tuple[int, int, int, int] | None):
        self.file_info = file_info
        self.slice_index = slice_index
        self.box = box

    def hash(self) -> str:
        base_dir = self.file_info.base_dir
        file_name = self.file_info.file_name
        file_name = f"{base_dir}/{file_name}"
        file_name_hash = hashlib.md5(file_name.encode("utf-8")).hexdigest()
        return f"{file_name_hash}_{self.slice_index}"


class MyDataset2d(MyDataset):
    """用于分割模型的CMRI数据库(2d)
    属性:
        base_dir: 数据集根目录
    """

    flattern_files: List[MyFileInfoFlettern] = []

    def __init__(
        self,
        base_dir: str,
        files: List[MyFileInfo] | None = None,
        item_transformer: Callable[[Any], Any] | None = None,
        file_filter: Callable[[MyFileInfo], bool] | None = None,
        on_file_loaded: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] | None = None,
    ):
        super().__init__(base_dir, files, item_transformer, file_filter, on_file_loaded)
        self._flattern_file()

    def get_item_info(self, index: int) -> MyFileInfoFlettern:
        return self.flattern_files[index]

    def _get_slice_image(self, index, key) -> Tensor:
        file: MyFileInfoFlettern = self.get_item_info(index)
        # [ch, d, h, w]
        tensor = self._get_file_tensor_key(file.file_info.file_name, key)
        max_deep = tensor.shape[1]
        if file.slice_index >= max_deep:
            raise ValueError(
                f"slice_index({file.slice_index}) >= max_deep({max_deep}), shape:{tensor.shape} file: {file.file_info.file_name}"
            )

        # [ch, 1, h, w]
        tensor = tensor[:, file.slice_index, :, :]
        # [ch, h, w]
        tensor = tensor.squeeze(1)

        return tensor

    def __len__(self) -> int:
        count = len(self.flattern_files)
        return count

    def _flattern_file_super(self):
        """将3d图像展开为2d图像"""
        files = []
        for file in self.files:
            if not file.segment:
                continue
            for slice_index in range(file.meta["shape"][0]):
                box = file.meta["segment_box_2d"].get(f"{slice_index}", None)
                files.append(MyFileInfoFlettern(file, slice_index, box))

        self.flattern_files = files

    def _flattern_file(self):
        """将3d图像展开为2d图像,  只保留有segment的图片"""
        self._flattern_file_super()
        files = []
        for file in self.flattern_files:
            if file.box is None:
                continue
            files.append(file)

        self.flattern_files = files

    def split_dataset_by_uids(self, uid: str | Set[str]) -> TMyDataset2d:
        """根据uid分割数据集"""
        (true_list, false_list) = self._split_dataset_by_uids_inner(uid)

        self.files = false_list

        self._flattern_file()

        return MyDataset2d(self.base_dir, self.ext, true_list)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        return {
            image: (ch, h, w),
            label: (h, w),
        }
        """
        image = self._get_slice_image(index, "image")
        segment = self._get_slice_image(index, "segment")
        segment = segment.squeeze(0)
        box = self.flattern_files[index].box

        result = {"image": image, "segment": segment, "box": np.array(box)}

        if self.item_transformer is not None:
            file: MyFileInfoFlettern = self.get_item_info(index)
            context = ItemTransformerContext(self, index, file_info=file)
            result = self.item_transformer(context, result)

        return result
