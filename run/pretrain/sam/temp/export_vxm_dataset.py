import json
import os
from typing import Any, Callable, Dict, List
from torch import ShortTensor, IntTensor, FloatTensor, HalfTensor, Tensor
import torch
from mos.datasets.acdc.acdc_dataset2d import AcdcDataset2d

from mos.datasets.cmri.cmri_dataset2d import CmriDataset2d
from mos.datasets.common.mydataset.my_dataset import ItemTransformerContext, MyDataset, MyFileInfo
from mos.datasets.common.mydataset.my_dataset2d import MyFileInfoFlettern
from mos.datasets.mnms.mnms_dataset2d import MnmsDataset2d
from torchvision import transforms
from mos.models.sam.configuration_sam import SamVisionConfig
from mos.models.sam.modeling_sam.embedding.image_embedding_sam import ImageEmbeddingSam

from mos.models.sam.modeling_sam.embedding.typing import GrayImageTensor, SegmentTensor
import numpy as np

# 数据集
# image_embedding f16
# image i8
# segment i8
# points kxnx2xi8
# cls: vec[]
# file: str
# slice_index: i8
# subject: i16

IMAGE_SIZE = 256
_image_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR
        ),
    ]
)
_segment_transform = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True, interpolation=transforms.InterpolationMode.NEAREST),
    ]
)


class MyDataset2d(MyDataset):
    flattern_files: List[MyFileInfoFlettern] = []

    def __init__(
        self,
        base_dir: str,
    ):
        super().__init__(base_dir)
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

    def _flattern_file(self):
        """将3d图像展开为2d图像"""
        files = []
        for file in self.files:
            for slice_index in range(file.meta["shape"][0]):
                files.append(MyFileInfoFlettern(file, slice_index, None))

        self.flattern_files = files

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        return {
            image: (ch, h, w),
        }
        """
        image = self._get_slice_image(index, "image")
        # (bs, ch, h, w)
        image: GrayImageTensor = image.unsqueeze(0)
        image = _image_transform(image)
        image = image.to(torch.float16)

        assert len(image.shape) == 4

        file_info: MyFileInfoFlettern = self.get_item_info(index)
        uid = file_info.file_info.uid
        phase = file_info.file_info.phase
        slice_index = file_info.slice_index

        return (uid, phase, slice_index, image)


def run1(_args):
    os.makedirs(".cache/dataset/vxm-dataset/", exist_ok=True)
    index = 0
    image_location = {}
    meta_list = []
    image_list = []
    dataset = MyDataset2d("cmri")
    for uid, phase, slice_index, image in dataset:
        image = image.cpu()

        phase = phase - 1  # to base 0

        image_location[f"{uid}-{phase}-{slice_index}"] = index
        meta_list.append({"uid": uid, "phase": phase, "slice_index": slice_index})
        print(f"{uid}-{phase}-{slice_index} {index} {image.shape}")

        image_list.append(image)
        index += 1

    image_list = torch.cat(image_list, dim=0)
    json.dump(meta_list, open(".cache/dataset/vxm-dataset/dataset.json", "w"))
    subjects = dataset.meta["subjects"]

    pair_list = []
    pair_index_list = []
    for subject in subjects:
        uid, seq_len, shape = subject["uid"], subject["seq_len"], subject["shape"]
        slice_count = shape[0]
        start_index_include = len(pair_list)
        for phase in range(seq_len):
            for slice_index in range(slice_count):
                current_index = image_location[f"{uid}-{phase}-{slice_index}"]
                # 当前image和接下来8个不同时刻的同一个slice组成pair
                for next_phase in range(1, 15 + 1):
                    next_phase = (phase + next_phase) % seq_len
                    next_index = image_location[f"{uid}-{next_phase}-{slice_index}"]
                    pair_list.append((current_index, next_index))
                    pair_list.append((next_index, current_index))
                # 当前image和下一时刻看slice+1组成pair
                for next_phase in range(1, 2 + 1):
                    next_slice = slice_index + next_phase
                    if next_slice >= slice_count:
                        break
                    next_phase = (phase + next_phase) % seq_len
                    next_index = image_location[f"{uid}-{next_phase}-{next_slice}"]
                    pair_list.append((current_index, next_index))
                    pair_list.append((next_index, current_index))
        end_index_exclude = len(pair_list)
        pair_index_list.append((uid, start_index_include, end_index_exclude))
    pair_list = np.array(pair_list)
    pair_list = torch.tensor(pair_list, dtype=torch.int32)
    pair_index_list = np.array(pair_index_list)
    pair_index_list = torch.tensor(pair_index_list, dtype=torch.int32)
    torch.save(
        image_list,
        ".cache/dataset/vxm-dataset/images.ot",
    )
    torch.save(
        pair_list,
        ".cache/dataset/vxm-dataset/pair-list.ot",
    )
    torch.save(
        pair_index_list,
        ".cache/dataset/vxm-dataset/pair-index-list.ot",
    )


def run(args):
    run1(args)
