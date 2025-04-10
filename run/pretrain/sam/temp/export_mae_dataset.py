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


_image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR),
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
            label: (h, w),
        }
        """
        image = self._get_slice_image(index, "image")
        # (bs, ch, h, w)
        image: GrayImageTensor = image.unsqueeze(0)
        image = _image_transform(image)
        image = image * 255
        image = image.to(torch.uint8)

        assert len(image.shape) == 4

        file_info: MyFileInfoFlettern = self.get_item_info(index)
        uid = file_info.file_info.uid
        slice_index = file_info.slice_index

        return (uid, slice_index, image)


def run1(_args):
    datasets = [
        ("cmri", MyDataset2d("cmri")),
        ("mnms", MyDataset2d("mnms")),
        ("acdc", MyDataset2d("acdc")),
    ]

    os.makedirs(".cache/dataset/mae-dataset/train", exist_ok=True)
    os.makedirs(".cache/dataset/mae-dataset/valid", exist_ok=True)
    index = 1
    while len(datasets) > 0:
        dataset_name, dataset = datasets.pop(0)
        for uid, slice_index, image in dataset:
            image = image.cpu().numpy()

            if dataset_name == "cmri" and uid % 10 == 0:
                dist_path = f".cache/dataset/mae-dataset/valid/{index}-{dataset_name}-{uid:03}-{slice_index:02}.npz"
            else:
                dist_path = f".cache/dataset/mae-dataset/train/{index}-{dataset_name}-{uid:03}-{slice_index:02}.npz"

            np.savez_compressed(
                dist_path,
                image=image,
            )
            index += 1


def list_all_npz_files(path):
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files if file.endswith(".npz")]
    return files


def run2(_args):
    files = list_all_npz_files(".cache/dataset/mae-dataset/train/")
    images = []
    for file in files:
        image = np.load(file)
        image = image["image"]
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.tensor(images)
    images = images.squeeze(1)
    torch.save(images, ".cache/dataset/mae-dataset/train.ot")

    files = list_all_npz_files(".cache/dataset/mae-dataset/valid/")
    images = []
    for file in files:
        image = np.load(file)["image"]
        images.append(image)
    images = np.stack(images, axis=0)
    images = torch.tensor(images)
    images = images.squeeze(1)
    torch.save(images, ".cache/dataset/mae-dataset/valid.ot")


def run(_args):
    run1(_args)
    run2(_args)
