import json
import os
from typing import List
from torch import Tensor
import torch

from mos.datasets.common.mydataset.my_dataset import MyDataset
from mos.datasets.common.mydataset.my_dataset2d import MyFileInfoFlettern
from torchvision import transforms
from mos.models.sam.modeling_sam.embedding.point_embedding import rand_get_segment_point

from mos.models.sam.modeling_sam.embedding.typing import GrayImageTensor, SegmentTensor
from torchvision.ops import masks_to_boxes
import json


# 数据集
# image_embedding f16
# image i8
# segment i8
# points kxnx2xi8
# cls: vec[]
# file: str
# slice_index: i8
# subject: i16

IMAGE_SIZE = 224
CROP_SIZE = 128


class MetaItem:
    def __init__(
        self,
        database_name: str,
        uid: str,
        phase: int,
        slice: int,
        cls: int,
        segment: SegmentTensor,
    ):
        """
        Args:
            database_name: 数据库名称
            uid: 文件uid
            phase: 数据集划分
            slice: 切片序号
            cls: 分割类别
            segment: 分割label, 0表示背景, 1表示前景, (1, 1, h, w)
            segment_points: 分割label的点, list[(x,y)]
            box: 分割label的box, (x1, y1, x2, y2)
            crop_start_range: 裁剪的起始范围, (x1, y1, x2, y2)
        """
        self.database_name = database_name
        self.uid = uid
        self.phase = phase
        self.slice = slice
        self.cls = cls
        self.segment_points = rand_get_segment_point(segment)
        x1, y1, x2, y2 = masks_to_boxes(segment).squeeze(0).tolist()
        self.box = [x1, y1, x2, y2]

        # calculate crop range
        start_x = x2 - CROP_SIZE
        start_x = min(x1, start_x)
        start_x = max(0, start_x)
        end_x = x1
        end_x = max(x2 - CROP_SIZE, end_x)
        end_x = min(IMAGE_SIZE - CROP_SIZE, end_x)

        start_y = y2 - CROP_SIZE
        start_y = min(y1, start_y)
        start_y = max(0, start_y)
        end_y = y1
        end_y = max(y2 - CROP_SIZE, end_y)
        end_y = min(IMAGE_SIZE - CROP_SIZE, end_y)

        self.crop_start_range = [start_x, start_y, end_x, end_y]

    def dict(self):
        return self.__dict__


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
        """将3d图像展开为2d图像(只保留有segment的图片)"""
        files = []
        for file in self.files:
            if file.segment is None or not file.segment:
                continue
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
        # image = image * 255
        # image = image.to(torch.uint8)
        image = image.to(torch.float16)

        segment = self._get_slice_image(index, "segment")
        segment = _segment_transform(segment)
        segment = segment.to(torch.uint8)

        assert len(image.shape) == 4

        file_info: MyFileInfoFlettern = self.get_item_info(index)
        uid = file_info.file_info.uid
        phase = file_info.file_info.phase
        slice = file_info.slice_index

        return (uid, phase, slice, image, segment)


acdc_segment_mapping = [0, 2, 3, 4]
mnms_segment_mapping = [0, 2, 3, 4]
cmri_segment_mapping = [0, 1]


def run1(_args):
    datasets = [
        ("cmri", MyDataset2d("cmri"), cmri_segment_mapping),
        ("mnms", MyDataset2d("mnms"), mnms_segment_mapping),
        ("acdc", MyDataset2d("acdc"), acdc_segment_mapping),
    ]

    os.makedirs(".cache/dataset/sam-dataset/", exist_ok=True)
    index = 0
    train_images = []
    valid_images = []
    train_segments = []
    valid_segments = []
    train_metas = []
    valid_metas = []

    while len(datasets) > 0:
        dataset_name, dataset, cls_mapping = datasets.pop(0)
        for uid, phase, slice, image, segment in dataset:
            # image (bs, 1, h, w)
            # segment (bs, h, w)
            # print("image", image.shape, "segment", segment.shape)
            clsses: list[int] = segment.unique().tolist()
            clsses.remove(0)
            for cls in clsses:
                cls = cls_mapping[cls]
                segment_cls = torch.where(segment == cls, 1, 0).to(torch.uint8)
                if segment_cls.sum() == 0:
                    continue
                index += 1
                assert len(image.shape) == 4
                assert len(segment_cls.shape) == 3
                meta = MetaItem(dataset_name, uid, phase, slice, cls, segment_cls).dict()

                print(f"{index} {dataset_name} {uid} {phase} {slice} {cls} {segment_cls.sum()}")

                if dataset_name == "cmri" and uid % 10 == 0:
                    valid_images.append(image)
                    valid_segments.append(segment_cls)
                    valid_metas.append(meta)
                else:
                    train_images.append(image)
                    train_segments.append(segment_cls)
                    train_metas.append(meta)
    # return
    train_images = torch.cat(train_images, dim=0)
    train_segments = torch.cat(train_segments, dim=0)

    valid_images = torch.cat(valid_images, dim=0)
    valid_segments = torch.cat(valid_segments, dim=0)

    torch.save(train_images, ".cache/dataset/sam-dataset/train-images.ot")
    torch.save(train_segments, ".cache/dataset/sam-dataset/train-segments.ot")
    json.dump(train_metas, open(".cache/dataset/sam-dataset/train-metas.json", "w", encoding="utf-8"), indent=2)

    torch.save(valid_images, ".cache/dataset/sam-dataset/valid-images.ot")
    torch.save(valid_segments, ".cache/dataset/sam-dataset/valid-segments.ot")
    json.dump(valid_metas, open(".cache/dataset/sam-dataset/valid-metas.json", "w", encoding="utf-8"), indent=2)


def run(_args):
    run1(_args)
