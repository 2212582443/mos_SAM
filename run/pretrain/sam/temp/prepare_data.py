import json
import os
from torch import ShortTensor, IntTensor, FloatTensor, HalfTensor
import torch
from mos.datasets.acdc.acdc_dataset2d import AcdcDataset2d

from mos.datasets.cmri.cmri_dataset2d import CmriDataset2d
from mos.datasets.common.mydataset.my_dataset import ItemTransformerContext
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


class TrainItem(object):
    image_embedding: HalfTensor = None
    image: ShortTensor = None
    segment: ShortTensor = None
    # 以下各个长度有差异
    valid_point: dict[str, list[(int, int)]] = None
    cls: ShortTensor = None
    dataset_name: str = None
    file_name: str = None
    slice_index: int = None
    uid: str = None

    def __init__(self, **args):
        for k, v in args.items():
            setattr(self, k, v)


vision_config = SamVisionConfig(image_size=1024)
_imagembed_image = ImageEmbeddingSam(vision_config)
_segment_transform = transforms.Compose(
    [
        transforms.Resize((224, 224), antialias=True, interpolation=transforms.InterpolationMode.NEAREST),
    ]
)
_image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR),
    ]
)


def new_item_tranformer(device, map):
    # 坐标是基于224*224的图像的, 需要缩放到1024*1024的图像上
    scaling = 4

    def transformer(context: ItemTransformerContext, item):
        # (ch, h, w)
        image = item["image"]
        # (bs, ch, h, w)
        image = image.to(device)
        image: GrayImageTensor = image.unsqueeze(0)
        image = _image_transform(image)
        image = image * 224
        image = image.to(torch.int8)

        # (h, w)
        segment = item["segment"]
        segment = segment.to(device)
        # (bs, h, w)
        segment: SegmentTensor = segment.unsqueeze(0)
        segment: SegmentTensor = _segment_transform(segment)
        segment = segment.to(torch.int8)

        cls: list[int] = segment.unique().tolist()
        valid_point = {}
        for c in cls:
            _, y_index, x_index = torch.nonzero(segment == c, as_tuple=True)
            point = [(x * scaling, y * scaling) for x, y in zip(x_index.tolist(), y_index.tolist())]
            valid_point[f"{c}"] = point

        file_info: MyFileInfoFlettern = context.file_info
        file_name = file_info.file_info.file_name
        uid = file_info.file_info.uid

        slice_index = file_info.slice_index

        cache_file = f".cache/dataset/sam-vit-embedding/{file_info.hash()}.pt"
        if os.path.exists(cache_file):
            image_embedding = torch.load(cache_file, map_location="cpu")
        else:
            image_embedding = _imagembed_image(image)

        image_embedding = image_embedding.to(torch.float16)

        item = TrainItem(
            image_embedding=image_embedding,
            image=image,
            segment=segment,
            valid_point=valid_point,
            cls=cls,
            dataset_name=file_info.file_info.base_dir,
            file_name=file_name,
            slice_index=slice_index,
            uid=uid,
        )
        return item

    return transformer


def run(_args):
    device = "cuda:0"
    acdc_segment_mapping = [0, 2, 3, 4]
    mnms_segment_mapping = [0, 2, 3, 4]
    cmri_segment_mapping = [0, 1]

    datasets = [
        ("cmri", CmriDataset2d("cmri", item_transformer=new_item_tranformer(device, cmri_segment_mapping))),
        ("mnms", MnmsDataset2d("mnms", item_transformer=new_item_tranformer(device, mnms_segment_mapping))),
        ("acdc", AcdcDataset2d("acdc", item_transformer=new_item_tranformer(device, acdc_segment_mapping))),
    ]

    item_infos = []

    os.makedirs(".cache/dataset/sam-dataset", exist_ok=True)
    index = 0
    while len(datasets) > 0:
        dataset_name, dataset = datasets.pop(0)
        for item in dataset:
            item_infos.append(
                {
                    "dataset_name": dataset_name,
                    "file_name": item.file_name,
                    "slice_index": item.slice_index,
                    "valid_point": item.valid_point,
                    "cls": item.cls,
                    "uid": f"{dataset_name}_{item.uid}",
                }
            )
            np.savez_compressed(
                f".cache/dataset/sam-dataset/{index}.npz",
                {
                    "image_embeddings": item.image_embedding.cpu().numpy(),
                    "image": item.image.cpu().numpy(),
                    "segment": item.segment.cpu().numpy(),
                },
            )
            index += 1

    json.dump(item_infos, open(".cache/dataset/sam-dataset/dataset.json", "w"))
