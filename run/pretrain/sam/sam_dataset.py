import os
import random
from typing import Any, Callable, TypeAlias

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms

from mos.datasets.acdc.acdc_dataset2d import AcdcDataset2d
from mos.datasets.cmri.cmri_dataset2d import CmriDataset2d
from mos.datasets.common.compose_dataset import ComposeDataset
from mos.datasets.common.mydataset.my_dataset import ItemTransformerContext
from mos.datasets.common.mydataset.my_dataset2d import MyFileInfoFlettern
from mos.datasets.mnms.mnms_dataset2d import MnmsDataset2d
from mos.models.sam.modeling_sam.embedding import Prompt
from mos.models.sam.modeling_sam.embedding.point_embedding import PointPrompt, PointType, rand_get_segment_point
from mos.models.sam.modeling_sam.embedding.typing import (
    GrayImageTensor,
    ImageEmbeddingTensor,
    SegmentTensor,
)
from run.pretrain.sam.token_text import get_cls_text_embedding

SAMDatasetItem: TypeAlias = dict[str, Tensor]
"""SAMDatasetItem 
{
    'image': ImageTensor|None, 图像 \n
    'original_sizes': (h, w), 原始图像的大小\n
    'image_embedding': ImageEmbeddingTensor, 图像的embedding\n
    'segment': SegmentTensor, 选择的segment\n
    'segment_cls': int, 选择的segment的类别\n
    'prompt': Prompt, prompt
}
"""


class SAMDataset(Dataset):
    def __init__(
        self,
        inner_dataset: ComposeDataset,
        device: str,
        image_embeddings: Callable[[GrayImageTensor], tuple[ImageEmbeddingTensor, tuple[int, int]]] | None = None,
        device_image_cache_count: int = 8000,
        **args,
    ):
        super().__init__(**args)
        self.inner_dataset = inner_dataset
        self.device = device
        self.image_embeddings = image_embeddings
        self.use_cache = False
        self.image_embeddings_cache: dict[str, dict[dict, Any]] = {}
        self.device_image_cache_count = device_image_cache_count
        """image_embeddings_cache:
           {
            'idx': {
                'image_embedding': ImageEmbeddingTensor,
                'original_sizes': (h, w),
            }
        """

    def set_image_embeddings_processor(self, image_embeddings):
        self.image_embeddings = image_embeddings

    def __len__(self) -> int:
        return len(self.inner_dataset)

    def __getitem__(self, idx) -> SAMDatasetItem:
        item = self.inner_dataset[idx]

        image = item["image"]

        if self.use_cache:
            if f"{idx}" in self.image_embeddings_cache:
                item.update(self.image_embeddings_cache[f"{idx}"])
            elif self.image_embeddings is not None:
                # (bs, ch, h, w)

                file_info: MyFileInfoFlettern = self.inner_dataset.get_item_info(idx)
                cache_file = f".cache/dataset/sam-vit-embedding/{file_info.hash()}.pt"
                if os.path.exists(cache_file):
                    image_embedding: Tensor = torch.load(cache_file, map_location="cpu")
                    if image_embedding.dtype != torch.float16:
                        image_embedding = image_embedding.half()
                        torch.save(image_embedding, cache_file)
                        image_embedding = image_embedding.pin_memory()
                    if idx <= self.device_image_cache_count:
                        image_embedding = image_embedding.to(self.device, non_blocking=True)
                    image_embedding_dict = {
                        "image_embeddings": image_embedding,
                        "original_sizes": image.shape[-2:-1],
                    }
                    item.update(image_embedding_dict)
                    self.image_embeddings_cache[f"{idx}"] = image_embedding_dict
                else:
                    with torch.no_grad():
                        # transforms.Normalize(
                        #     mean=[0.485, 0.456, 0.406],
                        #     std=[ 0.229, 0.224, 0.225]
                        # )
                        (image_embedding, original_sizes) = self.image_embeddings(image)
                        image_embedding = image_embedding.half()
                        image_embedding_dict = {
                            "image_embeddings": image_embedding.cpu().pin_memory(),
                            "original_sizes": original_sizes,
                        }
                        item.update(image_embedding_dict)

                        image_embedding_dict["image_embeddings"] = image_embedding
                        self.image_embeddings_cache[f"{idx}"] = image_embedding_dict

                        # print('save image_embedding: ', cache_file)
                        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                        torch.save(image_embedding, cache_file)
        else:
            with torch.no_grad(), torch.cuda.amp.autocast():
                (image_embedding, original_sizes) = self.image_embeddings(image)
                # image_embedding = image_embedding.half()
                image_embedding_dict = {
                    "image_embeddings": image_embedding,
                    "original_sizes": original_sizes,
                }
                item.update(image_embedding_dict)

        return item

    @staticmethod
    def collate_fn(batch):
        print("batch: ", batch)
        return default_collate(batch)


def _rand_select_segment(context: ItemTransformerContext, item: dict[str, Tensor], map: list[int]) -> dict[str, Tensor]:
    """随机选择一个segment
    Args:
        item: {
            'segment': SegmentTensor
        }
        map: cls map, 不同的数据集有不同的label, 这个map用于lable对齐, 更改不同数据集的标签, 以免冲突
            e.g. [0, 2, 3, 4] mapping form [0, 1, 2, 3] to [0, 2, 3, 4]
    Returns:
        {
            segment: SegmentTensor, 选择的segment
            segment_cls: int, 选择的segment的类别
        }
    """
    segment: SegmentTensor = item["segment"]
    if segment is None:
        return item

    cls: list[int] = segment.unique().tolist()
    if len(cls) <= 1:
        file: MyFileInfoFlettern = context.file_info
        file_name = file.file_info.file_name
        print("no cls, cls/map/slice: ", cls, map, file.slice_index)
        print("file: ", context.db.base_dir, file_name)
    cls.remove(0)  # remove background
    cls = random.choice(cls)
    segment = torch.where(segment == cls, 1, 0)
    item["segment"] = segment
    if cls >= len(map):
        file: MyFileInfoFlettern = context.file_info
        file_name = file.file_info.file_name
        print("cls/map: ", cls, map)
        print("file: ", context.db.base_dir, file_name)
    item["segment_cls"] = map[int(cls)]
    return item


def _create_prompt(
    context: ItemTransformerContext,
    item: dict[str, Tensor],
    cache: dict[str, Any],
) -> dict[str, Tensor]:
    """创建prompt
    Args:
        item: {
            'segment': SegmentTensor, 选择的segment
            'segment_cls': int, 选择的segment的类别
        }
    Returns:
        {
            'segment': SegmentTensor, 选择的segment
            'segment_cls': int, 选择的segment的类别
            'prompt': Prompt, prompt
        }
    """

    # construct = item["image"]
    construct = None

    segment: SegmentTensor = item["segment"]

    mask: Tensor = None  # 目前先不训练box, 置为空

    # box: BoxPrompt = rand_select_bounding_box(segment, 4)
    # boxes = [box]
    boxes = None  # 目前先不训练box, 置为空

    points = None
    text = None

    if segment is not None:
        segment_cls: int = item["segment_cls"]

        cache_key = context.file_info.hash()
        cache_key = f"{cache_key}-{segment_cls}"
        if False and (cache_key in cache):
            points = cache[cache_key]
            if len(points) <= 1:
                del cache[cache_key]
        else:
            # 这个操作需要同步GPU, 会比较慢
            points = rand_get_segment_point(segment, 100, 4)
            if len(points) > 1:
                cache[cache_key] = points

        # 提取一个随机点
        point = points.pop()
        points: list[PointPrompt] = [PointPrompt(point, PointType.OBJECT_POINT)]

        text: Tensor = get_cls_text_embedding(segment_cls)

    prompt = Prompt(mask, boxes, points, text, construct)

    item["prompt"] = prompt

    return item


_segment_transform = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True, interpolation=transforms.InterpolationMode.NEAREST),
    ]
)
_image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR),
        lambda x: (x - x.min()) / (x.max() - x.min()),
    ]
)
_rand_crop = transforms.RandomCrop(128)


def new_item_tranformer(device, map):
    cache = {}

    def transformer(context: ItemTransformerContext, item):
        # (ch,h,w), (h, w)
        from torchvision.transforms import functional as F

        image, segment = item["image"], item["segment"]

        crop_i, crop_j, crop_h, crop_w = _rand_crop.get_params(image, _rand_crop.size)
        if segment is None:
            print("segment is None!")
        if segment is not None:
            CROP_SIZE = 128
            file_info: MyFileInfoFlettern = context.file_info
            _, raw_h, raw_w = file_info.file_info.shape
            image_scale_w = 224 / raw_w
            image_scale_h = 224 / raw_h

            x1, y1, x2, y2 = file_info.box
            x1 = int(x1 * image_scale_w)
            y1 = int(y1 * image_scale_h)
            x2 = int(x2 * image_scale_w)
            y2 = int(y2 * image_scale_h)

            end_x = x2 - CROP_SIZE
            end_x = max(x1, end_x)
            begin_x = min(x1, x2 - CROP_SIZE)
            begin_x = max(0, begin_x)

            end_y = y2 - CROP_SIZE
            end_y = max(y1, end_y)
            begin_y = min(y1, y2 - CROP_SIZE)
            begin_y = max(0, begin_y)
            rand_start_x = random.randint(begin_x, end_x)
            rand_start_y = random.randint(begin_y, end_y)
            if rand_start_x + CROP_SIZE > 224:
                rand_start_x = 224 - CROP_SIZE
            if rand_start_y + CROP_SIZE > 224:
                rand_start_y = 224 - CROP_SIZE
            crop_i = rand_start_y
            crop_j = rand_start_x
            # print("box", (x1, y1, x2, x2), "crop", crop_j, crop_i, crop_j + crop_w, crop_i + crop_h)

        # (ch, h, w)
        # (bs, ch, h, w)
        image: GrayImageTensor = image.unsqueeze(0)
        image = F.crop(image, crop_i, crop_j, crop_h, crop_w)

        if segment is not None:
            segment = segment.to(device, non_blocking=True)
            # (bs, h, w)
            segment: SegmentTensor = segment.unsqueeze(0)
            # segment: SegmentTensor = _segment_transform(segment)
            segment = F.crop(segment, crop_i, crop_j, crop_h, crop_w)
            item["segment"] = segment

        item["image"] = image

        item = _rand_select_segment(context, item, map)
        item = _create_prompt(context, item, cache)
        return item

    return transformer


acdc_segment_mapping = [0, 2, 3, 4]
mnms_segment_mapping = [0, 2, 3, 4]
cmri_segment_mapping = [0, 1]


def pin_memory(item):
    if "segment" in item:
        item["segment"] = _segment_transform(item["segment"]).pin_memory()
    item["image"] = _image_transform(item["image"]).pin_memory()
    return item


def get_compose_datset_for_train(device):
    """组合多个不同的数据集进行训练"""
    cmri_dataset = CmriDataset2d(
        "cmri",
        item_transformer=new_item_tranformer(device, cmri_segment_mapping),
        file_filter=lambda x: int(x.uid) % 10 != 0,
        on_file_loaded=pin_memory,
    )

    mnms_dataset = MnmsDataset2d(
        "mnms",
        item_transformer=new_item_tranformer(device, mnms_segment_mapping),
        on_file_loaded=pin_memory,
    )

    acdc_dataset = AcdcDataset2d(
        "acdc",
        item_transformer=new_item_tranformer(device, acdc_segment_mapping),
        on_file_loaded=pin_memory,
    )

    compose_dataset = ComposeDataset([cmri_dataset, mnms_dataset, acdc_dataset])
    return compose_dataset


def get_compose_datset_for_validate(device):
    """组合多个不同的数据集进行验证"""
    cmri_dataset = CmriDataset2d(
        "cmri",
        item_transformer=new_item_tranformer(device, cmri_segment_mapping),
        file_filter=lambda x: int(x.uid) % 10 == 0,
        on_file_loaded=pin_memory,
    )

    compose_dataset = ComposeDataset([cmri_dataset])
    return compose_dataset
