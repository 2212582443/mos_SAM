import json
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
from mos.models.sam.modeling_sam.embedding import (
    BatchPrompt,
    PointBatchPrompt,
    Prompt,
    SamPromptEncoder,
    TextBatchPrompt,
)
from mos.models.sam.modeling_sam.embedding.point_embedding import (
    PointPrompt,
    PointType,
    point2tensor,
    rand_get_segment_point,
)
from mos.models.sam.modeling_sam.embedding.text_embedding import text2tensor
from mos.models.sam.modeling_sam.embedding.typing import (
    GrayImageTensor,
    ImageEmbeddingTensor,
    PointCoordTensor,
    PointLabelTensor,
    SegmentTensor,
    TextTokenEmbeddingTensor,
)
from run.pretrain.sam.token_text import get_cls_text_embedding
import asyncio

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

CROP_SIZE = 128

from torchvision.transforms import functional as F


class SAMDatasetCompat:
    def __init__(
        self,
        base_path: str,
        device: str,
    ):
        super().__init__()
        self.base_path = base_path
        self.device = device

        self.valid_meta: list[dict[str, Any]] = json.load(open(f"{base_path}/valid-metas.json"))
        self.valid_images: GrayImageTensor = torch.load(f"{base_path}/valid-images.ot", map_location=device)
        self.valid_segments: SegmentTensor = torch.load(f"{base_path}/valid-segments.ot", map_location=device)

        self.train_meta: list[dict[str, Any]] = json.load(open(f"{base_path}/train-metas.json"))
        self.train_images: GrayImageTensor = torch.load(f"{base_path}/train-images.ot", map_location=device)
        self.train_segments: SegmentTensor = torch.load(f"{base_path}/train-segments.ot", map_location=device)

    def filter_cmri_dataset(self):
        id_list = []
        meta_list = []
        for i, meta in enumerate(self.train_meta):
            if meta["database_name"] == "cmri":
                id_list.append(i)
                meta_list.append(meta)
        id_list = torch.tensor(id_list).to(self.device)
        self.train_images = torch.index_select(self.train_images, 0, id_list)
        self.train_segments = torch.index_select(self.train_segments, 0, id_list)
        self.train_meta = meta_list

        id_list = []
        meta_list = []
        for i, meta in enumerate(self.valid_meta):
            if meta["database_name"] == "cmri":
                id_list.append(i)
                meta_list.append(meta)
        id_list = torch.tensor(id_list).to(self.device)
        self.valid_images = torch.index_select(self.valid_images, 0, id_list)
        self.valid_segments = torch.index_select(self.valid_segments, 0, id_list)

    def train_len(self) -> int:
        return len(self.train_meta)

    def valid_len(self) -> int:
        return len(self.valid_meta)

    def shuffer_train_id(self, batch_size: int) -> list[torch.Tensor]:
        ids = list(range(self.train_len()))
        random.shuffle(ids)
        return list(torch.tensor(ids).split(batch_size, 0))

    def shuffer_valid_id(self, batch_size: int) -> list[torch.Tensor]:
        ids = list(range(self.valid_len()))
        random.shuffle(ids)
        return list(torch.tensor(ids).split(batch_size, 0))

    def batch_get_train(self, ids: torch.Tensor) -> tuple[GrayImageTensor, SegmentTensor, BatchPrompt]:
        return self._batch_get_items(self.train_images, self.train_segments, self.train_meta, ids)

    def batch_get_valid(self, ids: torch.Tensor) -> tuple[GrayImageTensor, SegmentTensor, BatchPrompt]:
        return self._batch_get_items(self.valid_images, self.valid_segments, self.valid_meta, ids)

    def _batch_get_items(
        self,
        batch_images: GrayImageTensor,
        batch_segments: SegmentTensor,
        meta_dic: list[dict[str, Any]],
        ids: torch.Tensor,
    ) -> tuple[GrayImageTensor, SegmentTensor, BatchPrompt]:
        ids_list = ids.tolist()

        ids = ids.to(self.device)
        images = torch.index_select(batch_images, 0, ids).split(1, 0)
        segments = torch.index_select(batch_segments, 0, ids).split(1, 0)

        images_list = []
        segment_list = []

        # prompt_type = 1 if random.random() < 0.9 else 0
        prompt_type = 0  # 只使用text prompt
        prompt_text_list: list[TextTokenEmbeddingTensor] = []
        points_list: list[list[PointPrompt]] = []

        for i, id in enumerate(ids_list):
            meta = meta_dic[id]

            start_x, start_y, end_x, end_y = meta["crop_start_range"]
            rand_start_x = random.randint(start_x, end_x)
            rand_start_y = random.randint(start_y, end_y)
            image, segment = images[i], segments[i]

            image = F.crop(image, rand_start_y, rand_start_x, CROP_SIZE, CROP_SIZE)
            segment = F.crop(segment, rand_start_y, rand_start_x, CROP_SIZE, CROP_SIZE)

            if prompt_type == 0:
                cls = meta["cls"]
                prompt_text: TextTokenEmbeddingTensor = get_cls_text_embedding(cls)
                prompt_text_list.append(prompt_text)
            else:
                segment_point: list[tuple[int, int]] = meta["segment_points"]
                # random select one point
                x1, y1, x2, y2 = rand_start_x, rand_start_y, rand_start_x + CROP_SIZE, rand_start_y + CROP_SIZE
                for _ in range(len(segment_point)):
                    x, y = random.choice(segment_point)
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        x = x - rand_start_x
                        y = y - rand_start_y
                        break
                prompt_points: list[PointPrompt] = [PointPrompt((x, y), PointType.OBJECT_POINT)]
                points_list.append(prompt_points)

            images_list.append(image)
            segment_list.append(segment)

        if prompt_type == 0:
            batch_prompt = TextBatchPrompt(prompt_text_list)
        else:
            batch_prompt = PointBatchPrompt(points_list)

        images_list = torch.cat(images_list, 0).float()
        segment_list = torch.cat(segment_list, 0).float()

        return (images_list, segment_list, batch_prompt)
