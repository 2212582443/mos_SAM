import json
import random
from typing import Any, TypeAlias

import torch
from torch import Tensor

from mos.models.sam.modeling_sam.embedding import (
    BatchPrompt,
)
from mos.models.sam.modeling_sam.embedding.typing import (
    GrayImageTensor,
    SegmentTensor,
)

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

CROP_SIZE = 160

from torchvision.transforms import functional as F


class SegNetDatasetCompat:
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

        self._filter_cmri_dataset()

    def _filter_cmri_dataset(self):
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
        self.valid_meta = meta_list

    def merge_train_valid_data(self):
        self.train_images = torch.cat([self.train_images, self.valid_images], 0)
        self.train_segments = torch.cat([self.train_segments, self.valid_segments], 0)
        self.train_meta = self.train_meta + self.valid_meta

        self.valid_images = None
        self.valid_segments = None
        self.valid_meta = None

    def split_systole_distole_as_train_valid(self):
        self.merge_train_valid_data()
        train_meta_list, train_image_list, train_segment_list = [], [], []
        valid_meta_list, valid_image_list, valid_segment_list = [], [], []
        has_systole = {}
        for meta, image, segment in zip(
            self.train_meta,
            self.train_images.split(1, 0),
            self.train_segments.split(1, 0),
        ):
            uid, phase = meta["uid"], meta["phase"]
            key = f"{uid}"
            if key not in has_systole:
                has_systole[key] = phase
            if phase != has_systole[key]:
                train_meta_list.append(meta)
                train_image_list.append(image)
                train_segment_list.append(segment)
            else:
                valid_meta_list.append(meta)
                valid_image_list.append(image)
                valid_segment_list.append(segment)

        self.train_images = torch.cat(train_image_list, 0)
        self.train_segments = torch.cat(train_segment_list, 0)
        self.train_meta = train_meta_list

        self.valid_images = torch.cat(valid_image_list, 0)
        self.valid_segments = torch.cat(valid_segment_list, 0)
        self.valid_meta = valid_meta_list

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

        for i, id in enumerate(ids_list):
            meta = meta_dic[id]

            start_x, start_y, end_x, end_y = meta["crop_start_range"]
            rand_start_x = random.randint(start_x, end_x)
            rand_start_y = random.randint(start_y, end_y)
            image, segment = images[i], segments[i]

            image = F.crop(image, rand_start_y, rand_start_x, CROP_SIZE, CROP_SIZE)
            segment = F.crop(segment, rand_start_y, rand_start_x, CROP_SIZE, CROP_SIZE)

            images_list.append(image)
            segment_list.append(segment)

        images_list = torch.cat(images_list, 0).float()
        segment_list = torch.cat(segment_list, 0).float()

        return (
            images_list,
            segment_list,
        )
