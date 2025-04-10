import json
import random
from typing import Any

import torch

from mos.models.sam.modeling_sam.embedding.typing import (
    GrayImageTensor,
)

CROP_SIZE = 160
IMAGE_SIZE = 288

from torchvision.transforms import functional as F


class VxmDatasetCompat:
    def __init__(
        self,
        base_path: str,
        device_count: int,
    ):
        super().__init__()
        self.base_path = base_path
        self.device = [torch.device("cuda", i) for i in range(device_count)]

        self.meta: list[dict[str, Any]] = json.load(open(f"{base_path}/dataset.json"))
        all_images: torch.Tensor = torch.load(f"{base_path}/images.ot", map_location="cpu")
        all_pair_list: torch.Tensor = torch.load(f"{base_path}/pair-list.ot", map_location="cpu")
        all_pair_index_list: torch.Tensor = torch.load(f"{base_path}/pair-index-list.ot", map_location="cpu")

        # self.pair_list = self.pair_list.reshape(-1, 2, 2)[:, 0, :]
        print(f"dataset {base_path} loaded, pair count: {all_pair_list.shape[0]}")
        device_images = []
        device_pair_list = []
        # 按照device平均分配数据
        # 注: pair_index_list是有序的, 保存有所有subject的pair的起始和结束index (uid, start, end)
        pair_index_list: list[torch.Tensor] = all_pair_index_list.chunk(device_count, 0)
        for pair_index, device in zip(pair_index_list, self.device):
            start = pair_index[0, 1].item()
            end = pair_index[-1, 2].item()
            pair_index = all_pair_list[start:end]

            idx = pair_index.reshape(-1).unique()
            start = idx.min().item()
            end = idx.max().item() + 1
            images = all_images[start:end].to(device)

            pair_index = pair_index - start
            pair_index = pair_index.to(device)

            device_images.append(images)
            device_pair_list.append(pair_index)

        self.device_images = device_images
        self.device_pair_list = device_pair_list

    def train_len(self) -> int:
        return self.device_pair_list[0].shape[0]

    def shuffer_train_id(self, batch_size: int) -> list[torch.Tensor]:
        batch_size = batch_size // len(self.device)
        device_pair_list = []
        for pair in self.device_pair_list:
            ids = list(range(pair.shape[0]))
            random.shuffle(ids)
            ids = torch.tensor(ids).to(pair.device)
            ids = pair.index_select(0, ids)
            ids = ids.reshape(-1).split(2 * batch_size, 0)
            device_pair_list.append(ids)

        # 返回每个device应该取的id [(device1, device2,...), ...]
        return [d for d in zip(*device_pair_list)]

    def batch_get_train(self, device_ids: list[torch.Tensor]) -> tuple[list[GrayImageTensor], list[GrayImageTensor]]:
        source_images_list = []
        target_images_list = []
        # 对应每个device取对应的数据
        for device, ids in enumerate(device_ids):
            source_images, target_images = self.batch_get_train_device(device, ids)

            source_images_list.append(source_images)
            target_images_list.append(target_images)

        return (
            source_images_list,
            target_images_list,
        )

    def batch_get_train_device(self, device: int, ids: torch.Tensor) -> tuple[GrayImageTensor, GrayImageTensor]:
        images = torch.index_select(self.device_images[device], 0, ids)

        bs, ch, h, w = images.shape

        assert ch == 1

        source_images, target_images = images.float().reshape(bs // 2, 2, h, w).split(1, 1)

        return (source_images, target_images)
