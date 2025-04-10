import json
import random
from typing import Any

import torch

from mos.models.sam.modeling_sam.embedding.typing import (
    GrayImageTensor,
)

CROP_SIZE = 128
IMAGE_SIZE = 214

from torchvision.transforms import functional as F


class MaeDatasetCompat:
    def __init__(
        self,
        load_finetune_index: int = -1,  # 加载微调数据集
        base_path: str = ".cache/dataset/mae-dataset",
        device_count: int = torch.cuda.device_count(),
    ):
        super().__init__()
        self.base_path = base_path
        self.device = [torch.device("cuda", i) for i in range(device_count)]

        # ======= 验证集 =======
        print("loading datast...")
        data = torch.load(f"{base_path}/dataset-valid.pt", map_location="cpu")
        all_images: torch.Tensor = data["image"]
        assert len(all_images.shape) == 3

        if load_finetune_index >= 0:
            print(f"loading finetune dataset {load_finetune_index}")
            cmri_image = data["cmri_image"]
            index = data[f"cmri_image_index/{load_finetune_index}"]
            cmri_image = cmri_image[index]
            all_images = torch.cat([all_images, cmri_image], dim=0)

        all_images = all_images.unsqueeze(1).pin_memory()  # bs, ch, h, w
        valid_count = all_images.shape[0]

        device_images = []
        # 按照device平均分配数据
        all_image_list: list[torch.Tensor] = all_images.chunk(device_count, 0)
        for images, device in zip(all_image_list, self.device):
            images = images.to(device, non_blocking=True)
            device_images.append(images)
        self.valid_images = device_images

        # ======= 训练集数据 =======
        data = torch.load(f"{base_path}/dataset-train.pt", map_location="cpu")
        all_images: torch.Tensor = data["image"]
        assert len(all_images.shape) == 3  # bs, h ,w

        if load_finetune_index >= 0:
            cmri_image = data["cmri_image"]
            index = data[f"cmri_image_index/{load_finetune_index}"]
            cmri_image = cmri_image[index]
            all_images = torch.cat([all_images, cmri_image], dim=0)

        all_images = all_images.unsqueeze(1).pin_memory()  # bs, ch, h, w
        train_count = all_images.shape[0]

        print(f"dataset {base_path} loaded, train image count: {train_count}, valid image count: {valid_count}")
        device_images = []
        # 按照device平均分配数据
        all_image_list: list[torch.Tensor] = all_images.chunk(device_count, 0)
        for images, device in zip(all_image_list, self.device):
            images = images.to(device, non_blocking=True)
            device_images.append(images)
        self.train_images = device_images

        from mos.utils.transforms.random_scale_intensity import RandomScaleIntensity

        from torchvision.transforms import (
            Compose,
            InterpolationMode,
            Lambda,
            RandomCrop,
            RandomVerticalFlip,
            RandomHorizontalFlip,
            RandomEqualize,
            RandomAdjustSharpness,
            RandomAutocontrast,
            RandomRotation,
        )

        self.train_image_transform = Compose(
            [
                RandomRotation(90, interpolation=InterpolationMode.BILINEAR, center=(IMAGE_SIZE // 2, IMAGE_SIZE // 2)),
                RandomCrop(CROP_SIZE),
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                # RandomEqualize(p=0.1),
                Lambda(lambda x: x.float() / 255.0),
                # RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
                # RandomAutocontrast(p=0.2),
                RandomScaleIntensity(scale_lower=0.8, scale_upper=1.3, p=0.2),
            ]
        )
        self.valid_image_transform = Compose(
            [
                RandomCrop(CROP_SIZE),
                Lambda(lambda x: x.float() / 255.0),
            ]
        )

    def train_len(self) -> int:
        return self.train_images[0].shape[0]

    def valid_len(self) -> int:
        return self.valid_images[0].shape[0]

    def shuffer_train_id(self, batch_size: int) -> list[torch.Tensor]:
        batch_size = batch_size // len(self.device)
        device_image_id_list = []
        for images in self.train_images:
            ids = list(range(images.shape[0]))
            random.shuffle(ids)
            ids = torch.tensor(ids).to(images.device)
            device_image_id_list.append(ids.split(batch_size, 0))

        # 返回每个device应该取的id [(device1, device2,...), ...]
        return [d for d in zip(*device_image_id_list)]

    def shuffer_valid_id(self, batch_size: int) -> list[torch.Tensor]:
        batch_size = batch_size // len(self.device)
        device_image_id_list = []
        for images in self.valid_images:
            ids = list(range(images.shape[0]))
            random.shuffle(ids)
            ids = torch.tensor(ids).to(images.device)
            device_image_id_list.append(ids.split(batch_size, 0))

        # 返回每个device应该取的id [(device1, device2,...), ...]
        return [d for d in zip(*device_image_id_list)]

    def batch_get_train(self, device_ids: list[torch.Tensor]) -> list[GrayImageTensor]:
        images_list = []
        # 对应每个device取对应的数据
        for device, ids in enumerate(device_ids):
            images = self.batch_get_train_device(device, ids)
            images_list.append(images)

        return images_list

    def batch_get_train_device(self, device: int, ids: torch.Tensor) -> GrayImageTensor:
        images = torch.index_select(self.train_images[device], 0, ids)
        images = self.train_image_transform(images)
        return images

    def batch_get_valid(self, device_ids: list[torch.Tensor]) -> list[GrayImageTensor]:
        images_list = []
        # 对应每个device取对应的数据
        for device, ids in enumerate(device_ids):
            images = self.batch_get_valid_device(device, ids)
            images_list.append(images)

        return images_list

    def batch_get_valid_device(self, device: int, ids: torch.Tensor) -> GrayImageTensor:
        images = torch.index_select(self.train_images[device], 0, ids)
        images = self.train_image_transform(images)
        return images
