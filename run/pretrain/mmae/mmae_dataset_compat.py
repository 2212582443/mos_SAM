import json
import random
from typing import Any

import torch
from mos.models.mmae.model_mmae import SrcImageType, TargetImageType

from mos.models.sam.modeling_sam.embedding.typing import (
    ContrastMatrixTensor,
    GrayImageTensor,
)

CROP_SIZE = 128
IMAGE_SIZE = 214

from torchvision.transforms import functional as F


class MmaeDatasetCompat:
    def __init__(
        self,
        base_path: str = ".cache/dataset/mmae-dataset",
        device_count: int = torch.cuda.device_count(),
        random_to_unknown_rate: float = 0.3,  # 随机把图片的type改成unknown
        load_finetune_index: int = -1,  # 加载微调数据集
    ):
        super().__init__()
        self.base_path = base_path
        self.random_to_unknown_rate = random_to_unknown_rate
        self.device = [torch.device("cuda", i) for i in range(device_count)]

        # ======= 加载数据集 =======
        print("loading dataset...")
        data = torch.load(f"{base_path}/dataset.pt", map_location="cpu")
        all_images, train_all_pair, valid_all_pair, all_cmri_pair = (
            data["image"],
            data["train_pair"],
            data["valid_pair"],
            data["all_cmri_pair"],
        )
        all_images = all_images.pin_memory()  # bs, h, w
        assert len(all_images.shape) == 3

        if load_finetune_index >= 0:
            print(f"loading finetune dataset {load_finetune_index}")
            train_cmri_pair, valid_cmri_pair = (
                data[f"train_cmri_pair/{load_finetune_index}"],
                data[f"valid_cmri_pair/{load_finetune_index}"],
            )  # (bs, [src type, src index, tgt type, tgt index]])
            valid_all_pair = torch.cat([valid_all_pair, valid_cmri_pair], dim=0)
            train_all_pair = torch.cat([train_all_pair, train_cmri_pair], dim=0)
        else:
            valid_all_pair = torch.cat([valid_all_pair, all_cmri_pair], dim=0)

        # 按照device平均分配数据
        device_images_list = [[] for _ in range(device_count)]
        device_image_index_list = [[] for _ in range(device_count)]
        device_image_index_dict = [{} for _ in range(device_count)]
        train_device_pair_list = [[] for _ in range(device_count)]
        valid_device_pair_list = [[] for _ in range(device_count)]
        target_modality_statics = dict()

        pair_index_list: list[torch.Tensor] = train_all_pair.chunk(device_count * 1000, 0)
        for i, pair_index in enumerate(pair_index_list):
            i = i % device_count
            pair_index = pair_index.tolist()
            for src_type, src_index, tgt_type, tgt_index in pair_index:
                if tgt_type not in target_modality_statics:
                    target_modality_statics[tgt_type] = 0
                target_modality_statics[tgt_type] += 1

                if src_index not in device_image_index_dict[i]:
                    device_image_index_dict[i][src_index] = len(device_image_index_list[i])
                    device_image_index_list[i].append(src_index)
                    device_images_list[i].append(src_index)
                src_index = device_image_index_dict[i][src_index]

                if tgt_index not in device_image_index_dict[i]:
                    device_image_index_dict[i][tgt_index] = len(device_image_index_list[i])
                    device_image_index_list[i].append(tgt_index)
                    device_images_list[i].append(tgt_index)
                tgt_index = device_image_index_dict[i][tgt_index]

                train_device_pair_list[i].append([src_type, src_index, tgt_type, tgt_index])

        pair_index_list: list[torch.Tensor] = valid_all_pair.chunk(device_count * 10, 0)
        for i, pair_index in enumerate(pair_index_list):
            i = i % device_count
            pair_index = pair_index.tolist()
            for src_type, src_index, tgt_type, tgt_index in pair_index:
                if src_index not in device_image_index_dict[i]:
                    device_image_index_dict[i][src_index] = len(device_image_index_list[i])
                    device_image_index_list[i].append(src_index)
                    device_images_list[i].append(src_index)
                src_index = device_image_index_dict[i][src_index]

                if tgt_index not in device_image_index_dict[i]:
                    device_image_index_dict[i][tgt_index] = len(device_image_index_list[i])
                    device_image_index_list[i].append(tgt_index)
                    device_images_list[i].append(tgt_index)
                tgt_index = device_image_index_dict[i][tgt_index]

                valid_device_pair_list[i].append([src_type, src_index, tgt_type, tgt_index])

        device_images_list = [
            torch.index_select(all_images, 0, torch.tensor(ids)).to(device)
            for ids, device in zip(device_images_list, self.device)
        ]
        train_device_pair_list = [
            torch.tensor(pair).to(device) for pair, device in zip(train_device_pair_list, self.device)
        ]
        valid_device_pair_list = [
            torch.tensor(pair).to(device) for pair, device in zip(valid_device_pair_list, self.device)
        ]

        # 按照device平均分配数据
        self.device_images = device_images_list
        self.train_device_pairs = train_device_pair_list
        self.valid_device_pairs = valid_device_pair_list

        print(
            f"dataset {base_path} loaded, train pair count: {train_all_pair.shape[0]}, valid pair count: {valid_all_pair.shape[0]}"
        )
        print(f"target modality statics: {target_modality_statics}")

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
            CenterCrop,
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
                CenterCrop(CROP_SIZE),
                Lambda(lambda x: x.float() / 255.0),
            ]
        )

    def train_len(self) -> int:
        return self.train_device_pairs[0].shape[0]

    def valid_len(self) -> int:
        return self.valid_device_pairs[0].shape[0]

    def shuffer_train_id(self, batch_size: int) -> list[torch.Tensor]:
        batch_size = batch_size // len(self.device)
        device_pair_id_list = []
        for pair in self.train_device_pairs:
            ids = list(range(pair.shape[0]))
            random.shuffle(ids)
            ids = torch.tensor(ids).to(pair.device)
            device_pair_id_list.append(ids.split(batch_size, 0))

        # 返回每个device应该取的id [(device1, device2,...), ...]
        return [d for d in zip(*device_pair_id_list)]

    def shuffer_valid_id(self, batch_size: int) -> list[torch.Tensor]:
        batch_size = batch_size // len(self.device)
        device_image_id_list = []
        for pair in self.valid_device_pairs:
            ids = list(range(pair.shape[0]))
            random.shuffle(ids)
            ids = torch.tensor(ids).to(pair.device)
            device_image_id_list.append(ids.split(batch_size, 0))

        # 返回每个device应该取的id [(device1, device2,...), ...]
        return [d for d in zip(*device_image_id_list)]

    def batch_get_train(
        self, device_ids: list[torch.Tensor]
    ) -> tuple[
        list[SrcImageType],
        list[GrayImageTensor],
        list[TargetImageType],
        list[GrayImageTensor],
        list[ContrastMatrixTensor],
    ]:
        source_image_types = []
        source_images_list = []
        target_image_types = []
        target_images_list = []
        contrast_matrix_list = []
        # 对应每个device取对应的数据
        for device, ids in enumerate(device_ids):
            source_types, source_images, target_types, target_images, contrast_matrix = self.batch_get_train_device(
                device,
                ids,
            )

            source_image_types.append(source_types)
            source_images_list.append(source_images)
            target_image_types.append(target_types)
            target_images_list.append(target_images)
            contrast_matrix_list.append(contrast_matrix)

        return (
            source_image_types,
            source_images_list,
            target_image_types,
            target_images_list,
            contrast_matrix_list,
        )

    def batch_get_train_device(
        self,
        device: int,
        ids: torch.Tensor,
    ) -> tuple[SrcImageType, GrayImageTensor, TargetImageType, GrayImageTensor, ContrastMatrixTensor]:
        bs = ids.shape[0]

        pairs = torch.index_select(self.train_device_pairs[device], 0, ids)
        source_types, target_types = torch.unbind(pairs[:, [0, 2]], 1)

        all_ids = pairs[:, [1, 3]].view(-1)
        images = torch.index_select(self.device_images[device], 0, all_ids)
        h, w = images.shape[-2:]
        images = images.reshape(bs, 2, h, w)  # bs, 2, h, w
        images = self.train_image_transform(images)

        source_images, target_images = images.split(1, 1)  # bs, 1, h, w

        if self.random_to_unknown_rate > 0:
            # 随机把target type 改成 unknown
            # 条件是 source type == target type
            mask = source_types == target_types
            mask = mask * (torch.rand_like(mask, dtype=torch.float) < self.random_to_unknown_rate)
            target_types[mask] = 0
            # 随机把source type 改成unknown
            mask = torch.rand_like(source_types, dtype=torch.float) < self.random_to_unknown_rate
            source_types[mask] = 0

        target_types: TargetImageType = target_types.unsqueeze(1)

        # 对比学习矩阵
        contrast_matrix = pairs[:, 1]  # (bs, )
        contrast_matrix = contrast_matrix.unsqueeze(1) == contrast_matrix.unsqueeze(0)  # (bs, bs)

        # print(source_types.shape, source_images.shape, target_types.shape, target_images.shape)
        return (source_types, source_images, target_types, target_images, contrast_matrix)

    def batch_get_valid(
        self, device_ids: list[torch.Tensor]
    ) -> tuple[
        list[SrcImageType],
        list[GrayImageTensor],
        list[TargetImageType],
        list[GrayImageTensor],
        list[ContrastMatrixTensor],
    ]:
        source_types_list = []
        source_images_list = []
        target_types_list = []
        target_images_list = []
        contrast_matrix_list = []

        for device, ids in enumerate(device_ids):
            source_types, source_images, target_types, target_images, contrast_matrix = self.batch_get_valid_device(
                device,
                ids,
            )

            source_types_list.append(source_types)
            source_images_list.append(source_images)
            target_types_list.append(target_types)
            target_images_list.append(target_images)
            contrast_matrix_list.append(contrast_matrix)

        return (source_types_list, source_images_list, target_types_list, target_images_list, contrast_matrix_list)

    def batch_get_valid_device(
        self,
        device: int,
        ids: torch.Tensor,
    ) -> tuple[SrcImageType, GrayImageTensor, TargetImageType, GrayImageTensor, ContrastMatrixTensor]:
        bs = ids.shape[0]

        # (bs, [src type, src index, tgt type, tgt index])
        pairs = torch.index_select(self.valid_device_pairs[device], 0, ids)
        source_types, target_types = torch.unbind(pairs[:, [0, 2]], 1)

        all_ids = pairs[:, [1, 3]].view(-1)
        images = torch.index_select(self.device_images[device], 0, all_ids)
        h, w = images.shape[-2:]
        images = images.reshape(bs, 2, h, w)  # bs, 2, h, w
        images = self.valid_image_transform(images)

        source_images, target_images = images.split(1, 1)  # (bs, 1, h, w)

        target_types: TargetImageType = target_types.unsqueeze(1)

        # 对比学习矩阵
        contrast_matrix = pairs[:, 1]  # (bs, )
        contrast_matrix = contrast_matrix.unsqueeze(1) == contrast_matrix.unsqueeze(0)  # (bs, bs)

        return (source_types, source_images, target_types, target_images, contrast_matrix)
