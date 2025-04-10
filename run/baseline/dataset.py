import random
from collections import namedtuple
from typing import Any, Callable
import torch
from torch import Tensor

from mos.models.sam.modeling_sam.embedding import (
    BatchPrompt,
)
from mos.models.sam.modeling_sam.embedding.typing import (
    DepthPositionInfoTensor,
    GrayImageTensor,
    SegmentTensor,
)
from enum import Enum

# image shape = (bs, 16, 214, 214)


class DATASET(Enum):
    MAIN = 1
    AUX = 2
    ZERO_SHOT = 3
    pseudo = 4


# FIXME: 改为可以动态调整训练的label
class Label(Enum):
    BG = 0
    EAT = 1
    LV = 2
    RV = 3
    MYO = 4


DataItem = namedtuple("DataItem", ["image", "segment", "dataset", "depth_info"])


def rand_crop_deep(axis: Tensor, data: Tensor, d: int, max_scale=1.0):
    """随机采样deep维度的数据
    Args:
        axis: (bs, d)
        data: (bs, 2, d, h, w)
        d: crop size
        max_scale: crop时采样的间隔
    """
    bs, _, raw_d, h, w = data.shape  # (bs,2,d,h,w)
    if raw_d <= d:
        return axis, data
    index = torch.arange(0, d, device=data.device).unsqueeze(0).repeat(bs, 1)
    begin = torch.randint(0, raw_d - d, (bs, 1), device=data.device).to(torch.float)
    bs_offset = (torch.arange(0, bs, device=data.device) * raw_d).unsqueeze(1).repeat(1, d)

    if max_scale > 1:
        scale = (raw_d - begin) / d
        scale = scale.minimum(torch.full((bs, 1), max_scale, device=data.device))
        scale = torch.rand((bs, 1), device=data.device) * scale
        scale = scale.repeat(1, d)

        index = index * scale + begin
    else:
        index = index + begin

    max = index.max().to(torch.long).item()
    assert max < raw_d, f"max:{max}, raw_d:{raw_d}"
    index = (index + bs_offset).to(torch.long).reshape(-1)
    axis = None if axis is None else axis.reshape(-1).index_select(0, index).reshape(bs, -1)
    data = (
        data.permute(0, 2, 1, 3, 4)
        .reshape(bs * raw_d, 2, h, w)
        .index_select(0, index)
        .reshape(bs, d, 2, h, w)
        .permute(0, 2, 1, 3, 4)
    )
    return axis, data


class BaselineDataset:
    def __init__(
        self,
        train_loop: int = 1,
        train_dataset: str = "cmri712",
        target_device: str = "cuda",
        dataset_device: str = "cuda",
        base_path: str = ".cache/dataset/baseline",
        aux_dataset: set[str] = None,
        image_enhance_level: int = 1,
        sample_count: int = 0,
        image_size=(214, 214),
        crop_size=(128, 128),
        crop_deep_maxscale=1,
        valid_crop_size=(16, 128, 128),
        valid_crop_deep_maxscale=1,
        train_all_data=False,
    ):
        super().__init__()

        self.base_path = base_path
        self.device = target_device

        if dataset_device is None or len(dataset_device) == 0:
            dataset_device = target_device
        self.dataset_device = dataset_device
        self.sample_count = sample_count

        self.is2d = len(crop_size) < 3
        self.train_loop = train_loop
        self.image_size = image_size
        self.crop_size = crop_size
        self.crop_deep_maxscale = crop_deep_maxscale
        self.valid_crop_size = valid_crop_size
        self.valid_crop_deep_maxscale = valid_crop_deep_maxscale

        cmri = torch.load(f"{base_path}/baseline-{train_dataset}.pt", map_location=dataset_device)

        # init default
        self.train2d = None
        self.train3d = None
        self.train3daxis = None
        self.aux3d = None
        self.aux3daxis = None

        if self.is2d:
            self.train2d = cmri["train2d"]  # (bs, 2, h, w)
        else:
            self.train3d = cmri["train3d"]  # (bs, 2, d, h, w)
            self.train3daxis = cmri["train3daxis"] if "train3daxis" in cmri else None  # (bs, d)
        self.valid3d = cmri["valid3d"]
        self.valid3daxis = cmri["valid3daxis"] if "valid3daxis" in cmri else None
        self.test3d = cmri["test3d"]
        self.test3daxis = cmri["test3daxis"] if "test3daxis" in cmri else None

        if aux_dataset is None or len(aux_dataset) == 0:
            if self.is2d:
                self.aux2d = self.train2d
            else:
                self.aux3d = self.train3d
        else:
            aux_2d_list = []
            aux_3d_list = []
            aux_3d_axis_list = []
            for aux in aux_dataset:
                db = torch.load(f"{base_path}/baseline-{aux}.pt", map_location="cpu")
                if self.is2d:
                    aux_2d_list.append(db["train2d"])
                else:
                    aux_3d_list.append(db["train3d"])
                if "train3daxis" in db:
                    aux_3d_axis_list.append(db["train3daxis"])
                print(f'数据集{aux}的大小：3d: {db["train3d"].shape}, 2d: {db["train2d"].shape}')
            if self.is2d:
                self.aux2d = torch.cat(aux_2d_list, dim=0).pin_memory().to(dataset_device)
            else:
                self.aux3d = torch.cat(aux_3d_list, dim=0).pin_memory().to(dataset_device)
                self.aux3daxis = (
                    torch.cat(aux_3d_axis_list, dim=0).to(dataset_device) if len(aux_3d_axis_list) > 0 else None
                )

        if train_all_data:
            print("警告！！！使用所有数据进行训练,包括验证集和测试集！！！")

            data = []
            if self.train2d is not None:
                data.append(self.train2d)
            if "valid2d" in cmri:
                data.append(cmri["valid2d"])
            if "test2d" in cmri:
                data.append(cmri["test2d"])
            if self.aux2d is not None:
                data.append(self.aux2d)
                self.aux2d = self.aux2d[:0]
            if len(data) > 1:
                data = torch.cat(data, dim=0)
                self.train2d = data
                print("合并2d数据集成功", data.shape)

            data = []
            axis = []
            if self.train3d is not None:
                data.append(self.train3d)
                if self.train3daxis is not None:
                    axis.append(self.train3daxis)
            if self.valid3d is not None:
                data.append(self.valid3d)
                self.valid3d = self.valid3d[:0]
                if self.valid3daxis is not None:
                    axis.append(self.valid3daxis)
                    self.valid3daxis = self.valid3daxis[:0]
            if self.test3d is not None:
                data.append(self.test3d)
                self.test3d = self.test3d[:0]
                if self.test3daxis is not None:
                    axis.append(self.test3daxis)
                    self.test3daxis = self.test3daxis[:0]
            if self.aux3d is not None:
                data.append(self.aux3d)
                self.aux3d = self.aux3d[:0]
                if self.aux3daxis is not None:
                    axis.append(self.aux3daxis)
                    self.aux3daxis = self.aux3daxis[:0]
            if len(data) > 1:
                data = torch.cat(data, dim=0)
                axis = torch.cat(axis, dim=0) if len(axis) > 0 else None
                self.train3d = data
                self.train3daxis = axis
                print("合并3d数据集成功", data.shape)

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

        if image_enhance_level >= 2:
            self.train_transform = Compose(
                [
                    RandomRotation(
                        90,
                        interpolation=InterpolationMode.NEAREST,
                        center=(self.image_size[-1] // 2, self.image_size[-1] // 2),
                    ),
                    RandomCrop(self.crop_size[-2:]),
                    RandomVerticalFlip(),
                    RandomHorizontalFlip(),
                ]
            )
        elif image_enhance_level == 1:
            self.train_transform = Compose(
                [
                    RandomCrop(self.crop_size[-2:]),
                ]
            )
        else:
            self.train_transform = Compose(
                [
                    CenterCrop(self.crop_size[-2:]),
                ]
            )

        self.train_image_transform = (
            Compose(
                [
                    RandomEqualize(p=0.1),
                    Lambda(lambda x: x.float() / 255.0),
                    RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
                    RandomAutocontrast(p=0.2),
                    RandomScaleIntensity(scale_lower=0.8, scale_upper=1.3, p=0.2),
                ]
            )
            if image_enhance_level >= 3
            else Compose(
                [
                    Lambda(lambda x: x.float() / 255.0),
                ]
            )
        )
        self.train_image_transform3d = Compose(
            [
                # RandomEqualize(p=0.1), # 报错，直接不用了
                Lambda(lambda x: x.float() / 255.0),
                # RandomScaleIntensity(scale_lower=0.8, scale_upper=1.3, p=0.2), # 报错，直接不用了
            ]
        )
        self.valid_transform = Compose(
            [
                CenterCrop(self.valid_crop_size[-2:]),
            ]
        )
        self.valid_image_transform = Compose(
            [
                Lambda(lambda x: x.float() / 255.0),
            ]
        )

        if sample_count > 0:
            print(f"每个epoch训练只采样{sample_count}个样本")

    def train_len(self) -> int:
        return len(self.train2d if self.is2d else self.train3d)

    def aux_len(self) -> int:
        aux = self.aux2d if self.is2d else self.aux3d
        return 0 if aux is None else len(aux)

    def valid_len(self) -> int:
        return len(self.valid3d)

    def test_len(self) -> int:
        return len(self.test3d)

    def shuffer_train_id(self, batch_size: int, use_loop=True) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cmri_len, aux_len = self.train_len(), self.aux_len()
        if cmri_len >= aux_len:
            cmri_ids = list(range(self.train_len()))
            random.shuffle(cmri_ids)
            if self.aux_len() > 0:
                aux_ids = list(range(self.aux_len()))
                # random sample aux_ids to have the same length as cmri_ids
                aux_ids = random.choices(aux_ids, k=cmri_len)
            else:
                aux_ids = None
        else:
            aux_ids = list(range(self.aux_len()))
            random.shuffle(aux_ids)
            cmri_ids = list(range(self.train_len()))
            # random sample cmri_ids to have the same length as aux_ids
            cmri_ids = random.choices(cmri_ids, k=aux_len)

        if use_loop and self.train_loop > 1:
            cmri_ids = cmri_ids * self.train_loop
            random.shuffle(cmri_ids)
            if aux_ids is not None:
                aux_ids = aux_ids * self.train_loop
                random.shuffle(aux_ids)

        if self.sample_count > 0:
            cmri_ids = cmri_ids[: self.sample_count]
            aux_ids = aux_ids[: self.sample_count] if aux_ids is not None else None

        cmri_ids = list(torch.tensor(cmri_ids).split(batch_size, 0))
        aux_ids = list(torch.tensor(aux_ids).split(batch_size, 0)) if aux_ids is not None else [None] * len(cmri_ids)
        return list(zip(cmri_ids, aux_ids))

    def shuffer_valid_id(self, batch_size: int) -> list[torch.Tensor]:
        if self.valid_len() == 0:
            return []

        return list(torch.arange(self.valid_len(), device=self.dataset_device).split(batch_size, 0))

    def shuffer_test_id(self, batch_size: int) -> list[torch.Tensor]:
        if self.test_len() == 0:
            return []
        return list(torch.arange(self.test_len(), device=self.dataset_device).split(batch_size, 0))

    def batch_get_train(self, ids: torch.Tensor) -> DataItem:
        if self.is2d:
            dataset = DATASET.MAIN
            ids = ids.to(self.dataset_device)

            data = self.train2d.index_select(0, ids).to(self.device)

            data = self.train_transform(data)

            image, segment = data.unbind(1)
            image = image.unsqueeze(1)  # (bs, 1, h, w) or (bs, 1, d, h, w)
            image = self.train_image_transform(image)

            segment = segment.long()

            return DataItem(image, segment, dataset, None)
        else:
            dataset = DATASET.MAIN
            ids = ids.to(self.dataset_device)

            axis = self.train3daxis.index_select(0, ids).to(self.device) if self.train3daxis is not None else None
            data = self.train3d.index_select(0, ids).to(self.device)

            axis, data = rand_crop_deep(axis, data, self.crop_size[0], self.crop_deep_maxscale)

            bs, _, d, h, w = data.shape

            data = data.reshape(bs, -1, h, w)
            data = self.train_transform(data)
            data = data.reshape(bs, 2, *self.crop_size)

            image, segment = data.unbind(1)
            image = image.reshape(bs, -1, *self.crop_size[-2:])
            image = self.train_image_transform3d(image)
            image = image.reshape(bs, 1, *self.crop_size)

            segment = segment.long()

            return DataItem(image, segment, dataset, axis)

    def batch_get_aux(self, ids: torch.Tensor) -> DataItem:
        if ids is None:
            return None
        if self.is2d:
            dataset = DATASET.AUX
            ids = ids.to(self.dataset_device)

            data = self.aux2d.index_select(0, ids).to(self.device)
            data = self.train_transform(data)

            image, segment = data.unbind(1)
            image = image.unsqueeze(1)  # (bs, 1, h, w) or (bs, 1, d, h, w)
            image = self.train_image_transform(image)

            segment = segment.long()

            return DataItem(image, segment, dataset, None)
        else:
            dataset = DATASET.AUX
            ids = ids.to(self.dataset_device)

            axis = self.aux3daxis.index_select(0, ids).to(self.device) if self.aux3daxis is not None else None
            data = self.aux3d.index_select(0, ids).to(self.device)
            axis, data = rand_crop_deep(axis, data, self.crop_size[0], self.crop_deep_maxscale)

            bs, _, d, h, w = data.shape
            data = data.reshape(bs, -1, h, w)
            data = self.train_transform(data)
            data = data.reshape(bs, 2, *self.crop_size)

            image, segment = data.unbind(1)
            image = image.reshape(bs, -1, *self.crop_size[-2:])
            image = self.train_image_transform3d(image)
            image = image.reshape(bs, 1, *self.crop_size)

            segment = segment.long()

            return DataItem(image, segment, dataset, axis)

    def batch_get_valid(self, ids: torch.Tensor) -> DataItem:
        # 不管什么情况，都以3d的方式预测
        return self._batch_get_items(self.valid3d, self.valid3daxis, ids)

    def batch_get_test(self, ids: torch.Tensor) -> DataItem:
        # 不管什么情况，都以3d的方式预测
        return self._batch_get_items(self.test3d, self.test3daxis, ids)

    def _batch_get_items(
        self,
        data,
        axis,
        ids: torch.Tensor,
    ) -> DataItem:
        if ids is None:
            return None
        ids = ids.to(self.dataset_device)

        axis = axis.index_select(0, ids).to(self.device) if axis is not None else None
        data = data.index_select(0, ids).to(self.device)
        axis, data = rand_crop_deep(axis, data, self.valid_crop_size[0], self.valid_crop_deep_maxscale)

        bs, _, d, h, w = data.shape
        data = data.reshape(bs, -1, h, w)
        data = self.valid_transform(data)
        data = data.reshape(bs, 2, *self.valid_crop_size)

        image, segment = data.unbind(1)
        image = image.reshape(bs, -1, *self.valid_crop_size[-2:])
        image = self.valid_image_transform(image)
        image = image.reshape(bs, 1, *self.valid_crop_size)

        segment = segment.long()

        return DataItem(image, segment, DATASET.MAIN, axis)


class ZeroshotDataset:
    def __init__(
        self,
        zeroshot_dataset: str = "zeroshot",
        target_device: str = "cuda",
        dataset_device: str = "cuda",
        base_path: str = ".cache/dataset/baseline",
        image_size=(214, 214),
        valid_crop_size=(16, 128, 128),
        valid_crop_deep_maxscale=1,
    ):
        super().__init__()

        self.base_path = base_path
        self.device = target_device
        self.valid_crop_size = valid_crop_size
        self.valid_crop_deep_maxscale = valid_crop_deep_maxscale
        self.image_size = image_size

        if dataset_device is None or len(dataset_device) == 0:
            dataset_device = target_device
        self.dataset_device = dataset_device

        zeroshot_db = torch.load(f"{base_path}/baseline-{zeroshot_dataset}.pt", map_location=dataset_device)

        self.zeroshot3d = zeroshot_db["crop3d"]  # (bs, [img,seg], d, h, w)
        self.zeroshot3daixs = zeroshot_db["zeroshot3daxis"] if "zeroshot3daxis" in zeroshot_dataset else None  # (bs, d)

        from torchvision.transforms import (
            Compose,
            Lambda,
            CenterCrop,
        )

        self.valid_image_transform = Compose(
            [
                Lambda(lambda x: x.float() / 255.0),
            ]
        )

    def zeroshot_len(self) -> int:
        return len(self.zeroshot3d)

    def shuffer_zeroshot_id(self, batch_size: int) -> list[torch.Tensor]:
        return list(torch.arange(self.zeroshot_len(), device=self.dataset_device).split(batch_size, 0))

    def batch_get_zeroshot(self, ids: torch.Tensor) -> DataItem:
        # 不管什么情况，都以3d的方式预测
        return self._batch_get_items(self.zeroshot3d, self.zeroshot3daixs, ids)

    def _batch_get_items(
        self,
        data,
        axis,
        ids: torch.Tensor,
    ) -> DataItem:

        ids = ids.to(self.dataset_device)
        axis = axis.index_select(0, ids).to(self.device) if axis is not None else None
        data = data.index_select(0, ids).to(self.device)

        axis, data = rand_crop_deep(axis, data, self.valid_crop_size[0], self.valid_crop_deep_maxscale)

        bs, _, d, h, w = data.shape

        image, segment = data.unbind(1)
        image = image.reshape(bs, -1, *self.valid_crop_size[-2:])
        image = self.valid_image_transform(image)
        image = image.reshape(bs, 1, *self.valid_crop_size)

        segment = segment.long()

        return DataItem(image, segment, DATASET.ZERO_SHOT, axis)


class PseudoDataset:
    def __init__(
        self,
        pseudo_dataset: str = "pseudo",
        pseudo_dataset_key: str = "crop3d",
        target_device: str = "cuda",
        dataset_device: str = "cuda",
        base_path: str = ".cache/dataset/baseline",
        image_size=(214, 214),
        valid_crop_size=(16, 128, 128),
        valid_crop_deep_maxscale=1,
    ):
        super().__init__()

        self.base_path = base_path
        self.device = target_device
        self.valid_crop_size = valid_crop_size
        self.valid_crop_deep_maxscale = valid_crop_deep_maxscale
        self.image_size = image_size

        if dataset_device is None or len(dataset_device) == 0:
            dataset_device = target_device
        self.dataset_device = dataset_device

        pseudo_db = torch.load(f"{base_path}/baseline-{pseudo_dataset}.pt", map_location=dataset_device)

        self.zeroshot3d = pseudo_db[pseudo_dataset_key]  # (bs, [img,seg], d, h, w)
        self.zeroshot3daixs = (
            pseudo_db[f"{pseudo_dataset_key}axis"] if f"{pseudo_dataset_key}axis" in pseudo_db else None
        )  # (bs, d)

        from torchvision.transforms import (
            Compose,
            Lambda,
            CenterCrop,
        )

        self.valid_transform = Compose(
            [
                CenterCrop(self.valid_crop_size[-2:]),
            ]
        )
        self.valid_image_transform = Compose(
            [
                Lambda(lambda x: x.float() / 255.0),
            ]
        )

    def preduso_len(self) -> int:
        return len(self.zeroshot3d)

    def shuffer_pseudo_id(self, batch_size: int) -> list[torch.Tensor]:
        return list(torch.arange(self.preduso_len(), device=self.dataset_device).split(batch_size, 0))

    def batch_get_pseudo(self, ids: torch.Tensor) -> DataItem:
        # 不管什么情况，都以3d的方式预测
        return self._batch_get_items(self.zeroshot3d, self.zeroshot3daixs, ids)

    def _batch_get_items(
        self,
        data,
        axis,
        ids: torch.Tensor,
    ) -> DataItem:
        ids = ids.to(self.dataset_device)
        axis = axis.index_select(0, ids).to(self.device) if axis is not None else None
        data = data.index_select(0, ids).to(self.device)
        axis, data = rand_crop_deep(axis, data, self.valid_crop_size[0], self.valid_crop_deep_maxscale)

        bs, _, d, h, w = data.shape
        data = data.reshape(bs, -1, h, w)
        data = self.valid_transform(data)
        data = data.reshape(bs, 2, *self.valid_crop_size)

        image, segment = data.unbind(1)
        image = image.reshape(bs, -1, *self.valid_crop_size[-2:])
        image = self.valid_image_transform(image)
        image = image.reshape(bs, 1, *self.valid_crop_size)

        segment = segment.long()

        return DataItem(image, segment, DATASET.pseudo, axis)


class DatasetLazyLoader:
    def __init__(self, lazy_init: Callable) -> None:
        self.lazy_init = lazy_init
        self.is_loaded = False
        self.dataset = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if not self.is_loaded:
            self.dataset = self.lazy_init(*args, **kwds)
            self.is_loaded = True
        return self.dataset
