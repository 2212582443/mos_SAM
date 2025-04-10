import random, torch
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

from mos.models.sam.modeling_sam.embedding.typing import (
    GrayImageTensor,
    SegmentTensor,
    TextTokenEmbeddingTensor,
)


CROP_SIZE = 128
IMAGE_SIZE = 214


class SAMDatasetCompatSimple:
    def __init__(
        self,
        device: str,
        base_path: str = ".cache/dataset/text-mae-sam-dataset/dataset-all-label-0.pt",
        only_cmri_dataset: bool = False,
    ):
        super().__init__()
        self.base_path = base_path
        self.device = device
        print(f"loading dataset from {base_path}")

        data = torch.load(base_path)

        (
            image,  # (bs, h, w)
            _train_mae_pair,  # (bs, [src_image_index, token_index, target_image_index*5])
            train_mae_pair_simple,  # (bs, [src_image_index, token_index, target_image_index])
            train_cmri_pair_simple,  # (bs, [src_image_index, token_index, target_image_index])
            _valid_mae_pair,  # (bs, [src_image_index, token_index, target_image_index*5])
            valid_mae_pair_simple,  # (bs, [src_image_index, token_index, target_image_index])
            token_list,  # (N, seq_len, hidden_size)
            token_selector_list,  # (N, [offset, len])
        ) = (
            data["image"],
            data["train_mae_pair"],
            data["train_mae_pair_simple"],
            data["train_cmri_pair_simple"],
            data["valid_mae_pair"],
            data["valid_mae_pair_simple"],
            data["token_list"],
            data["token_selector_list"],
        )

        global IMAGE_SIZE
        IMAGE_SIZE = image.shape[-1]

        self.image: GrayImageTensor = image.to(device)
        self.token_list = token_list.to(device)
        self.token_selector_list = token_selector_list.to(device)

        if only_cmri_dataset:
            self.train_pair = train_cmri_pair_simple.to(device).repeat(10, 1)
        else:
            self.train_pair = train_mae_pair_simple.to(device)
        self.valid_pair = valid_mae_pair_simple.to(device)

        self.train_image_transform = Compose(
            [
                # RandomRotation(90, interpolation=InterpolationMode.BILINEAR, center=(IMAGE_SIZE // 2, IMAGE_SIZE // 2)),
                RandomCrop(CROP_SIZE),
                # RandomVerticalFlip(),
                # RandomHorizontalFlip(),
                # RandomEqualize(p=0.1),
                Lambda(lambda x: x.float() / 255.0),
                # RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
                # RandomAutocontrast(p=0.2),
                # RandomScaleIntensity(scale_lower=0.8, scale_upper=1.3, p=0.2),
            ]
        )
        self.valid_image_transform = Compose(
            [
                CenterCrop(CROP_SIZE),
                # RandomCrop(CROP_SIZE),
                Lambda(lambda x: x.float() / 255.0),
            ]
        )

        print(f"dataset loaded! train_len: {self.train_len()}, valid_len: {self.valid_len()}")

    def train_len(self) -> int:
        return len(self.train_pair)

    def valid_len(self) -> int:
        return len(self.valid_pair)

    def shuffer_train_id(self, batch_size: int) -> list[torch.Tensor]:
        ids = list(range(self.train_len()))
        remain = batch_size - (self.train_len() % batch_size)
        if remain > 0:
            ids += random.choices(ids, k=remain)
        random.shuffle(ids)
        return list(torch.tensor(ids).to(self.device).split(batch_size, 0))

    def shuffer_valid_id(self, batch_size: int) -> list[torch.Tensor]:
        ids = list(range(self.valid_len()))
        random.shuffle(ids)
        return list(torch.tensor(ids).to(self.device).split(batch_size, 0))

    def batch_get_train(self, ids: torch.Tensor) -> tuple[GrayImageTensor, SegmentTensor, TextTokenEmbeddingTensor]:
        return self._batch_get_items(self.train_pair, ids, True)

    def batch_get_valid(self, ids: torch.Tensor) -> tuple[GrayImageTensor, SegmentTensor, TextTokenEmbeddingTensor]:
        return self._batch_get_items(self.valid_pair, ids, False)

    def _batch_get_items(
        self,
        pair: torch.Tensor,
        ids: torch.Tensor,
        is_train: bool,
    ) -> tuple[GrayImageTensor, SegmentTensor, TextTokenEmbeddingTensor]:
        bs = ids.shape[0]

        # [src_image_index, token_index, target_image_index]
        pair = torch.index_select(pair, 0, ids)

        # (bs, 2, h, w)
        image = torch.index_select(self.image, 0, pair[:, [0, 2]].reshape(-1))
        h, w = image.shape[-2:]
        image = image.reshape(bs, 2, h, w)

        if is_train:
            image = self.train_image_transform(image)
        else:
            image = self.valid_image_transform(image)
        image, segment = image.unbind(dim=1)
        # (bs, 1, h ,w)
        image = image.unsqueeze(1)

        # (N, [offset, len])
        token_index = torch.index_select(self.token_selector_list, 0, pair[:, 1])
        token_index = torch.rand(bs, device=self.device) * token_index[:, 1] + token_index[:, 0]
        token: TextTokenEmbeddingTensor = torch.index_select(self.token_list, 0, token_index.long())

        return (image, segment, token)
