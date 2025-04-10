from dataclasses import dataclass, field
from typing import List
from PIL import Image
from mos.datasets.cmri.cmri_dataset2d import CmriDataset2d
from mos.datasets.common.mydataset.my_dataset import ItemTransformerContext
from mos.datasets.common.mydataset.my_dataset2d import MyFileInfoFlettern
from mos.models.sam import (
    SamVisionConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
    SamConfig,
    SamModel,
    SamProcessor,
    SamImageProcessor,
)
from transformers import (
    SamModel as SamModelPretrain,
    HfArgumentParser,
    TrainingArguments,
)

import os
from mos.models.sam.modeling_sam.embedding.image_embedding_sam import ImageEmbeddingSam
from mos.models.sam.modeling_sam.embedding.typing import (
    GrayImageTensor,
    SparseEmbeddingsTensor,
    SparseEmbeddingsTensor,
    DenseEmbeddingsTensor,
    PredIouScoresTensor,
    PredMasksTensor,
    ImageEmbeddingTensor,
    SegmentTensor,
)
from mos.models.sam.modeling_sam.embedding import Prompt
from mos.models.sam.modeling_sam.sam_model import SamImageSegmentationOutput
from mos.utils.plot.plot_file import save_image_tensor
from run.pretrain.sam.token_text import get_cls_text_embedding
from .model_factory import ModelFactory
from .batch_soft_dice import BatchSoftIoU, DiceScore

from .sam_dataset import SAMDataset, SAMDatasetItem, get_compose_datset_for_train
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

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
        lambda x: (x - x.min()) / (x.max() - x.min()),
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
        image = image.to(torch.float)
        item["image"] = image

        # (h, w)
        segment = item["segment"]
        segment = segment.to(device)
        # (bs, h, w)
        segment: SegmentTensor = segment.unsqueeze(0)
        segment: SegmentTensor = _segment_transform(segment)
        segment = segment.to(torch.int8)
        item["segment"] = segment

        cls: list[int] = segment.unique().tolist()
        cls.remove(0)
        item["cls"] = cls

        valid_point = {}
        for c in cls:
            _, y_index, x_index = torch.nonzero(segment == c, as_tuple=True)
            point = [(x * scaling, y * scaling) for x, y in zip(x_index.tolist(), y_index.tolist())]
            valid_point[f"{c}"] = point
        item["valid_point"] = valid_point

        file_info: MyFileInfoFlettern = context.file_info
        file_name = file_info.file_info.file_name
        item["file_name"] = file_name

        uid = file_info.file_info.uid
        item["uid"] = uid

        slice_index = file_info.slice_index
        item["slice_index"] = slice_index

        cache_file = f".cache/dataset/sam-vit-embedding/{file_info.hash()}.pt"
        if os.path.exists(cache_file):
            image_embedding = torch.load(cache_file)
        else:
            image_embedding = _imagembed_image(image)

        image_embedding = image_embedding.to(device)
        item["image_embeddings"] = image_embedding

        text: torch.Tensor = get_cls_text_embedding(cls[0])
        prompt = Prompt(None, None, None, text)
        item["prompt"] = prompt

        return item

    return transformer


def run(_):
    run_name = "sam0804"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_factory = ModelFactory(device, run_name)

    model: SamModel = model_factory.load_model(
        # f".cache/model/{run_name}/sam_model-2100.pt"
    )
    calc_dice_score = DiceScore()
    cmri_segment_mapping = [0, 1]
    dataset = CmriDataset2d(
        "cmri",
        item_transformer=new_item_tranformer(device, cmri_segment_mapping),
        file_filter=lambda x: int(x.uid) % 10 == 0,
    )

    def collate_fn(batch: list[SAMDatasetItem]) -> dict[str, torch.Tensor]:
        item = batch[0]
        prompt = item["prompt"]
        sparse_embedding, dense_embedding = model.prompt(prompt, prompt_to_keep=1)
        item["sparse_embedding"] = sparse_embedding
        item["dense_embedding"] = dense_embedding

        return item

    valid_dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )

    model.eval()
    losses = []
    for batch in tqdm(valid_dataloader):
        _pred_iou_scores, pred_masks = model(
            image_embeddings=batch["image_embeddings"],
            sparse_embeddings=batch["sparse_embedding"],
            dense_embeddings=batch["dense_embedding"],
            multimask_output=False,
        )
        labeled_segment = batch["segment"]
        first_pred_masks = pred_masks[:, 0, 0, :, :]
        dice_loss = calc_dice_score(first_pred_masks, labeled_segment)
        dice_loss = dice_loss.item()

        uid = batch["uid"]
        slice_index = batch["slice_index"]
        dice_level = int(dice_loss * 10)
        image_file = f".checkpoint/sam-perf/{run_name}-latest/{dice_level}/{uid}-{slice_index}.png"

        image = batch["image"].repeat(1, 3, 1, 1)

        image[0, 0, :, :] = torch.where(first_pred_masks >= 0.5, image[0, 0, :, :] / 2 + 0.5, image[0, 0, :, :])
        image[0, 1, :, :] = torch.where(labeled_segment == 1, image[0, 1, :, :] / 2 + 0.5, image[0, 1, :, :])

        save_image_tensor(image, image_file)

        losses.append(dice_loss)
    losses.sort()
    print("dice_loss: ", mean(losses), losses)
