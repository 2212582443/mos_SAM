from dataclasses import dataclass, field
import torch.nn as nn
import torch
import torch.nn.functional as F

from run.baseline.model_arguments import ModelArguments
from .label_utils import get_config_train_labels, get_config_eval_labels
from run.baseline.result_wrapper import SeparetionMaskResult, parse_custom_args

from ..dataset import DataItem
from ..models.sammed3d import build_sam3D_vit_b_ori
from ..models.sammed3d.modeling import ImageEncoderViT3D, MaskDecoder3D, PromptEncoder3D, Sam3D
import random
from torch.nn.functional import threshold, normalize
from scipy.ndimage import zoom
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp


def get_point_prompt2(label: torch.Tensor, scale=1.0, valid_model=False):
    # (bs, d, h, w)
    # 去掉边缘像素
    bs, d, h, w = label.shape
    # 根据项目的label需求调整要腐蚀的像素大小，避免取到边缘像素的点
    label = F.max_pool3d(-label.float(), kernel_size=7, stride=1, padding=3)
    point_label: list[torch.Tensor] = label.split(1, 0)
    points = []
    for i, x in enumerate(point_label):
        # (d, h, w)
        x = x.squeeze(0).nonzero()

        if len(x) == 0:
            # 没有label
            x = torch.zeros(3, dtype=torch.int64, device=label.device)
        else:
            if valid_model:
                pos = len(x) // 2
            else:
                pos = random.randint(0, len(x) - 1)
            x = x[pos]

        # [(z,y,x)] -> [(x,y,z)]
        # x = x[[2, 1, 0]]
        points.append(x)

    # (bs, 3)
    point_coords = torch.stack(points, dim=0)
    # (bs, N, 3)
    point_coords = point_coords.unsqueeze(1).float()
    point_coords *= scale  # 缩放到image的坐标系

    # (bs, N)
    point_labels = torch.ones(bs, 1, dtype=torch.int64)

    return point_coords, point_labels


def scale_image_to_sammed_input_3d(image, label):
    import torchio as tio

    # image = F.interpolate(image, size=[128, 128, 128], mode="trilinear", align_corners=False)
    image = image * 255.0
    image = tio.ZNormalization(masking_method=lambda x: x > 0)(image.squeeze(dim=1))
    image = image.unsqueeze(1)
    return image, label


@dataclass
class SammedWrapperArguments(object):
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    freeze_vit: bool = field(
        default=False,
        metadata={"help": ("冻结vit")},
    )
    freeze_prompt: bool = field(
        default=False,
        metadata={"help": ("冻结prompt encoder")},
    )
    freeze_decoder: bool = field(
        default=False,
        metadata={"help": ("冻结mask decoder")},
    )


class SammedWrapper3d(nn.Module):
    def __init__(
        self,
        device,
        model_args: ModelArguments,
        args: list[str] = None,
        checkpoint=".cache/dataset/baseline/sam_med3d_turbo.pth",
    ) -> None:
        super().__init__()

        if args is None or len(args) > 0:
            self.sammed_args: SammedWrapperArguments = parse_custom_args(args, SammedWrapperArguments)[0]
        else:
            self.sammed_args = SammedWrapperArguments()

        self.device = device
        self.model: Sam3D = build_sam3D_vit_b_ori(checkpoint).to(device)

        if self.sammed_args.freeze_vit:
            print("freeze vit encoder")
            for p in self.model.image_encoder.parameters():
                p.requires_grad = False

        if self.sammed_args.freeze_prompt:
            print("freeze prompt encoder")
            for p in self.model.prompt_encoder.parameters():
                p.requires_grad = False

        mask_decoder_grad = not self.sammed_args.freeze_decoder
        for p in self.model.mask_decoder.parameters():
            p.requires_grad = mask_decoder_grad

        self.crop_size = model_args.get_dataset_crop_size()
        self.label_count = model_args.label_count

        _target_label_id_list_cuda, target_label_id_list = get_config_train_labels(model_args, device)
        self.train_target_label_id_list = target_label_id_list
        _target_label_id_list_cuda, target_label_id_list = get_config_eval_labels(model_args, device)
        self.eval_target_label_id_list = target_label_id_list

    def forward(self, data_item: DataItem):
        return self._forward3d(data_item)

    def forward_valid(self, data_item: DataItem):
        return self._forward3d(data_item, True)

    def forward_test(self, data_item: DataItem):
        return self._forward3d(data_item, True, True)

    def forward_zeroshot(self, data_item: DataItem):
        return self._forward3d(data_item, True, True)

    def forward_presudo(self, data_item: DataItem):
        return self._forward3d(data_item, True)

    def forward_sammed_3d(
        self,
        label,
        image_embeddings,
        eval_model,
    ):
        """
        Args:
            image: (bs, 1, 128, 128, 128)
            label: (bs, 128, 128, 128)
        """
        # (bs, 1, d, h, w)
        low_res_masks = torch.zeros((label.shape[0], 1, 128 // 4, 128 // 4, 128 // 4), device=label.device)

        point_coords, point_labels = get_point_prompt2(label, 1.0, eval_model)
        pt = [point_coords, point_labels]

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=pt,
            boxes=None,
            masks=low_res_masks,
        )
        logit, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,  # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        logit = logit.sigmoid()
        logit = logit.squeeze(1)

        # 只保留有label的部分
        label_only = (label.sum((-3, -2, -1)) > 0).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        logit = logit * label_only

        pred_mask = logit > 0.5  # (bs, d, h, w)
        assert logit.dim() == 4
        return pred_mask, logit

    def _forward3d(
        self,
        data_item: DataItem,
        eval=False,
        all_label=False,
    ):
        image, label, db = data_item.image, data_item.segment, data_item.dataset

        # image: (bs, 1, d, h, w)
        # label: (bs, d, h, w)

        result_logit = [None] * self.label_count
        result_mask = [None] * self.label_count

        index = 0 if all_label else db.value
        target_label_ids = self.eval_target_label_id_list[index] if eval else self.train_target_label_id_list[index]

        input_images, input_labels = scale_image_to_sammed_input_3d(image, label)

        image_embeddings = self.model.image_encoder(input_images)
        for k in target_label_ids:
            masks, logit = self.forward_sammed_3d(
                input_labels == k,
                image_embeddings,
                eval,
            )

            # (bs, d, h ,w)
            masks = masks.squeeze(1)
            logit = logit.squeeze(1)

            result_logit[k] = logit
            result_mask[k] = masks

        return SeparetionMaskResult(result_logit, result_mask)
