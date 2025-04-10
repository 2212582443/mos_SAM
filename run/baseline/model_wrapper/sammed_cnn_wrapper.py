import torch.nn as nn
import torch
import torch.nn.functional as F

from run.baseline.model_arguments import ModelArguments
from .label_utils import get_config_train_labels, get_config_eval_labels
from run.baseline.result_wrapper import SeparetionMaskResult

from ..dataset import DataItem
from ..models.sammed_cnn_3d import build_sam3D_vit_b_ori
from ..models.sammed_cnn_3d.modeling import ImageEncoderViT3D, MaskDecoder3D, PromptEncoder3D, Sam3D
import random
from torch.nn.functional import threshold, normalize
from scipy.ndimage import zoom
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def get_point_prompt(prev_seg, label: torch.Tensor, scale=1.0):
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    scale = scale.to(label.device)

    pred_masks = prev_seg > mask_threshold
    true_masks = label > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)
    for i in range(label.shape[0]):
        points = torch.argwhere(to_point_mask[i])
        if len(points) > 0:
            point = random.choice(points)
            is_positive = fn_masks[i, 0, point[1], point[2], point[3]]
            bp = point[1:].reshape(1, 1, 3)
            bp = (bp * scale.unsqueeze(0).unsqueeze(0)).long()
            bl = torch.tensor([int(is_positive)]).reshape(1, 1)
            batch_points.append(bp)
            batch_labels.append(bl)

    return batch_points, batch_labels


import torchio as tio


def scale_image_to_sammed_input_3d(image, label):
    image = F.interpolate(image, size=[128, 128, 128], mode="trilinear", align_corners=False) * 255.0
    image = tio.ZNormalization(masking_method=lambda x: x > 0)(image.squeeze(dim=1))
    image = image.unsqueeze(1)
    return image, label.unsqueeze(1)


class SammedCNNWrapper3d(nn.Module):
    def __init__(
        self,
        device,
        model_args: ModelArguments,
        checkpoint=".cache/dataset/baseline/sam_med3d_turbo.pth",
    ) -> None:
        super().__init__()

        self.device = device
        self.model: Sam3D = build_sam3D_vit_b_ori(checkpoint).to(device)
        # # Create a SummaryWriter object

        # writer = SummaryWriter(log_dir=".cache/dataset/baseline")

        # # Write the model graph to TensorBoard
        # writer.add_graph(self.model.image_encoder, input_to_model=torch.randn(1, 1, 128, 128, 128).to(self.device))

        # # Close the SummaryWriter
        # writer.close()

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
    ):
        """
        Args:
            image: (bs, 1, 128, 128, 128)
            label: (bs, 1, 128, 128, 128)
        """
        label_size = label.shape[-3:]
        prev_masks = torch.zeros_like(label).float()
        low_res_masks = F.interpolate(prev_masks.float(), size=[128 // 4, 128 // 4, 128 // 4])
        # low_res_masks = None

        # prev_masks = prev_masks.squeeze(1)
        # low_res_masks = low_res_masks.squeeze(1)
        scale = torch.tensor([128 / label_size[0], 128 / label_size[1], 128 / label_size[2]])
        point_coords, point_labels = get_point_prompt(prev_masks, label, scale)
        # print(point_coords, point_labels)
        point_coords = torch.cat(point_coords, dim=0)
        point_labels = torch.cat(point_labels, dim=0)
        pt = [point_coords, point_labels]

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=pt,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,  # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        logit = F.interpolate(low_res_masks, size=label_size, mode="trilinear", align_corners=False)

        # logit = threshold(logit, 0.0, 0)
        logit = logit.sigmoid()

        # 只保留有label的部分
        label_only = (label.sum((-3, -2, -1)) > 0).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        logit = logit * label_only

        pred_mask = logit > 0.5  # (bs, d, h, w)
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
            )

            masks = masks.squeeze(1)
            logit = logit.squeeze(1)

            result_logit[k] = logit
            result_mask[k] = masks

        return SeparetionMaskResult(result_logit, result_mask)
