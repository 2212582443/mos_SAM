import torch.nn as nn
import torch
import torch.nn.functional as F
from run.baseline.model_arguments import ModelArguments
from run.baseline.result_wrapper import SeparetionMaskResult

from ..dataset import DataItem
from ..models import EfficientSam
from .label_utils import get_config_train_labels, get_config_eval_labels
import random


def get_point_prompt(label: torch.Tensor, scale=8.0):
    # (bs, h, w)
    # 去掉边缘像素
    bs, h, w = label.shape
    point_label: list[torch.Tensor] = F.max_pool2d(label.float(), kernel_size=3, stride=1, padding=1).split(1, 0)
    points = []
    for i, x in enumerate(point_label):
        # (h, w)
        x = x.squeeze(0).nonzero()

        if len(x) == 0:
            # 没有label
            x = torch.zeros(2, dtype=torch.int64, device=label.device)
        else:
            pos = random.randint(0, len(x) - 1)
            x = x[pos]

        # [(y,x)] -> [(x,y)]
        x = x[[1, 0]]
        points.append(x)

    # (bs, 2)
    point_coords = torch.stack(points, dim=0)
    # (bs, 1, N, 2)
    point_coords = point_coords.unsqueeze(1).unsqueeze(1).float()
    point_coords *= scale  # 缩放到1024的坐标系

    # (bs, 1, N)
    point_labels = torch.ones(bs, 1, 1, dtype=torch.int64).to(label)

    return point_coords, point_labels


def scale_image_to_sam_input_3d(image):
    image = F.interpolate(image, size=1024, mode="bilinear", align_corners=False) * 255
    image = image.repeat(1, 3, 1, 1)
    return image


def scale_sam_segment_output_3d(output, size):
    return F.interpolate(output, size=size, mode="bilinear", align_corners=False)


class EfficientSamWrapper3d(nn.Module):
    def __init__(self, device, model: EfficientSam, model_args: ModelArguments) -> None:
        super().__init__()

        self.device = device
        self.model: EfficientSam = model.to(device)
        self.aux_labels: list[int] = model_args.get_aux_dataset_labels()
        self.crop_size = model_args.get_dataset_crop_size()[-2:]

        _target_label_id_list_cuda, target_label_id_list = get_config_train_labels(model_args, device)
        self.train_target_label_id_list = target_label_id_list
        _target_label_id_list_cuda, target_label_id_list = get_config_eval_labels(model_args, device)
        self.eval_target_label_id_list = target_label_id_list

    def forward(self, data_item: DataItem):
        return self._forward3d(data_item)

    def forward_valid(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True)

    def forward_test(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True, True)

    def forward_zeroshot(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True, True)

    def forward_pseudo(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True)

    def forward_sam_3d(self, image, label: torch.Tensor, image_embeddings):
        """
        Args:
            image: (bs, 1, h, w)
            label: (bs, h, w)
        """

        label_size = label.shape[-2:]
        scale = 1024 / image.shape[-1]
        point_coords, point_labels = get_point_prompt(label, scale)
        logit, iou_predictions = self.model.predict_masks(
            image_embeddings=image_embeddings,
            batched_points=point_coords,
            batched_point_labels=point_labels,
            multimask_output=False,
            input_h=1024,
            input_w=1024,
        )

        # 根据iou排序，然后依次返回iou最大的几个
        sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
        iou_predictions = torch.take_along_dim(iou_predictions, sorted_ids, dim=2)
        logit = torch.take_along_dim(logit, sorted_ids[..., None, None], dim=2)

        logit = logit[:, 0:1, 0, :, :]  # (bs,ch, h, w)
        logit = scale_sam_segment_output_3d(logit, label_size)
        logit = logit[:, 0, :, :]  # (bs, h, w)
        # logit = normalize(threshold(logit, 0.0, 0))
        logit = logit.sigmoid()

        pred_mask = (logit > 0.5).long()  # (bs, h, w)

        return pred_mask, iou_predictions, logit

    def _forward3d(self, data_item: DataItem, eval=False, all_label=False):
        # image: (bs, 1, d, h, w)
        # label: (bs, d, h, w)
        # 需要拆分为4x4的大小拼接起来,结果再拆分开

        image, label, db = data_item.image, data_item.segment, data_item.dataset

        bs, d, h, w = label.shape

        result_logit = [None] * 5
        result_mask = [None] * 5

        image = image.reshape(bs, 4, 4, h, w).permute(0, 1, 3, 2, 4).reshape(bs, 1, 4 * h, 4 * w)
        label = label.reshape(bs, 4, 4, h, w)
        # 只计算有label的
        label_only = (label.reshape(bs * 4 * 4, h, w).sum((1, 2)) > 0).unsqueeze(1).unsqueeze(1).float()
        label = label.permute(0, 1, 3, 2, 4).reshape(bs, 4 * h, 4 * w)

        index = 0 if all_label else db.value
        target_label_ids = self.eval_target_label_id_list[index] if eval else self.train_target_label_id_list[index]

        input_images = scale_image_to_sam_input_3d(image)
        image_embeddings = self.model.get_image_embeddings(input_images)
        for k in target_label_ids:
            masks, _iou_predictions, logit = self.forward_sam_3d(image, label == k, image_embeddings)
            masks = masks.reshape(bs, 4, h, 4, w).permute(0, 1, 3, 2, 4).reshape(bs * 4 * 4, h, w)
            logit = logit.reshape(bs, 4, h, 4, w).permute(0, 1, 3, 2, 4).reshape(bs * 4 * 4, h, w)

            masks = (masks * label_only).reshape(bs, 4 * 4, h, w)
            logit = (logit * label_only).reshape(bs, 4 * 4, h, w)

            result_logit[k] = logit
            result_mask[k] = masks

        return SeparetionMaskResult(result_logit, result_mask)


def scale_image_to_sam_input_2d(image):
    # 放大到512x512其余填充0， 否则很难训练
    image = F.interpolate(image, size=512, mode="bilinear", align_corners=False) * 255
    image = F.pad(image, [0, 512, 0, 512])
    image = image.repeat(1, 3, 1, 1)
    return image


def scale_sam_segment_output_2d(output, out_size):
    # 由于输入为填充，对应mask只需要corp即可
    return output[:, :, :, : out_size[0], : out_size[1]]


class EfficientSamWrapper2d(EfficientSamWrapper3d):
    def __init__(self, device, model: EfficientSam, model_args: ModelArguments) -> None:
        super().__init__(device, model, model_args)

    def forward_sam_2d(self, image, label, image_embeddings):
        """
        Args:
            image: (bs, 1, h, w)
            label: (bs, h, w)
        """

        scale = 512 / image.shape[-1]
        point_coords, point_labels = get_point_prompt(label, scale)
        logit, iou_predictions = self.model.predict_masks(
            image_embeddings=image_embeddings,
            batched_points=point_coords,
            batched_point_labels=point_labels,
            multimask_output=False,
            input_h=1024,
            input_w=1024,
        )

        # 根据iou排序，然后依次返回iou最大的几个
        sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
        iou_predictions = torch.take_along_dim(iou_predictions, sorted_ids, dim=2)
        logit = torch.take_along_dim(logit, sorted_ids[..., None, None], dim=2)
        logit = scale_sam_segment_output_2d(logit, self.crop_size)
        logit = logit[:, 0, 0, :, :]  # (bs, h, w)
        # logit = normalize(threshold(logit, 0.0, 0))
        logit = logit.sigmoid()

        # 只计算有label的
        label_only = (label.sum((1, 2)) > 0).unsqueeze(1).unsqueeze(1).float()
        logit = logit * label_only

        pred_mask = (logit > 0.5).long()  # (bs, h, w)
        return pred_mask, iou_predictions, logit

    def forward(self, data_item: DataItem):
        # image: (bs, 1, h, w)
        # label: (bs, h, w)
        # 每次只能分割一个类别

        image, label, db = data_item.image, data_item.segment, data_item.dataset

        bs, h, w = label.shape

        result_logit = [None] * 5
        rusult_mask = [None] * 5

        target_label_ids = self.train_target_label_id_list[db.value]

        input_images = scale_image_to_sam_input_2d(image)
        image_embeddings = self.model.get_image_embeddings(input_images)
        for k in target_label_ids:
            masks, _iou_predictions, logit = self.forward_sam_2d(image, label == k, image_embeddings)
            result_logit[k] = logit
            rusult_mask[k] = masks

        return SeparetionMaskResult(result_logit, rusult_mask)

    def forward_valid(self, data_item: DataItem):
        return self._forward_2d(data_item)

    def forward_test(self, data_item: DataItem):
        return self._forward_2d(data_item, True, True)

    def forward_zeroshot(self, data_item: DataItem):
        return self._forward_2d(data_item, True, True)

    def forward_pseudo(self, data_item: DataItem):
        return self._forward_2d(data_item, True)

    def _forward_2d(self, data_item: DataItem, eval=False, all_label=False):
        images, labels, db = data_item.image, data_item.segment, data_item.dataset
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        bs, d, h, w = labels.shape
        all_logit = [None] * 5
        all_mask = [None] * 5

        point_labels = labels.reshape(bs * d, h, w)

        images = images.reshape(bs * d, 1, h, w)

        logit_list = {}
        mask_list = {}

        index = 0 if all_label else db.value
        target_label_ids = self.eval_target_label_id_list[index] if eval else self.train_target_label_id_list[index]

        for img, label in zip(images.split(1, 0), point_labels.split(1, 0)):
            input_images = scale_image_to_sam_input_2d(img)
            image_embeddings = self.model.get_image_embeddings(input_images)

            for k in target_label_ids:
                masks, _iou_predictions, logit = self.forward_sam_2d(img, label == k, image_embeddings)

                if k not in logit_list:
                    logit_list[k] = []
                    mask_list[k] = []

                logit_list[k].append(logit)
                mask_list[k].append(masks)

        logit_list = {k: torch.cat(v, dim=0).reshape(bs, d, h, w) for k, v in logit_list.items()}

        mask_list = {k: torch.cat(v, dim=0).reshape(bs, d, h, w) for k, v in mask_list.items()}

        for k in target_label_ids:
            all_logit[k] = logit_list[k]
            all_mask[k] = mask_list[k]

        return SeparetionMaskResult(all_logit, all_mask)
