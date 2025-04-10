import torch.nn as nn
import torch
import torch.nn.functional as F

from run.baseline.model_arguments import ModelArguments
from .label_utils import get_config_train_labels, get_config_eval_labels
from run.baseline.result_wrapper import SeparetionMaskResult

from ..dataset import DataItem
from ..models.sam import build_sam_vit_b, Sam
import random
from torch.nn.functional import threshold, normalize


def get_point_prompt(label: torch.Tensor, scale=8.0):
    # (bs, h, w)
    # 去掉边缘像素
    bs, h, w = label.shape
    point_label: list[torch.Tensor] = F.max_pool2d(-label.float(), kernel_size=3, stride=1, padding=1).split(1, 0)
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
    # (bs, N, 2)
    point_coords = point_coords.unsqueeze(1).float()
    point_coords *= scale  # 缩放到1024的坐标系

    # (bs, N)
    point_labels = torch.ones(bs, 1, dtype=torch.int64)

    return point_coords, point_labels


def scale_image_to_sam_input_3d(image):
    image = F.interpolate(image, size=1024, mode="bilinear", align_corners=False) * 255
    image = image.repeat(1, 3, 1, 1)
    return image


def scale_sam_segment_output_3d(output, size):
    return F.interpolate(output, size=size, mode="bilinear", align_corners=False)


class SamWrapper3d(nn.Module):
    def __init__(
        self,
        device,
        model_args: ModelArguments,
        checkpoint=".cache/dataset/baseline/sam_vit_b_01ec64.pth",
    ) -> None:
        super().__init__()

        self.device = device
        self.model: Sam = build_sam_vit_b(checkpoint).to(device)
        self.crop_size = model_args.get_dataset_crop_size()[-2:]
        self.label_count = model_args.label_count

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

    def forward_sam_3d(self, image, label, image_embeddings):
        """
        Args:
            image: (bs, 1, h, w)
            label: (bs, h, w)
        """

        label_size = label.shape[-2:]
        scale = 1024 / image.shape[-1]
        point_coords, point_labels = get_point_prompt(label, scale)
        _masks, iou_predictions, logit = self.model(None, point_coords, point_labels, image_embeddings=image_embeddings)
        logit = scale_sam_segment_output_3d(logit, label_size)
        logit = logit[:, 0, :, :]  # (bs, h, w)
        # logit = normalize(threshold(logit, 0.0, 0))
        logit = logit.sigmoid()

        pred_mask = logit > 0.5  # (bs, h, w)
        return pred_mask, iou_predictions, logit

    def _forward3d(self, data_item: DataItem, eval=False, all_label=False):
        image, label, db = data_item.image, data_item.segment, data_item.dataset

        # image: (bs, 1, d, h, w)
        # label: (bs, d, h, w)
        # 需要拆分为4x4的大小拼接起来,结果再拆分开

        bs, d, h, w = label.shape

        result_logit = [None] * self.label_count
        result_mask = [None] * self.label_count

        index = 0 if all_label else db.value
        target_label_ids = self.eval_target_label_id_list[index] if eval else self.train_target_label_id_list[index]

        image = image.reshape(bs, 4, 4, h, w).permute(0, 1, 3, 2, 4).reshape(bs, 1, 4 * h, 4 * w)
        label = label.reshape(bs, 4, 4, h, w)
        # 只计算有label的
        label_only = (label.reshape(bs * 4 * 4, h, w).sum((1, 2)) > 0).unsqueeze(1).unsqueeze(1).float()
        label = label.permute(0, 1, 3, 2, 4).reshape(bs, 4 * h, 4 * w)

        input_images = scale_image_to_sam_input_3d(image)
        input_images = self.model.preprocess(input_images)
        image_embeddings = self.model.image_encoder(input_images)
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


def scale_sam_segment_output_2d(output, size):
    # 由于输入为填充，对应mask只需要corp即可
    return output[:, :, : size[0], : size[1]]


class SamWrapper2d(SamWrapper3d):
    def __init__(
        self,
        device,
        model_args: ModelArguments,
        checkpoint=".cache/dataset/baseline/sam_vit_b_01ec64.pth",
    ) -> None:
        super().__init__(device, model_args, checkpoint)

    def forward_sam_2d(self, image, label, image_embeddings):
        """
        Args:
            image: (bs, 1, h, w)
            label: (bs, h, w)
        """

        label_size = label.shape[-2:]
        scale = 512 / image.shape[-1]
        point_coords, point_labels = get_point_prompt(label, scale)
        _masks, iou_predictions, logit = self.model(None, point_coords, point_labels, image_embeddings=image_embeddings)
        logit = scale_sam_segment_output_2d(logit, self.crop_size)
        logit = logit[:, 0, :, :]  # (bs, h, w)
        # logit = normalize(threshold(logit, 0.0, 0))
        logit = logit.sigmoid()

        # 只保留有label的部分
        label_only = (label.sum((1, 2)) > 0).unsqueeze(1).unsqueeze(1).float()
        logit = logit * label_only

        pred_mask = logit > 0.5  # (bs, h, w)
        return pred_mask, iou_predictions, logit

    def forward(self, data_item: DataItem):
        image, label, db = data_item.image, data_item.segment, data_item.dataset
        # image: (bs, 1, h, w)
        # label: (bs, h, w)
        # 每次只能分割一个类别

        bs, h, w = label.shape

        result_logit = [None] * self.label_count
        rusult_mask = [None] * self.label_count

        target_label_ids = self.train_target_label_id_list[db.value]

        input_images = scale_image_to_sam_input_2d(image)
        input_images = self.model.preprocess(input_images)
        image_embeddings = self.model.image_encoder(input_images)
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
        all_logit = [None] * self.label_count
        all_mask = [None] * self.label_count

        point_labels = labels.reshape(bs * d, h, w)

        images = images.reshape(bs * d, 1, h, w)

        logit_list = {}
        mask_list = {}

        index = 0 if all_label else db.value
        target_label_ids = self.eval_target_label_id_list[index] if eval else self.train_target_label_id_list[index]

        # 该模型只能一个个处理，不支持bs>1
        for img, label in zip(images.split(1, 0), point_labels.split(1, 0)):
            input_images = scale_image_to_sam_input_2d(img)
            input_images = self.model.preprocess(input_images)
            image_embeddings = self.model.image_encoder(input_images)

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
