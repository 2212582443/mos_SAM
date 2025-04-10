import torch.nn as nn
import torch
import torch.nn.functional as F

from run.baseline.model_arguments import ModelArguments
from .label_utils import get_config_train_labels, get_config_eval_labels
from run.baseline.result_wrapper import SeparetionMaskResult

from ..dataset import DataItem
from ..models.samus import build_samus_vit_b
from ..models.samus.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Samus, TwoWayTransformer
import random
from torch.nn.functional import threshold, normalize


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
    # (bs, N, 2)
    point_coords = point_coords.unsqueeze(1).float()
    scale = scale[[1, 0]].to(label.device)
    point_coords *= scale.unsqueeze(0).unsqueeze(0)  # 缩放到256*256的坐标系

    # (bs, N)
    point_labels = torch.ones(bs, 1, dtype=torch.int64)

    return point_coords, point_labels


def scale_image_to_samus_input_3d(image):
    image = F.interpolate(image, size=256, mode="bilinear", align_corners=False) * 255
    image = image.repeat(1, 3, 1, 1)
    return image


def scale_samus_segment_output_3d(output, size):
    return F.interpolate(output, size=size, mode="nearest")


class SamusWrapper3d(nn.Module):
    def __init__(
        self,
        device,
        model_args: ModelArguments,
        checkpoint=".cache/dataset/baseline/sam_vit_b_01ec64.pth",
    ) -> None:
        super().__init__()

        self.device = device
        self.model: Samus = build_samus_vit_b(checkpoint).to(device)
        self.crop_size = model_args.get_dataset_crop_size()
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

    def forward_presudo(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True)

    def forward_samus_3d(self, image, label, image_embeddings):
        """
        Args:
            image: (bs, 1, h, w)
            label: (bs, h, w)
        """

        label_size = label.shape[-2:]
        scale = 256 / image.shape[-1]
        point_coords, point_labels = get_point_prompt(label, scale)
        pt = (point_coords, point_labels)
        output = self.model(None, pt, None, image_embeddings)
        logit = output["low_res_logits"]
        logit = scale_samus_segment_output_3d(logit, label_size)
        logit = logit[:, 0, :, :]  # (bs, h, w)
        logit = normalize(threshold(logit, 0.0, 0))

        pred_mask = logit > 0  # (bs, h, w)
        return pred_mask, logit

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

        image = image.reshape(bs * d, 1, h, w)
        label = label.reshape(bs * d, h, w)
        # 只计算有label的
        label_only = (label.sum((1, 2)) > 0).unsqueeze(1).unsqueeze(1).float()

        input_images = scale_image_to_samus_input_3d(image)
        image_embeddings = self.model.image_encoder(input_images)
        for k in target_label_ids:
            masks, logit = self.forward_samus_3d(image, label == k, image_embeddings)

            masks = (masks * label_only).reshape(bs, d, h, w)
            logit = (logit * label_only).reshape(bs, d, h, w)

            result_logit[k] = logit
            result_mask[k] = masks

        return SeparetionMaskResult(result_logit, result_mask)


def scale_image_to_samus_input_2d(image):
    image = F.interpolate(image, size=[256, 256], mode="bilinear", align_corners=False) * 255
    image = image.repeat(1, 3, 1, 1)
    return image


class SamusWrapper2d(SamusWrapper3d):
    def __init__(
        self,
        device,
        model_args: ModelArguments,
        checkpoint=".cache/dataset/baseline/sam_vit_b_01ec64.pth",
    ) -> None:
        super().__init__(device, model_args, checkpoint)

    def forward_samus_2d(self, label, image_embeddings):
        """
        Args:
            image: (bs, 1, h, w)
            label: (bs, h, w)
        """

        label_size = label.shape[-2:]
        scale = torch.tensor([256 / label_size[0], 256 / label_size[1]])
        point_coords, point_labels = get_point_prompt(label, scale)
        pt = (point_coords, point_labels)
        output = self.model(None, pt, None, image_embeddings)
        logit = output["low_res_logits"]
        logit = F.interpolate(logit, size=label_size, mode="bilinear", align_corners=False)
        logit = logit[:, 0, :, :]  # (bs, h, w)
        # logit = threshold(logit, 0.0, 0)
        logit = logit.sigmoid()

        # 只保留有label的部分
        label_only = (label.sum((1, 2)) > 0).unsqueeze(1).unsqueeze(1).float()
        logit = logit * label_only

        pred_mask = logit > 0.5  # (bs, h, w)
        return pred_mask, logit

    def forward(self, data_item: DataItem):
        image, label, db = data_item.image, data_item.segment, data_item.dataset
        # image: (bs, 1, h, w)
        # label: (bs, h, w)
        # 每次只能分割一个类别

        # bs, h, w = label.shape

        result_logit = [None] * self.label_count
        rusult_mask = [None] * self.label_count

        target_label_ids = self.train_target_label_id_list[db.value]

        input_images = scale_image_to_samus_input_2d(image)
        image_embeddings = self.model.image_encoder(input_images)
        for k in target_label_ids:
            masks, logit = self.forward_samus_2d(label == k, image_embeddings)
            result_logit[k] = logit
            rusult_mask[k] = masks

        return SeparetionMaskResult(result_logit, rusult_mask)

    def forward_valid(self, data_item: DataItem):
        return self._forward_2d(data_item)

    def forward_test(self, data_item: DataItem):
        return self._forward_2d(data_item, True, True)

    def forward_zeroshot(self, data_item: DataItem):
        return self._forward_2d(data_item, True, True)

    def forward_presudo(self, data_item: DataItem):
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
            input_images = scale_image_to_samus_input_2d(img)
            image_embeddings = self.model.image_encoder(input_images)

            for k in target_label_ids:
                masks, logit = self.forward_samus_2d(label == k, image_embeddings)

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
