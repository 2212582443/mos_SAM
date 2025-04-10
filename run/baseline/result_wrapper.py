from typing import Any, Callable, List
import torch.nn.functional as F
import torch
from transformers import HfArgumentParser

from run.baseline.dataset import DATASET, DataItem
from run.baseline.metric import Metric, get_onehost_label


def parse_custom_args(args: List[str] | None, option_cls: Any) -> tuple[Any, list[str]]:
    parser = HfArgumentParser(option_cls)
    options, others = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)
    return options, others


class ModelResult:
    def calc_loss(self, data_item: DataItem, valid_label_ids: list[int], num_classes) -> torch.Tensor:
        raise NotImplementedError("calc_loss not implemented")

    def calc_metric(
        self,
        metric: Metric,
        metric_segment: Any,
        data_item: DataItem,
        valid_label_ids: list[int],
        num_classes,
    ) -> torch.Tensor:
        raise NotImplementedError("calc_loss not implemented")

    def merge_incomplete_label(
        self, metric: Metric, db: DATASET, avaliable_label: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("calc_loss not implemented")


# 模型使用多个chanel输出结果
class MultiChanelResult(ModelResult):
    def __init__(
        self,
        pred_softmax,
        logit=None,
        aux_data=None,
        loss_fn: Callable[[ModelResult, DataItem, list[int], int], torch.Tensor] = None,
    ):
        super().__init__()
        self.logit = logit
        self.pred_softmax = pred_softmax
        self.aux_data = aux_data
        self.loss_fn = loss_fn

    def calc_loss(self, data_item: DataItem, valid_label_ids: list[int], num_classes) -> torch.Tensor:
        # return (bs)
        if self.loss_fn is not None:
            return self.loss_fn(self, data_item, valid_label_ids, num_classes)

        loss = F.nll_loss(torch.log(self.pred_softmax + 1e-4), data_item.segment, reduction="none")
        return loss.mean(list(range(1, loss.dim())))

    def calc_metric(
        self,
        metric: Metric,
        metric_segment: torch.Tensor,
        data_item: DataItem,
        valid_label_ids: list[int],
        num_classes,
    ) -> torch.Tensor:
        return metric.calc_metric(
            metric_segment.long(),
            data_item.segment,
            num_classes,
            valid_labels=valid_label_ids,
        )

    def merge_incomplete_label(
        self, metric: Metric, db: DATASET, avaliable_label: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        merger = metric.get_merge_label_config(avaliable_label)
        pred_softmax = merger.merge_multi_chanel_labels(self.pred_softmax)
        logit = merger.merge_multi_chanel_labels(self.logit) if self.logit is not None else None

        pred_segment_all = self.pred_softmax.argmax(1, keepdim=False)
        # (bs, h, w) or (bs, d, h, w)
        metric_segment = pred_softmax.argmax(1, keepdim=False)

        pred_segment_all_to_be_ploted = metric.color_map.apply(pred_segment_all)

        self.logit = logit
        self.pred_softmax = pred_softmax
        return metric_segment, pred_segment_all_to_be_ploted


# 模型每次只输出一个结果，所以这里的结果是一个list， 每个元素对应一个label类别
class SeparetionMaskResult(ModelResult):
    def __init__(self, logit, mask, loss_fn: Callable[[ModelResult, DataItem, list[int], int], torch.Tensor] = None):
        super().__init__()
        # list[(bs, h, w)] or list[(bs, d, h, w)]
        self.logit: List[torch.Tensor] = logit
        self.mask: List[torch.Tensor] = mask
        self.loss_fn = loss_fn

    def get_onehot_mask(self) -> torch.Tensor:
        sample = None
        for m in self.mask:
            if m is None:
                continue
            sample = m
            break
        if sample is None:
            return None
        result = []
        for m in self.mask:
            if m is None:
                result.append(torch.zeros_like(sample, dtype=torch.uint8))
            else:
                result.append(m.to(torch.uint8))
        return torch.stack(result, dim=1)

    def calc_loss(self, data_item: DataItem, valid_label_ids: list[int], num_classes) -> torch.Tensor:
        # return (bs)
        if self.loss_fn is not None:
            return self.loss_fn(self, data_item, valid_label_ids, num_classes)

        label = data_item.segment
        logit: List[torch.Tensor] = self.logit

        # return F.mse_loss(logit[1], label.float(), reduction="none")

        # 分别计算各标签的loss
        loss = None
        label[label >= num_classes] = 0
        segment_label_onehot_list = F.one_hot(label, num_classes=num_classes).float().split(1, -1)

        for id in valid_label_ids:
            r, segment_label_onehot = logit[id], segment_label_onehot_list[id]

            if r is None:
                continue

            segment_label_onehot = segment_label_onehot.squeeze(-1)
            # print(r.shape, segment_label_onehot.shape, (r - segment_label_onehot).abs().sum())
            current_loss = F.mse_loss(r, segment_label_onehot, reduction="none")
            if loss is None:
                loss = current_loss
            else:
                loss += current_loss

        return loss.mead(list(range(1, loss.dim()))) if loss.dim() > 1 else loss

    def calc_metric(
        self,
        metric: Metric,
        metric_segment: List[torch.Tensor],
        data_item: DataItem,
        valid_label_ids: list[int],
        num_classes,
    ) -> torch.Tensor:
        # one hot, 只保留valid_label
        segment_onehot = get_onehost_label(data_item.segment, num_classes, valid_labels=valid_label_ids)
        return metric.calc_metric_onehot(metric_segment, segment_onehot, num_classes)

    def merge_incomplete_label(
        self, metric: Metric, db: DATASET, avaliable_label: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_mask: List[torch.Tensor] = self.mask
        # logit: List[torch.Tensor] = self.logit

        pred_segment_all_to_be_ploted = None
        for i, r in enumerate(all_mask):
            if r is None:
                continue
            r = (r * i).long()
            color = metric.color_map.apply(r)

            if pred_segment_all_to_be_ploted is None:
                pred_segment_all_to_be_ploted = color.to(torch.long)
            else:
                pred_segment_all_to_be_ploted += color

        if pred_segment_all_to_be_ploted is not None:
            pred_segment_all_to_be_ploted = pred_segment_all_to_be_ploted.clamp(0, 255).to(torch.uint8)

        padding = None
        for mask in all_mask:
            if mask is not None:
                padding = torch.zeros_like(mask)
                break

        # 支持多个label
        # (bs, label_count, h, w) or (bs, label_count, d, h, w)
        metric_segment = []  # all_mask[Label.EAT.value]
        avaliable_label = set(avaliable_label)
        for i in range(0, metric.label_count):
            mask = all_mask[i] if i < len(all_mask) else None
            if i in avaliable_label and mask is not None:
                metric_segment.append(mask)
            else:
                metric_segment.append(padding)
        metric_segment = torch.stack(metric_segment, 1).long()

        return metric_segment, pred_segment_all_to_be_ploted
