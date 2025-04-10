from typing import Any, List
import torch

from .dataset import DATASET, Label
from typing import List
import torch.nn.functional as F


def get_onehost_label(segment_label, num_classes, valid_labels: list[int] = None):
    """
    Args:
        segment_lable: (bs, h, w) or (bs, d, h ,w)
        valid_label: list[int]
    """
    max_cls_count = segment_label.max().item() + 1
    max_cls_count = max(max_cls_count, num_classes)
    segment_label_onehot = F.one_hot(segment_label.long(), num_classes=max_cls_count).float()
    if len(segment_label_onehot.shape) == 4:
        segment_label_onehot = segment_label_onehot.permute(0, 3, 1, 2)
    else:
        segment_label_onehot = segment_label_onehot.permute(0, 4, 1, 2, 3)
    # 过滤掉不需要的lable
    if valid_labels is not None and len(valid_labels) > 0:
        idx = []
        valid_labels = set(valid_labels)
        for i in range(1, num_classes):
            if i not in valid_labels:
                idx.append(i)
        segment_label_onehot[:, idx] = 0

    return segment_label_onehot[:, :num_classes, ::]


class LabelMergeConfig:
    def __init__(self, device, avaliable_label: list[int], label_count) -> None:
        self.label_count = label_count
        self.avaliable_labels = avaliable_label

        missing_labels = []
        for i in range(1, label_count):
            if i in avaliable_label:
                continue
            missing_labels.append(i)
        self.missing_labels = missing_labels

        labels_permute = []
        temp = [0] + missing_labels + avaliable_label
        for position in range(label_count):
            for index in range(label_count):
                if temp[index] == position:
                    labels_permute.append(index)
                    break
        labels_permute = torch.tensor(labels_permute, device=device)

        self.labels_permute = labels_permute

        self.need_merge = len(avaliable_label) < label_count - 1

    def merge_multi_chanel_labels(self, after_softmax):
        if not self.need_merge:
            return after_softmax

        # 这个数据集有部分标签，需要把没有的标签合并到0
        pred_bg = [0] + self.missing_labels
        pred_bg = after_softmax[:, pred_bg].sum(1, keepdim=True)

        pred_label = after_softmax[:, self.avaliable_labels]

        padding = list(pred_bg.shape)
        padding[1] = self.label_count - len(self.avaliable_labels) - 1
        padding = torch.zeros(padding, dtype=pred_bg.dtype, device=pred_bg.device)
        pred = torch.cat([pred_bg, padding, pred_label], 1)
        pred = pred.index_select(1, self.labels_permute)

        return pred


# 生成不同的色彩， 把label转换为彩色图片，以方便显示
class ColorPlotMap:
    def __init__(self, device, label_count):
        import colorsys

        assert label_count <= 100

        self.label_count = label_count
        color_map = [0, 0, 0]

        for i in range(0, label_count):
            h = i * 62
            s = 100 - (h / 360) * 5
            h = h % 360
            v = 100
            r, g, b = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)
            assert 0 <= r <= 1
            assert 0 <= g <= 1
            assert 0 <= b <= 1
            color_map.extend([r * 255, g * 255, b * 255])

        self.color_map = torch.tensor(color_map, dtype=torch.uint8, device=device).reshape(-1, 3)

    def apply(self, pred: torch.Tensor, color_map=None) -> torch.Tensor:
        """
        Args:
            pred: (bs, h, w) or (bs, d, h, w)
            color_map: (label_count, [r, g, b])
        Returns:
            (bs, 3, h, w) or (bs, 3, d, h, w)
        """
        shape = pred.shape

        if color_map is None:
            color_map = self.color_map

        colorlize = color_map.index_select(0, pred.reshape(-1)).reshape(*shape, 3)
        if len(shape) == 3:  # bs, h, w
            # (bs, h, w, 3) -> (bs, 3, h, w)
            colorlize = colorlize.permute(0, 3, 1, 2)
        else:  # bs, d, h, w
            # (bs, d, h, w, 3) -> (bs, 3, d, h, w)
            colorlize = colorlize.permute(0, 4, 1, 2, 3)

        return colorlize.to(torch.uint8)


class Metric:
    def __init__(
        self,
        device,
        aux_dataset_labels: list[int],
        zeroshot_dataset_labels: list[int],
        label_count=5,
        spacing=[10, 1.5, 1.5],
    ):
        self.aux_dataset_labels = aux_dataset_labels
        self.zeroshot_dataset_labels = zeroshot_dataset_labels
        self.label_count = label_count

        self.merge_config: dict[str, LabelMergeConfig] = {}
        self.color_map = ColorPlotMap(device, 100)
        self.spacing = spacing

    @torch.no_grad()
    def calc_metric(self, pred, label, num_classes=2, mean_channels=True, valid_labels: list[int] = None):
        if pred.min() < 0:
            print(pred)
        pred_one_hot = get_onehost_label(pred, num_classes, valid_labels)
        label_one_hot = get_onehost_label(label, num_classes, valid_labels)

        return self.calc_metric_onehot(pred_one_hot, label_one_hot, num_classes, mean_channels)

    def calc_metric_onehot(self, pred_one_hot, label_one_hot, mean_channels=True, spacing=None):
        import monai.metrics as monaimetrics

        num_classes = pred_one_hot.shape[1]

        # 心脏图像各向异性
        # 2D: (bs, h, w) 3D: (bs, d, h, w)
        if spacing is None:
            spacing = self.spacing[-2:] if len(pred_one_hot.shape) == 4 else self.spacing
        metric = {}
        # mean surface distance
        # (bs, label_count)
        msd = monaimetrics.compute_average_surface_distance(
            pred_one_hot,
            label_one_hot,
            symmetric=True,
            spacing=spacing,
        )
        metric["raw_msd"] = msd
        if mean_channels:
            msd = msd.nanmean(1)
        metric["msd"] = msd.tolist()
        metric["notnan_msd"] = msd[~torch.logical_or(msd.isnan(), msd.isinf())].tolist()

        # hausdorff distance
        hd = monaimetrics.compute_hausdorff_distance(
            pred_one_hot,
            label_one_hot,
            spacing=spacing,
        )
        metric["raw_hd"] = hd
        if mean_channels:
            hd = hd.nanmean(1)
        metric["hd"] = hd.tolist()
        metric["notnan_hd"] = hd[~torch.logical_or(hd.isnan(), hd.isinf())].tolist()

        # dice score
        dice = monaimetrics.compute_dice(
            pred_one_hot,
            label_one_hot,
            include_background=False,
            num_classes=num_classes,
        )
        metric["raw_dice"] = dice
        if mean_channels:
            dice = dice.nanmean(1)
        metric["dice"] = dice.tolist()
        metric["notnan_dice"] = dice[~torch.logical_or(dice.isnan(), dice.isinf())].tolist()

        # iou score
        iou = monaimetrics.compute_iou(
            pred_one_hot,
            label_one_hot,
            include_background=False,
        )
        metric["raw_iou"] = iou
        if mean_channels:
            iou = iou.nanmean(1)
        metric["iou"] = iou.tolist()
        metric["notnan_iou"] = iou[~torch.logical_or(iou.isnan(), iou.isinf())].tolist()

        return metric

    def get_merge_label_config(self, avaliable_label: list[int]):
        key = ",".join(map(str, avaliable_label))
        if key not in self.merge_config:
            self.merge_config[key] = LabelMergeConfig(
                avaliable_label=avaliable_label, device="cuda", label_count=self.label_count
            )
        return self.merge_config[key]

    def update_metric(self, metric, metrics_to_show, metrics_to_saved: dict[str, list[Any]]):
        # merge metric
        for k, v in metric.items():
            if k.startswith("notnan_"):
                if metrics_to_show is not None:
                    k = k[7:]
                    metrics_to_show[k].extend(v)
            elif metrics_to_saved is not None:
                if k.startswith("raw_"):
                    if k not in metrics_to_saved:
                        metrics_to_saved[k] = []
                    metrics_to_saved[k].append(v)
                else:
                    metrics_to_saved[k].extend(v)
