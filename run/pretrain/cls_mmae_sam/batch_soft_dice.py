import torch
from torch import nn, Tensor


class BatchSoftIoU(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self):
        super().__init__()

    def forward(self, probs: Tensor, labels: Tensor):
        """
        inputs:
            probs: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        bs = probs.shape[0]
        intersection = (probs * labels).view(bs, -1).sum(dim=1)
        union = (probs + labels).view(bs, -1).sum(dim=1) + 1e-5
        iou = (2 * intersection) / union
        return iou


class DiceScore(nn.Module):
    """
    hard-dice loss, useful in binary segmentation
    """

    def __init__(self):
        super().__init__()

    def forward(self, probs: Tensor, labels: Tensor):
        """
        inputs:
            probs: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """

        probs = probs.round()

        bs = probs.shape[0]
        intersection = (probs * labels).view(bs, -1).sum(dim=1)
        union = (probs + labels).view(bs, -1).sum(dim=1)

        dice = (2 * intersection) / torch.clamp(union, min=1e-5)
        dice = torch.mean(dice)
        return dice
