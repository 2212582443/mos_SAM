import numpy as np
import torch
from torch import tensor, FloatTensor
from torch.nn import Module, Embedding, MaxPool2d

from .typing import PointCoordTensor, PointLabelTensor, SegmentTensor, SparseEmbeddingsTensor

from ...configuration_sam import SamPromptEncoderConfig
from .positional_embedding import SamPositionalEmbedding
from enum import Enum


class PointEmbedding(Module):
    def __init__(
        self,
        config: SamPromptEncoderConfig,
        positional_embedding: SamPositionalEmbedding,
    ):
        super().__init__()
        self.positional_embedding = lambda input_coords, input_shape: positional_embedding(input_coords, input_shape)
        self.image_size = config.image_size

        # 使用不同的embedding来区分不同的点
        self.object_point = Embedding(1, config.hidden_size)
        self.not_object_point = Embedding(1, config.hidden_size)
        self.background_point = Embedding(1, config.hidden_size)

    def forward(self, points: PointCoordTensor, labels: PointLabelTensor) -> SparseEmbeddingsTensor:
        """
        Args:
            points: (bs, nb_points, 2)
                2: x, y
            labels: (bs, nb_points) 点的类型 PointType

        returns:
            point_embedding(bs, nb_points, hidden_size)
        """
        points = points + 0.5  # Shift to center of pixel

        # 1. 把点替换为对应的位置embedding
        input_shape = (self.image_size, self.image_size)  # (h, w)

        # (bs, nb_points, 256)
        point_embedding = self.positional_embedding(points, input_shape)  # (**, 256)

        # 2. 针对不同的label, 修改对应的embedding

        # 2.1 背景点的embedding替换为特定的embedding(background_point)
        # note: torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = torch.where(
            labels[..., None] == PointType.BACKGROUND_POINT.value, self.background_point.weight, point_embedding
        )

        # 2.2 不包含物体的点, 加上特定的embedding, 表示不包含物体
        point_embedding = torch.where(
            (labels == PointType.NOT_OBJECT_POINT.value)[:, :, None],
            point_embedding + self.not_object_point.weight[None, :, :],
            point_embedding,
        )

        # 2.3 包含物体的点, 加上特定的embedding, 表示包含物体
        point_embedding = torch.where(
            (labels == PointType.OBJECT_POINT.value)[:, :, None],
            point_embedding + self.object_point.weight[None, :, :],
            point_embedding,
        )

        return point_embedding


class PointType(Enum):
    # 包含有物体的点
    OBJECT_POINT = 1
    # 不包含物体的点
    NOT_OBJECT_POINT = 0
    # 背景, 没有物体的点
    BACKGROUND_POINT = -1


class PointPrompt(object):
    def __init__(
        self,
        point: tuple[int, int],
        point_type: PointType,
    ):
        # points: [(x, y), ...]
        # point_type: [PointType, ...]
        self.point = point
        self.point_type = point_type


def point2tensor(points: list[PointPrompt]) -> tuple[PointCoordTensor, PointLabelTensor]:
    point_type = [point.point_type.value for point in points]
    points = [point.point for point in points]
    """
    Args:
        points: [(x, y), ...]
        point_type: [PointType, ...]

    Returns:
        (points, labels)
            points: (1, nb_points, 2)
            labels: (1, nb_points)
    """
    coords: PointLabelTensor = tensor(points).unsqueeze(0)
    labels: PointLabelTensor = tensor(point_type).unsqueeze(0)
    return (coords, labels)


_erosion_op = MaxPool2d(kernel_size=3, stride=1, padding=1)


def rand_get_segment_point(
    segment: SegmentTensor, return_count: int = -1, scaling: float = 1, erosion=True
) -> list[tuple[int, int]]:
    """随机获取segment中的一个有效的分割的坐标点
    Args:
        segment: (1, h, w)
        return count: 返回的点的数量
        scaling: 缩放因子, 图像和mask大小不一致(1024*1024 vs 256*256), 需要缩放系数
        erosion: 腐蚀边缘

    Returns:
        (x, y)
    """
    if erosion:
        erosioned = -_erosion_op(-segment.float()).long()
        if erosioned.sum() > 0:
            segment = erosioned
        # else:
        # print("not erosioned, count:", segment.sum().item())

    bs, h, w = segment.shape
    assert bs == 1, "bs must be 1"
    _, y_indices, x_indices = segment.nonzero(as_tuple=True)
    count = y_indices.size().numel()
    if count == 0:
        return [w / 2, h / 2]
    if return_count == -1:
        return_count = count
    index = np.random.choice(count, return_count, replace=True)
    x, y = x_indices[index].tolist(), y_indices[index].tolist()
    result = [(x * scaling, y * scaling) for x, y in zip(x, y)]
    return result
