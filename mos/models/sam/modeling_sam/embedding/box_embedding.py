import numpy as np
from torch import Tensor, tensor, FloatTensor
from torch.nn import Module, Embedding, Linear

from .typing import BoxCoordTensor, SparseEmbeddingsTensor, BoxPrompt
from ...configuration_sam import SamPromptEncoderConfig

from .positional_embedding import SamPositionalEmbedding
from transformers.activations import ACT2FN


def rand_select_bounding_box(ground_truth_map: Tensor, scaling: float = 1) -> tuple[float, float, float, float]:
    """随机选择一个bounding box
    Args:
        ground_truth_map: (1, h, w)

        scaling: 缩放因子, 图像和mask大小不一致(1024*1024 vs 256*256), 需要缩放系数
    Returns:
        bounding box: (x_min, y_min, x_max, y_max)
    """
    bs, H, W = ground_truth_map.shape

    assert bs == 1, 'bs must be 1'

    # get bounding box from mask
    _, y_indices, x_indices = ground_truth_map.nonzero(as_tuple=True)
    if y_indices.size().numel() == 0:
        return [W/2, H/2, W/2, H/2]

    x_min, x_max = x_indices.min().item(), x_indices.max().item()
    y_min, y_max = y_indices.min().item(), y_indices.max().item()
    # add perturbation to bounding box coordinates
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min*scaling, y_min*scaling, x_max*scaling, y_max*scaling]

    return bbox


def box2tensor(boxes: list[BoxPrompt]) -> BoxCoordTensor:
    """处理bounding box, 把box坐标转换为box tensor,
    Args:
        boxes: [(x_min, y_min, x_max, y_max), ...]
    returns:  bounding box的tensor的标准形式(1, nb_boxes, 4))
    """
    box: FloatTensor = tensor(boxes).float()
    return box.unsqueeze(0)


class BoxEmbedding(Module):
    def __init__(
        self,
        config: SamPromptEncoderConfig,
        positional_embedding: SamPositionalEmbedding,
    ):
        super().__init__()
        self.positional_embedding = lambda **args: positional_embedding(**args)
        self.image_size = config.image_size
        self.left_top_embedding = Embedding(1, config.hidden_size)
        self.right_bottom_embedding = Embedding(1, config.hidden_size)
        self.mlp = BoxMlp(config.hidden_size*4, config.hidden_size, config.hidden_act)

    def forward(self, boxes: FloatTensor) -> SparseEmbeddingsTensor:
        """ 处理bounding box, 返回bounding box的embedding
        Args:
            box: (bs, nb_boxes, 4)
                4: x0, y0, x1, y1
        returns:
            box_embedding(bs, nb_boxes, hidden_size)
        """

        # move to center
        boxes = boxes + 0.5
        # 变成两个坐标点
        bs, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(bs, nb_boxes, 2, 2)
        # 两个坐标点的embedding
        input_shape = (self.image_size, self.image_size)
        # (bs, num_boxes, 2, hidden_size)
        corner_embedding = self.positional_embedding(coords, input_shape)
        # 加上box坐标点的专属embedding, 用于区分别的embedding
        corner_embedding[:, :, 0, :] += self.left_top_embedding.weight
        corner_embedding[:, :, 1, :] += self.right_bottom_embedding.weight

        return self.mlp(corner_embedding)


class BoxMlp(Module):
    def __init__(self, mlp_dim: int, hidden_size: int, hidden_act: str):
        super().__init__()
        self.lin1 = Linear(hidden_size*2, mlp_dim)
        self.lin2 = Linear(mlp_dim, hidden_size)
        self.act = ACT2FN[hidden_act]

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (bs, nb_boxes, 2, hidden_size)
        Returns: (bs, nb_boxes, hidden_size)
        """
        bs, nb_boxes = hidden_states.shape[:2]
        hidden_states = hidden_states.reshape(bs, nb_boxes, -1)
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states
