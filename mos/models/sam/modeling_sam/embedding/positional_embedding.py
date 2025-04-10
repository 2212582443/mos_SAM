from torch import nn, Tensor
import torch
import numpy as np

from .typing import PointCoordTensor

from ...configuration_sam import SamVisionConfig


class SamPositionalEmbedding(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.scale = config.hidden_size // 2
        # (2, 128)
        self.register_buffer("positional_embedding", self.scale * torch.randn((2, config.num_pos_feats)))

    def forward(self, input_coords: PointCoordTensor, input_shape: Tensor | None = None) -> Tensor:
        """Positionally encode points that are normalized to [0,1].
        Args:
            input_coords: [bs, *num_points, 2], 最后一个维度为 (x, y)
            input_shape: (h, w)

        Returns:
            outputs: [bs, *num_points, 2 * num_pos_feats]
                example, (bs, 1, 2, 256) or (64, 64, 256)
        """
        coordinates = input_coords.clone()
        if input_shape is not None:
            coordinates[:, :,  0] = coordinates[:, :,  0] / input_shape[1]  # x or w
            coordinates[:, :,  1] = coordinates[:, :,  1] / input_shape[0]  # y or h

        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        # outputs d_1 x ... x d_n x channel shape
        outputs = torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)

        return outputs
