from torch import Tensor
from torch.nn import Module, Embedding, Conv2d

from .typing import DenseEmbeddingsTensor, MaskPromptTensor
from ...configuration_sam import SamPromptEncoderConfig
from ..blocks import SamChannelNorm

from transformers.activations import ACT2FN


class MaskEmbedding(Module):
    def __init__(self, config: SamPromptEncoderConfig):
        super().__init__()
        self.mask_input_channels: int = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = SamChannelNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = SamChannelNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.no_mask_embed = Embedding(1, config.hidden_size)

    def embedding_mask(self, masks: MaskPromptTensor) -> Tensor:
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)

        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings

    def embedding_none(self) -> Tensor:
        return self.no_mask_embed.weight\
            .reshape(1, -1, 1, 1)\
            .expand(1, - 1, self.image_embedding_size[0], self.image_embedding_size[1])

    def forward(self, masks: MaskPromptTensor | None = None) -> DenseEmbeddingsTensor:
        """
        Args:
            masks: (bs, 1, h, w)

        Returns:
            mask_embedding: (bs, hidden_size, win_h, win_w)
        """
        if masks is None:
            return self.embedding_none()
        else:
            return self.embedding_mask(masks)
