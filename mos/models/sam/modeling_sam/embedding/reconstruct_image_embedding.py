from torch import Tensor
from torch.nn import Module, Linear, ReLU, Linear, ModuleList, functional as F

from .typing import SparseEmbeddingsTensor, TextTokenEmbeddingTensor

from ...configuration_sam import SamPromptEncoderConfig
from transformers import BertTokenizer, BertModel
from torch.nn import Embedding, Parameter, init
import torch


class ReconstructImageEmbedding(Module):
    def __init__(
        self,
        config: SamPromptEncoderConfig,
    ):
        super().__init__()
        self.reconstruct_image_embedding = Parameter(
            torch.empty(
                (1, 1, config.hidden_size),
                dtype=torch.float,
            ),
            requires_grad=True,
        )
        init.normal_(self.reconstruct_image_embedding)

    def forward(self, batch_size: int) -> SparseEmbeddingsTensor:
        """
        Args:
            batch_size: batch size

        Returns:
            token_embedding: (bs, 1, hidden_size)
        """

        token_embedding = self.reconstruct_image_embedding.repeat(batch_size, 1, 1)
        return token_embedding
