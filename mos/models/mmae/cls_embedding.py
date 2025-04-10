from typing import TypeAlias
from torch import nn, FloatTensor, LongTensor
from torch.nn import Module
import torch


ClsEmbeddingsTensor: TypeAlias = FloatTensor
"""Sparse embeddings, (bs, 1, hidden_size)"""

ClsTypeTokenEmbeddingTensor: TypeAlias = LongTensor
"""(bs, descriptor_count)"""


class ClsEmbedding(Module):
    def __init__(
        self,
        hidden_size: int = 256,
        cls_count: int = 50,  # 分类的数量
        cls_descriptor_count: int = 5,  # 分类描述符的数量
        enable_pos_embed: bool = False,
        padding_index: int = 30,
    ):
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(cls_count, 1, hidden_size))
        torch.nn.init.normal_(self.cls_embedding, std=0.02)
        self.pos_embedding = nn.Parameter(torch.zeros(1, cls_descriptor_count, hidden_size))
        self.descriptor_count = cls_descriptor_count
        self.hidden_size = hidden_size
        self.enable_pos_embed = enable_pos_embed
        self.padding_index = padding_index

    def forward(self, cls_type: ClsTypeTokenEmbeddingTensor) -> ClsEmbeddingsTensor:
        bs, descriptor_count = cls_type.shape

        # assert descriptor_count == self.descriptor_count

        cls_index = cls_type.reshape(-1)
        cls_token = self.cls_embedding.index_select(dim=0, index=cls_index)

        if descriptor_count > 1:
            cls_token *= (cls_index != self.padding_index).int()  # 30 是 padding 的 cls
            cls_token = cls_token.reshape(bs, descriptor_count, self.hidden_size)

        if self.enable_pos_embed:
            pos = self.pos_embedding[:, :descriptor_count, :].repeat(bs, 1, 1)
            cls_token = cls_token + pos

        if descriptor_count > 1:
            cls_token = cls_token.sum(dim=1, keepdim=True)

        return cls_token
