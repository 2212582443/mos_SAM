import torch
from torch.nn import Module, Parameter
from mos.models.sam.modeling_sam.embedding.text_embedding import TextEmbeddingGRU


from mos.models.sam.modeling_sam.embedding.typing import (
    SparseEmbeddingsTensor,
    ImageEmbeddingTensor,
    TextTokenEmbeddingTensor,
)
from mos.models.sam.configuration_sam import SamVisionConfig, SamPromptEncoderConfig
from mos.models.sam.modeling_sam.embedding.positional_embedding import SamPositionalEmbedding
from mos.models.sam.modeling_sam.embedding.image_embedding_mae import ImageEmbeddingMae
from mos.models.mmae.cls_embedding import ClsEmbedding, ClsTypeTokenEmbeddingTensor


class SamPromptEncoder(Module):
    def __init__(
        self,
        prompt_config: SamPromptEncoderConfig,
        vision_config: SamVisionConfig,
        adapter_rank_3d=0,
    ):
        super().__init__()
        self.hidden_size = prompt_config.hidden_size
        self.image_embedding_size = prompt_config.image_embedding_size
        self.embed_image = ImageEmbeddingMae(vision_config, adapter_rank_3d)
        self.positional_embedding = SamPositionalEmbedding(vision_config)

        self.text_embedding = TextEmbeddingGRU(prompt_config)

        self.masked_img_embd = Parameter(torch.empty((1, prompt_config.hidden_size, 1, 1)), requires_grad=True)

    def patch_img_embd(self, img_embd: ImageEmbeddingTensor, ratial: float = 0.5) -> ImageEmbeddingTensor:
        """
        Args:
            ratial: 0.0 ~ 1.0, 0.0表示不patch, 1.0表示全patch
        """
        bs, hidden_size, h, w = img_embd.shape
        condition = torch.rand((bs, 1, h, w), device=img_embd.device).repeat(1, hidden_size, 1, 1)
        img_embd = torch.where(
            condition < ratial,
            self.masked_img_embd,
            img_embd,
        )
        return img_embd

    def get_device(self):
        device = self.parameters().__next__().device
        return device

    def forward(self, text_token: TextTokenEmbeddingTensor) -> SparseEmbeddingsTensor:
        sparse_embeddings = self.text_embedding(text_token)

        return sparse_embeddings

    def get_image_wide_positional_embeddings(self) -> ImageEmbeddingTensor:
        """
        Returns: (1, 256, 64, 64), (1, 256, height, width)
        """
        size = self.image_embedding_size
        target_device = self.positional_embedding.positional_embedding.device
        target_dtype = self.positional_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        # (64, 64, 256)
        positional_embedding = self.positional_embedding(torch.stack([x_embed, y_embed], dim=-1))
        # 1xchannel x height x width
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)
