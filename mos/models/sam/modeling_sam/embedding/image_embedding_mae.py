from torch.nn import Module
from mos.models.sam.configuration_sam import SamVisionConfig
from mos.models.sam.modeling_sam.embedding.typing import (
    ColorImageTensor,
    DepthPositionInfoTensor,
    GrayImageTensor,
    ImageEmbeddingTensor,
)


import torch.nn as nn
from torch import Tensor


from mos.models.mae.model_mae import ViT
from ..sam_vision_encoder import SamVisionNeck


class MaeVisionEncoder(nn.Module):
    def __init__(
        self,
        config: SamVisionConfig,
        adapter_rank_3d: int = 0,
    ):
        super().__init__()

        self.vit = ViT(
            img_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            mlp_ratio=config.mlp_ratio,
            num_attention_heads=config.num_attention_heads,
            norm_layer=nn.LayerNorm,
            norm_pix_loss=False,
            adapter_rank_3d=adapter_rank_3d,
        )

        h, w = config.image_size // config.patch_size, config.image_size // config.patch_size
        self.reshape_mask_embed = lambda x: x.reshape(x.shape[0], h, w, -1)
        self.neck = SamVisionNeck(config)

    # 不使用mask进行embedding
    def forward(self, imgs: ColorImageTensor, depth_info: DepthPositionInfoTensor = None) -> Tensor:
        """
        Args:
            imgs: (bs, 3, h, w)
            depth_info: (bs, d), d为3d图像depth的维度
        """
        # 把图像打成块(16x16), 并映射成embedding(bs, N+1, 768)
        latent = self.vit(imgs, depth_info)
        # remote cls
        latent = latent[:, 1:, :]
        # reshape to image(bs, 16, 16, 768)
        latent = self.reshape_mask_embed(latent)
        # image embedding(bs, 256, 16, 16)
        pred = self.neck(latent)
        return pred


class ImageEmbeddingMae(Module):
    def __init__(self, config: SamVisionConfig, adapter_rank_3d=0) -> None:
        super().__init__()
        self.vision_encoder = MaeVisionEncoder(config, adapter_rank_3d)

    def forward(
        self,
        image: GrayImageTensor,
        depth_info: DepthPositionInfoTensor = None,
    ) -> ImageEmbeddingTensor:
        """处理图像, 返回图像的embedding和图像的original_sizes
        Args:
            image: (bs, 1, h, w)
        """
        # (h, w)
        image_embedding: ImageEmbeddingTensor = self.vision_encoder(image, depth_info)
        return image_embedding
