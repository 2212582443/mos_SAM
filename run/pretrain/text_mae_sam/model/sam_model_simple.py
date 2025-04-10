import torch
from mos.models.mmae.cls_embedding import ClsTypeTokenEmbeddingTensor

from mos.models.sam.configuration_sam import SamConfig
from mos.models.sam.modeling_sam.sam_mask_decoder_simple import SamMaskDecoderSimple

from .embedding import SamPromptEncoder
from mos.models.sam.modeling_sam.embedding.typing import (
    GrayImageTensor,
    SparseEmbeddingsTensor,
    PredMasksTensor,
    ImageEmbeddingTensor,
    TextTokenEmbeddingTensor,
)


class SamModelSimple(torch.nn.Module):
    def __init__(self, config: SamConfig):
        super().__init__()
        self.mask_decoder = SamMaskDecoderSimple(config.mask_decoder_config)
        self.prompt = SamPromptEncoder(config.prompt_encoder_config, config.vision_config)

    def forward_image(
        self, image: GrayImageTensor, text_token: TextTokenEmbeddingTensor
    ) -> tuple[ImageEmbeddingTensor, tuple[int, int]]:
        image_embeddings = self.prompt.embed_image(image)
        sparse_embeddinsg = self.prompt(text_token=text_token)
        return self.forward(image_embeddings, sparse_embeddinsg)

    def forward(
        self,
        image_embeddings: ImageEmbeddingTensor,
        sparse_embeddings: SparseEmbeddingsTensor | None = None,
    ) -> PredMasksTensor:
        # (1, 256, 64, 64)
        image_positional_embeddings = self.prompt.get_image_wide_positional_embeddings()
        # repeat with batch size

        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        pred_mask: PredMasksTensor = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
        )

        pred_mask = torch.sigmoid(pred_mask.float())

        return pred_mask
