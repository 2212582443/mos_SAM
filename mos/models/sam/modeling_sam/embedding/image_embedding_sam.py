from torch.nn import Module
import torch

from ...configuration_sam import (
    SamVisionConfig,
)
from ..sam_vision_encoder import SamVisionEncoder
from .typing import ColorImageTensor, ImageEmbeddingTensor, GrayImageTensor

from torchvision import transforms


class ImageEmbeddingSam(Module):
    def __init__(self, config: SamVisionConfig) -> None:
        super().__init__()
        image_size = config.image_size
        self.image_transform = transforms.Compose(
            [
                # Resize to the size the model expects
                transforms.Resize((image_size, image_size), antialias=True),
                # transforms.ToTensor(),
                # Normalization values for pre-trained PyTorch models
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[ 0.229, 0.224, 0.225]
                # )
            ]
        )
        self.is_rgb_input = config.num_channels > 1

        self.vision_encoder = SamVisionEncoder(config)

    @torch.no_grad()
    def forward(self, image: GrayImageTensor) -> tuple[ImageEmbeddingTensor, tuple[int, int]]:
        """处理图像, 返回图像的embedding和图像的original_sizes
        Args:
            image: (bs, 1, h, w)

        return (image_embedding, original_sizes)
            image_embedding: (bs, hidden_size, window_size, window_size)
                Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
                efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
                method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
            original_sizes: (h, w)

        """
        # (h, w)
        original_sizes = image.shape[-2:-1]
        image = self.image_transform(image)
        if self.is_rgb_input:
            image: ColorImageTensor = image.repeat(1, 3, 1, 1)  # to rgb
        image_embedding: ImageEmbeddingTensor = self.vision_encoder(image)
        return (image_embedding, original_sizes)
