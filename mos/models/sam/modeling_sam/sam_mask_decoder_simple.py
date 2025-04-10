from torch import nn
import torch
from ..configuration_sam import SamMaskDecoderConfig
import torch.nn.functional as F
from mos.utils.model_utils import LambdaModule

from .embedding.typing import (
    PredMasksTensor,
    SparseEmbeddingsTensor,
    ImageEmbeddingTensor,
    PromptEmbeddingTensor,
)
from .blocks import SamChannelNorm

from .sam_mask_decoder import (
    SamFeedForward,
    SamTwoWayTransformer,
)


class SamMaskDecoderSimple(nn.Module):
    """
    对SamMaskDecoder的简化,
    1. 没有IOU pred, 只输出一个mask
    2. 没有dense_prompt_embeddings, 只有sparse_prompt_embeddings
    3. 没有multimask_output
    """

    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        assert config.num_multimask_outputs == 0
        self.num_multimask_outputs = config.num_multimask_outputs
        # 4
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.mask_tokens: PromptEmbeddingTensor = nn.Parameter(
            torch.zeros([1, 1, self.num_mask_tokens, self.hidden_size]),
            requires_grad=True,
        )

        self.transformer = SamTwoWayTransformer(config)

        # should we create a new class for this?
        self.upscale_count = 3
        self.upscale_conv = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    self.hidden_size // (2**i), self.hidden_size // (2 ** (i + 1)), kernel_size=2, stride=2
                )
                for i in range(self.upscale_count)
            ]
        )
        self.upscale_layer_norm = nn.ModuleList(
            [
                SamChannelNorm(self.hidden_size // (2 ** (i + 1)), data_format="channels_first")
                for i in range(self.upscale_count - 1)
            ]
        )
        self.upscale_layer_norm.append(LambdaModule(lambda x: x))

        # self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        # self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        # self.upscale_layer_norm = SamChannelNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // (2**self.upscale_count), 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

    def forward(
        self,
        image_embeddings: ImageEmbeddingTensor,
        image_positional_embeddings: ImageEmbeddingTensor,
        sparse_prompt_embeddings: SparseEmbeddingsTensor,
    ) -> PredMasksTensor:
        # (bs, hidden_size, window_size, window_size) , (2, 256, 64, 64)
        batch_size, num_channels, height, width = image_embeddings.shape
        # 2, from (bs, out_counts, hidden_size)
        out_counts = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        # (bs, out_counts, n, hidden_size)
        output_tokens: PromptEmbeddingTensor = self.mask_tokens.repeat(batch_size, out_counts, 1, 1)

        # 按理说, 每次输入都要有prompt, 才能对应一个输出
        # to (bs, out_counts, 1, hidden_size)
        sparse_prompt_embeddings: PromptEmbeddingTensor = sparse_prompt_embeddings.unsqueeze(2)

        # (bs, out_counts, n, hidden_size)
        # n: mask_tokens + sparse_prompt_embeddings
        prompt_embeddings: PromptEmbeddingTensor = torch.cat([output_tokens, sparse_prompt_embeddings], dim=2)

        # Expand per-image data in batch direction to be per-point
        # (bs*out_counts, 256, 64, 64)
        # 也就是说out_counts多少, image_embeddings就要重复多少次! WTF....我还不如一个点来pred, 内存不用钱吗?!!
        if out_counts > 1:
            image_embeddings = image_embeddings.repeat(out_counts, 1, 1, 1)
        # (bs*out_counts, 256, 64, 64)
        image_positional_embeddings = image_positional_embeddings.repeat(out_counts, 1, 1, 1)

        # Run the transformer, image_positional_embedding are consumed
        # prompt_emask_tokens_outmbedding (bs, cout_counts, n, hidden_size)
        #   n: mask_tokens + sparse_prompt_embeddings
        # image_embeddings (bs*out_counts, hidden_size, win_size, win_size)
        mask_tokens_out, image_embeddings = self.transformer(
            prompt_embeddings=prompt_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * out_counts, num_channels, height, width
        )

        # 两次反卷积, 得到(bs, 32, h, w)的mask
        for i in range(self.upscale_count):
            image_embeddings = self.activation(self.upscale_layer_norm[i](self.upscale_conv[i](image_embeddings)))

        # 需要输出多个mask
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        # (bs, out_counts, n_masks, 32)
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = image_embeddings.shape  # (bs, 32, h, w)
        image_embeddings = image_embeddings.reshape(batch_size, out_counts, num_channels, height * width)
        # (bs, out_counts, n_masks, h, w) = (bs, out_counts, n_masks, 32) @ (bs, out_counts, 32, h * w)
        masks: PredMasksTensor = (hyper_in @ image_embeddings).reshape(batch_size, out_counts, -1, height, width)

        return masks
