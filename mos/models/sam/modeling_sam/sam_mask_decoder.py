from typing import Optional, Tuple, TypeAlias
import torch.nn.functional as F
from torch import FloatTensor, Tensor, nn
import torch

from mos.utils.model_utils import LambdaModule

from .embedding.typing import (
    PredIouScoresTensor,
    PredMasksTensor,
    SparseEmbeddingsTensor,
    DenseEmbeddingsTensor,
    ImageEmbeddingTensor,
    ImageTransformerEmbeddingTensor,
    PromptEmbeddingTensor,
)
from ..configuration_sam import SamMaskDecoderConfig
import math
from .blocks import SamChannelNorm, SamMLPBlock


class SamFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))

        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class SamAttention(nn.Module):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """

    def __init__(self, config, downsample_rate=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate

        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    def _separate_heads(
        self, hidden_states: ImageTransformerEmbeddingTensor | PromptEmbeddingTensor, num_attention_heads: int
    ) -> Tensor:
        batch, output_count, n_tokens, token_size = hidden_states.shape
        c_per_head = token_size // num_attention_heads
        hidden_states = hidden_states.reshape(batch * output_count, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(
        self,
        query: PromptEmbeddingTensor | ImageTransformerEmbeddingTensor,
        key: ImageTransformerEmbeddingTensor | PromptEmbeddingTensor,
        value: ImageTransformerEmbeddingTensor | PromptEmbeddingTensor,
    ) -> PromptEmbeddingTensor | ImageTransformerEmbeddingTensor:
        # Input projections
        # (bs, point_batch_size, hidden_size)
        # (2, 1, 6, 256) -> (2, 1, 6, 128)
        query = self.q_proj(query)
        # (bs, point_batch_size, hidden_size)
        # (2, 1, 4096, 256) -> (2, 1, 4096, 128)
        key = self.k_proj(key)
        # (2, 1, 4096, 256) -> (2, 1, 4096, 128)
        value = self.v_proj(value)

        output_count = query.shape[1]
        # Separate into heads
        # (12,1, 8, 16)
        query = self._separate_heads(query, self.num_attention_heads)
        # (2, 4096, 8, 16)
        key = self._separate_heads(key, self.num_attention_heads)
        # (2, 4096, 8, 16)
        value = self._separate_heads(value, self.num_attention_heads)

        # SamAttention
        _, _, _, c_per_head = query.shape
        # batch_size * point_batch_size  x N_heads x N_tokens x N_tokens

        # pytorch 的 Attention 不会用吗??!
        attn = query @ key.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out = attn @ value

        out = self._recombine_heads(out, output_count)
        out = self.out_proj(out)

        return out


class SamTwoWayAttentionBlock(nn.Module):
    def __init__(
        self, config: SamMaskDecoderConfig, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False
    ):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optionalk*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn = SamAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.cross_attn_token_to_image = SamAttention(config, downsample_rate=attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.mlp = SamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamAttention(config, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: PromptEmbeddingTensor,
        keys: ImageTransformerEmbeddingTensor,
        query_point_embedding: PromptEmbeddingTensor,
        key_point_embedding: ImageTransformerEmbeddingTensor,
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            # (bs, out_counts, n, hidden_size)
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            # (bs, out_counts, n, hidden_size)
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        # (2, 1, 6, 256)
        query: PromptEmbeddingTensor = queries + query_point_embedding
        # (2, 1, 4096, 256)
        key: ImageTransformerEmbeddingTensor = keys + key_point_embedding

        attn_out = self.cross_attn_token_to_image(
            query=query,
            key=key,
            value=keys,
        )
        queries = queries + attn_out

        queries = self.layer_norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        query = queries + query_point_embedding
        key = keys + key_point_embedding  # 重复了

        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out

        keys = self.layer_norm4(keys)

        outputs = (queries, keys)

        return outputs


class SamTwoWayTransformer(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_hidden_layers):
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = SamAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        prompt_embeddings: PromptEmbeddingTensor,
        image_embeddings: ImageEmbeddingTensor,
        image_positional_embeddings: ImageEmbeddingTensor,
        return_dict: Optional[bool] = None,
    ) -> tuple[PromptEmbeddingTensor, ImageTransformerEmbeddingTensor]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        # (bs, 1, window_size*window_size, hidden_size) from (2, 256, 64, 64)
        # (2, 256, 64, 64) -> (2, 256, 64*64) -> (2, 64*64, 256) -> (2, 1, 64*64, 256)
        image_embeddings: ImageTransformerEmbeddingTensor = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings: ImageTransformerEmbeddingTensor = (
            image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        )

        # Prepare queries
        # (bs, out_counts, n, hidden_size)
        queries = prompt_embeddings
        keys = image_embeddings

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=prompt_embeddings,
                key_point_embedding=image_positional_embeddings,
            )

        # Apply the final attenion layer from the points to the image
        query: PromptEmbeddingTensor = queries + prompt_embeddings
        key = keys + image_positional_embeddings

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return queries, keys


class SamMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        # 4
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

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

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: ImageEmbeddingTensor,
        image_positional_embeddings: ImageEmbeddingTensor,
        sparse_prompt_embeddings: SparseEmbeddingsTensor,
        dense_prompt_embeddings: DenseEmbeddingsTensor,
        multimask_output: bool,
    ) -> Tuple[PredMasksTensor, PredIouScoresTensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings: the embeddings from the image encoder
            image_positional_embedding: positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings: The embeddings of the points and boxes
            dense_prompt_embeddings: the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.

        Returns: (masks, iou_score)
        """
        # (bs, hidden_size, window_size, window_size) , (2, 256, 64, 64)
        batch_size, num_channels, height, width = image_embeddings.shape
        # 2, from (2, 1, 256)
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        # (5, 256), 1+4=5
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # (2, 1, 5, 256)
        output_tokens: PromptEmbeddingTensor = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        # 按理说, 每次输入都要有prompt, 才能对应一个输出
        sparse_prompt_embeddings: PromptEmbeddingTensor = sparse_prompt_embeddings.unsqueeze(2)
        # (2, 1, 6, 256)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)

        # (2, 1, 6, 256)
        # iou_token(1), mask_tokens(4), sparse_prompt_embeddings(1)
        prompt_embeddings: PromptEmbeddingTensor = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        # (2, 256, 64, 64)
        image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
        # (2, 256, 64, 64)
        image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

        # Run the transformer, image_positional_embedding are consumed
        prompt_embedding, image_embeddings = self.transformer(
            prompt_embeddings=prompt_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        iou_token_out = prompt_embedding[:, :, 0, :]
        mask_tokens_out = prompt_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        # 两次反卷积, 得到(256, 256)的mask
        upscaled_embedding = image_embeddings
        for i in range(self.upscale_count):
            upscaled_embedding = self.activation(self.upscale_layer_norm[i](self.upscale_conv[i](upscaled_embedding)))

        # 需要输出多个mask
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        # (2, 1, 4, 32)
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape  # (2, 32, 256, 256)
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        # (2, 1, 4, 256, 256)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        return outputs
