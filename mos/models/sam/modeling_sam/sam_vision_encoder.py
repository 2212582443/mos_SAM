import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple
import collections

from .embedding.typing import ColorImageTensor, ImageEmbeddingTensor

from ..configuration_sam import SamVisionConfig
from .blocks import SamChannelNorm, SamMLPBlock
import loralib as lora


class SamPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    把图片按照块大小(16x16)进行编码, 返回(bs, blk_h, blk_w, 768)
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class SamVisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config: SamVisionConfig, window_size: int):
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )

        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout

        # self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.qkv = lora.MergedLinear(config.hidden_size, config.hidden_size * 3, r=8, enable_lora=[True, False, True])

        # 多头的头数虽然多了, 但是每个头的维度变小了, 而且参数要保证总的参数量不变
        # 所以最后输出的projection的维度和原来的hidden_size一样
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("Input size must be provided if using relative positional encoding.")

            # initialize relative positional embeddings
            # (27, 64), (N, embedd)
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))

            # (27, 64), (N, embedd)
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (N, embedd).

        Returns:
            Extracted positional embeddings according to relative positions.
        """

        # 按照max_rel_dist的个数对坐标rel_pos进行坐标插值, 得到max_rel_dist个坐标
        # max_rel_dist = 27
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized: torch.Tensor = F.interpolate(
            # [N, embedd] -> [1, N, embedd] -> [1, embedd, N]
            rel_pos.unsqueeze(0).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        # [1, embedd, max_rel_dist] -> [embedd, max_rel_dist] -> [max_rel_dist, embedd]
        rel_pos_resized = rel_pos_resized.squeeze(0).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        # 结果为一个矩阵 [q_size, k_size]
        # [
        #   [ 13, 12, ....0],
        #   [ 14, 13, ....1],
        #   ...
        #   [ 26, 25, ....13]
        # ]
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        query: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        # 14, 14
        query_height, query_width = q_size
        # 14, 14
        key_height, key_width = k_size
        # (14,14,64) , (q_height, k_height, hidden_size)
        relative_position_height = self.get_rel_pos(query_height, key_height, self.rel_pos_h)
        # (14,14,64) , (q_width, k_width, hidden_size)
        relative_position_width = self.get_rel_pos(query_width, key_width, self.rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        # 这里同样模仿了QK^T的乘法, 消除了hidden_size的维度
        # 这样就能得到一个和attn相同(QK^T/sqrt(d))的维度(需要扩充维度, h或者w相同)
        # (bs, &q_height, q_width, &hidden_size) * (&q_height, k_height, &hidden_size)^T -> (bs, &q_height, q_width, k_height)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        # (bs, q_height, &q_width, &hidden_size) * (&q_width, k_width, &hidden_size)^T -> (bs, q_height, &q_width, k_width
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)

        # attn = QK^T/sqrt(d)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        # 注意rel_h, rel_w这里进行了维度扩充
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)

        # 可以化简为
        #  let R = relative_position_height, S = relative_position_width
        # QK^T/sqrt(d) + QR^T + QS^T
        # = Q(K/sqrt(d) + R + S)^T
        # 这样化简后再用flash attention速度应该会快很多
        return attn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对一个窗口内的hidden_state使用注意力
        # batch_size = bs*原先的num_windows
        batch_size, height, width, channel = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        qkv = (
            # qkv project(multi head)
            # hidden_states: (batch_size, 14, 14, 768)
            self.qkv(hidden_states.reshape(-1, channel))
            # qkv: (batch_size, 14,14, 768*3)
            # num_attention_heads=12, 结果为 (batch_size, 14*14,3, 12, 64)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
            # (qkv, batch_size, num_attention_heads, 14*14, qkv_hidden_size)
            # 结果为 (3, batch_size, 12, 14*14, 64)
        )
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        # QK^T/sqrt(d)
        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        # add position embedding, 用相对位置编码, 可以化简,见add_decomposed_rel_pos的推导
        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(attn_weights, query, (height, width), (height, width))

        # softmax, 此为q对k的注意力矩阵, 论文中可能会监控该注意力
        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        # dropout
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # attention = QK^T/sqrt(d)+pos -> softmax -> dropout -> *V
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)

        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        return attn_output


class SamVisionLayer(nn.Module):
    def __init__(self, config: SamVisionConfig, window_size: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SamVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SamMLPBlock(config)
        self.window_size = window_size

    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        按照14x14的窗口进行划分, 返回(bs*num_windows, 14, 14, 768),
        注, [2, 5, 8, 11]层不进行划分, 其他层按照window_size=14进行划分. 各划分之间不重叠
        Args:
        Partition into non-overlapping windows with padding if needed.
            hidden_states (tensor): input tokens with [batch_size, height, width, channel]. window_size (int): window
            size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width): padded height and width before partition
        """
        batch_size, height, width, channel = hidden_states.shape

        # height 64, widow_size 14, pad_h 0
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w

        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
        Window unpartition into original sequences and removing padding.
            hidden_states (tensor):
                input tokens with [batch_size * num_windows, window_size, window_size, channel].
            window_size (int):
                window size.
            padding_shape (Tuple):
                padded height and width (pad_height, pad_width).
            original_shape (Tuple): original height and width (height, width) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [batch_size, height, width, channel].
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        )

        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # Window partition
        # 按照窗口划分区域, 窗口内使用注意力
        if self.window_size > 0:
            height, width = hidden_states.shape[1:3]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        # 窗口内使用注意力
        hidden_states = self.attn(hidden_states=hidden_states)
        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))

        # 残差连接
        hidden_states = residual + hidden_states
        # layer norm
        layernorm_output = self.layer_norm2(hidden_states)
        # mlp
        hidden_states = hidden_states + self.mlp(layernorm_output)

        return hidden_states


# 对hidden embedding(768)转换为image embedding(256)
class SamVisionNeck(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        # 对channel进行normalize
        self.layer_norm1 = SamChannelNorm(config.output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = SamChannelNorm(config.output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


class SamVisionEncoder(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            # (1, 64,64, 768)
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = SamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        pixel_values: ColorImageTensor,
    ) -> ImageEmbeddingTensor:
        """
        流程为:
            1. 把图像打成块(16x16), 并映射成embedding(bs, 64,64, 768)
            2. 加上位置编码
            3. 进行多层注意力计算
            4. 转换为图像的embedding(256)

        returns:
            image_embedding(bs, 256, h, w)

        """

        # 把图像打成块(16x16), 并映射成embedding(bs, 64,64, 768)
        hidden_states = self.patch_embed(pixel_values)
        # 加上位置编码
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        # 以下进行多层注意力计算
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states)

        # 转换为图像的embedding(256) (bs, hidden_size, window_size, window_size)
        image_embedding = self.neck(hidden_states)

        return image_embedding
