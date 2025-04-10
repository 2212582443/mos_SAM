# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch, math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp, LayerScale, DropPath

from .pos_embed import get_2d_sincos_pos_embed


class Vit3dAtapter(nn.Module):
    def __init__(self, input_channels: int, adapter_rank: int = 64) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.atapter_rank = adapter_rank

        self.encoder = nn.Sequential(
            *[
                nn.Conv3d(in_channels=input_channels, out_channels=adapter_rank, kernel_size=3, padding=1),
                nn.BatchNorm3d(adapter_rank),
                nn.ReLU(inplace=True),
            ]
        )
        self.middle = nn.Sequential(
            *[
                nn.Conv3d(in_channels=adapter_rank, out_channels=adapter_rank, kernel_size=3, padding=1),
                nn.BatchNorm3d(adapter_rank),
                nn.ReLU(inplace=True),
            ]
        )
        self.decoder = nn.Sequential(
            *[nn.ConvTranspose3d(in_channels=adapter_rank, out_channels=self.input_channels, kernel_size=3, padding=1)]
        )
        nn.init.constant_(self.decoder[-1].weight, 0)
        nn.init.constant_(self.decoder[-1].bias, 0)

    def forward(self, x_input: torch.Tensor, depth: int = 1):
        if depth <= 1:
            return x_input

        cls_token, img_token = x_input[:, :1, ::], x_input[:, 1:, ::]
        # n = 256, ch=768
        bs, n, ch = img_token.shape
        w = int(math.sqrt(n))  # 假定图像是正方形
        h = w

        assert w * h == n
        x = img_token.reshape((-1, depth, h, w, ch)).permute(0, 4, 1, 2, 3)
        dim_0 = x.shape

        x = self.encoder(x)
        x, indices_0 = F.max_pool3d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.middle(x)

        x = F.max_unpool3d(x, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x = F.relu(self.decoder(x))
        x = x.permute(0, 2, 3, 4, 1).reshape(bs, n, ch)
        img_token = img_token + x
        return torch.cat([cls_token, img_token], dim=1)


class SparsePositionalEncoding(nn.Module):
    def __init__(self, max_deep=30, hidden_size=1024):
        super().__init__()
        self.max_deep = max_deep
        self.positional = nn.Parameter(torch.zeros(max_deep, 1, hidden_size), requires_grad=True)

    def forward(self, location: torch.Tensor) -> torch.Tensor:
        """
        Args:
            location: (bs, d)
        Returns: (bs*d, 1, hidden_size)
        """
        bs, d = location.shape
        location = location.reshape(bs * d)
        pos = location / 10.0
        low_pos = pos.floor_().clamp_min_(0.0)
        high_pos = pos.ceil_().clamp_max_(self.max_deep - 1.0)

        resulta = self.positional.index_select(0, low_pos.long())
        resultb = self.positional.index_select(0, high_pos.long())
        positional = (pos - low_pos)[:, None, None] * resulta + (high_pos - pos)[:, None, None] * resultb
        return positional


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        adapter_rank_3d=0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.adapter_rank_3d = adapter_rank_3d
        if adapter_rank_3d > 0:
            self.vit3d_adapter = Vit3dAtapter(dim, adapter_rank_3d)

    def _mixed_3d_info(self, x: torch.Tensor, depth: int = 1) -> torch.Tensor:
        if depth == 1 or self.adapter_rank_3d < 1:
            return x
        return self.vit3d_adapter(x, depth)

    def freeze_adapter(self, is_freezed=True):
        if self.vit3d_adapter is None:
            return
        for p in self.vit3d_adapter.parameters():
            p.requires_grad = not is_freezed

    def forward(self, x: torch.Tensor, depth: int = 1) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self._mixed_3d_info(self.attn(self.norm1(x)), depth)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MaeDecoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        num_channels=3,
        hidden_size=1024,
        patch_size=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        adapter_rank_3d=0,
    ) -> None:
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(hidden_size, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=True)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    adapter_rank_3d=adapter_rank_3d,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * num_channels, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_adapter(self, is_freezed=True):
        for blk in self.decoder_blocks:
            blk.freeze_adapter(is_freezed)

    def forward_resotre(self, x: Tensor, ids_restore) -> Tensor:
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        return x

    def forward(self, x: Tensor, ids_restore: Tensor | None) -> Tensor:
        if ids_restore is not None:
            x = self.forward_resotre(x, ids_restore)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class ViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=1024,
        num_hidden_layers=24,
        mlp_ratio=4.0,
        num_attention_heads=16,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        adapter_rank_3d=0,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, num_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_size,
                    num_attention_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    adapter_rank_3d=adapter_rank_3d,
                )
                for i in range(num_hidden_layers)
            ]
        )
        self.norm = norm_layer(hidden_size)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.adapter_rank_3d = adapter_rank_3d
        if adapter_rank_3d > 0:
            self.sparse_positional = SparsePositionalEncoding(max_deep=30, hidden_size=hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: Tensor):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        ch = imgs.shape[1]
        x = (
            imgs.reshape(imgs.shape[0], ch, h, p, w, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(imgs.shape[0], h * w, p**2 * ch)
        )
        return x

    def unpatchify(self, x: Tensor):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        imgs = x.reshape(x.shape[0], h, w, p, p, -1).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], -1, h * p, w * p)
        return imgs

    def freeze_adapter(self, is_freezed=True):
        for blk in self.blocks:
            blk.freeze_adapter(is_freezed)
        if self.sparse_positional is not None:
            for p in self.sparse_positional.parameters():
                p.requires_grad = not is_freezed

    def random_masking(self, x: Tensor, mask_ratio: float):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 保存的是原始顺序
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor, depth_info: torch.Tensor = None):
        """
        Args:
            x: (bs, ch, H, W) or (bs, ch, d, h w)
            depth_info: (bs, d)
        Returns:
            embedding: (bs, N+1, 768) or (bs*d, N+1, 768)
        """

        shape = x.shape

        if len(x.shape) == 5:
            assert depth_info is not None
            bs, ch, d, h, w = shape
            x = x.permute(0, 2, 1, 3, 4).reshape(bs * d, ch, h, w)
        else:
            bs, ch, h, w = shape
            d = 1

        if self.adapter_rank_3d > 0 and depth_info is not None:  # 3d
            bs, d = depth_info.shape
            is_3d = True
            # (bs*d, 1, hidden_size)
            deep_pos_embed = self.sparse_positional(depth_info)
        else:
            is_3d = False

        # embed patches, (bs*d, N, 768)
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if is_3d:
            x = x + deep_pos_embed

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, d)
        x = self.norm(x)

        return x

    def forward_mask_ratio(self, x: Tensor, mask_ratio: float):
        """目前只支持2d
        Returns:
            (
                embedding: (bs, N+1, 768),
                mask: (bs, N),
                ids_restore: (bs, N)
            )

        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, 1)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = torch.mean(loss, dim=[1, 2])  # [N, L], mean loss per patch

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


class MaeVit(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=1024,
        num_hidden_layers=24,
        mlp_ratio=4.0,
        num_attention_heads=16,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
    ):
        super().__init__()
        self.vit = ViT(
            img_size=img_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            mlp_ratio=mlp_ratio,
            num_attention_heads=num_attention_heads,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
        )
        self.decoder = MaeDecoder(
            num_patches=self.vit.patch_embed.num_patches,
            num_channels=num_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            decoder_depth=decoder_depth,
            decoder_embed_dim=decoder_embed_dim,
            decoder_num_heads=decoder_num_heads,
        )

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.vit.forward_mask_ratio(imgs, mask_ratio)
        pred = self.decoder(latent, ids_restore)  # [N, L, p*p*3]
        reconstruction_loss = self.vit.forward_loss(imgs, pred, mask)

        return pred, reconstruction_loss, mask, latent


def sam_vit():
    return ViT(
        img_size=128,
        patch_size=8,
        num_channels=1,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    )


def sam_mae():
    return MaeVit(
        img_size=128,
        patch_size=8,
        num_channels=1,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=16,
    )
