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
from typing import TypeAlias

import torch
import torch.nn as nn
from torch import LongTensor, Tensor

from timm.models.vision_transformer import PatchEmbed, Block

from mos.models.sam.modeling_sam.embedding.typing import ContrastMatrixTensor

from .pos_embed import get_2d_sincos_pos_embed
from .cls_embedding import ClsEmbedding, ClsTypeTokenEmbeddingTensor

SrcImageType: TypeAlias = LongTensor
"""(bs)"""

TargetImageType: TypeAlias = ClsTypeTokenEmbeddingTensor
"""(bs, descriptor_count)"""


class MMaeDecoder(nn.Module):
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
        image_type_num=50,
        image_type_descriptor_len=5,
    ) -> None:
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(hidden_size, decoder_embed_dim, bias=True)

        self.image_types = ClsEmbedding(
            cls_count=image_type_num,
            cls_descriptor_count=image_type_descriptor_len,
            hidden_size=hidden_size,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            # bs, {image_type + cls_token + numpatched}, hidden
            torch.zeros(1, 1 + 1 + num_patches, decoder_embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
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
            self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=True, image_type_token=True
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

    def forward_resotre(self, x: Tensor, ids_restore) -> Tensor:
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        return x

    def forward(self, image_type: TargetImageType, image: Tensor, ids_restore: Tensor | None) -> Tensor:
        if ids_restore is not None:
            image = self.forward_resotre(image, ids_restore)

        # add image type
        image_type = self.image_types(image_type)
        image = torch.cat([image_type, image], dim=1)

        # add pos embed
        image = image + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            image = blk(image)
        image = self.decoder_norm(image)

        # predictor projection
        image = self.decoder_pred(image)

        # remove image_type & cls token
        image = image[:, 2:, :]

        return image


class MViT(nn.Module):
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
        image_type_num=50,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, num_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.image_types = nn.Parameter(torch.zeros(image_type_num, 1, hidden_size))
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
                )
                for i in range(num_hidden_layers)
            ]
        )
        self.norm = norm_layer(hidden_size)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

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
        # torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.image_types, std=0.02)

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

    def _get_cls_token(self, image_type: SrcImageType):
        cls_token = self.image_types[image_type]
        pos_embed = self.pos_embed[:, :1, :]
        pos_embed = pos_embed.expand(cls_token.shape[0], -1, -1)
        cls_token = cls_token + pos_embed
        return cls_token

    def forward(self, image_type: SrcImageType, image):
        # embed patches, (bs, N, 768)
        image = self.patch_embed(image)

        # add pos embed w/o cls token
        image = image + self.pos_embed[:, 1:, :]

        # append cls token
        cls_tokens = self._get_cls_token(image_type)
        image = torch.cat((cls_tokens, image), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            image = blk(image)
        image = self.norm(image)

        return image

    def forward_mask_ratio(self, image_type: SrcImageType, images: Tensor, mask_ratio: float):
        """
        Returns:
            (
                embedding: (bs, N+1, 768),
                mask: (bs, N),
                ids_restore: (bs, N)
            )

        """
        # embed patches
        images = self.patch_embed(images)

        # add pos embed w/o cls token
        images = images + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        images, mask, ids_restore = self.random_masking(images, mask_ratio)

        # append cls token
        cls_tokens = self._get_cls_token(image_type)
        images = torch.cat((cls_tokens, images), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            images = blk(images)
        images = self.norm(images)

        return images, mask, ids_restore

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
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


class MMaeVit(nn.Module):
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
        image_type_num=20,
    ):
        super().__init__()
        self.vit = MViT(
            img_size=img_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            mlp_ratio=mlp_ratio,
            num_attention_heads=num_attention_heads,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
            image_type_num=image_type_num,
        )
        self.decoder = MMaeDecoder(
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

    def forward(
        self,
        image_type: SrcImageType,
        imgs,
        result_type: TargetImageType,
        result_image,
        mask_ratio=0.75,
    ):
        # latent: cls_token + patch_token
        latent, mask, ids_restore = self.vit.forward_mask_ratio(image_type, imgs, mask_ratio)
        pred = self.decoder(result_type, latent, ids_restore)  # [N, L, p*p*3]
        loss = self.vit.forward_loss(result_image, pred, mask)
        return pred, loss, mask

    def forward_contrast_loss(
        self,
        image_type: SrcImageType,
        imgs,
        result_type: TargetImageType,
        result_image,
        contrast_matrix: ContrastMatrixTensor,
        mask_ratio=0.75,
    ):
        # latent: cls_token + patch_token
        latent, mask, ids_restore = self.vit.forward_mask_ratio(image_type, imgs, mask_ratio)
        pred = self.decoder(result_type, latent, ids_restore)  # [N, L, p*p*3]
        reconstruction_loss = self.vit.forward_loss(result_image, pred, mask)

        # 对比学习的loss
        latent2, _mask2, _ids_restore2 = self.vit.forward_mask_ratio(image_type, imgs, mask_ratio)

        latent1, latent2 = latent[:, 0], latent2[:, 0]
        latent1 = latent1 / latent1.norm(dim=-1, keepdim=True)
        latent2 = latent2 / latent2.norm(dim=-1, keepdim=True)

        similarity = latent1 @ latent2.T

        if True:
            bs = latent1.shape[0]
            scale = torch.ones((bs, bs), dtype=torch.float, device=latent1.device)
            # instance norm为F范数, dim为768, 大致cos相似度极端情况下缩放值为 sqrt(768) = 27 才能归一化到1
            scale[contrast_matrix] = 27
            similarity *= scale

        target_similarity = contrast_matrix.float()

        contrast_loss = ((similarity - target_similarity) ** 2).mean()

        return pred, reconstruction_loss, mask, contrast_loss


def sam_vit():
    return MViT(
        img_size=128,
        patch_size=8,
        num_channels=1,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    )


def sam_mae():
    return MMaeVit(
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
