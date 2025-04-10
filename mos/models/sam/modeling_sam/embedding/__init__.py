from abc import abstractmethod
from typing_extensions import Self
import torch
from torch import Tensor
from torch.nn import Module, Parameter


from .typing import (
    BoxCoordTensor,
    GrayImageTensor,
    MaskPromptTensor,
    PointCoordTensor,
    PointLabelTensor,
    SparseEmbeddingsTensor,
    ImageEmbeddingTensor,
    DenseEmbeddingsTensor,
    TextTokenEmbeddingTensor,
)
from ...configuration_sam import SamVisionConfig, SamPromptEncoderConfig
import numpy as np
from .positional_embedding import SamPositionalEmbedding
from .box_embedding import BoxEmbedding, BoxPrompt, box2tensor
from .image_embedding_sam import ImageEmbeddingSam
from .mask_embedding import MaskEmbedding
from .point_embedding import PointEmbedding, PointPrompt, point2tensor
from .text_embedding import TextEmbeddingMLP, text2tensor
from .reconstruct_image_embedding import ReconstructImageEmbedding


class Prompt(object):
    """单张图像的prompt"""

    def __init__(
        self,
        masks: MaskPromptTensor | None = None,
        boxes: list[BoxPrompt] | None = None,
        points: list[PointPrompt] | None = None,
        texts: TextTokenEmbeddingTensor | str | None = None,
        construct: GrayImageTensor = None,
    ):
        self.masks = masks
        self.boxes = boxes
        self.points = points
        self.texts = texts
        self.reconstruct_image = construct

    def rand_drop_prompt(self, prompt_to_keep: int = -1) -> Self:
        """随机丢弃prompt中的一些元素, 最多保留prompt_to_keep个元素
        Args:
            prompt_to_keep: 保留prompt_to_keep个prompt
                -1 表示保留所有prompt
        """
        if prompt_to_keep == -1 or prompt_to_keep >= 4:
            return self

        prompt_flag = []
        if self.masks is not None:
            prompt_flag.append(0)
        if self.boxes is not None:
            prompt_flag.append(1)
        if self.points is not None:
            prompt_flag.append(2)
        if self.texts is not None:
            prompt_flag.append(3)
        if self.reconstruct_image is not None:
            prompt_flag.append(4)

        if len(prompt_flag) <= prompt_to_keep:
            return self

        prompt_flag = np.random.choice(prompt_flag, prompt_to_keep, replace=False)
        prompt = Prompt()
        for flag in prompt_flag:
            if flag == 0:
                prompt.masks = self.masks
            elif flag == 1:
                prompt.boxes = self.boxes
            elif flag == 2:
                prompt.points = self.points
            elif flag == 3:
                prompt.texts = self.texts
            elif flag == 4:
                prompt.reconstruct_image = self.reconstruct_image
        return prompt


class SamPromptEncoder(Module):
    def __init__(
        self,
        prompt_config: SamPromptEncoderConfig,
        vision_config: SamVisionConfig,
    ):
        super().__init__()
        self.hidden_size = prompt_config.hidden_size
        self.image_embedding_size = prompt_config.image_embedding_size
        self.embed_image = ImageEmbeddingSam(vision_config)
        self.positional_embedding = SamPositionalEmbedding(vision_config)

        self.embed_mask = MaskEmbedding(prompt_config)
        self.embed_point = PointEmbedding(prompt_config, self.positional_embedding)
        self.embed_box = BoxEmbedding(prompt_config, self.positional_embedding)
        self.embed_text = TextEmbeddingMLP(prompt_config)
        self.constrcut_embedding = ReconstructImageEmbedding(prompt_config)

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

    def forward(
        self, prompt: Prompt | None = None, prompt_to_keep: int = -1
    ) -> tuple[SparseEmbeddingsTensor, DenseEmbeddingsTensor]:
        """处理prompt, 返回prompt的embedding
        Args:
            prompt: Prompt
            prompt_to_keep: 保留prompt_to_keep个prompt
                -1 表示保留所有prompt
        returns:
            (sparse_embeddings, dense_embeddings)
            sparse_embeddings: (bs, n, hidden_size)
            dense_embeddings: (bs, hidden_size, h, w)
        """
        sparse_embeddings = []  # 稀疏的embedding (bs, n, hidden_size)
        device = self.get_device()

        if prompt is None:
            prompt = Prompt()

        masks = prompt.masks
        if masks is not None:
            masks = masks.to(device)
        dense_embeddings: DenseEmbeddingsTensor = self.embed_mask(masks)

        prompt = prompt.rand_drop_prompt(prompt_to_keep)

        if prompt.boxes is not None:
            boxes: BoxCoordTensor = box2tensor(prompt.boxes)
            boxes = boxes.to(device)
            box_embedding = self.embed_box(boxes)
            sparse_embeddings.append(box_embedding)

        if prompt.points is not None:
            points, labels = point2tensor(prompt.points)
            points: PointCoordTensor = points.to(device)
            labels: PointLabelTensor = labels.to(device)
            point_embedding = self.embed_point(points, labels)
            sparse_embeddings.append(point_embedding)

        prompt_texts = prompt.texts
        if prompt_texts is not None:
            if prompt_texts is str:
                prompt_texts = text2tensor(prompt_texts)
            prompt_texts: TextTokenEmbeddingTensor = prompt_texts.to(device)
            text_embedding = self.embed_text(prompt_texts)
            sparse_embeddings.append(text_embedding)

        construct_embedding = prompt.reconstruct_image
        if construct_embedding is not None:
            bs = construct_embedding.shape[0]
            construct_embedding: ImageEmbeddingTensor = self.constrcut_embedding(bs)
            sparse_embeddings.append(construct_embedding)

        while len(sparse_embeddings) < prompt_to_keep:
            sparse_embeddings.append(torch.zeros((1, 1, self.hidden_size), device=device))

        sparse_embeddings = torch.cat(sparse_embeddings, dim=1)

        return (sparse_embeddings, dense_embeddings)

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


class BatchPrompt(object):
    @abstractmethod
    def encode(self, prompt_encoder: SamPromptEncoder) -> SparseEmbeddingsTensor:
        pass


class TextBatchPrompt(BatchPrompt):
    def __init__(self, text_embeddings: list[TextTokenEmbeddingTensor]):
        super().__init__()
        self.text_embeddings = text_embeddings

    def encode(self, prompt_encoder: SamPromptEncoder) -> SparseEmbeddingsTensor:
        text_embeddings: list[SparseEmbeddingsTensor] = [
            prompt_encoder.embed_text(text_embedding) for text_embedding in self.text_embeddings
        ]
        text_embeddings = torch.cat(text_embeddings, dim=0)
        return text_embeddings


class PointBatchPrompt(BatchPrompt):
    def __init__(self, point_embeddings: list[list[PointPrompt]]):
        super().__init__()
        self.point_embeddings = point_embeddings

    def encode(self, prompt_encoder: SamPromptEncoder) -> SparseEmbeddingsTensor:
        point_list = []
        label_list = []
        for point in self.point_embeddings:
            points, labels = point2tensor(point)
            point_list.append(points)
            label_list.append(labels)

        point_list = torch.cat(point_list, dim=0).to(prompt_encoder.get_device())
        label_list = torch.cat(label_list, dim=0).to(prompt_encoder.get_device())
        point_embedding = prompt_encoder.embed_point(point_list, label_list)
        return point_embedding
