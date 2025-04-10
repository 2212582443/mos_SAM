from typing import TypeAlias

from torch import LongTensor, Tensor, FloatTensor, BoolTensor


SparseEmbeddingsTensor: TypeAlias = FloatTensor
"""Sparse embeddings, (bs, output_count, hidden_size)"""

# (x_min, y_min, x_max, y_max)
BoxPrompt: TypeAlias = tuple[int, int, int, int]

BoxCoordTensor: TypeAlias = FloatTensor
"""(bs, nb_boxes, 4):

    4: (x_min, y_min, x_max, y_max)
"""

# (bs, nb_boxes, hidden_size)

DenseEmbeddingsTensor: TypeAlias = Tensor
"""Dense embeddings, (bs, hidden_size, window_size, window_size)"""

PointCoordTensor: TypeAlias = FloatTensor
"""(bs, nb_points, 2)"""
PointLabelTensor: TypeAlias = LongTensor
"""(bs, nb_points)"""

TextTokenEmbeddingTensor: TypeAlias = FloatTensor
"""(bs, seq_len, token_hidden_size)"""

ImageEmbeddingTensor: TypeAlias = FloatTensor
"""Image embeddings, (bs, hidden_size, window_size, window_size)"""

GrayImageTensor: TypeAlias = FloatTensor
"""(bs, 1, h, w)"""

ContrastMatrixTensor: TypeAlias = BoolTensor
"""(n,n)
对比学习矩阵, 同一个图像为true, 否则为false
"""

MaskPromptTensor: TypeAlias = FloatTensor
"""(bs, 1, h, w)"""

ColorImageTensor: TypeAlias = FloatTensor
"""(bs, 3, h, w)"""

DepthPositionInfoTensor: TypeAlias = FloatTensor
"""(bs, d)"""

SegmentTensor: TypeAlias = FloatTensor
"""(bs, h, w)"""

PredIouScoresTensor: TypeAlias = FloatTensor
"""shape `(batch_size, output_count, num_masks)`:
    The iou scores of the predicted masks.
"""
PredMasksTensor: TypeAlias = FloatTensor
"""shape `(batch_size, output_count, num_masks, height, width)`:
    The predicted low resolutions masks. Needs to be post-processed by the processor

    output_count: 每个提示对一一个输出

    num_masks: 每个输出对应的mask数量(默认为3个最优匹配)

    height, width: mask的高和宽
"""

PromptEmbeddingTensor: TypeAlias = FloatTensor
""" 用于attention提升的tensor (batch, output_count, n_tokens, token_size)

    batch: batch size

    output_count: 一个batch中的提示的数量, 没个提示对应输出一个mask

    n_tokens: 一个点对应的token数量, 一个token对应一个embedding
        点的构成由: iou_token (1), mask_tokens (4), sparse_prompt_embeddings (1) 组成

    token_size: embedding的维度, 256
"""
ImageTransformerEmbeddingTensor: TypeAlias = FloatTensor
"""Image Transformer embedding, (bs, output_count, win*win, hidden_size)

因为需要cross attention, 所以ImageTransformerEmbeddingTensor和PromptEmbeddingTensor的定义是一样的
"""
