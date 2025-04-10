import random

import torch

from mos.models.sam.modeling_sam.embedding.text_embedding import text2tensor
from mos.models.sam.modeling_sam.embedding.typing import TextTokenEmbeddingTensor


# 类别的token字典, 同一个类别的token有多种表达方式
# todo: 词汇扩充为形状描述(医学字典)
_token_text = {
    1: [
        "心包脂肪",
        "EAT",
        "位于心肌和心包膜脏层之间的脂肪组织",
        "心包膜壁层之外的脂肪组织",
        "心包脂肪组织",
        " EAT及心包外脂肪组织",
        "位于房室沟以及室间沟,覆盖大部分的心脏血管,随着EAT的增多,脂肪开始分布于心室游离壁,甚至可能覆盖整个心脏",
        "起源于胚外中胚层，与肠系膜、大网膜脂肪组织相同都由胚胎时期的棕色脂肪组织发育而来",
    ],
    2: ["左心房", "Left ventricular cavity", "LV cavity"],
    3: [
        "心肌环",
        "心肌层",
        "Myocardium",
        "myocardium",
        "cardiac muscle",
        "heart muscle",
        "one of three types of vertebrate muscle tissues, with the other two being skeletal muscle and smooth muscle",
    ],
    4: ["右心房", "Right ventricular cavity", "RV cavity"],
}

_cls2id = {
    "EAT": 1,
    "LV": 2,
    "MYO": 3,
    "RV": 4,
}


@torch.no_grad()
def _parse_token_text(token_text) -> dict[str, list[TextTokenEmbeddingTensor]]:
    return {
        f"{k}": [
            # [1, n,768]
            text2tensor(v)
            for v in vs
        ]
        for k, vs in token_text.items()
    }


_token_embedding = None


def prepare_token_embedding_to_device(device):
    global _token_embedding
    if _token_embedding is None:
        _token_embedding = _parse_token_text(_token_text)
    _token_embedding = {k: [v.to(device) for v in vs] for k, vs in _token_embedding.items()}


def get_cls_text_embedding(cls_id: int) -> TextTokenEmbeddingTensor:
    """获取token的embedding

    Args:
        token_id (int): token id 分割的类别id
    Returns:
        shape: [1, n, 768] embedding
    """
    global _token_embedding
    if _token_embedding is None:
        _token_embedding = _parse_token_text(_token_text)

    cls_id = f"{cls_id}"
    embedding_list: list[TextTokenEmbeddingTensor] = _token_embedding[cls_id]
    embedding = random.choice(embedding_list)
    return embedding
