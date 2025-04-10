import random

import torch

from mos.models.sam.modeling_sam.embedding.text_embedding import text2tensor
from mos.models.sam.modeling_sam.embedding.typing import TextTokenEmbeddingTensor


target_modality = {
    "Restore": 0,
    "T1": 1,
    "T2": 2,
    "CINE": 3,
    "LGE": 4,
    "SSFP": 5,
    "CT": 6,
    "Flair": 7,
    "Ultrasound": 9,
    "MergeLabelPadding": 10,  # 合并输出几个label的tag
    "LV": 11,
    "RV": 12,
    "MYO": 13,
    "LA": 14,
    "RA": 15,
    "AO": 16,
    "PA": 17,
    "PV": 18,
    "VA": 19,
    "EAT": 20,
}

# 类别的token字典, 同一个类别的token有多种表达方式
# todo: 词汇扩充为形状描述(医学字典)
_token_text = {
    0: [
        "请还原出原图",
        "不知道输出什么, 随便输出一个图吧",
        "Restore the original image",
    ],
    1: ["请输出T1模态的图像", "T1", "MRI T1 image"],
    2: ["请输出T2模态的图像", "T2", "MRI T2 image"],
    3: ["请输出CINE模态的图像", "CINE", "MRI CINE image"],
    4: ["请输出LGE模态的图像", "LGE", "MRI LGE image"],
    5: ["请输出SSFP模态的图像", "SSFP", "MRI SSFP image"],
    6: ["请输出CT模态的图像", "CT", "CT image"],
    7: ["请输出Flair模态的图像", "Flair", "MRI Flair image"],
    9: [
        "请输出超声模态的图像",
        "超声",
        "Untrasound",
        "Ultrasound image",
    ],
    10: [""],
    11: ["左心房", "Left ventricular cavity", "LV cavity", "Left Ventricle", "LV"],
    12: ["右心房", "Right ventricular cavity", "RV cavity", "Right Ventricle", "RV"],
    13: [
        "心肌环",
        "心肌层",
        "Myocardium",
        "Myocardium of LV",
        "左心室心肌",
        "cardiac muscle",
        "heart muscle",
        "one of three types of vertebrate muscle tissues, with the other two being skeletal muscle and smooth muscle",
    ],
    14: ["左心房", "Left Atrium", "LA"],
    15: ["右心房", "Right Atrium", "RA"],
    16: ["主动脉", "Ascending Aorta", "升主动脉", "AO"],
    17: ["肺动脉", "Pulmonary Artery", "PA"],
    18: ["肺静脉", "Pulmonary Vein", "PV"],
    19: ["Vena Cava", "腔静脉", "VA"],
    20: [
        "心包脂肪",
        "EAT",
        "Epicardial Adipose Tissue",
        "PAT",
        "Pericardial Adipose Tissue",
        "位于心肌和心包膜脏层之间的脂肪组织",
        "心包膜壁层之外的脂肪组织",
        "心包脂肪组织",
        "EAT及心包外脂肪组织",
        "位于房室沟以及室间沟,覆盖大部分的心脏血管,随着EAT的增多,脂肪开始分布于心室游离壁,甚至可能覆盖整个心脏",
        "起源于胚外中胚层，与肠系膜、大网膜脂肪组织相同都由胚胎时期的棕色脂肪组织发育而来",
    ],
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
