import random
from typing import TypeAlias

import torch

from mos.models.mmae.cls_embedding import ClsTypeTokenEmbeddingTensor

from .text_embedding import text2tensor
from .typing import TextTokenEmbeddingTensor

target_modality = {
    "Restore": 0,  # 只是恢复原始图片
    "T1": 1,
    "T2": 2,
    "CINE": 3,
    "LGE": 4,
    "SSFP": 5,
    "CT": 6,
    "Flair": 7,
    "Ultrasound": 9,
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
    # 30-39 为图片属性描述
    "Empty": 30,  # 属性为空
    "CNT1": 31,  # 31-36 为计数器, 索引, 6进制, 两位可以表达0~35之间的数字
    "CNT2": 32,
    "CNT3": 33,
    "CNT4": 34,
    "CNT5": 35,
    "CNT6": 36,
    "NoAffined": 37,  # 未配准, 或者对齐
}

# 类别的token字典, 同一个类别的token有多种表达方式
# todo: 词汇扩充为形状描述(医学字典)
_token_text = {
    0: [
        "Restore",
        "Restore the original image",
    ],
    1: ["T1", "MRI T1 image"],
    2: ["T2", "MRI T2 image"],
    3: ["CINE", "MRI CINE image"],
    4: ["LGE", "MRI LGE image"],
    5: ["SSFP", "MRI SSFP image"],
    6: ["CT", "CT image"],
    7: ["Flair", "MRI Flair image"],
    9: [
        "Untrasound",
        "Ultrasound image",
    ],
    11: [
        "Left ventricular cavity",
        "LV cavity",
        "Left Ventricle",
        "LV segment",
    ],
    12: [
        "Right ventricular cavity",
        "RV cavity",
        "Right Ventricle",
        "RV segment",
    ],
    13: [
        "Myocardium of LV",
        "cardiac muscle",
        "heart muscle",
        "Myocardium segment",
    ],
    14: ["Left Atrium", "LA segment"],
    15: ["Right Atrium", "RA segment"],
    16: ["Ascending Aorta", "AO segment"],
    17: ["Pulmonary Artery", "PA segment"],
    18: ["Pulmonary Vein", "PV segment"],
    19: ["Vena Cava", "VA segment"],
    20: [
        # "EAT",
        "Epicardial Adipose Tissue",
        # "PAT",
        "Pericardial Adipose Tissue",
        "positioned between the myocardium and visceral pericardium",
        "tissue located between the myocardium and visceral pericardium",
        "located between the myocardium and visceral pericardium",
        "EAT segment",
        # "Epicardial Adipose Tissue segment",
        "PAT segment",
        # "Pericardial Adipose Tissue segment",
    ],
    30: [""],
    31: ["index 1"],
    32: ["index 2"],
    33: ["index 3"],
    34: ["index 4"],
    35: ["index 5"],
    36: ["index 6"],
    37: ["no affined", "unaligned"],
}
# convert key to int
_token_text = {int(k): v for k, v in _token_text.items()}

ClsIndex: TypeAlias = torch.Tensor


def make_cls_text_compose(cls: list[int]) -> list[str]:
    """把cls转换成文本描述
    Args:
        cls: [0~9, 31~37, 31~36*]
    Returns:
        str: 文本描述
    """
    result = []
    prompt_list = []
    cls.sort()
    for id in cls:
        if id == 30:
            continue
        prompt_list.append(_token_text[id])
    for a in prompt_list[0]:
        if len(prompt_list) == 1:
            result.append(f"Please output {a}.")
            continue
        for b in prompt_list[1]:
            if len(prompt_list) == 2:
                result.append(f"Please output {a} and {b}.".strip())
                continue
            for c in prompt_list[2]:
                if len(prompt_list) == 3:
                    result.append(f"Please output {a} and {b} and {c}.".strip())
                    continue
                for d in prompt_list[3]:
                    if len(prompt_list) == 4:
                        result.append(f"Please output {a} and {b} and {c} and {d}.".strip())
                        continue
                    for e in prompt_list[4]:
                        result.append(f"Please output {a} and {b} and {c} and {d} and {e}.".strip())

    return result


map_flag = [0] * 40
for i in range(11, 21):
    map_flag[i] = 1

DICT_TOKEN_SIZE = 40
_dict_token_weight = [1, DICT_TOKEN_SIZE, DICT_TOKEN_SIZE**2, DICT_TOKEN_SIZE**3, DICT_TOKEN_SIZE**4]


def cls2index_key(cls: list[int]):
    r = 0
    for i, v in enumerate(cls):
        r += v * _dict_token_weight[i]
    return r


def _make_cls_embedding_index() -> dict[int, int]:
    """
    所有可能组合为
    1. [0~9,31~37,31~36*]
    2. [0~9,30*]
    3. [11~20*,30*]
    映射公式为:
      map_number[30], map_flag[30], offset[30] -> 0
      map_number[cls] * map_flag[cls] + offset[cls]
    """
    index = dict()
    for a in range(10):
        for b in range(31, 38):
            b = b * _dict_token_weight[1]
            for c in range(31, 37):
                for d in range(31, 37):
                    for e in range(31, 37):
                        key = a + b + c * _dict_token_weight[2] + d * _dict_token_weight[3] + e * _dict_token_weight[4]
                        index.setdefault(key, len(index))
            c = 30 * _dict_token_weight[2]
            d = 30 * _dict_token_weight[3]
            e = 30 * _dict_token_weight[4]
            key = a + b + c + d + e
            index.setdefault(key, len(index))
    for a in range(11, 21):
        for b in range(11, 21):
            b = b * _dict_token_weight[1]
            _c = 30 * _dict_token_weight[2]
            _d = 30 * _dict_token_weight[3]
            _e = 30 * _dict_token_weight[4]
            key = a + b + _c + _d + _e
            index.setdefault(key, len(index))
            for c in range(11, 21):
                c = c * _dict_token_weight[2]
                key = a + b + c + _d + _e
                index.setdefault(key, len(index))
                for d in range(11, 21):
                    d = d * _dict_token_weight[3]
                    key = a + b + c + d + _e
                    index.setdefault(key, len(index))
                    for e in range(11, 21):
                        e = e * _dict_token_weight[4]
                        key = a + b + c + d + e
                        index.setdefault(key, len(index))
    return index


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
