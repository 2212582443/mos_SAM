import os
import random
import numpy as np
import torch.nn as nn
import torch


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types

        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def half_model(model: nn.Module):
    """将模型转换为半精度模型

    In PyTorch, batch-norm layers have convergence issues with half precision floats.
    If that's the case with you, make sure that batch norm layers are float32.
    """
    model = model.half()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d) or isinstance(layer, nn.BatchNorm1d):
            layer.float()

    return model


def use_cudnn():
    """使用cudnn加速

    You can make use of the cudnn benchmark instead of the vanilla benchmark.
    CuDNN can provided a lot of optimisation which can bring down your space usage,
    especially when the input to your neural network is of fixed size.
    """
    import torch.backends.cudnn as cudnn

    cudnn.enabled = True

    # if benchmark=True, deterministic will be False
    cudnn.benchmark = False
    cudnn.deterministic = True


def seed_all(seed: int | None = None):
    """设置随机种子"""
    if seed is not None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    use_cudnn()


def empyt_cache():
    """清空缓存"""
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def print_trainable_parameters(model, title: str = None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if title is None:
        title = ""
    print(
        f"{title}, trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
