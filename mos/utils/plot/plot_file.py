from torchvision import utils as vutils
from torch import Tensor
import os


def save_image_tensor(input_tensor: Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor, [bs, ch, h, w], float, 0~1
    :param filename: 保存的文件名
    """

    file_dir = os.path.dirname(filename)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    assert len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1

    input_tensor = input_tensor.cpu()
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)
