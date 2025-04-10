from typing import Set, Dict
from .files import get_file_ext
import torch
import SimpleITK as sitk
import numpy as np


def load_tensor_file(file: str) -> Dict[str, torch.Tensor]:
    """加载tensor文件"""

    ext = get_file_ext(file)

    match ext:
        case ".npy":
            data = np.load(file)
            data = torch.from_numpy(data)
            return {"": data}
        case ".npz":
            data = np.load(file)
            data = {k: torch.from_numpy(data[k]) for k in data}
            return data
        case ".pt":
            data = torch.load(file)
            return dict(data)
        case ".ot":
            data = torch.load(file)
            return data
        case ".nii" | ".nii.gz" | ".dcm":
            np_data = sitk.GetArrayFromImage(sitk.ReadImage(file))
            if np_data.dtype == np.uint16:
                np_data = np_data.astype(np.int32)
            data = torch.from_numpy(np_data)
            return {"": data}
        case _:
            raise ValueError(f"unsupported file type {ext}")


def save_tensor_file(data: Dict[str, torch.Tensor] | torch.Tensor, file: str):
    """保存tensor文件"""

    ext = get_file_ext(file)

    match ext:
        case ".npy":
            np.save(file, data)
        case ".npz":
            np.savez_compressed(file, **data)
        case ".pt":
            torch.save(data, file)
        case ".nii" | ".nii.gz" | ".dcm":
            np_data = data.numpy()
            sitk.WriteImage(sitk.GetImageFromArray(np_data), file)
        case _:
            raise ValueError(f"unsupported file type {ext}")


def generate_grid_2d(h: int, w: int, space: int):
    grid = torch.ones(1, 1, h, w)
    h_index = torch.tensor(range(0, h, space))
    grid = grid.index_put_(indices=[None, None, h_index, None], values=0)
    w_index = torch.tensor(range(0, w, space))
    grid = grid.index_put_(indices=[None, None, None, w_index], values=0)
    return grid
