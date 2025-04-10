import torch
from torch import Tensor


def distance_single(a: Tensor, b: Tensor):
    """
    Compute distance between each pair of the two collections of inputs.
    注意, 不支持批量计算
    Args:
        a: (N, d), d为[x,y,z]的坐标
        b: (M, d)
    Returns:
        (N, M, d) matrix
        where dist[i,j] is the norm between a[i,:] and b[j,:],
          i.e. dist[i,j] = ||a[i,:]-b[j,:]||

    """
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    diff = torch.sum(diff**2, dim=-1).sqrt()
    return diff


def hausdorff_distance(x_mask: Tensor, y_mask: Tensor, reduce=True) -> tuple[Tensor, Tensor]:
    # Args:
    #     x_mask: (N, H, W) or (N, D, H, W)
    #     y_mask: (M, H, W) or (M, D, H, W)
    # Returns:
    #       (avg hausdorff distance, max hausdorff distance) of each mask
    #       shape: (bs)

    x_mask, y_mask = x_mask >= 0.5, y_mask >= 0.5
    avg_result, max_result = [], []
    for x, y in zip(x_mask.split(1, 0), y_mask.split(1, 0)):
        x, y = torch.nonzero(x.squeeze(0)), torch.nonzero(y.squeeze(0))

        # mask至少只少要有一个点
        if x.shape[0] == 0 or y.shape[0] == 0:
            avg_result.append(torch.zeros(1).to(x.device))
            max_result.append(torch.tensor(0.0, device=x.device))
            continue
        # 如果其中一个点集不存在, 那么就对应到0点好了
        if len(x.shape) == 0 or x.shape[0] == 0:
            shape = y.shape
            shape[0] = 1
            x = torch.zeros(shape).to(y.device)
        if len(y.shape) == 0 or y.shape[0] == 0:
            shape = x.shape
            shape[0] = 1
            y = torch.zeros(shape).to(x.device)

        diff = distance_single(x, y)
        if len(diff.shape) == 0 or diff.shape[1] == 0:
            avg_result.append(torch.zeros(1).to(x.device))
            max_result.append(torch.tensor(0.0, device=x.device))
            continue

        min_x, min_y = torch.min(diff, dim=1).values, torch.min(diff, dim=0).values

        avg_x, avg_y = torch.mean(min_x, 0, keepdim=True), torch.mean(min_y, 0, keepdim=True)
        dist = (avg_x + avg_y) / 2
        avg_result.append(dist)

        max_x, max_y = torch.max(min_x, dim=0).values, torch.max(min_y, dim=0).values
        dist = max_x if max_x > max_y else max_y
        max_result.append(dist)

    avg_result = torch.cat(avg_result).to(x_mask.device)
    max_result = torch.stack(max_result).to(x_mask.device)

    assert len(avg_result.shape) == 1, f"len({avg_result.shape}) != 1"
    assert len(max_result.shape) == 1, f"len({max_result.shape}) != 1"
    assert avg_result.shape == max_result.shape, f"{avg_result.shape} != {max_result.shape}"

    if reduce:
        avg_result = torch.mean(avg_result)
        max_result = torch.mean(max_result)

    return avg_result, max_result


def avg_hausdorff_distance(x_mask: Tensor, y_mask: Tensor, reduce=True):
    """
    计算两个mask的平均hausdorff距离
    Args:
        x_mask: (N, H, W) or (N, D, H, W)
        y_mask: (M, H, W) or (M, D, H, W)
    """

    avg_result, _max_result = avg_hausdorff_distance(x_mask, y_mask, reduce=False)
    return avg_result
