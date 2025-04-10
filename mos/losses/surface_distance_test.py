import unittest
import monai.metrics as monaimetrics
import torch
from torch.nn import functional as F


class SurfaceDistanceTest(unittest.TestCase):
    def test_mean_surface_distance(self):
        x_pred = torch.zeros(3, 16, 128, 128)
        y_label = torch.zeros(3, 16, 128, 128)

        x_pred[:, 2, 10:11, 1:2] = 1
        y_label[:, 2, 11:12, 2:3] = 1

        # (bs, d,h,w,c)
        x_onehot = F.one_hot(x_pred.long(), num_classes=2).float()
        y_onehot = F.one_hot(y_label.long(), num_classes=2).float()

        # 注意本函数的chanel 放在第二个维度
        x_onehot = x_onehot.permute(0, 4, 1, 2, 3)
        y_onehot = y_onehot.permute(0, 4, 1, 2, 3)

        d1 = monaimetrics.compute_average_surface_distance(
            x_onehot,
            y_onehot,
        )

        print(d1)
