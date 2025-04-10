import unittest
import monai.metrics as monaimetrics
import torch
from torch.nn import functional as F
from .hausdorff import hausdorff_distance


class HausforffTest(unittest.TestCase):
    def test_hausdorff_distance_with_monai(self):
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
        d1 = monaimetrics.compute_hausdorff_distance(
            x_onehot,
            y_onehot,
        )

        avg_hd, max_hd = hausdorff_distance(
            x_pred,
            y_label,
        )
        print(x_onehot.shape, y_onehot.shape)
        print("hd:", d1.shape, d1)
        print("-------")
        print(avg_hd, max_hd)

        assert max_hd == d1[0].item()
