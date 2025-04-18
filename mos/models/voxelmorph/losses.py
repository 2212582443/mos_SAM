import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_true)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = 1
            padding = pad_no
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class NCC2d:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win: tuple[int, int] = None):
        self.win = win
        self.element_count = win[0] * win[1]

    def loss(self, y_true, y_pred):
        x, y = y_true, y_pred
        win_size = self.element_count
        win = self.win

        x2 = x * x
        y2 = y * y
        xy = x * y

        ch = x.shape[1]
        sum_filt = torch.ones([1, ch, *win]).to(y_true).float()
        stride = (1, 1)
        padding = (win[0] // 2, win[1] // 2)

        sum_x = F.conv2d(x, sum_filt, stride=stride, padding=padding)
        sum_y = F.conv2d(y, sum_filt, stride=stride, padding=padding)
        sum_x2 = F.conv2d(x2, sum_filt, stride=stride, padding=padding)
        sum_y2 = F.conv2d(y2, sum_filt, stride=stride, padding=padding)
        sum_xy = F.conv2d(xy, sum_filt, stride=stride, padding=padding)

        cov_xy = sum_xy - sum_x * sum_y / win_size
        var_x = sum_x2 - sum_x * sum_x / win_size
        var_y = sum_y2 - sum_y * sum_y / win_size

        # cov_xy, var_x, var_y = cov_xy.float(), var_x.float(), var_y.float()

        # nvidia 半精度浮点数的范围是 6.10x10^-5 ~ 6.55x10^4
        # 为了不溢出, 分母加上一个很小的数至少为 1.6x10^-5
        cc = cov_xy * cov_xy / (var_x * var_y + 1.6e-5)

        return 1 - torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l1", loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == "l1":
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == "l2", "penalty can only be l1 or l2. Got: %s" % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
