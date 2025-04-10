import torch.nn as nn
import torch

from run.baseline.result_wrapper import MultiChanelResult


from ..dataset import DATASET, DataItem


class ModelWrapper3d(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model: nn.Module = model

    def forward(self, data_item: DataItem):
        image, label, db = data_item.image, data_item.segment, data_item.dataset

        logit, after_softmax = self.model(image)

        # 只计算有label的
        label_only = (label.sum((2, 3)) > 0).unsqueeze(1).unsqueeze(3).unsqueeze(3).float()
        # 对于置0的，bg channel需要设置一定的值，给于补偿
        compensation = torch.zeros_like(logit)
        compensation[:, 0:1, :, :, :] = 1 - label_only

        logit = logit * label_only + compensation
        after_softmax = after_softmax * label_only + compensation

        return MultiChanelResult(after_softmax, logit)

    def forward_valid(self, data_item: DataItem):
        image, label = data_item.image, data_item.segment
        logit, after_softmax = self.model(image)

        # 只计算有label的
        label_only = (label.sum((2, 3)) > 0).unsqueeze(1).unsqueeze(3).unsqueeze(3).float()
        # 对于置0的，bg channel需要设置一定的值，给于补偿
        compensation = torch.zeros_like(logit)
        compensation[:, 0:1, :, :, :] = 1 - label_only

        logit = logit * label_only + compensation
        after_softmax = after_softmax * label_only + compensation

        return MultiChanelResult(after_softmax, logit)

    def forward_test(self, data_item: DataItem):
        return self.forward_valid(data_item)

    def forward_zeroshot(self, data_item: DataItem):
        return self.forward_valid(data_item)

    def forward_pseudo(self, data_item: DataItem):
        return self.forward_valid(data_item)


class ModelWrapper2d(nn.Module):
    def __init__(self, model, label_count: int):
        super().__init__()
        self.model: nn.Module = model
        self.label_count = label_count

    def forward(self, data_item: DataItem):
        image, label = data_item.image, data_item.segment

        # 训练使用2d，推理时使用3d，以公平的方式比较
        logit, after_softmax = self.model(image)

        # 只计算有label的
        label_only = (label.sum((1, 2)) > 0).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        # 对于置0的，bg channel需要设置一定的值，给于补偿
        compensation = torch.zeros_like(logit)
        compensation[:, 0:1, :, :] = 1 - label_only

        logit = logit * label_only + compensation
        after_softmax = after_softmax * label_only + compensation

        return MultiChanelResult(after_softmax, logit)

    def forward_valid(self, data_item: DataItem):
        image, label = data_item.image, data_item.segment

        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        # image: (bs, 1, d, h ,w)
        bs, d, h, w = label.shape
        image = image.reshape(bs * d, 1, h, w)
        logit, after_softmax = self.model(image)

        # 只计算有label的
        label_only = (label.reshape(bs * d, h, w).sum((1, 2)) > 0).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        # 对于置0的，bg channel需要设置一定的值，给于补偿
        compensation = torch.zeros_like(logit)
        compensation[:, 0:1, :, :] = 1 - label_only

        logit = logit * label_only + compensation
        after_softmax = after_softmax * label_only + compensation

        logit = logit.reshape(bs, d, self.label_count, h, w).permute(0, 2, 1, 3, 4)
        after_softmax = after_softmax.reshape(bs, d, self.label_count, h, w).permute(0, 2, 1, 3, 4)
        return MultiChanelResult(after_softmax, logit)

    def forward_test(self, data_item: DataItem):
        return self.forward_valid(data_item)

    def forward_zeroshot(self, data_item: DataItem):
        return self.forward_valid(data_item)

    def forward_pseudo(self, data_item: DataItem):
        return self.forward_valid(data_item)
