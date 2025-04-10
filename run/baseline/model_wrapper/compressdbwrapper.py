from dataclasses import dataclass, field
from typing import Callable
import torch.nn as nn
import torch
import torch.nn.functional as F

from mos.models.vae import VanillaVAE
from run.baseline.metric import Metric
from run.baseline.result_wrapper import ModelResult, MultiChanelResult, parse_custom_args


from ..dataset import DATASET, DataItem


@dataclass
class CompressDbWrapperArguments(object):
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    m_in_channels: int = field(
        default=2,
        metadata={"help": ("数据集输出通道个数")},
    )

    m_latent_dim: int = field(
        default=128,
        metadata={"help": ("特征向量的维度")},
    )

    m_hidden_dims: str = field(
        default="16,32,64,128,256",
        metadata={"help": ("用于编码/解码的向量维度")},
    )
    m_shape: str = field(
        default="320,320",
        metadata={"help": ("输入图片的大小")},
    )
    m_use_bin_channel: bool = field(
        default=False,
        metadata={"help": ("是否使用二进制通道")},
    )

    def get_hidden_dims(self):
        return [int(x) for x in self.m_hidden_dims.split(",")]

    def get_shape(self):
        return [int(x) for x in self.m_shape.split(",")]


class BinChannelParser:
    def get_out_channel_count(n: int):
        count = 1
        value = 1
        while value < n:
            value <<= 1
            value |= 1
            count += 1
        return count

    def dec2bin(self, x):
        shape = list(x.shape)
        x = x.view(-1).long()
        x = self.dec2bin_mapper.index_select(0, x)
        x = x.reshape(*shape, self.bits)

        # move bits to the channel
        shape = list(range(len(shape)))
        shape.insert(1, len(shape))
        x = x.permute(*shape)

        return x

    def bin2dec(self, x, keepdim=False):
        x = (x > 0.5).int()
        return torch.sum(self.bin2dec_mask * x, 1, keepdim=keepdim)

    def __init__(self, channel_count: int, device) -> None:
        self.device = device
        self.bits = BinChannelParser.get_out_channel_count(channel_count)

        channel_count = 2**self.bits
        x = torch.arange(channel_count, device=device).unsqueeze(-1)  # (channel_count, 1)
        mask = 2 ** torch.arange(self.bits).to(x.device, x.dtype).unsqueeze(0)  # (1, bits)
        self.dec2bin_mapper = x.bitwise_and(mask).ne(0).float()  # (channel_count, bits)

        # (1, bits, 1, 1)
        self.bin2dec_mask = mask.unsqueeze(-1).unsqueeze(-1).int()


class CompressDbModelBinChannelResult(MultiChanelResult):
    def __init__(
        self,
        pred_softmax,
        channel_parser: BinChannelParser,
        logit=None,
        aux_data=None,
        loss_fn: Callable[[ModelResult, DataItem, list[int], int], torch.Tensor] = None,
    ):
        super().__init__(pred_softmax, logit, aux_data, loss_fn)
        self.channel_parser = channel_parser

    def calc_loss(self, data_item: DataItem, valid_label_ids: list[int], num_classes) -> torch.Tensor:
        image, label = data_item.image, data_item.segment
        label = self.channel_parser.dec2bin(label)
        target = torch.cat([image, label], 1)

        recons_loss = F.mse_loss(self.pred_softmax, target, reduction="none").mean(list(range(1, target.dim())))
        if True or self.aux_data is None:
            return recons_loss

        mu, log_var = self.aux_data["mu"], self.aux_data["log_var"]
        # https://medium.com/@outerrencedl/variational-autoencoder-and-a-bit-kl-divergence-with-pytorch-ce04fd55d0d7
        # N(mu, sigma^2) ~ N(mu, var)
        # kl = -log sigma + 0.5(sigma^2 +mu^2) - 0.5
        #    = -0.5 (1 + log sigma^2 - sigma^2 - mu^2)
        #    = -0.5 (1 + log var - var - mu^2)
        #    = -0.5 (1 + log_var - ext(log_var) - mu^2)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        kl_weight = 0.7

        return (1 - kl_weight) * recons_loss + kl_weight * kld_loss

    def calc_metric(
        self,
        metric: Metric,
        metric_segment: torch.Tensor,
        data_item: DataItem,
        valid_label_ids: list[int],
        num_classes,
    ) -> torch.Tensor:
        return metric.calc_metric(
            metric_segment.long(),
            data_item.segment,
            num_classes,
            valid_labels=valid_label_ids,
        )

    def merge_incomplete_label(
        self, metric: Metric, db: DATASET, avaliable_label: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        pred_segment_all = self.channel_parser.bin2dec(self.pred_softmax[:, 1:])
        # (bs, h, w) or (bs, d, h, w)
        metric_segment = pred_segment_all

        pred_segment_all_to_be_ploted = metric.color_map.apply(pred_segment_all)
        return metric_segment, pred_segment_all_to_be_ploted


class CompressDbModelMixChannelResult(MultiChanelResult):
    def __init__(
        self,
        pred_softmax,
        logit=None,
        aux_data=None,
        loss_fn: Callable[[ModelResult, DataItem, list[int], int], torch.Tensor] = None,
    ):
        super().__init__(pred_softmax, logit, aux_data, loss_fn)

    def calc_loss(self, data_item: DataItem, valid_label_ids: list[int], num_classes) -> torch.Tensor:
        image, label = data_item.image, data_item.segment

        recons_loss = F.mse_loss(self.pred_softmax[:, :1], image, reduction="none").mean(list(range(1, image.dim())))
        # F.cross_entropy()
        # label_loss = F.nll_loss(torch.log(self.logit[:, 1:].softmax(dim=1) + 1e-4), label, reduction="none").mean(
        #     list(range(1, label.dim()))
        # )
        label_loss = F.cross_entropy(self.logit[:, 1:], label, reduction="none").mean(list(range(1, label.dim())))
        recons_loss = recons_loss + label_loss

        if True or self.aux_data is None:
            return recons_loss

        mu, log_var = self.aux_data["mu"], self.aux_data["log_var"]
        # https://medium.com/@outerrencedl/variational-autoencoder-and-a-bit-kl-divergence-with-pytorch-ce04fd55d0d7
        # N(mu, sigma^2) ~ N(mu, var)
        # kl = -log sigma + 0.5(sigma^2 +mu^2) - 0.5
        #    = -0.5 (1 + log sigma^2 - sigma^2 - mu^2)
        #    = -0.5 (1 + log var - var - mu^2)
        #    = -0.5 (1 + log_var - ext(log_var) - mu^2)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        kl_weight = 0.7

        return (1 - kl_weight) * recons_loss + kl_weight * kld_loss

    def merge_incomplete_label(
        self, metric: Metric, db: DATASET, avaliable_label: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        pred_segment_all = self.pred_softmax[:, 1:].argmax(1)
        # (bs, h, w) or (bs, d, h, w)
        metric_segment = pred_segment_all

        pred_segment_all_to_be_ploted = metric.color_map.apply(pred_segment_all)
        return metric_segment, pred_segment_all_to_be_ploted


# 用于数据集压缩
# 使用VAE学习数据集压缩
class CompressDbWrapper2d(nn.Module):
    def __init__(self, device, label_count: int, args: list[str] = None):
        super().__init__()
        if len(args) > 0:
            self.model_args: CompressDbWrapperArguments = parse_custom_args(args, CompressDbWrapperArguments)[0]
        else:
            self.model_args = CompressDbWrapperArguments()

        self.use_bin_channel = self.model_args.m_use_bin_channel
        if self.use_bin_channel:
            # 因为label数可能比较多，预测通道数使用二进制编码。第一个通道为image
            self.chanel_parser = BinChannelParser(label_count, device)

            # image + segment(bit channel)
            self.out_channel = 1 + BinChannelParser.get_out_channel_count(label_count)
        else:
            self.out_channel = 1 + label_count

        self.label_count = label_count
        self.model: VanillaVAE = VanillaVAE(
            self.model_args.m_in_channels,
            self.out_channel,
            self.model_args.m_latent_dim,
            shape=self.model_args.get_shape(),
            hidden_dims=self.model_args.get_hidden_dims(),
            use_reparameterize=False,
        ).to(device)

    def forward(self, data_item: DataItem):
        image, label = data_item.image, data_item.segment

        # normalize label (0~1)
        label = label / (self.label_count - 1)
        input = torch.cat([image, label.unsqueeze(1)], dim=1)

        logit, mu, log_var = self.model(input)

        after_softmax = logit.sigmoid()

        if self.use_bin_channel:
            return CompressDbModelBinChannelResult(
                after_softmax,
                self.chanel_parser,
                logit,
                aux_data={
                    "mu": mu,
                    "log_var": log_var,
                },
            )
        else:
            return CompressDbModelMixChannelResult(
                after_softmax,
                logit,
                aux_data={
                    "mu": mu,
                    "log_var": log_var,
                },
            )

    def forward_valid(self, data_item: DataItem):
        return self.forward(data_item)

    def forward_test(self, data_item: DataItem):
        return self.forward_valid(data_item)

    def forward_zeroshot(self, data_item: DataItem):
        return self.forward_valid(data_item)

    def forward_pseudo(self, data_item: DataItem):
        return self.forward_valid(data_item)
