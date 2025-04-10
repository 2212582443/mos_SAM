import torch.nn as nn
import torch, os
from mos.models.sam.configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
from mos.models.sam.modeling_sam.embedding.typing import ImageEmbeddingTensor

from mos.models.txtsam.txtsam import TxtSam
from run.baseline.model_arguments import ModelArguments
from run.baseline.result_wrapper import SeparetionMaskResult, parse_custom_args

from ..dataset import DataItem
from dataclasses import dataclass, field
from .label_utils import get_config_train_labels, get_config_eval_labels


@dataclass
class TxtSamArguments(object):
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    load_pretrained: bool = field(
        default=False,
        metadata={"help": ("加载预训练数据")},
    )

    freeze_adapter: bool = field(
        default=False,
        metadata={"help": ("冻结adapter")},
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": ("冻结vit")},
    )
    adapter_rank: int = field(
        default=64,
        metadata={"help": ("adapter大小")},
    )


def create_txtsam(device, image_size: tuple[int, int], adapter_rank_3d=0) -> TxtSam:
    vision_config = SamVisionConfig(
        image_size=image_size,
        patch_size=8,
        num_channels=1,
    )
    prompt_encoder_config = SamPromptEncoderConfig(
        image_size=image_size,
        patch_size=8,
    )
    mask_decoder_config = SamMaskDecoderConfig(num_multimask_outputs=0)
    config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)
    model = TxtSam(config, adapter_rank_3d)
    model = model.to(device)

    return model


class TxtSamWrapper3d(nn.Module):
    def __init__(
        self,
        device,
        model_args: ModelArguments,
        args: list[str] = None,
        token_file=".cache/dataset/baseline/token.pt",
        is_3d=True,
    ) -> None:
        super().__init__()

        if len(args) > 0:
            self.txtsam_args: TxtSamArguments = parse_custom_args(args, TxtSamArguments)[0]
        else:
            self.txtsam_args = TxtSamArguments()
        if not is_3d:
            self.txtsam_args.adapter_rank = 0

        self.label_count = model_args.label_count

        self.crop_size = model_args.get_dataset_crop_size()
        self.device = device
        self.model: TxtSam = create_txtsam(
            device,
            # FIXME: 支持不同比例的输入
            self.crop_size[-1],
            self.txtsam_args.adapter_rank if is_3d else 0,
        )
        data = torch.load(token_file)
        self.token: torch.Tensor = data["token"].to(device)
        self.token_text = data["text"]
        self.aux_dataset_labels = model_args.get_aux_dataset_labels()
        self.model_args = model_args

        target_label_id_list_cuda, target_label_id_list = get_config_train_labels(model_args, device)
        self.train_target_label_id_list_cuda = target_label_id_list_cuda
        self.train_target_label_id_list = target_label_id_list

        target_label_id_list_cuda, target_label_id_list = get_config_eval_labels(model_args, device)
        self.eval_target_label_id_list_cuda = target_label_id_list_cuda
        self.eval_target_label_id_list = target_label_id_list

        if self.txtsam_args.load_pretrained:
            self.init_from_pretrain(".cache/dataset/baseline/")

    def on_model_loaded(self):
        if self.txtsam_args.freeze_vit:
            print("注意：冻结vit")
            for p in self.model.prompt.embed_image.vision_encoder.vit.parameters():
                p.requires_grad = False
            if self.txtsam_args.adapter_rank > 0 and not self.txtsam_args.freeze_adapter:
                self.model.prompt.embed_image.vision_encoder.vit.freeze_adapter(False)

        if self.txtsam_args.adapter_rank > 0 and self.txtsam_args.freeze_adapter:
            self.model.prompt.embed_image.vision_encoder.vit.freeze_adapter(True)

        print("trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("\t", name, param.shape)

    def init_from_pretrain(self, base_path):
        pretrained_file = f"{base_path}/mae-model.pt"
        if not os.path.exists(pretrained_file):
            raise Exception(f"pretrain model:{pretrained_file} not found")

        print("load pretrained model:", pretrained_file)

        mae_state_dict = dict(
            torch.load(
                pretrained_file,
                map_location=self.device,
            )
        )
        state = {}
        for k, v in mae_state_dict.items():
            if k.startswith("vit."):
                k = k[4:]
                state[k] = v
        self.model.prompt.embed_image.vision_encoder.vit.load_state_dict(state, strict=False)

    def forward(self, data_item: DataItem):
        return self._forward3d(data_item)

    def forward_valid(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True)

    def forward_test(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True, True)

    def forward_zeroshot(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True, True)

    def forward_pseudo(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True)

    def _forward3d(self, data_item: DataItem, eval=False, all_label=False):
        image, label, db, depth_info = data_item.image, data_item.segment, data_item.dataset, data_item.depth_info

        # image: (bs, 1, d, h, w)
        # label: (bs, d, h, w)
        # 因为每次只能分割一个类别
        # 需要转换为2d的onehot,结果再拼接为3d

        bs, d, h, w = label.shape

        result_logit = [None] * self.label_count
        result_mask = [None] * self.label_count

        index = 0 if all_label else db.value
        target_label_ids_cuda = (
            self.eval_target_label_id_list_cuda[index] if eval else self.train_target_label_id_list_cuda[index]
        )
        target_label_ids = self.eval_target_label_id_list[index] if eval else self.train_target_label_id_list[index]
        label_count = target_label_ids_cuda.shape[0]

        # 只计算有label的
        label_only: torch.Tensor = (
            (label.reshape(bs * d, h, w).sum((1, 2)) > 0)
            .bitwise_and((image.reshape(bs * d, h, w).sum((1, 2)) > 0))
            .unsqueeze(1)
            .unsqueeze(1)
            .float()
        )
        if label_count > 1:
            label_only = label_only.repeat(label_count, 1, 1)

        # 训练时随机进行位置偏移，以学习到相对位置信息
        if torch.is_grad_enabled() and depth_info is not None:
            random_offset = torch.randint(0, 8, (bs, 1), device=self.device).float()
            random_offset = random_offset.repeat(1, d)
            # (bs, d)
            depth_info = depth_info + random_offset

        image = image.reshape(bs * d, 1, h, w)  # (bs * d , 1, h, w)
        # (bs*d, 256, 16, 16)
        image_embeddings: ImageEmbeddingTensor = self.model.prompt.embed_image(image, depth_info)
        # print("image_embeddings:", image_embeddings.shape)
        if label_count > 1:
            # (labelcount*bs*d, 256, 16, 16)
            image_embeddings = image_embeddings.repeat(label_count, 1, 1, 1)

        # (labelcount, 40, 768)
        token = self.token.index_select(0, target_label_ids_cuda)
        # (labelcount, 1, 256)
        sparse_embeddinsg = self.model.prompt(text_token=token)
        tn, th = sparse_embeddinsg.shape[-2:]
        # (labelcount, bs*d*tn, 256)
        sparse_embeddinsg = sparse_embeddinsg.repeat(1, bs * d, 1).reshape(label_count * bs * d, tn, th)
        logit = self.model(image_embeddings, sparse_embeddinsg)

        logit = logit * label_only

        logit = logit.reshape(label_count * bs, d, h, w)  # bs, d, h, w
        logit = logit.sigmoid()
        mask = logit > 0.5

        for k, logit, mask in zip(target_label_ids, logit.split(bs, 0), mask.split(bs, 0)):
            result_logit[k] = logit
            result_mask[k] = mask

        return SeparetionMaskResult(result_logit, result_mask)


class TxtSamWrapper2d(TxtSamWrapper3d):
    def __init__(
        self,
        device,
        model_args: ModelArguments,
        args: list[str] = None,
        token_file=".cache/dataset/baseline/token.pt",
    ) -> None:
        super().__init__(device, model_args, args, token_file, False)

    def forward(self, data_item: DataItem):
        image, label, db = data_item.image, data_item.segment, data_item.dataset

        # image: (bs, 1, h, w)
        # label: (bs, h, w)
        # 因为每次只能分割一个类别
        # 需要转换为2d的onehot,结果再拼接为3d

        bs, h, w = label.shape

        result_logit = [None] * self.label_count
        result_mask = [None] * self.label_count

        target_label_ids_cuda = self.train_target_label_id_list_cuda[db.value]
        target_label_ids = self.train_target_label_id_list[db.value]
        label_count = len(target_label_ids)

        # 只计算有label的
        label_only = (label.sum((1, 2)) > 0).unsqueeze(1).unsqueeze(1).float()
        if label_count > 1:
            label_only = label_only.repeat(label_count, 1, 1)

        image_embeddings = self.model.prompt.embed_image(image)
        if label_count > 1:
            image_embeddings = image_embeddings.repeat(label_count, 1, 1, 1)

        token = self.token.index_select(0, target_label_ids_cuda)
        sparse_embeddinsg = self.model.prompt(text_token=token)
        tn, th = sparse_embeddinsg.shape[-2:]
        sparse_embeddinsg = sparse_embeddinsg.repeat(1, bs, 1).reshape(label_count * bs, tn, th)

        logit = self.model(image_embeddings, sparse_embeddinsg)
        logit = logit * label_only
        logit = logit.sigmoid()
        mask = logit > 0.5

        for k, logit, mask in zip(target_label_ids, logit.split(bs, 0), mask.split(bs, 0)):
            result_logit[k] = logit
            result_mask[k] = mask

        return SeparetionMaskResult(result_logit, result_mask)

    def forward_valid(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True)

    def forward_test(self, data_item: DataItem):
        # 3d 输入，转换为2d进入模型，结果再拼接为3d
        return self._forward3d(data_item, True, True)

    def forward_zeroshot(self, data_item: DataItem):
        return self._forward3d(data_item, True, True)

    def forward_pseudo(self, data_item: DataItem):
        return self._forward3d(data_item, True)
