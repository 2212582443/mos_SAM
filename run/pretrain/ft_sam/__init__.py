# fine tune encoder
import os
import torch
from mos.datasets.acdc.acdc_dataset2d import AcdcDataset2d
from mos.datasets.cmri.cmri_dataset2d import CmriDataset2d
from mos.datasets.common.mydataset.my_dataset import ItemTransformerContext
from mos.datasets.mnms.mnms_dataset2d import MnmsDataset2d
from mos.models.sam.configuration_sam import SamVisionConfig
from mos.models.sam.modeling_sam.embedding.image_embedding_sam import ImageEmbeddingSam
from mos.models.sam.modeling_sam.embedding.typing import GrayImageTensor
from torchvision import transforms
from torch import nn

_image_transform = transforms.Compose(
    [
        transforms.Resize((256, 256), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR),
    ]
)


def new_item_tranformer(device, _map):
    # 坐标是基于256*256的图像的, 需要缩放到1024*1024的图像上
    scaling = 4

    def transformer(_context: ItemTransformerContext, item):
        # (ch, h, w)
        image = item["image"]
        # (bs, ch, h, w)
        image = image.to(device)
        image: GrayImageTensor = image.unsqueeze(0)
        image = _image_transform(image)
        image = image * 256
        image = image.to(torch.int8)

        return image

    return transformer


class FtSamModel(nn.Module):
    def __init__(self, vision_config: SamVisionConfig) -> None:
        self.embed_image = ImageEmbeddingSam(vision_config)

    def forward(self, image):
        return self.embed_image(image)


class FtSamModelFactory:
    def __init__(
        self,
        device: str,
        run_name: str,
        pretrain_model="facebook/sam-vit-base",
        model_cache_file="./.cache/models/sam_model.pt",
        model_latest_file="./.cache/models/{run_name}/sam_model-{part}-latest.pt",
    ):
        self.device: torch.device = torch.device(device)
        self.run_name = run_name
        self.pretrain_model = pretrain_model
        self.model_cache_file = model_cache_file.format(run_name=run_name)
        self.model_latest_file = model_latest_file.format(run_name=run_name)
        if not os.path.exists(self.model_cache_file):
            os.makedirs(os.path.dirname(self.model_cache_file), exist_ok=True)

    def init_pretrain(self):
        if not os.path.exists(self.model_latest_file):
            state = torch.load(self.model_latest_file, map_location="cpu")
            #  fileter out the embed_image
            state = {
                k[len("prompt.embed_image.") :]: v for k, v in state.items() if not k.startswith("prompt.embed_image.")
            }
            file_name = self.model_latest_file.format(run_name=self.run_name, part="encoder")
            torch.save(state, file_name)

    def create_model(self) -> FtSamModel:
        return FtSamModel(SamVisionConfig())

    def load_model(self, model_path=None) -> ImageEmbeddingSam:
        self.init_pretrain()
        model = self.create_model()
        if model_path is None:
            model_path = self.model_latest_file
        if not os.path.exists(model_path):
            model_path = self.model_cache_file
        print("load model...", model_path)
        image_embedding_state = torch.load(
            model_path.format(run_name=self.run_name, part="encoder"), map_location="cpu"
        )
        model.embed_image.load_state_dict(image_embedding_state)
        return model.to(self.device)


def run(_args):
    device = "cuda:0"
    acdc_segment_mapping = [0, 2, 3, 4]
    mnms_segment_mapping = [0, 2, 3, 4]
    cmri_segment_mapping = [0, 1]

    datasets = [
        ("cmri", CmriDataset2d("cmri", item_transformer=new_item_tranformer(device, cmri_segment_mapping))),
        ("mnms", MnmsDataset2d("mnms", item_transformer=new_item_tranformer(device, mnms_segment_mapping))),
        ("acdc", AcdcDataset2d("acdc", item_transformer=new_item_tranformer(device, acdc_segment_mapping))),
    ]

    item_infos = []
    image_config = SamVisionConfig()
    vit = ImageEmbeddingSam(image_config)
