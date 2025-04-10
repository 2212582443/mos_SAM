from typing import Any
from mos.models.sam.configuration_sam import (
    SamVisionConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
    SamConfig,
)

import os

import torch

from torch.optim import Optimizer


from mos.utils.files import relative_symlink_file
from .model.sam_model_simple import SamModelSimple


class ModelFactory(object):
    def __init__(
        self,
        device: str,
        run_name: str,
    ):
        self.device: torch.device = torch.device(device)
        self.run_name = run_name

    def create_model(self) -> SamModelSimple:
        image_size = 128
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
        model = SamModelSimple(config)
        model = model.to(self.device)
        return model

    def init_model_from_mae(self, model: SamModelSimple, mae_state_dict: dict[str, Any]):
        state = {}
        for k, v in mae_state_dict.items():
            if k.startswith("vit."):
                k = k[4:]
                state[k] = v
        model.prompt.embed_image.vision_encoder.vit.load_state_dict(state)

    def freeze_mae_encoder(self, model: SamModelSimple, freeze: bool):
        model.prompt.embed_image.vision_encoder.vit.requires_grad_(not freeze)

    def load_optimizer(self, optimizer: Optimizer):
        if optimizer is None:
            return
        if not os.path.exists(f"{self.run_name}/optimizer.pt"):
            return
        optimizer.load_state_dict(torch.load(f"{self.run_name}/optimizer.pt"))

    def load_model(self, model_path) -> SamModelSimple:
        model = self.create_model()

        file = f"{model_path}/model.pt"
        if os.path.exists(file):
            print(f"loading model: {file}")
            model.load_state_dict(torch.load(file, map_location=self.device))

        return model

    def save_model(self, model: SamModelSimple, optimizer: Optimizer, target_dir: str | None = None):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        torch.save(model.state_dict(), f"{target_dir}/model.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), f"{target_dir}/optimizer.pt")

        parent_path = os.path.dirname(target_dir)
        model_latest_dir = f"{parent_path}/latest"

        relative_symlink_file(target_dir, model_latest_dir)
