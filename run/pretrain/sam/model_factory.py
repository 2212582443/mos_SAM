from mos.models.sam.configuration_sam import (
    SamVisionConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
    SamConfig,
)
from transformers import (
    SamModel as SamModelPretrain,
)

import os

import torch
import torch.nn as nn
from mos.models.sam.modeling_sam.sam_model import SamModel

from mos.models.sam.modeling_sam.sam_vision_encoder import SamVisionEncoder
import loralib as lora

from mos.utils.files import relative_symlink_file


class ModelFactory(object):
    def __init__(
        self,
        device: str,
        run_name: str,
        pretrain_model="facebook/sam-vit-base",
        model_cache_dir="./.cache/models/sam_model",
        model_latest_dir="./.checkpoint/{run_name}/latest",
    ):
        self.device: torch.device = torch.device(device)
        self.run_name = run_name
        self.pretrain_model = pretrain_model
        self.model_cache_dir = model_cache_dir.format(run_name=run_name)
        self.model_latest_dir = model_latest_dir.format(run_name=run_name)
        if not os.path.exists(self.model_cache_dir):
            os.makedirs(self.model_cache_dir, exist_ok=True)

    def create_model(self) -> SamModel:
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
        mask_decoder_config = SamMaskDecoderConfig()
        config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)
        model = SamModel(config)
        return model

    def init_model_from_pretrain(self):
        print("init model...")

        MAE_PRETRAIN = False
        if MAE_PRETRAIN:
            mae_model_dir = "./.cache/models/mae_model"
            os.makedirs(mae_model_dir, exist_ok=True)
            if not os.path.exists(f"{mae_model_dir}/mae_pretrain_vit_base.pth"):
                os.system(
                    f"aria2c -x 16 -c https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -d {mae_model_dir}"
                )
            os.makedirs(self.model_cache_dir, exist_ok=True)
            if not os.path.exists(f"{self.model_cache_dir}/mae_pretrain_vit_base.pth"):
                relative_symlink_file(
                    f"{mae_model_dir}/mae_pretrain_vit_base.pth", f"{self.model_cache_dir}/mae_pretrain_vit_base.pth"
                )

        pretrained_model = SamModelPretrain.from_pretrained(self.pretrain_model)

        model = self.create_model().to(self.device)

        # copy pretrain model weights
        print("visiton_encoder_state_dice:")
        if False:
            visiton_encoder_state_dict = model.prompt.embed_image.vision_encoder.state_dict()
            for var_name in pretrained_model.vision_encoder.state_dict():
                src = pretrained_model.vision_encoder.state_dict()[var_name]
                print("\tparameter name:", var_name, src.shape)
                visiton_encoder_state_dict[var_name].copy_(src)
        else:
            mae_model = torch.load(f"{mae_model_dir}/mae_pretrain_vit_base.pth", map_location="cpu")
            model.prompt.embed_image.vision_encoder.load_state_dict(mae_model, strict=False)

        position_embedding_state_dict = model.prompt.positional_embedding.state_dict()
        print("position_embedding_state_dice:")
        for var_name in pretrained_model.shared_image_embedding.state_dict():
            src = pretrained_model.shared_image_embedding.state_dict()[var_name]
            print("\tparameter name:", var_name, src.shape)
            position_embedding_state_dict[var_name].copy_(src)

        print("save model...")
        self.save_model(model, self.model_cache_dir)

    def load_model(self, model_path=None) -> SamModel:
        model = self.create_model()
        if model_path is None:
            model_path = self.model_latest_dir
        if not os.path.exists(model_path):
            print("load model from cache dir: ${self.model_cache_dir}")
            model_path = self.model_cache_dir
        print("load model...", model_path)

        if os.path.exists(f"{model_path}/model.pt"):
            print("load model.pt")
            model_state = torch.load(f"{model_path}/model.pt", map_location="cpu")
            for name in model_state:
                print(f"\t{name}: {model_state[name].shape}")
            model.load_state_dict(model_state, strict=False)

        if os.path.exists(f"{model_path}/vision_encoder.pt"):
            print("load vision_encoder.pt")
            image_state = torch.load(f"{model_path}/vision_encoder.pt", map_location="cpu")
            for name in image_state:
                print(f"\t{name}: {image_state[name].shape}")
            model.prompt.embed_image.vision_encoder.load_state_dict(image_state, strict=False)

        if os.path.exists(f"{model_path}/lora.pt"):
            print("load lora.pt")
            lora_state = torch.load(f"{model_path}/lora.pt", map_location="cpu")
            for name in lora_state:
                print(f"\t{name}: {lora_state[name].shape}")
            model.load_state_dict(lora_state, strict=False)

        # lora.mark_only_lora_as_trainable(model.prompt.embed_image.vision_encoder)
        # load lora

        # model.prompt = model.module.prompt

        model = model.to(self.device)
        # model = torch.compile(model)

        # model = nn.DataParallel(model)

        return model

    def save_model(self, model, target_dir: str | None = None):
        if target_dir is None:
            target_dir = self.model_latest_dir
        else:
            target_dir = target_dir.format(run_name=self.run_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        # image_embed = {}
        model_embed = {}
        lora_embed = lora.lora_state_dict(model)
        for name, param in model.named_parameters():
            if name in lora_embed:
                continue
            # elif name.startswith("prompt.embed_image.vision_encoder."):
            #     name = name.replace("prompt.embed_image.vision_encoder.", "")
            #     image_embed[name] = param
            else:
                model_embed[name] = param

        # torch.save(image_embed, f"{target_dir}/vision_encoder.pt")
        torch.save(model_embed, f"{target_dir}/model.pt")
        if len(lora_embed) > 0:
            torch.save(lora_embed, f"{target_dir}/lora.pt")

        if target_dir != self.model_latest_dir:
            relative_symlink_file(target_dir, self.model_latest_dir)
