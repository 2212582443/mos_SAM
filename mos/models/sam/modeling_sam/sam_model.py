from dataclasses import dataclass

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ..configuration_sam import SamConfig
from .embedding import SamPromptEncoder
from .embedding.typing import (
    SparseEmbeddingsTensor,
    DenseEmbeddingsTensor,
    PredIouScoresTensor,
    PredMasksTensor,
    ImageEmbeddingTensor,
)
from .sam_mask_decoder import SamMaskDecoder


class SamPreTrainedModel(PreTrainedModel):
    config_class = SamConfig
    base_model_prefix = "sam"
    main_input_name = "image_embeddings"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class SamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`PredIouScoresTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`PredMasksTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
    """

    iou_scores: PredIouScoresTensor = None
    pred_masks: PredMasksTensor = None


class SamModel(SamPreTrainedModel):
    """Segment Anything Model (SAM) for generating segmentation masks, given an input image and optional 2D location and bounding boxes.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SamConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    _keys_to_ignore_on_load_missing = [r"prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config: SamConfig):
        super().__init__(config)
        self.mask_decoder = SamMaskDecoder(config.mask_decoder_config)
        self.prompt = SamPromptEncoder(config.prompt_encoder_config, config.vision_config)

        self.post_init()

    def forward(
        self,
        image_embeddings: ImageEmbeddingTensor,
        sparse_embeddings: SparseEmbeddingsTensor | None = None,
        dense_embeddings: DenseEmbeddingsTensor | None = None,
        multimask_output: bool = True,
    ) -> tuple[PredIouScoresTensor, PredMasksTensor]:
        """
        Args:
            image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
                Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
                efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
                method, and then feed them to the `forward` method instead of feeding the `pixel_values`.

            sparse_embeddings (`torch.FloatTensor` of shape `(batch_size, num_points, hidden_size)`):

            dense_embeddings (`torch.FloatTensor` of shape `(batch_size, hidden_size, window_size, window_size)`):

            multimask_output (`bool`, *optional*):
                In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
                bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
                "best" mask, by specifying `multimask_output=False`.

        """

        # (1, 256, 64, 64)
        image_positional_embeddings = self.prompt.get_image_wide_positional_embeddings()
        # repeat with batch size

        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        low_res_masks = torch.sigmoid(low_res_masks.float())

        return (
            iou_predictions,
            low_res_masks,
        )
