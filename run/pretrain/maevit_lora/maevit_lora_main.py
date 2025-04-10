from dataclasses import dataclass
from typing import List, Tuple


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
    ViTMAEConfig,
    ViTMAEForPreTraining,
    AutoImageProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from peft import LoraConfig, get_peft_model
from mos.datasets.cmri.cmri_dataset2d import CmriDataset2d

from mos.utils.model_utils import print_trainable_parameters

logger = logging.getLogger(__name__)

check_min_version("4.28.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt"
)

_IMG_SIZE = 224


@dataclass
class DataTrainingArguments(object):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="cifar10", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    image_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of the images in the files."}
    )
    train_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the training data."}
    )
    validation_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the validation data."}
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        data_files = {}
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments(object):
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    mask_ratio: float = field(
        default=0.75, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    norm_pix_loss: bool = field(
        default=False, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=1e-3, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )


def collate_fn(examples):
    pixel_values = torch.stack(examples)
    return {"pixel_values": pixel_values}


def _migrate_from_pretrained(model):
    dict = model.state_dict()
    # src_image_processor: ViTImageProcessor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    src_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    src_dict = src_model.state_dict()

    # migrate from facebook/vit-mae-base to 1 channel & size 144x144
    for name in src_dict:
        src_state = src_dict[name]
        state = dict[name]

        # vit.embeddings.patch_embeddings.projection.weight torch.Size([768, 3, 16, 16])
        if name == 'vit.embeddings.patch_embeddings.projection.weight':
            src_state = src_state.sum(dim=1, keepdim=True)
        # [1, 197, -1] -> [1, 82, -1]
        if name == 'vit.embeddings.position_embeddings' or name == 'decoder.decoder_pos_embed':
            cls = src_state[:, 0:1, :]
            src_state = src_state[:, 1:, :].reshape(1, 14, 14, -1)
            src_state = src_state[:, 0:9, 0:9, :].reshape(1, 81, -1)
            src_state = torch.cat([cls, src_state], dim=1)

        if name == 'decoder.decoder_pred.weight':  # [768, 512] -> [256, 512]
            # shape (patch_size*patch_size*num_channels, 512)
            src_state = src_state.reshape(1, 16, 16, 3, 512)
            src_state = src_state.mean(dim=3)
            src_state = src_state.reshape(16*16, 512)

        if name == 'decoder.decoder_pred.bias':  # [768] -> [256]
            src_state = src_state.reshape(1, 16, 16, 3)
            src_state = src_state.mean(dim=3)
            src_state = src_state.reshape(-1)

        if state.shape != src_state.shape:
            print('state NOT same', name, state.shape, src_state.shape)
            continue

        dict[name] = src_state

    return model


def run_train(argv: List[str] = sys.argv):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    print('run_train', argv)

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    if len(argv) == 2 and argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )  # type: Tuple[ModelArguments, DataTrainingArguments, CustomTrainingArguments]
    else:
        # type: Tuple[ModelArguments, DataTrainingArguments, CustomTrainingArguments]
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    dataset = CmriDataset2d('cmri')

    # Load pretrained model and image processor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = ViTMAEConfig.from_pretrained(
            model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ViTMAEConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs)
    else:
        config = ViTMAEConfig()
        logger.warning(
            "You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
            "num_channels": 1,
            "image_size": _IMG_SIZE,
        }
    )

    # create image processor
    if model_args.image_processor_name:
        image_processor = ViTImageProcessor.from_pretrained(
            model_args.image_processor_name, **config_kwargs
        )
    elif model_args.model_name_or_path:
        image_processor = ViTImageProcessor.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        # image_processor = ViTImageProcessor(size=config.image_size)
        image_processor: ViTImageProcessor = AutoImageProcessor.from_pretrained(
            "facebook/vit-mae-base")

    # create model
    if model_args.model_name_or_path:
        model = ViTMAEForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        # model = ViTMAEForPreTraining(config)
        # model = _migrate_from_pretrained(model)
        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

    print_trainable_parameters(model, 'original model')
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["decode_head"],
    )
    lora_model = get_peft_model(model, lora_config)
    print_trainable_parameters(lora_model, 'lora model')

    # transformations as done in original MAE paper
    # source: https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    transforms = Compose([
        Lambda(lambda img: img.to('cuda:0').expand(3, -1, -1)),
        RandomResizedCrop(
            size,
            scale=(0.2, 1.0),
            interpolation=InterpolationMode.BICUBIC
        ),
        RandomHorizontalFlip(),
        # ToTensor(),
        Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        ),
    ])

    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size *
        training_args.gradient_accumulation_steps *
        training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * \
            total_train_batch_size / 256

    # Initialize our trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    lora_model.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "tasks": "masked-auto-encoding",
        "dataset": data_args.dataset_name,
        "tags": ["masked-auto-encoding"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def run(args: List[str] | None):
    """ 预训练ViTMAE模型

    """
    # use_cudnn()
    run_train(args)
    # dataset = CmriFlatternDataset('final2/cmri')
    # train_loader = DataLoader(dataset=dataset,  # 要传递的数据集
    #                           batch_size=32,  # 一个小批量数据的大小是多少
    #                           shuffle=True,  # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
    #                           num_workers=0)  # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。
