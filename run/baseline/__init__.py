from typing import List


from .dataset import BaselineDataset, PseudoDataset, ZeroshotDataset, DatasetLazyLoader
from .model_arguments import ReEvalType, get_args
from .model_factory import ModelFactory
from typing import List
from .run_re_eval import run_re_eval
from .run_train import run_train
from .run_pseudo import run_pseudo
from .run_plot_test import run_plot_test


# 脚本入口
def run(args: List[str] | None):
    print("args:", args)

    model_args, training_args, other_args = get_args(args)

    if len(other_args) > 0:
        print("other_args:", other_args)

    get_dataset = DatasetLazyLoader(
        lambda: BaselineDataset(
            train_loop=model_args.loop,
            train_dataset=model_args.train_dataset,
            target_device=training_args.device,
            dataset_device=model_args.dataset_device,
            aux_dataset=model_args.get_aux_dataset(),
            image_enhance_level=model_args.image_enhance_level,
            sample_count=model_args.sample_count,
            image_size=model_args.get_dataset_image_size(),
            crop_size=model_args.get_dataset_crop_size(),
            crop_deep_maxscale=model_args.dataset_crop_deep_maxscale,
            valid_crop_size=model_args.get_dataset_valid_crop_size(),
            valid_crop_deep_maxscale=model_args.dataset_valid_crop_deep_maxscale,
            train_all_data=model_args.dataset_train_all,
        )
    )

    if len(model_args.zeroshot_dataset) > 0 and len(model_args.get_zeroshot_dataset_labels()) > 0:
        get_zeroshot_dataset = DatasetLazyLoader(
            lambda: ZeroshotDataset(
                model_args.zeroshot_dataset,
                target_device=training_args.device,
                dataset_device=model_args.dataset_device,
                image_size=model_args.get_dataset_image_size(),
                valid_crop_size=model_args.get_dataset_valid_crop_size(),
                valid_crop_deep_maxscale=model_args.dataset_valid_crop_deep_maxscale,
            )
        )
    else:
        get_zeroshot_dataset = DatasetLazyLoader(lambda: None)

    model_factory = ModelFactory(model_args, training_args, other_args)

    re_eval_type = ReEvalType.from_str(model_args.re_eval)
    if re_eval_type in (ReEvalType.Normal, ReEvalType.Force, ReEvalType.Complete):
        return run_re_eval(
            model_factory,
            get_dataset,
            get_zeroshot_dataset,
            model_args,
            training_args,
            re_eval_type,
        )

    if model_args.run_plot_test:
        return run_plot_test(
            model_factory,
            get_dataset,
            model_args,
            training_args,
        )

    if model_args.pseudo:
        get_pseudo_dataset = DatasetLazyLoader(
            lambda: PseudoDataset(
                pseudo_dataset=model_args.pseudo_dataset,
                pseudo_dataset_key=model_args.pseudo_dataset_key,
                target_device=training_args.device,
                dataset_device=model_args.dataset_device,
                image_size=model_args.get_dataset_image_size(),
                valid_crop_size=model_args.get_dataset_valid_crop_size(),
                valid_crop_deep_maxscale=model_args.dataset_valid_crop_deep_maxscale,
            )
        )
        return run_pseudo(
            model_factory,
            get_pseudo_dataset,
            model_args,
            training_args,
            model_args.pseudo_output,
        )

    return run_train(
        args,
        model_args,
        training_args,
        model_factory,
        get_dataset,
        get_zeroshot_dataset,
    )
