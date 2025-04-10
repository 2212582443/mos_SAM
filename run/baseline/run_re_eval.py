from dataclasses import dataclass, field
import os
import shutil
from glob import glob
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import TrainingArguments

from .dataset import DATASET, BaselineDataset, DataItem, ZeroshotDataset
from .metric import Metric
from .model_arguments import ModelArguments, ReEvalType
from .model_factory import ModelFactory
from .model_wrapper.label_utils import get_config_eval_labels
from .result_wrapper import ModelResult


def should_re_eval(
    stage: str,
    model_path: str,
    re_eval_type: ReEvalType,
) -> bool:
    if re_eval_type == ReEvalType.NoReEval:
        return False
    if re_eval_type == ReEvalType.Force:
        return True

    backup_metric_file = f"{model_path}/{stage}-metrics.pt.bk"
    metric_file = f"{model_path}/{stage}-metrics.pt"

    if re_eval_type == ReEvalType.Normal and not os.path.exists(backup_metric_file):
        return True

    if re_eval_type == ReEvalType.Complete and not os.path.exists(metric_file):
        return True

    return False


@torch.no_grad()
def run_re_eval(
    model_factory: ModelFactory,
    get_dataset: Callable[[], BaselineDataset],
    get_zeroshot_dataset: Callable[[], ZeroshotDataset],
    model_args: ModelArguments,
    training_args: TrainingArguments,
    re_eval_type: ReEvalType,
):
    print("重新对所有指标进行计算！")
    out_dir = model_factory.get_output_dir()
    metricer = Metric(
        device=training_args.device,
        aux_dataset_labels=model_args.get_aux_dataset_labels(),
        zeroshot_dataset_labels=model_args.get_zeroshot_dataset_labels(),
        label_count=model_args.label_count,
        spacing=model_args.get_dataset_spacing(),
    )

    _target_label_id_list_cuda, eval_target_label_id_list = get_config_eval_labels(model_args, training_args.device)

    model_list = glob("**/model.pt", root_dir=out_dir)
    model_list.sort(reverse=True)
    for model in model_list:
        print(model)
        model_path = f"{out_dir}/{model[:-9]}"

        model_wrapper = None

        # eval_labels: list[list[int]] = model_args.get_eval_labels()

        if should_re_eval("valid", model_path, re_eval_type):
            if model_wrapper is None:  # lazy load model
                model_wrapper = model_factory.load_model(model_path)
                model_wrapper.model.eval()

            # valid metrics
            epoch_losses = []
            metrics_to_saved = {
                "msd": [],
                "hd": [],
                "dice": [],
                "iou": [],
                "epoch_losses": epoch_losses,
            }
            for cmri_batch in get_dataset().shuffer_valid_id(training_args.eval_batch_size):
                data_item: DataItem = get_dataset().batch_get_valid(cmri_batch)
                with torch.autocast(device_type="cuda", enabled=training_args.fp16):
                    after_softmax: ModelResult = model_wrapper.forward_valid(data_item)

                valid_label = eval_target_label_id_list[data_item.dataset.value]
                metric_segment, _pred_segment_all_to_be_ploted = after_softmax.merge_incomplete_label(
                    metricer,
                    data_item.dataset,
                    valid_label,
                )

                loss = after_softmax.calc_loss(data_item, valid_label, model_args.label_count)
                epoch_losses.extend(loss.tolist())

                current_metric = after_softmax.calc_metric(
                    metricer,
                    metric_segment,
                    data_item,
                    valid_label,
                    model_args.label_count,
                )
                metricer.update_metric(current_metric, None, metrics_to_saved)

            metric_file = f"{model_path}/valid-metrics.pt"
            backup_metric_file = f"{model_path}/valid-metrics.pt.bk"
            if not os.path.exists(backup_metric_file) and os.path.exists(metric_file):
                shutil.copy(metric_file, backup_metric_file)
            metrics_to_saved = {k: torch.tensor(v) for k, v in metrics_to_saved.items()}
            torch.save(metrics_to_saved, metric_file)

        # test metrics
        if should_re_eval("test", model_path, re_eval_type):
            if model_wrapper is None:  # lazy load model
                model_wrapper = model_factory.load_model(model_path)
                model_wrapper.model.eval()

            epoch_losses = []
            metrics_to_saved = {
                "msd": [],
                "hd": [],
                "dice": [],
                "iou": [],
                "epoch_losses": epoch_losses,
            }
            for cmri_batch in get_dataset().shuffer_test_id(training_args.eval_batch_size):
                data_item = get_dataset().batch_get_test(cmri_batch)
                with torch.autocast(device_type="cuda", enabled=training_args.fp16):
                    after_softmax: ModelResult = model_wrapper.froward_test(data_item)

                valid_label = eval_target_label_id_list[data_item.dataset.value]
                metric_segment, pred_segment_all_to_be_ploted = after_softmax.merge_incomplete_label(
                    metricer,
                    data_item.dataset.value,
                    valid_label,
                )

                loss = after_softmax.calc_loss(data_item, valid_label, model_args.label_count)
                epoch_losses.extend(loss.tolist())

                current_metric = after_softmax.calc_metric(
                    metricer,
                    metric_segment,
                    data_item,
                    valid_label,
                    model_args.label_count,
                )
                metricer.update_metric(current_metric, None, metrics_to_saved)

            backup_metric_file = f"{model_path}/test-metrics.pt.bk"
            metric_file = f"{model_path}/test-metrics.pt"
            if not os.path.exists(backup_metric_file) and os.path.exists(metric_file):
                shutil.copy(metric_file, backup_metric_file)

            metrics_to_saved = {k: torch.tensor(v) for k, v in metrics_to_saved.items()}
            torch.save(metrics_to_saved, metric_file)

        # zeroshot metrics
        if (
            should_re_eval("zeroshot", model_path, re_eval_type)
            and (zeroshot_dataset := get_zeroshot_dataset()) is not None
        ):
            if model_wrapper is None:  # lazy load model
                model_wrapper = model_factory.load_model(model_path)
                model_wrapper.model.eval()

            epoch_losses = []
            metrics_to_saved = {
                "msd": [],
                "hd": [],
                "dice": [],
                "iou": [],
                "epoch_losses": epoch_losses,
            }
            for cmri_batch in zeroshot_dataset.shuffer_zeroshot_id(training_args.eval_batch_size):
                data_item = zeroshot_dataset.batch_get_zeroshot(cmri_batch)
                with torch.autocast(device_type="cuda", enabled=training_args.fp16):
                    after_softmax: ModelResult = model_wrapper.forward_zeroshot(data_item)
                valid_label = eval_target_label_id_list[DATASET.ZERO_SHOT.value]
                onthot_metric_segment, _pred_segment_all_to_be_ploted = after_softmax.merge_incomplete_label(
                    metricer,
                    DATASET.ZERO_SHOT,
                    valid_label,
                )

                loss = after_softmax.calc_loss(data_item, valid_label, model_args.label_count)
                epoch_losses.extend(loss.tolist())

                current_metric = after_softmax.calc_metric(
                    metricer,
                    onthot_metric_segment,
                    data_item,
                    valid_label,
                    model_args.label_count,
                )
                metricer.update_metric(current_metric, None, metrics_to_saved)

            metric_file = f"{model_path}/zeroshot-metrics.pt"
            backup_metric_file = f"{model_path}/zeroshot-metrics.pt.bk"
            if not os.path.exists(backup_metric_file) and os.path.exists(metric_file):
                shutil.copy(metric_file, backup_metric_file)

            metrics_to_saved = {k: torch.tensor(v) for k, v in metrics_to_saved.items()}
            torch.save(metrics_to_saved, metric_file)

    print("计算完毕！")
