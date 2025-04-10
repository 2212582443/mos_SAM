import os
from glob import glob
from typing import Callable

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import TrainingArguments

from run.baseline.result_wrapper import ModelResult

from .dataset import DATASET, BaselineDataset
from .metric import Metric
from .model_arguments import ModelArguments
from .model_factory import ModelFactory
from .model_wrapper.label_utils import get_config_eval_labels


def find_best_epoch(experiment_root: str) -> str:
    best_dice, best_epoch = 0, ""

    metric_file_list = glob("epoch-*/valid-metrics.pt", root_dir=experiment_root)
    metric_file_list.sort(reverse=True)

    for metric_path in metric_file_list:
        epoch = f"{experiment_root}/{metric_path[:-17]}/model.pt"
        if not os.path.exists(epoch):
            continue
        epoch = f"{experiment_root}/{metric_path[:-17]}/test-metrics.pt"
        if not os.path.exists(epoch):
            continue

        metric = torch.load(f"{experiment_root}/{metric_path}")
        dice: torch.Tensor = metric["dice"]
        dice = dice.mean().item()
        if dice > best_dice:
            best_dice = dice
            best_epoch = f"{experiment_root}/{metric_path[:-17]}"

    if len(best_epoch) == 0:
        print("best metric NOT found!", experiment_root)
    else:
        print("best metric:", best_dice, best_epoch)
    return best_epoch


@torch.no_grad()
def run_plot_test(
    model_factory: ModelFactory,
    get_dataset: Callable[[], BaselineDataset],
    model_args: ModelArguments,
    training_args: TrainingArguments,
):
    print("重新绘制最佳epoch的测试结果！")
    out_dir = model_factory.get_output_dir()
    best_model = find_best_epoch(out_dir)
    if best_model == "":
        print("best model NOT found!")
        return

    metricer = Metric(
        device=training_args.device,
        aux_dataset_labels=model_args.get_aux_dataset_labels(),
        zeroshot_dataset_labels=model_args.get_zeroshot_dataset_labels(),
        label_count=model_args.label_count,
        spacing=model_args.get_dataset_spacing(),
    )
    _target_label_id_list_cuda, eval_target_label_id_list = get_config_eval_labels(model_args, training_args.device)

    model_wrapper = model_factory.load_model(best_model)
    model_wrapper.model.eval()
    test_result_all_pred = []
    test_result_metric = []
    test_image = []
    test_label = []
    for cmri_batch in get_dataset().shuffer_test_id(training_args.eval_batch_size):
        data_item = get_dataset().batch_get_test(cmri_batch)
        with torch.autocast(device_type="cuda", enabled=training_args.fp16):
            after_softmax: ModelResult = model_wrapper.forward_test(data_item)

        metric_segment, pred_segment_all_to_be_ploted = after_softmax.merge_incomplete_label(
            metricer,
            DATASET.MAIN,
            eval_target_label_id_list[DATASET.MAIN.value],
        )

        # (bs, [rgb], h, w) or (bs, [rgb], d, h, w)
        # 理论上都为3d图像
        test_result_all_pred.append(pred_segment_all_to_be_ploted)
        # (bs,h,w) or (bs, d, h, w) -> (bs, 3, h, w) or (bs, 3, d, h, w)
        test_result_metric.append(metric_segment)

        test_image.append(data_item.image)
        test_label.append(data_item.segment)
    if len(test_result_all_pred) == 0:
        print("没有测试结果！")
        return

    test_image = torch.cat(test_image, dim=0).repeat(1, 3, 1, 1, 1)  # (bs, 3, d, h, w)
    test_label = torch.cat(test_label, dim=0)
    test_result_all_pred = torch.cat(test_result_all_pred, dim=0)
    test_result_metric = torch.cat(test_result_metric, dim=0)
    test_result_metric_overlap = test_result_metric + test_label * 2
    print(test_result_metric_overlap.unique())

    test_label = metricer.color_map.apply(test_label)  # (bs, [rgb], d, h, w)
    test_result_metric = metricer.color_map.apply(test_result_metric.long())
    overlap_color_map = torch.tensor(
        [
            [0, 0, 0],
            [255, 0, 0],
            [36, 70, 255],
            [0, 255, 0],
        ],
        dtype=torch.uint8,
    ).to(test_label.device)
    test_result_metric_overlap = metricer.color_map.apply(test_result_metric_overlap.long(), overlap_color_map)
    assert len(test_result_all_pred.shape) == 5  # (bs, [rgb], d, h, w)

    bs, _, d, h, w = test_result_all_pred.shape
    # (bs, [rgb], d, h, w) -> ([rgb], d*h, bs*w)
    test_result_all_pred = test_result_all_pred.permute(1, 2, 3, 0, 4).reshape(3, d * h, bs * w)
    test_result_metric = test_result_metric.permute(1, 2, 3, 0, 4).reshape(3, d * h, bs * w)
    test_result_metric_overlap = test_result_metric_overlap.permute(1, 2, 3, 0, 4).reshape(3, d * h, bs * w)
    test_label = test_label.permute(1, 2, 3, 0, 4).reshape(3, d * h, bs * w)
    test_image = test_image.permute(1, 2, 3, 0, 4).reshape(3, d * h, bs * w) * 255

    to_pil_image(test_result_all_pred.cpu()).save(f"{best_model}/test_all_pred[label].png")
    to_pil_image(test_result_metric.cpu()).save(f"{best_model}/test_metric_only[label].png")
    to_pil_image(test_result_metric_overlap.cpu()).save(f"{best_model}/test_metric_only[label][overlap].png")
    to_pil_image(apply_label(test_image, test_result_metric_overlap, 0.90)).save(
        f"{best_model}/test_metric_only[image+label][overlap].png"
    )
    to_pil_image(test_label.cpu()).save(f"{best_model}/test[label].png")
    to_pil_image(test_image.cpu()).save(f"{best_model}/test[image].png")
    to_pil_image(apply_label(test_image, test_label)).save(f"{best_model}/test[image+label].png")
    to_pil_image(apply_label(test_image, test_result_all_pred)).save(f"{best_model}/test_all_pred[image+label].png")
    to_pil_image(apply_label(test_image, test_result_metric)).save(f"{best_model}/test_metric_only[image+label].png")

    print(f"图像保存完毕！目录：{best_model}")


def apply_label(image, label, alpha=0.8):
    label_mask = label.sum(dim=0, keepdim=True) > 0
    label_mask = label_mask.repeat(3, 1, 1)
    image, label = image.float(), label.float()
    image[label_mask > 0] *= 1 - alpha
    label = label * alpha
    image = image + label
    return image.clamp(0, 255).to(torch.uint8)
