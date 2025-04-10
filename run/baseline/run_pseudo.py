import os
from typing import Callable
import torch
from transformers import TrainingArguments

from .model_arguments import ModelArguments

from .dataset import DataItem, Label, PseudoDataset
from .model_factory import ModelFactory
from .result_wrapper import SeparetionMaskResult
from .metric import get_onehost_label
from glob import glob


@torch.no_grad()
def run_pseudo(
    model_factory: ModelFactory,
    get_pseudo_dataset: Callable[[], PseudoDataset],
    model_args: ModelArguments,
    training_args: TrainingArguments,
    output_dir: str,
):
    print("生成伪标签...")
    root_dir = model_factory.get_output_dir()

    # find best model
    best_dice, best_model = -1, None
    for metric in glob("**/test-metrics.pt", root_dir=root_dir):
        model_path = root_dir + "/" + os.path.dirname(metric) + "/model.pt"
        if not os.path.exists(model_path):
            continue
        dice: torch.Tensor = torch.load(f"{root_dir}/{metric}")["dice"]
        dice = dice.nanmean(dim=0).item()

        if dice > best_dice:
            best_dice = dice
            best_model = model_path

    if best_model is None:
        models = glob("epoch-*/model.pt", root_dir=root_dir)
        models.sort(reverse=True)
        if len(models) <= 0:
            print("没有找到模型，不生成伪标签")
            return
        best_model = f"{root_dir}/{models[0]}"

        print("模型评分未知，选择最新的模型", best_model)
        best_dice = 0.74  # 模型未知，设置一个默认值

    if best_dice < 0.74:
        print(f"最好的模型的dice={best_dice}小于0.74，不生成伪标签")
        return
    print(f"best_dice: {best_dice}, best_model: {best_model}")

    # setting/modelname/epoch
    model_name: list[str] = best_model.split("/")[-3:-2]
    model_name = "_".join(model_name)

    dst_file_name = f"{output_dir}/{best_dice:0.4f}@{model_name}.pt"
    if False and os.path.exists(dst_file_name):
        print(f"伪标签已经存在，不重新生成: {dst_file_name}")
        return

    model_path = os.path.dirname(best_model)
    model_wrapper = model_factory.load_model(model_path)
    model_wrapper.model.eval()

    pseudo_dataset = get_pseudo_dataset()
    with torch.autocast(device_type="cuda", enabled=training_args.fp16):
        onehot_label_list = []
        for batch in pseudo_dataset.shuffer_pseudo_id(training_args.eval_batch_size):
            data_item: DataItem = pseudo_dataset.batch_get_pseudo(batch)

            after_softmax = model_wrapper.forward_pseudo(data_item)

            if isinstance(after_softmax, SeparetionMaskResult):
                onehot_label = after_softmax.get_onehot_mask()
            else:
                onehot_label = after_softmax.argmax(dim=1)
                onehot_label = get_onehost_label(onehot_label, model_args.label_count)

            onehot_label = onehot_label.to(torch.uint8)
            onehot_label_list.append(onehot_label)

        onehot_label_list = torch.cat(onehot_label_list, dim=0)
        print("label_list:", onehot_label_list.shape)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(onehot_label_list, dst_file_name)
