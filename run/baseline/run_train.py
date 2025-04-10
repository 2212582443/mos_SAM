import json, math, random, os
from typing import Callable
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments

from mos.utils.tensors import save_tensor_file

from .dataset import DATASET, BaselineDataset, DataItem, ZeroshotDataset
from .model_arguments import ModelArguments
from .model_factory import ModelFactory
import torch.nn.functional as F
import numpy as np
from .result_wrapper import ModelResult, SeparetionMaskResult
from .metric import Metric
from torch.optim.lr_scheduler import MultiStepLR
from .model_wrapper.label_utils import get_config_eval_labels, get_config_train_labels


def mix_image_label(image, label, alpha=0.8):
    """
    Args:
        image: (bs, 3, h, w)
        label: (bs, 3, h, w)
        alpha: float
    """
    label_mask = label.sum(dim=1, keepdim=True) > 0
    label_mask = torch.cat([label_mask, label_mask, label_mask], 1)
    image, label = image.clone().float(), label.float()
    image[label_mask > 0] *= 1 - alpha
    label = label * alpha
    image = image + label
    return image.clamp(0, 255).to(torch.uint8)


class Trainer:
    def __init__(
        self,
        model_wrapper,
        scaler,
        optimizer,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        dataset,
        zeroshot_dataset,
        log_writer,
        model_factory,
    ):
        self.model_wrapper: torch.nn.Module = model_wrapper
        self.scaler = scaler
        self.optimizer: AdamW = optimizer
        self.model_args: ModelArguments = model_args
        self.training_args: TrainingArguments = training_args
        self.dataset: BaselineDataset = dataset
        self.zeroshot_dataset: ZeroshotDataset | None = zeroshot_dataset
        self.log_writer: SummaryWriter = log_writer
        self.model_factory: ModelFactory = model_factory
        self.mini_batch_size = training_args.gradient_accumulation_steps
        self.label_count = model_args.label_count

        _target_label_id_list_cuda, target_label_id_list = get_config_train_labels(model_args, training_args.device)
        self.train_target_label_id_list = target_label_id_list
        _target_label_id_list_cuda, target_label_id_list = get_config_eval_labels(model_args, training_args.device)
        self.eval_target_label_id_list = target_label_id_list

        self.metric = Metric(
            device=training_args.device,
            aux_dataset_labels=model_args.get_aux_dataset_labels(),
            zeroshot_dataset_labels=model_args.get_zeroshot_dataset_labels(),
            label_count=model_args.label_count,
            spacing=model_args.get_dataset_spacing(),
        )

        metric_dic = {}
        if os.path.exists(f"{model_factory.get_output_dir()}/metric.json"):
            with open(f"{model_factory.get_output_dir()}/metric.json", "r") as f:
                for line in f.readlines():
                    metric = json.loads(line)
                    metric_dic[metric["epoch"]] = metric["dice"]
        best_metric, best_epoch = 0, -1
        for epoch, dice in metric_dic.items():
            if dice > best_metric:
                best_metric = dice
                best_epoch = epoch
        print(f"目前最优epoch:{best_epoch}, dice:{best_metric}")
        self.best_metric = best_metric

    def log_to_tensorboard(
        self,
        stage,
        epoch,
        epoch_losses,
        metrics_to_show,
        source_image,
        segment_label,
        pred_segment_all_to_be_ploted,
        pred_eat_label,
    ):
        print(f"{stage} epoch: {epoch:04}, loss: {mean(epoch_losses):0.8f} ±{np.std(epoch_losses):0.8f}")

        self.log_writer.add_scalar(f"{stage}/loss/mean", mean(epoch_losses), epoch)
        self.log_writer.add_scalar(f"{stage}/loss/std", np.std(epoch_losses), epoch)

        if stage == "train" and epoch % self.model_args.log_every_n_epochs != 0:
            return

        PLOT_IMG_COUNT = 10

        source_image = source_image[:PLOT_IMG_COUNT]
        segment_label = segment_label[:PLOT_IMG_COUNT]
        pred_segment_all_to_be_ploted = pred_segment_all_to_be_ploted[:PLOT_IMG_COUNT]
        pred_eat_label = pred_eat_label[:PLOT_IMG_COUNT] if pred_eat_label is not None else None

        # print("labels:", segment_label.unique(), pred_eat_label.unique(), pred_segment_all.unique())
        if len(source_image.shape) == 4:
            bs, c, h, w = source_image.shape

            source_image = source_image.repeat(1, 3, 1, 1) * 255
            segment_label = self.metric.color_map.apply(segment_label)
            segment_label = mix_image_label(source_image, segment_label, 0.8)
            segment_label = segment_label.permute(1, 2, 0, 3).reshape(3, h, bs * w)

            if pred_eat_label is not None:
                # (bs, 3, h, w)
                pred_eat_label = self.metric.color_map.apply(pred_eat_label)
                pred_eat_label = mix_image_label(source_image, pred_eat_label)
                pred_eat_label = pred_eat_label.permute(1, 2, 0, 3).reshape(3, h, bs * w)

            source_image = source_image.permute(1, 2, 0, 3).reshape(3, h, bs * w).to(torch.uint8)
            pred_segment_all_to_be_ploted = pred_segment_all_to_be_ploted.permute(1, 2, 0, 3).reshape(3, h, bs * w)
        else:
            bs, c, d, h, w = source_image.shape

            source_image = source_image.repeat(1, 3, 1, 1, 1) * 255
            segment_label = self.metric.color_map.apply(segment_label)
            segment_label = mix_image_label(source_image, segment_label, 0.8)
            segment_label = segment_label.permute(1, 2, 3, 0, 4).reshape(3, d * h, bs * w)

            if pred_eat_label is not None:
                # (bs, 3, d, h, w)
                pred_eat_label = self.metric.color_map.apply(pred_eat_label)
                pred_eat_label = mix_image_label(source_image, pred_eat_label)
                pred_eat_label = pred_eat_label.permute(1, 2, 3, 0, 4).reshape(3, d * h, bs * w)

            source_image = source_image.permute(1, 2, 3, 0, 4).reshape(3, d * h, bs * w).to(torch.uint8)
            pred_segment_all_to_be_ploted = pred_segment_all_to_be_ploted.permute(1, 2, 3, 0, 4).reshape(
                3, d * h, bs * w
            )

        self.log_writer.add_image(f"{stage}/image/source", source_image, epoch, dataformats="CHW")
        self.log_writer.add_image(f"{stage}/image/label", segment_label, epoch, dataformats="CHW")
        if pred_eat_label is not None:
            self.log_writer.add_image(f"{stage}/image/pred", pred_eat_label, epoch, dataformats="CHW")
        self.log_writer.add_image(f"{stage}/image/pred_all", pred_segment_all_to_be_ploted, epoch, dataformats="CHW")

        for k, v in metrics_to_show.items():
            if len(v) == 0:
                print(f"{stage} metric {k} is empty")
                continue
            self.log_writer.add_scalar(f"{stage}/metric/{k}/mean", mean(v), epoch)
            self.log_writer.add_scalar(f"{stage}/metric/{k}/std", np.std(v), epoch)

    # @autocast()
    def do_train(self, epoch):
        epoch_losses = []
        self.model_wrapper.model.train()
        metrics_to_show = {
            "msd": [],
            "hd": [],
            "dice": [],
            "iou": [],
        }
        update_count = 0
        need_log_metric = epoch % self.model_args.log_every_n_epochs == 0
        for cmri_batch, aux_batch in tqdm(
            self.dataset.shuffer_train_id(self.training_args.train_batch_size, use_loop=not need_log_metric)
        ):
            if self.model_args.main_dataset_only or self.dataset.aux_len() == 0:
                batch: list[DataItem] = [self.dataset.batch_get_train(cmri_batch)]
            elif self.model_args.aux_dataset_only:
                batch: list[DataItem] = [self.dataset.batch_get_aux(aux_batch)]
            else:
                batch: list[DataItem] = (
                    [
                        self.dataset.batch_get_aux(aux_batch),
                        self.dataset.batch_get_train(cmri_batch),
                    ]
                    if random.randint(0, 10) <= 5
                    else [
                        self.dataset.batch_get_train(cmri_batch),
                        self.dataset.batch_get_aux(aux_batch),
                    ]
                )

            for data_item in batch:
                if data_item is None:
                    continue
                with torch.autocast(device_type="cuda", enabled=self.training_args.fp16):
                    after_softmax: ModelResult = self.model_wrapper(data_item)

                    valid_label: list[int] = self.train_target_label_id_list[data_item.dataset.value]
                    pred_segment_eat, pred_segment_all_to_be_ploted = after_softmax.merge_incomplete_label(
                        self.metric,
                        data_item.dataset,
                        valid_label,
                    )
                    if False and data_item.dataset == DATASET.AUX:
                        save_tensor_file(
                            {
                                "pred_softmax": pred_softmax,
                                "pred_segment_eat": pred_segment_eat,
                                "pred_segment_all_to_be_ploted": pred_segment_all_to_be_ploted,
                                "after_softmax": after_softmax,
                                "image": data_item.image,
                                "label": data_item.segment,
                                "depth_info": data_item.depth_info,
                            },
                            "test.pt",
                        )
                        raise 999

                    loss = after_softmax.calc_loss(data_item, valid_label, self.model_args.label_count)

                epoch_losses.extend(loss.tolist())

                loss = loss.mean()

                if need_log_metric:
                    metric = after_softmax.calc_metric(
                        self.metric,
                        pred_segment_eat,
                        data_item,
                        valid_label,
                        self.label_count,
                    )
                    self.metric.update_metric(metric, metrics_to_show, None)

                if loss.isnan():
                    raise "loss is nan"
                if self.training_args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                update_count += 1

            if update_count % self.mini_batch_size == 0:
                update_count = 0
                if self.training_args.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

        if update_count > 0:
            if self.training_args.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        self.model_factory.save_epoch_model(epoch, self.model_wrapper, self.optimizer)

        if isinstance(after_softmax, SeparetionMaskResult):
            pred_to_be_plot = pred_segment_eat.argmax(1)
        else:
            pred_to_be_plot = pred_segment_eat

        self.log_to_tensorboard(
            "train",
            epoch,
            epoch_losses,
            metrics_to_show,
            data_item.image,
            data_item.segment,
            pred_segment_all_to_be_ploted,
            pred_to_be_plot,
        )

    @torch.no_grad()
    def do_valid(self, epoch):
        if self.dataset.valid_len() < 1:
            return
        epoch_losses = []
        self.model_wrapper.model.eval()
        metrics_to_show = {
            "msd": [],
            "hd": [],
            "dice": [],
            "iou": [],
        }
        metrics_to_saved = {
            "msd": [],
            "hd": [],
            "dice": [],
            "iou": [],
            "epoch_losses": epoch_losses,
        }
        data_item = None
        for cmri_batch in self.dataset.shuffer_valid_id(self.training_args.eval_batch_size):
            data_item: DataItem = self.dataset.batch_get_valid(cmri_batch)
            with torch.autocast(device_type="cuda", enabled=self.training_args.fp16):
                after_softmax: ModelResult = self.model_wrapper.forward_valid(data_item)
            valid_label = self.eval_target_label_id_list[data_item.dataset.value]
            pred_segment_eat, pred_segment_all_to_be_ploted = after_softmax.merge_incomplete_label(
                self.metric,
                data_item.dataset,
                valid_label,
            )

            loss = after_softmax.calc_loss(data_item, valid_label, self.model_args.label_count)
            epoch_losses.extend(loss.tolist())

            metric = after_softmax.calc_metric(
                self.metric,
                pred_segment_eat,
                data_item,
                valid_label,
                self.label_count,
            )
            self.metric.update_metric(metric, metrics_to_show, metrics_to_saved)

        self.model_factory.save_epoch_metric(epoch, "valid", metrics_to_saved)

        out_dir = self.model_factory.get_output_dir()
        dice = mean(metrics_to_show["dice"]) if len(metrics_to_show["dice"]) > 0 else 0.0
        record = json.dumps({"stage": "valid", "epoch": epoch, "loss": mean(epoch_losses), "dice": dice})
        with open(f"{out_dir}/metric.json", "a") as f:
            f.writelines([record, "\n"])

        if dice > self.best_metric:
            self.best_metric = dice
            print(f"最优epoch:{epoch}, validation dice:{dice}, 保存模型")
            self.model_factory.persist_model(epoch)

        if isinstance(after_softmax, SeparetionMaskResult):
            pred_to_be_plot = pred_segment_eat.argmax(1)
        else:
            pred_to_be_plot = pred_segment_eat

        self.log_to_tensorboard(
            "valid",
            epoch,
            epoch_losses,
            metrics_to_show,
            data_item.image,
            data_item.segment,
            pred_segment_all_to_be_ploted,
            pred_to_be_plot,
        )

    @torch.no_grad()
    def do_test(self, epoch):
        if self.dataset.test_len() < 1:
            return
        epoch_losses = []
        self.model_wrapper.model.eval()
        metrics_to_show = {
            "msd": [],
            "hd": [],
            "dice": [],
            "iou": [],
        }
        metrics_to_saved = {
            "msd": [],
            "hd": [],
            "dice": [],
            "iou": [],
            "epoch_losses": epoch_losses,
        }
        data_item = None
        for cmri_batch in self.dataset.shuffer_test_id(self.training_args.eval_batch_size):
            data_item: DataItem = self.dataset.batch_get_test(cmri_batch)
            valid_label = self.eval_target_label_id_list[data_item.dataset.value]
            with torch.autocast(device_type="cuda", enabled=self.training_args.fp16):
                after_softmax: ModelResult = self.model_wrapper.forward_test(data_item)
            pred_segment_eat, pred_segment_all_to_be_ploted = after_softmax.merge_incomplete_label(
                self.metric,
                data_item.dataset,
                valid_label,
            )

            loss = after_softmax.calc_loss(data_item, valid_label, self.model_args.label_count)
            epoch_losses.extend(loss.tolist())

            metric = after_softmax.calc_metric(
                self.metric,
                pred_segment_eat,
                data_item,
                valid_label,
                self.label_count,
            )
            self.metric.update_metric(metric, metrics_to_show, metrics_to_saved)

        self.model_factory.save_epoch_metric(epoch, "test", metrics_to_saved)

        dice = mean(metrics_to_show["dice"]) if len(metrics_to_show["dice"]) > 0 else 0.0
        out_dir = self.model_factory.get_output_dir()
        record = json.dumps({"stage": "test", "epoch": epoch, "loss": mean(epoch_losses), "dice": dice})
        with open(f"{out_dir}/metric.json", "a") as f:
            f.writelines([record, "\n"])

        print(f"test epoch: {epoch:04}, dice: {dice}")

        if isinstance(after_softmax, SeparetionMaskResult):
            pred_to_be_plot = pred_segment_eat.argmax(1)
        else:
            pred_to_be_plot = pred_segment_eat

        self.log_to_tensorboard(
            "test",
            epoch,
            epoch_losses,
            metrics_to_show,
            data_item.image,
            data_item.segment,
            pred_segment_all_to_be_ploted,
            pred_to_be_plot,
        )

    @torch.no_grad()
    def do_zeroshot(self, epoch):
        if self.zeroshot_dataset is None:
            return

        epoch_losses = []
        self.model_wrapper.model.eval()
        metrics_to_show = {
            "msd": [],
            "hd": [],
            "dice": [],
            "iou": [],
        }
        metrics_to_saved = {
            "msd": [],
            "hd": [],
            "dice": [],
            "iou": [],
            "epoch_losses": epoch_losses,
        }
        data_item = None
        for cmri_batch in self.zeroshot_dataset.shuffer_zeroshot_id(self.training_args.eval_batch_size):
            data_item: DataItem = self.dataset.batch_get_zeroshot(cmri_batch)
            after_softmax: ModelResult = self.model_wrapper.forward_zeroshot(data_item)
            valid_label = self.eval_target_label_id_list[data_item.dataset.value]
            pred_segment_eat, pred_segment_all_to_be_ploted = after_softmax.merge_incomplete_label(
                self.metric,
                data_item.dataset,
                valid_label,
            )

            loss = after_softmax.calc_loss(data_item, valid_label, self.model_args.label_count)
            epoch_losses.extend(loss.tolist())

            metric = after_softmax.calc_metric(
                self.metric,
                pred_segment_eat,
                data_item,
                valid_label,
                self.label_count,
            )
            self.metric.update_metric(metric, metrics_to_show, metrics_to_saved)

        self.model_factory.save_epoch_metric(epoch, "zeroshot", metrics_to_saved)

        out_dir = self.model_factory.get_output_dir()
        record = json.dumps({"stage": "zeroshot", "epoch": epoch, "loss": mean(epoch_losses)})
        with open(f"{out_dir}/metric.json", "a") as f:
            f.writelines([record, "\n"])

        self.log_to_tensorboard(
            "zeroshot",
            epoch,
            epoch_losses,
            metrics_to_show,
            data_item.image,
            data_item.segment,
            pred_segment_all_to_be_ploted,
            pred_segment_eat=None,
        )


def get_warmup_learning_rate(init_lr, warmup_steps, epoch, total_epoch):
    """Warmup learning rate for 5 epoch"""
    factor = warmup_steps // 30
    lr = init_lr * (0.1**factor)
    """Warmup"""
    lr = lr * float(1 + epoch + warmup_steps * total_epoch) / (5.0 * total_epoch)
    return lr


def run_train(
    args: list[str],
    model_args: ModelArguments,
    training_args: TrainingArguments,
    model_factory: ModelFactory,
    get_dataset: Callable[[], BaselineDataset],
    get_zeroshot_dataset: Callable[[], ZeroshotDataset],
):
    model_wrapper = model_factory.load_model(
        model_args.model_path,
        pretrain_model_path=model_args.pretrain_model_path,
        strict=model_args.load_model_strict,
    )

    if model_args.init:
        if "init_from_pretrain" in model_wrapper:
            print("model initing.. from pretrain!")
            model_wrapper.init_from_pretrain(model_factory.get_output_dir())
            model_factory.save_epoch_model(0, model_wrapper, None)
            print("init complete & exit!")
        else:
            print("model not support init")
        return

    log_writer = SummaryWriter(log_dir=f"{model_factory.get_output_dir()}/logs")
    log_writer.add_text(f"cmd_args", str(args), model_args.start_epoch)

    if model_args.start_epoch < 1:
        print("auto detect start epoch...")
        model_args.start_epoch = model_factory.get_start_epoch_number()

    if model_args.increase_epoch is None or len(model_args.increase_epoch) == 0:
        print(f"start_epoch={model_args.start_epoch}")
    elif model_args.increase_epoch == "fix":
        model_args.start_epoch = model_factory.get_start_epoch_number()
        if model_args.start_epoch % 10 == 1:
            print("已经是10的倍数，无需训练")
            return
        end_epoch = math.ceil(model_args.start_epoch / 10) * 10
        training_args.num_train_epochs = end_epoch
        print(f"训练到最近整10的epoch: {model_args.start_epoch}到{end_epoch}")
    elif int(model_args.increase_epoch) > 0:
        new_epoches = model_args.start_epoch + int(model_args.increase_epoch) - 1
        new_epoches = math.ceil(new_epoches / 10) * 10  # 向上取整到10的倍数
        print(f"增量训练，从{model_args.start_epoch}到{new_epoches}的epochs.")
        training_args.num_train_epochs = new_epoches
    elif int(model_args.increase_epoch) < 0:  # 重新训练最后几个epoch
        start_epoch = model_args.start_epoch + int(model_args.increase_epoch)
        start_epoch = math.floor(start_epoch / 10) * 10  # 向下取整到10的倍数
        start_epoch = max(start_epoch, 1)
        end_epoch = start_epoch + 9
        end_epoch = math.ceil(end_epoch / 10) * 10  # 向上取整到10的倍数
        print(f"重新训练{start_epoch}到{end_epoch}的epochs.")
        model_args.start_epoch = start_epoch
        training_args.num_train_epochs = end_epoch
    else:
        raise ValueError(f"increase_epoch 参数错误: {model_args.increase_epoch}")

    num_epochs = int(training_args.num_train_epochs)
    if model_args.start_epoch > num_epochs:
        print("训练结束， 无需训练")
        return

    optimizer = AdamW(
        model_wrapper.model.parameters(),
        lr=training_args.learning_rate,
        eps=1e-4,
        weight_decay=training_args.weight_decay,
    )
    if model_args.load_optimizer:
        model_factory.load_optimizer(
            optimizer,
            pretrain_model_path=model_args.pretrain_model_path,
        )

    scaler = GradScaler() if training_args.fp16 else None
    warmup_steps = training_args.warmup_steps
    trainer = Trainer(
        model_wrapper,
        scaler,
        optimizer,
        model_args,
        training_args,
        get_dataset(),
        get_zeroshot_dataset(),
        log_writer,
        model_factory,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=model_args.get_multi_step_lr_scheduler_parameter(),
        gamma=model_args.lr_scheduler_gamma,
    )

    if model_args.use_profile:
        print("profiling...")
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{model_factory.get_output_dir()}/logs/"),
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    for epoch in range(model_args.start_epoch, num_epochs + 1):
        if model_args.use_profile:
            prof.step()
        if training_args.do_train:
            if epoch <= warmup_steps:
                lr = get_warmup_learning_rate(training_args.learning_rate, warmup_steps, epoch, num_epochs)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            elif epoch == warmup_steps + 1:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = training_args.learning_rate
            else:
                scheduler.step(epoch=epoch)
            trainer.do_train(epoch)
        if training_args.do_eval:
            trainer.do_valid(epoch)
        if training_args.do_predict:
            trainer.do_test(epoch)
        if get_zeroshot_dataset() is not None:
            trainer.do_zeroshot(epoch)

    if model_args.use_profile:
        prof.stop()
