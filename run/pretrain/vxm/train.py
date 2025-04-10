import os, torch
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from mos.models.voxelmorph.networks import VxmDense
import mos.models.voxelmorph.losses as vxm_loss
from mos.utils.files import relative_symlink_file
from mos.utils.model_utils import use_cudnn

from tqdm import tqdm
from statistics import mean
from torch import nn, Tensor
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from .vxm_dataset_compat import VxmDatasetCompat


class DiceScore(nn.Module):
    """
    hard-dice loss, useful in binary segmentation
    """

    def __init__(self):
        super().__init__()

    def forward(self, probs: Tensor, labels: Tensor):
        """
        inputs:
            probs: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """

        probs = probs.round()

        bs = probs.shape[0]
        intersection = (probs * labels).view(bs, -1).sum(dim=1)
        union = (probs + labels).view(bs, -1).sum(dim=1)

        dice = (2 * intersection) / torch.clamp(union, min=1e-5)
        dice = torch.mean(dice)
        return dice


global_last_result = (None, None, None)


class CustomDataParallel(torch.nn.DataParallel):
    def __init__(
        self,
        module: Any,
        device_ids: Sequence[int | torch.device] | None = None,
        output_device: int | torch.device | None = None,
        dim: int = 0,
    ) -> None:
        super().__init__(module, device_ids, output_device, dim)

    def scatter(
        self,
        inputs: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]],
        device_ids: Sequence[Union[int, torch.device]],
    ) -> Any:
        all_device_param_list = []
        for device_index in device_ids:
            param_list = []
            for input in inputs:
                param_list.append(input[device_index])
            all_device_param_list.append(param_list)
        return all_device_param_list, ([kwargs for _ in device_ids])


class VxmTrainModel(nn.Module):
    def __init__(
        self,
        model: VxmDense,
        dataset: VxmDatasetCompat,
    ):
        super().__init__()
        self.vxm = model
        self.dataset = dataset
        self.image_loss = vxm_loss.MSE()
        self.image_grad_loss = vxm_loss.Grad("l2")
        self.ncc_loss = vxm_loss.NCC2d(win=[9, 9])

    def forward(self, device: int, ids: torch.Tensor):
        source, target = self.dataset.batch_get_train_device(device, ids)
        return self.forward_2(source, target, device)

    def forward_2(self, source: Tensor, target: Tensor, device: int):
        # with autocast(enabled=False):
        y_moved, pos_flow = self.vxm(source, target)

        mse_loss = self.image_loss.loss(target, y_moved)
        grad_loss = self.image_grad_loss.loss(None, pos_flow)
        ncc_loss = self.ncc_loss.loss(target, y_moved)

        loss = mse_loss

        if not grad_loss.isnan():
            loss = loss + grad_loss * 1e-3
        else:
            del grad_loss
            print("grad loss is nan", grad_loss.dtype)

        if False:
            if not ncc_loss.isnan():
                if ncc_loss.item() < 1:
                    loss = loss + ncc_loss
                else:
                    del ncc_loss
                    print(f"ncc loss too big, skip, loss:{ncc_loss}")
            else:
                del ncc_loss
                print("ncc_loss is nan", ncc_loss.dtype)

        if device == 0:
            global global_last_result
            global_last_result = (source, target, y_moved, pos_flow)

        return (loss.unsqueeze(0),)


def run(_):
    use_cudnn()

    global global_last_result

    run_name = "vxm1010"
    BATCH_SIZE = 640

    log_writer = SummaryWriter(log_dir=f".checkpoint/{run_name}/logs")

    log_writer.add_text("description", "对全部数据集进行训练目的在于补全缺失的segment")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    new_dataset = VxmDatasetCompat(
        base_path=".cache/dataset/vxm-dataset",
        device_count=torch.cuda.device_count(),
    )
    # new_dataset.split_systole_distole_as_train_valid()

    model: VxmDense = VxmDense(
        inshape=(256, 256),
        nb_unet_features=[
            [16, 32, 32, 32],
            [32, 32, 32, 32, 32, 16, 16],
        ],
        int_downsize=1,
    ).to(device)

    def save_model(path):
        target_dir = os.path.dirname(path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        torch.save(model.state_dict(), path)

    def load_model_if_exists(path):
        if os.path.exists(path):
            print("load model...", path)
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
            model.to(device)

    load_model_if_exists(f".checkpoint/{run_name}/model-latest.pth")

    scaler = GradScaler()
    train_model = VxmTrainModel(model, new_dataset)
    # dp_model = torch.nn.DataParallel(train_model).cuda()
    dp_model = CustomDataParallel(train_model).cuda()
    optimizer = AdamW(dp_model.parameters(), lr=3e-3, eps=1e-5, weight_decay=0.05)
    device_list = range(torch.cuda.device_count())
    for epoch in range(1481, 9001):
        dp_model.train()
        epoch_losses = []
        for batch in tqdm(new_dataset.shuffer_train_id(BATCH_SIZE)):
            # source, target = new_dataset.batch_get_train(batch)
            # (bs,1,h,w), (bs,2,h,w)

            (loss,) = dp_model(device_list, batch)

            loss = loss.mean()

            if loss.isnan():
                raise "loss is nan"

            if loss.item() > 3:
                print(f"loss abnormal! skip, loss: {loss}")
                continue

            if True:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_losses.append(loss.item())

        epoch_losses = mean(epoch_losses)
        print(f"epoch: {epoch}, loss: {epoch_losses:.6f} \t {batch[0][:10].cpu().tolist()}")
        source, target, pred, pos_flow = global_last_result
        log_writer.add_scalar("train/loss", epoch_losses, epoch)
        log_writer.add_image("train/source_image", source[0, :, :, :], epoch, dataformats="CHW")

        pos_flow = pos_flow[0, :, :, :]
        _, h, w = pos_flow.shape
        # padd 0 in dim 1
        pos_flow = torch.cat([pos_flow, torch.zeros(1, h, w, device=pos_flow.device)], dim=0)
        pos_flow = pos_flow.abs() / h
        max = pos_flow.max()
        if max < 0.8:
            max = max / 0.8
            pos_flow = pos_flow / max
        pos_flow = pos_flow.clamp(0, 1)
        log_writer.add_image("train/pos_flow", pos_flow, epoch, dataformats="CHW")

        log_writer.add_image("train/target_image", target[0, :, :, :], epoch, dataformats="CHW")
        log_writer.add_image("train/moved_image", pred[0, :, :, :], epoch, dataformats="CHW")
        log_writer.add_image(
            "train/diff_image_before", source[0, :, :, :] - target[0, :, :, :], epoch, dataformats="CHW"
        )
        log_writer.add_image("train/diff_image_after", pred[0, :, :, :] - target[0, :, :, :], epoch, dataformats="CHW")
        # log_writer.add_tensor("train/pos_flow", pos_flow[0, :, :, :], epoch)

        if epoch % 10 == 0:
            save_model(f".checkpoint/{run_name}/model-{epoch:04}.pth")
            relative_symlink_file(
                f".checkpoint/{run_name}/model-{epoch:04}.pth",
                f".checkpoint/{run_name}/model-latest.pth",
            )
        else:
            if os.path.exists(f".checkpoint/{run_name}/model-latest.pth"):
                os.remove(f".checkpoint/{run_name}/model-latest.pth")
            save_model(f".checkpoint/{run_name}/model-latest.pth")
