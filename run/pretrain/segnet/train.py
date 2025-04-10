import os
from mos.utils.files import relative_symlink_file
from mos.utils.model_utils import use_cudnn
from run.pretrain.segnet.model import SegNet

from tqdm import tqdm
from statistics import mean
import torch
from torch import nn, Tensor
from torch.optim import AdamW


from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean
import torch
from torch.utils.tensorboard import SummaryWriter

from run.pretrain.segnet.segnet_dataset_compat import SegNetDatasetCompat


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


def run(_):
    use_cudnn()

    run_name = "segnet-train-all"
    BATCH_SIZE = 60

    log_writer = SummaryWriter(log_dir=f".checkpoint/{run_name}/logs")

    log_writer.add_text("description", "对全部数据集进行训练目的在于补全缺失的segment")

    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    new_dataset = SegNetDatasetCompat(
        base_path=".cache/dataset/sam-dataset",
        device=device,
    )
    # new_dataset.split_systole_distole_as_train_valid()

    model: SegNet = SegNet(1, 1).to(device)
    calc_dice_score = DiceScore()
    calc_iou_loss = torch.nn.MSELoss()

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

    optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-5)
    for epoch in range(9001, 19001):
        model.train()
        epoch_losses = []
        dice_losses = []
        iou_losses = []
        for batch in tqdm(new_dataset.shuffer_train_id(BATCH_SIZE)):
            image, labeled_segment = new_dataset.batch_get_train(batch)
            predicted_tensor, _softmaxed_tensor = model(image)
            predicted_tensor = predicted_tensor.sigmoid().squeeze(1)

            iou_loss = calc_iou_loss(predicted_tensor, labeled_segment)
            dice_loss = calc_dice_score(predicted_tensor, labeled_segment)

            loss = iou_loss
            if loss.isnan():
                raise "loss is nan"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            dice_losses.append(dice_loss.item())
            iou_losses.append(iou_loss.item())

        print(f"epoch: {epoch}, loss: {mean(epoch_losses)}")
        log_writer.add_image("train/image", image[0, :, :, :], epoch, dataformats="CHW")
        log_writer.add_image("train/segment_pred", predicted_tensor[0, :, :], epoch, dataformats="HW")
        segment_label = image[0, :, :, :].repeat(3, 1, 1)
        segment_label[1, :, :] += labeled_segment[0, :, :]
        segment_label = segment_label.clamp(0, 1)
        log_writer.add_image("train/segment_label", segment_label, epoch, dataformats="CHW")

        log_writer.add_scalar("train/loss", mean(epoch_losses), epoch)
        log_writer.add_scalar("train/iou", mean(iou_losses), epoch)
        log_writer.add_scalar("train/dice", mean(dice_losses), epoch)

        if epoch % 100 == 0:
            save_model(f".checkpoint/{run_name}/model-{epoch:04}.pth")
            relative_symlink_file(
                f".checkpoint/{run_name}/model-{epoch:04}.pth",
                f".checkpoint/{run_name}/model-latest.pth",
            )

        continue
        valid_losses = []
        model.eval()
        for batch in new_dataset.shuffer_valid_id(BATCH_SIZE):
            (
                image,
                labeled_segment,
            ) = new_dataset.batch_get_valid(batch)
            predicted_tensor, _softmaxed_tensor = model(image)
            predicted_tensor = predicted_tensor.sigmoid().squeeze(1)

            dice_loss = calc_dice_score(predicted_tensor, labeled_segment)
            dice_loss = dice_loss.item()

            valid_losses.append(dice_loss)

        log_writer.add_image("valid/image", image[0, :, :, :], epoch, dataformats="CHW")

        segment_label = image[0, :, :, :].repeat(3, 1, 1)
        segment_label[1, :, :] += labeled_segment[0, :, :]
        segment_label = segment_label.clamp(0, 1)
        log_writer.add_image("valid/segment_label", segment_label, epoch, dataformats="CHW")

        log_writer.add_image("valid/segment_pred", predicted_tensor[0, :, :], epoch, dataformats="HW")
        log_writer.add_scalar("valid/dice", mean(valid_losses), epoch)
