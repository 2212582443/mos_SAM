import os, torch, glob

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

convert_tensor = transforms.ToTensor()


def open_image(path):
    return convert_tensor(Image.open(path))


def find_best_epoch(experiment_root: str) -> str:
    best_dice, best_epoch = 0, ""

    metric_file_list = glob.glob("epoch-*/valid-metrics.pt", root_dir=experiment_root)
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


def run(args: list[str] | None):
    plot_ours()
    plot_baseline()


def plot_baseline():
    model_list = []
    for aux in ["none", "acdc,mnms"]:
        for dim in ["2d", "3d"]:
            for m in [
                "resunet",
                "resunet++",
                "segnet",
                "unet",
                "unet[s]",
                "unetr",
                "sam",
                "effsams",
                "effsamti",
                "txtsam[ps]",
            ]:
                if aux == "acdc,mnms" and ("[s]" in m or "[ps]" in m):
                    model_list.append(f"{m}-{dim}-pseudo")
                else:
                    model_list.append(f"{m}-{dim}-{aux}")

    ROOT = ".checkpoint"

    for setting in [
        # "baseline",
        # "baseline-cmri712-1",
        "baseline-cmri712-2",
        # "baseline-cmri712-3",
        # "baseline-cmri712-4",
    ]:
        # 绿色为正确的预测，红色为多余的预测, 黄色为缺失的预测
        plot_all_image(model_list, ROOT, setting, "test_metric_only[image+label][overlap].png", 4, 10, False, "dice")


def plot_ours():
    # fmt: off
    model_list = [
        "txtsam-2d-none", "txtsam-3d-none", "txtsam-2d-acdc,mnms", "txtsam-3d-acdc,mnms",
        "txtsam[p]-2d-none", "txtsam[p]-3d-none", "txtsam[p]-2d-acdc,mnms", "txtsam[p]-3d-acdc,mnms",
        "txtsam[s]-2d-none", "txtsam[s]-3d-none", "txtsam[s]-2d-pseudo", "txtsam[s]-3d-pseudo",
        "txtsam[ps]-2d-none", "txtsam[ps]-3d-none", "txtsam[ps]-2d-pseudo", "txtsam[ps]-3d-pseudo",
    ]
    # fmt: on

    ROOT = ".checkpoint"

    for setting in [
        # "baseline",
        # "baseline-cmri712-1",
        "baseline-cmri712-2",
        # "baseline-cmri712-3",
        # "baseline-cmri712-4",
    ]:
        # plot_all_image(model_list, ROOT, setting, "test_metric_only[image+label].png", 4, 4, add_metric="dice")
        # 绿色为正确的预测，红色为多余的预测, 黄色为缺失的预测
        plot_all_image(
            model_list, ROOT, setting, "test_metric_only[image+label][overlap].png", 2, 8, False, add_metric="dice"
        )


def plot_all_image(model_list, ROOT, setting, image_file_name, row=4, col=4, append_gt=True, add_metric=None):
    assert row * col == len(model_list)

    test_image = None
    image_list = []
    metric_list = []
    for model in model_list:
        file = f"{ROOT}/{setting}/{model}"
        file = find_best_epoch(file)
        if file == "":
            print("best metric NOT found!", file)
            continue
        if test_image is None:
            test_image = open_image(f"{file}/test[image+label].png")
        image = open_image(f"{file}/{image_file_name}")
        image_list.append(image)
        if add_metric is not None:
            metric = torch.load(f"{file}/test-metrics.pt")[add_metric]
            metric = metric.unsqueeze(0).repeat(16, 1)
            metric_list.append(metric)

    image_list = torch.stack(image_list, dim=0)
    image_list = image_list.reshape(row, col, 3, 16, 128, -1, 128)
    image_list = image_list.permute(5, 3, 2, 0, 4, 1, 6)
    image_list = image_list.reshape(-1, 3, row * 128, col * 128)

    test_image = test_image.unsqueeze(0).repeat(row, 1, 1, 1)
    test_image = test_image.reshape(row, 3, 16, 128, -1, 128)
    test_image = test_image.permute(4, 2, 1, 0, 3, 5)
    test_image = test_image.reshape(-1, 3, row * 128, 128)

    mask = (image_list[:, 0] - image_list[:, 1]).abs().sum(dim=(1, 2)) > 0
    mask += (image_list[:, 1] - image_list[:, 2]).abs().sum(dim=(1, 2)) > 0
    image_list = image_list[mask]
    test_image = test_image[mask]

    if add_metric is not None:
        metric_list = torch.stack(metric_list, dim=0)
        metric_list = metric_list.reshape(row, col, 16, -1)
        metric_list = metric_list.permute(3, 2, 0, 1)
        metric_list = metric_list.reshape(-1, row, col)
        metric_list = metric_list[mask]
    else:
        metric_list = torch.zeros((image_list.shape[0], row, col))

    os.makedirs(f".checkpoint/summary/images/{row}-{col}/{setting}", exist_ok=True)
    for i, (file, gt, metric) in enumerate(
        zip(image_list.split(1, 0), test_image.split(1, 0), metric_list.split(1, 0))
    ):
        file, gt = file.squeeze(0), gt.squeeze(0)
        if append_gt:
            file = torch.cat([file, gt], dim=2)
        image = to_pil_image(file)
        if add_metric is not None:
            metric = metric.squeeze(0)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 24)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 24)

            for r in range(row):
                for c in range(col):
                    value = metric[r, c].item() * 100
                    value = f"{value:3.1f}"
                    position = (c * 128 + 6, r * 128 + 128 - 25)
                    bbox = draw.textbbox(position, value, font=font)
                    draw.rectangle(bbox, fill=(0, 0, 0))
                    draw.text(position, value, font=font, fill="#3CFD4F")

        image.save(f".checkpoint/summary/images/{row}-{col}/{setting}/image_{i:003}.png")
