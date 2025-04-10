import matplotlib.pyplot as plt
import torch, glob, os

from mos.utils.tensors import load_tensor_file, save_tensor_file


def make_pseudo_dataset_init():
    # 生成pseudo数据集
    acdc = torch.load(".cache/dataset/baseline/baseline-acdc.pt", map_location="cpu")
    mnms = torch.load(".cache/dataset/baseline/baseline-mnms.pt", map_location="cpu")

    train2d = torch.cat([acdc["train2d"], mnms["train2d"]], dim=0)
    train3d = torch.cat([acdc["train3d"], mnms["train3d"]], dim=0)
    train3daxis = torch.cat([acdc["train3daxis"], mnms["train3daxis"]], dim=0)
    center2d_yx = torch.cat([acdc["center2d_yx"], mnms["center2d_yx"]], dim=0)
    crop2d_yx = torch.cat([acdc["crop2d_yx"], mnms["crop2d_yx"]], dim=0)
    center3d_yx = torch.cat([acdc["center3d_yx"], mnms["center3d_yx"]], dim=0)
    crop3d_yx = torch.cat([acdc["crop3d_yx"], mnms["crop3d_yx"]], dim=0)
    crop3d = torch.cat([acdc["crop3d"], mnms["crop3d"]], dim=0)

    torch.save(
        {
            "center2d_yx": center2d_yx,
            "crop2d_yx": crop2d_yx,
            "center3d_yx": center3d_yx,
            "crop3d_yx": crop3d_yx,
            "crop3d": crop3d,
            "crop3daxis": train3daxis,
            "train3d_raw": train3d,
            "train2d": train2d[0:0, ::],
            "train3d": train3d[0:0, ::],
            "train3daxis": train3daxis[0:0, ::],
        },
        ".cache/dataset/baseline/baseline-pseudo.pt",
    )


def get_mamual_preset():
    manuals = [
        [i * 2 for i in range(0, 5)],
        [i * 2 for i in range(5, 10)],
    ]
    for i in range(2, 20):
        item = [j * 2 + 1 for j in range(i * 5 - 5 - 5, i * 5 - 5)]  # 上上一批的手动修复学习结果确认
        item += [j * 2 for j in range(i * 5, i * 5 + 5)]  # 这一批待处理的标注
        manuals.append(item)
    return manuals


def gen_pseudo(args):
    # 生成acdc和mnms的pseudo

    device = "cpu"

    db = torch.load(".cache/dataset/baseline/baseline-pseudo.pt", map_location=device)
    train3d = db["crop3d"]
    bs, _, d, h, w = train3d.shape

    crop_image, crop_label = train3d[:, 0, :, :, :], train3d[:, 1, :, :, :]

    print(crop_image.shape, crop_label.shape)
    crop_image: torch.Tensor = crop_image.reshape(-1, 128, 128)
    crop_label = crop_label.reshape(-1, 128, 128)

    valiable_label_mask = crop_label.sum(dim=(1, 2)) > 0

    src_root_dir = ".checkpoint/pseudo/aux"
    file_list = []
    weights = []
    for pseudo_file in glob.glob("*.pt", root_dir=src_root_dir):
        dice = float(pseudo_file.split("@")[0])
        file_list.append(pseudo_file)
        weights.append(dice)

    weights = torch.tensor(weights).softmax(dim=0).tolist()

    onehot_pseudo_labels = None

    for i, file in enumerate(file_list):
        print(file)
        pseudo = torch.load(os.path.join(src_root_dir, file), map_location=device).float()
        if onehot_pseudo_labels is None:
            onehot_pseudo_labels = pseudo
        else:
            onehot_pseudo_labels += pseudo * weights[i]
    onehot_pseudo_labels = (onehot_pseudo_labels > 0.5).int()

    # onehot to value
    onehot_pseudo_labels[:, 0] = 0
    onehot_pseudo_labels[:, 1] *= 4
    onehot_pseudo_labels[:, 2] *= 3
    onehot_pseudo_labels[:, 3] *= 2
    onehot_pseudo_labels[:, 4] *= 1
    pseudo_labels = onehot_pseudo_labels.argmax(dim=1)

    pseudo_labels = pseudo_labels.reshape(-1, 128, 128)
    pseudo_labels[pseudo_labels > 1] = 0  # 去掉 LV,RV,MYO
    pseudo_labels[crop_label > 0] = 0
    pseudo_labels += crop_label
    pseudo_labels = pseudo_labels.to(torch.uint8)
    print(pseudo_labels.unique(), pseudo_labels.shape)

    pseudo_labels_nii = pseudo_labels[valiable_label_mask]
    pseudo_labels_nii = pseudo_labels_nii.permute(1, 2, 0)
    pseudo_image_nii = crop_image[valiable_label_mask]
    pseudo_image_nii = pseudo_image_nii.permute(1, 2, 0)
    save_tensor_file(pseudo_image_nii.cpu(), f"{src_root_dir}/image.nii.gz")
    save_tensor_file(pseudo_labels_nii.cpu(), f"{src_root_dir}/label.nii.gz")

    # 生成下一个要手动修复的数据
    valiable_label_mask_3d = valiable_label_mask.reshape(bs, d)
    valiable_label_mask_3d_index = torch.arange(bs * d, device=device).reshape(bs, d)
    manuals = get_mamual_preset()
    print(manuals)
    # find latest id
    tmp_ids: list[str] = glob.glob("merged/label-*.nii.gz", root_dir=src_root_dir)
    tmp_ids.sort(reverse=True)
    if len(tmp_ids) > 0:
        current_id = int(tmp_ids[0].split("-")[1].split(".")[0]) + 1
    else:
        current_id = 0
    print("生成待标注数据集", current_id)
    current_valiable_label = valiable_label_mask_3d[manuals[current_id]].reshape(-1)
    current_valiable_label = valiable_label_mask_3d_index[manuals[current_id]].reshape(-1)[current_valiable_label]
    pseudo_labels_nii = pseudo_labels.index_select(0, current_valiable_label)
    pseudo_labels_nii = pseudo_labels_nii.permute(1, 2, 0)
    pseudo_image_nii = crop_image.index_select(0, current_valiable_label)
    pseudo_image_nii = pseudo_image_nii.permute(1, 2, 0)
    save_tensor_file(pseudo_image_nii.cpu(), f"{src_root_dir}/image-{current_id:03}.nii.gz")
    save_tensor_file(pseudo_labels_nii.cpu(), f"{src_root_dir}/label-{current_id:03}.nii.gz")


def merge_manul_label():
    device = "cpu"
    # merge manual label
    src_root_dir = ".checkpoint/pseudo/aux/merged"
    labels: list[str] = glob.glob("label-*.nii.gz", root_dir=src_root_dir)
    labels.sort()
    if len(labels) == 0:
        print("没有待合并的标注数据集")
        return

    db = torch.load(".cache/dataset/baseline/baseline-pseudo.pt", map_location=device)

    crop3d_label = db["crop3d"][:, 1, ::]
    bs, d, h, w = crop3d_label.shape
    valiable_label_mask = crop3d_label.reshape(-1, h, w).sum(dim=(1, 2)) > 0
    valiable_label_mask_3d = valiable_label_mask.reshape(bs, d)
    manuals = get_mamual_preset()

    crop3daxis, crop3d_yx, train3d_raw = db["crop3daxis"], db["crop3d_yx"], db["train3d_raw"]
    train2d, train3d, train3daxis = [], [], []
    for label in labels:
        print(f"合并标注数据集 {label}")
        current_id = int(label.split("-")[1].split(".")[0])
        subject_ids = manuals[current_id]
        # (n, h, w)
        label = load_tensor_file(f"{src_root_dir}/{label}")[""].permute(2, 0, 1)
        current_label_index = 0
        label = (label == 1).int()  # 只保留EAT

        for yx, axis, pair, flag in zip(
            crop3d_yx[subject_ids].split(1),
            crop3daxis[subject_ids].split(1),
            train3d_raw[subject_ids].split(1),
            valiable_label_mask_3d[subject_ids].split(1),
        ):
            top, left = yx[0, 0], yx[0, 1]
            bottom, right = top + 128, left + 128
            axis = axis.squeeze(0)
            pair = pair.squeeze(0)  # (2, d, h ,w)
            train3daxis.append(axis)
            flag = flag.squeeze(0).tolist()

            for slice in range(d):
                if flag[slice] == 0:
                    continue
                mask = pair[1, slice, top:bottom, left:right]
                label[current_label_index][mask > 0] = 0
                pair[1, slice, top:bottom, left:right] += label[current_label_index]
                current_label_index += 1
                train2d.append(pair[:, slice, ::])

            train3d.append(pair)

    train2d = torch.stack(train2d, dim=0).to(torch.uint8)
    train3d = torch.stack(train3d, dim=0).to(torch.uint8)
    train3daxis = torch.stack(train3daxis, dim=0).to(torch.uint8)
    print(train2d.shape, train3d.shape, train3daxis.shape)

    db["train2d"] = train2d
    db["train3d"] = train3d
    db["train3daxis"] = train3daxis

    torch.save(db, ".cache/dataset/baseline/baseline-pseudo.pt")
