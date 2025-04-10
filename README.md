# MOS
Medical Open Source, 医学影像分析开源代码.

## 代码下载和运行
---
<!-- 
本库使用git lfs, 需要提前安装好git lfs, 请参考[git lfs](https://git-lfs.github.com/), 
使用文档: [配置 Git Large File Storage](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)

```bash
sudo apt-get install git-lfs
git lfs install
```
 -->

本项目需要python3.10以上版本, 请先安装python3.10 或 3.11
```bash
conda create -n py311 python=3.11 -c conda-forge -y

conda activate py311

conda install -y -c conda-forge jupyterlab notebook voila jupyterlab-link-share jupyterlab-language-pack-zh-CN

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade

pip3 install -r requirements.txt
pip3 install -r requirements_dev.txt
```



## 编码规范
---
+ 本项目使用google的命名规范, 请参考[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) OR [中文版](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)
+ 编辑代码时, 请使用vscode, 并启用black自动格式化工具
+ 请使用python3.10以上版本, 更高的版本对性能有更大提升, 更好的消除训练时python性能瓶颈
+ 每个函数参数需要指定类型注释, 请参考[typing](https://docs.python.org/3/library/typing.html), 无需注解模块中的所有函数.
    + 至少需要注解你的公开 API.
    + 你可以自行权衡, 一方面要保证代码的安全性和清晰性, 另一方面要兼顾灵活性.
    + 应该注解那些容易出现类型错误的代码 (比如曾经出现过错误或疑难杂症).
    + 应该注解晦涩难懂的代码.
    + 应该注解那些类型已经确定的代码. 多数情况下，即使注解了成熟的代码中所有的函数，也不会丧失太多灵活性.

+ vscode 设置每行字符数 (已经自动设置，无需下面操作)
  - 安装格式化插件 
    `> ext install aslamanver.vsc-export`
    `> ext install ms-python.black-formatter`
    `> vsc extensions import`
  - 启用 Format on Save
  - 设置 format with black

## 医学图像坐标系统
---
MRI图像坐标系统遵循 LPS 坐标系, 请参考[坐标系1](https://www.slicer.org/wiki/Coordinate_systems) 

LPS = Left-Posterior-Superior, 也就是左手系, 也就是左手拇指指向左边, 食指指向前方, 中指指向上方, 请参考[坐标系2](https://theaisummer.com/medical-image-coordinates/):
x = from right towards left 
y = from anterior towards posterior 
z = from inferior towards superior 
​
对于心脏MRI图像需要注意: 心脏MRI扫描前需要医生先进行定位操作, 扫描的方位以心脏的位置和方向为准, 每个人的心脏位置和方向都不一样, 会有所差异, 导致扫描的图像角度看起来都不大一样, 比较难进行配准和对齐. 这是正常的. [心脏MRI解剖结构](https://www.imaios.com/cn/e-anatomy/1/5)
[心脏定位以及其他心脏MRI基本知识](https://mp.weixin.qq.com/s?__biz=MzI1NzU4Njg2OQ==&mid=2247485105&idx=1&sn=d6bc9c3943fa6eb01e86e31a0142ee20&chksm=ea14656bdd63ec7dc7aa995d6d7d89e907fc4579843132cd8b3311eaa9c053c3ce6d13934695&cur_album_id=1319709675184783360&scene=189#wechat_redirect) 
[MRI规范化扫描](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI1NzU4Njg2OQ==&action=getalbum&album_id=1319709675184783360&scene=173&from_msgid=2247495068&from_itemidx=1&count=3&nolastread=1#wechat_redirect)

x,y,z,t

ch, d, h, w -> ch, z, y, x


## 数据格式约定
---
所有函数的出参和入参shape都遵循以下规则, 对应关系如下, 以下名称对应同一意思

### **图像格式**:

1d:
  - (channel, width)
  - (ch, w)
  - (ch, i)
  - (ch, x)

2d: 
  - (channel, height, width)
  - (ch, h, w)
  - (ch, i, j)
  - (ch, y, x)

3d: 
  - (channel, deep, height, width)
  - (ch, d, h, w)
  - (ch, i, j, k)
  - (ch, z, y, x)

### **Segment格式**:
2d:
  - (bs, h, w)
  - (bs, label_count, h, w) # one-hot
3d:
  - (bs, d, h, w)
  - (bs, label_count, d, h, w) # one-hot



### **向量场(配准等)**:

  - 2d ([xy] , h, w)
  - 3d ([xyz], d, h, w)

### **对于NLP来说, 遵循以下规则**:

1d: 
  - (seq_lens, e_dim)

> 注意: 
>
> 1. 省略了batch_size, 但是在代码中实际上是有batch_size, 除了以下情况, 其他情况都需要加上batch_size
>   - 1. dataset的tensor不包含bs
> 2. 如果格式和以上定义不一致, 需要在函数注释中特别说明

## 目录结构
```txt
.cache/      训练数据
.checkpoint/ 训练结果
.scripts/    训练启动脚本
mos/         公共类库
run/         每个训练模型代码
tests/       测试目录
``` 

### baseline 训练例子
```python
cd 当前项目根目录
.scripts/baseline/unet-2d-none
.scripts/baseline/unet-2d-none --note '这是3d的模型，测试'
# 脚本中设置了一些默认值，可以在环境变量中改变默认值，
# 如，跑1个epoch，batch size=1可以设置环境变量COUTN=100 BS=1
COUNT=100 BS=1 .scripts/baseline/unet-2d-none
# 也可以在脚本后面加上参数，覆盖脚本的默认值
# 如强制再跑10个epoch
.script/baseline/unet-2d-none --increase_epoch 10
# 使用辅助数据集oasis(此时目录后缀为oasis,可以在脚本中直接修改此值)
.script/baseline/unet-2d-acdc,mnms --aux_dataset oasis
# 跑zeroshot(具体需求请修改下面脚本)
.script/baseline/cmd/zeroshot
# 交互式生成和学习伪标签(具体见脚本)
.script/baseline/cmd/pseudo-gen
# 添加了新的指标，重新生成所有训练的新指标数据(具体见脚本)
.script/baseline/cmd/re_eval
# 输出所有的训练报表到result.html
# 包含论文需要的latex表格（在.checkpoint/summary/目录下）
.script/baseline/cmd/summary
# 把模型最优的结果导出为图片
.script/baseline/cmd/plot-test
# 把模型消融实验的最优结果图片合并为4x4/2x8等组合的图片(可以附加各种指标)，方便对比（论文插图需要用到）
.script/baseline/cmd/summary-image

# =====预训练模型======
# 只使用辅助数据集进行训练（主数据集cmri712-1,辅助数据集acdc和mnms，目录后缀[pretrain]）)
SETTING=cmri712-1 .scripts/baseline/resunet-2d-acdc,mnms --dir_postfix [pretrain] --aux_dataset_only
# # PRETRAIN2D='.checkpoint/baseline-cmri712-1/resunet-2d-acdc,mnms[pretrain]/latest'
# 从预训练模型中加载模型，并开始训练
SETTING=cmri712-1 COUNT=100 .scripts/baseline/resunet-2d-acdc,mnms  \
  --note '从预训模型开始' \
  --pretrain_model_path '.checkpoint/baseline-cmri712-1/resunet-2d-acdc,mnms[pretrain]/latest'


```

## baseline基准模型详解
### 数据集的准备
数据集存放位置： `.cache/dataset/baseline/`

1. 训练数据集：
  文件名`baseline-xxx.pt`, 
  ```json
  {
    "train2d": Tensor, // (bs,2,h,w), 2为image+segmentation,uint8
    "train3d": Tensor, // (bs,2,d,h,w), 2为image+segmentation,uint8
    "train3daxis": Tensor|None, // # (bs, d) 每个slice对应的物理坐标, int32
    "valid3d": Tensor, // 同train3d
    "valid3daxis": Tensor|None, // # 同valid3d
    "test3d": Tensor, // 同train3d
    "test3daxis": Tensor|None, // 同train3d
  }
  ```
使用`SETTING=xxx .scripts/baseline/....` 来加载数据集，比如5折交叉验证就需要制作5个数据集。
建议使用`xxx-0`, `xxx-1`, ..., `xxx-4`这样的命名规则，方便后续对训练结果进行统一处理。
对应生成的checkpoint存放在`.checkpoint/xxx/....`下面。

2. 辅助数据集:
  文件名`baseline-xxx.pt`, 
  ```json
  {
    "train2d": Tensor, // (bs,2,h,w), 2为image+segmentation,uint8
    "train3d": Tensor, // (bs,2,d,h,w), 2为image+segmentation,uint8
    "train3daxis": Tensor|None, // # (bs, d) 每个slice对应的物理坐标, int32
  }
  ```

3. Zeroshot数据集:
  文件名`baseline-zeroshot.pt`, 
  ```json
  {
    "crop3d": Tensor, // (bs,2,d,h,w), 2为image+segmentation,uint8, 因为目标位置不确定在图像的什么位置，即不知道要crop什么地方才能代表准确评估，因此需要手动crop3d
    "zeroshot3daxis": Tensor|None, // # (bs, d) 每个slice对应的物理坐标, int32
  }
  ```

Pseudo数据集:
3. Zeroshot数据集:
  文件名`baseline-zeroshot.pt`, key为`pseudo`,可以通过`--pseudo_dataset_key`指定 
  ```json
  {
    "crop3d": Tensor, // (bs,2,d,h,w), 2为image+segmentation,uint8, 因为目标位置不确定在图像的什么位置，即不知道要crop什么地方才能代表准确评估，因此需要手动crop3d
    "crop3daxis": Tensor|None, // # (bs, d) 每个slice对应的物理坐标, int32
  }
  ```

#### 数据集生成demo
```python
# 生成训练数据
# baseline dataset
# images: H,W
# train2d: (bs, [image,label], d, h, w)
# train3d: (bs, [image,label], h, w)
# valid2d: (bs, [image,label], d, h, w)
# valid3d: (bs, [image,label], h, w)
# test2d: (bs, [image,label], d, h, w)
# test3d: (bs, [image,label], h, w)
# train:valid:test = 7:1:2

from typing import List
import os

# EAT = target_modality["EAT"]
from enum import Enum

os.makedirs("1-dataset", exist_ok=True)

settings712 = [
    ([7], [8, 9]),
    ([9], [0, 1]),
    ([1], [2, 3]),
    ([3], [4, 5]),
    ([5], [6, 7]),
    ([], []),  # all
]
labels = glob.glob(f"*.nii.gz", root_dir=label_dir)
# get uid
uids = {}
user_list = []
for label in labels:
    user_name = label.split("_")[1]
    if user_name not in uids:
        uids[user_name] = len(uids)
    uid = uids[user_name]
    user_list.append((uid, label))

IMAGE_SIZE = (192, 160)

for setting_index in range(6):
    valid_set = set(settings712[setting_index][0])
    test_set = set(settings712[setting_index][1])

    image_list: List[torch.Tensor] = []
    train2d_list: List[torch.Tensor] = []
    train3d_list: List[torch.Tensor] = []
    valid2d_list: List[torch.Tensor] = []
    valid3d_list: List[torch.Tensor] = []
    test2d_list: List[torch.Tensor] = []
    test3d_list: List[torch.Tensor] = []

    train3d_axis_list: List[torch.Tensor] = []
    valid3d_axis_list: List[torch.Tensor] = []
    test3d_axis_list: List[torch.Tensor] = []

    for uid, file_name in user_list:
        meta = read_dicom_meta(f"{image_dir}/{file_name}")
        dim = int(meta["dim[3]"]), int(meta["dim[1]"]), int(meta["dim[2]"])
        spacing = (
            float(meta["pixdim[3]"]),
            float(meta["pixdim[1]"]),
            float(meta["pixdim[2]"]),
        )
        scale = spacing[1]
        vspacing, vcount = spacing[0], dim[0]

        axis = torch.arange(16, dtype=torch.float) * vspacing

        images = read_dicom_file(f"{image_dir}/{file_name}")
        segments = read_dicom_file(f"{label_dir}/{file_name}")
        images = normalize_image_size(images, scale, True, IMAGE_SIZE)
        images = (images * 255).clamp(0, 255).to(torch.uint8)

        segments = normalize_segment_size(segments, scale, IMAGE_SIZE)
        segments = (segments > 0).to(torch.uint8)

        print(images.shape, segments.shape)

        d, h, w = segments.shape
        if d >= 16:
            segments = segments[-16:]
            images = images[-16:]
        else:
            segments = torch.cat([segments, torch.zeros(16 - d, h, w)])
            images = torch.cat([images, torch.zeros(16 - d, h, w)])
            axis[vcount:] = 28  # index为填充

        partition = uid % 10
        if partition in test_set:
            test3d_list.append(torch.stack([images, segments]))
            test3d_axis_list.append(axis)
        elif partition in valid_set:
            valid3d_list.append(torch.stack([images, segments]))
            valid3d_axis_list.append(axis)
        else:
            train3d_list.append(torch.stack([images, segments]))
            train3d_axis_list.append(axis)

        for segment, image in zip(segments.split(1, 0), images.split(1, 0)):
            if segment.max() < 1:
                continue
            if partition in test_set:
                test2d_list.append(torch.cat([image, segment]))
            elif partition in valid_set:
                valid2d_list.append(torch.cat([image, segment]))
            else:
                train2d_list.append(torch.cat([image, segment]))

    train2d_list = torch.stack(train2d_list).to(torch.uint8)
    train3d_list = torch.stack(train3d_list).to(torch.uint8)
    valid2d_list = (
        torch.stack(valid2d_list).to(torch.uint8)
        if len(valid2d_list) > 0
        else torch.zeros(0, 2, 16, IMAGE_SIZE15, IMAGE_SIZE15).to(torch.uint8)
    )
    valid3d_list = (
        torch.stack(valid3d_list).to(torch.uint8)
        if len(valid3d_list) > 0
        else torch.zeros(0, 2, 16, IMAGE_SIZE15, IMAGE_SIZE15).to(torch.uint8)
    )
    test2d_list = (
        torch.stack(test2d_list).to(torch.uint8)
        if len(test2d_list) > 0
        else torch.zeros(0, 2, 16, IMAGE_SIZE15, IMAGE_SIZE15).to(torch.uint8)
    )
    test3d_list = (
        torch.stack(test3d_list).to(torch.uint8)
        if len(test3d_list) > 0
        else torch.zeros(0, 2, 16, IMAGE_SIZE15, IMAGE_SIZE15).to(torch.uint8)
    )
    train3d_axis_list = torch.stack(train3d_axis_list).float()
    valid3d_axis_list = (
        torch.stack(valid3d_axis_list).float()
        if len(valid3d_list) > 0
        else torch.zeros(0, 16).float()
    )
    test3d_axis_list = (
        torch.stack(test3d_axis_list).float()
        if len(test3d_list) > 0
        else torch.zeros(0, 16).float()
    )

    file_tag = "all" if setting_index == 5 else f"{setting_index}"

    torch.save(
        {
            "train2d": train2d_list,
            "train3d": train3d_list,
            "train3daxis": train3d_axis_list,
            "valid2d": valid2d_list,
            "valid3d": valid3d_list,
            "valid3daxis": valid3d_axis_list,
            "test2d": test2d_list,
            "test3d": test3d_list,
            "test3daxis": test3d_axis_list,
        },
        f"1-dataset/baseline-pa-{file_tag}.pt",
    )
```

```python
# convert baseline aux dataset
from typing import List
import torch, numpy as np, json, os

from torchvision.transforms.functional import crop

os.makedirs("1-dataset", exist_ok=True)

IMAGE_SIZE = (192, 160)


train2d_list: List[torch.Tensor] = []
train3d_list: List[torch.Tensor] = []

train3d_axis_list: List[torch.Tensor] = []

center_position_list_2d = []  # [(y,x)]
crop_position_list_2d = []  # [(top, left)]
center_position_list_3d = []  # [(y,x)]
crop_position_list_3d = []  # [(top, left)]
crop3d_list = []
CROP_SIZE = 128

for uid in range(1, 458):
    root = f"oasis/OASIS_OAS1_{uid:04}_MR1"
    if os.path.exists(f"{root}/aligned_orig.nii.gz") is False:
        continue

    meta = read_dicom_meta(f"{root}/aligned_orig.nii.gz")
    scale, vspacing, vcount = (
        float(meta["pixdim[1]"]),
        float(meta["pixdim[3]"]),
        int(meta["dim[3]"]),
    )
    print(f"processing {uid}...{scale}, {vspacing}, {vcount}")
    images = read_dicom_file(f"{root}/aligned_orig.nii.gz")
    segments = read_dicom_file(f"{root}/aligned_seg35.nii.gz")

    images = (images * 255).to(torch.uint8)
    axis = torch.arange(224, dtype=torch.float) * vspacing

    # 0,1,2,3,4..35 -> 0, 2,3,4,5..36
    segments = segments + 1
    segments[segments == 1] = 0
    segments = segments.to(torch.uint8)

    train3d_list.append(torch.stack([images, segments]))
    train3d_axis_list.append(axis)

    for slice, label in zip(images.split(1, 0), segments.split(1, 0)):
        if label.max() == 0:
            continue
        train2d_list.append(torch.cat([slice, label]))


print(len(train2d_list))

train2d_list = torch.stack(train2d_list).to(torch.uint8)
train3d_list = torch.stack(train3d_list).to(torch.uint8)
train3d_axis_list = torch.stack(train3d_axis_list).float()

# 224个slice 按照vspacing=8拆分为多个3d图像
bs, _, d, h, w = train3d_list.shape
# (bs, 2, d, h, w)
train3d_list = train3d_list.split(8, 2)
# (bs, 16, 2, 16, h, 2)
train3d_list = torch.cat(
    [
        # (bs, 16, 2, 8, h, w)
        torch.stack(train3d_list[:16], 1),
        torch.stack(train3d_list[-16:], 1),
    ],
    3,
)
train3d_list = train3d_list.permute(0, 3, 2, 1, 4, 5).reshape(-1, 2, 16, h, w)

# 位置坐标也按照vspacing=8拆分为多个3d图像
train3d_axis_list = train3d_axis_list.split(8, 1)
# (bs, 16, 16)
train3d_axis_list = torch.cat(
    [
        # (bs, 16, 8)
        torch.stack(train3d_axis_list[:16], 1),
        torch.stack(train3d_axis_list[-16:], 1),
    ],
    2,
)
train3d_axis_list: torch.Tensor = train3d_axis_list.permute(0, 2, 1).reshape(-1, 16)
train3d_axis_list = train3d_axis_list - train3d_axis_list.min(1, keepdim=True).values


torch.save(
    {
        "train2d": train2d_list,
        "train3d": train3d_list,
        "train3daxis": train3d_axis_list,
    },
    f"1-dataset/baseline-oasis.pt",
)
```


## baseline 参数说明
- label参数
  + `label_count: 5` # 标签的类别数量
  + `dataset_train_labels:1` # 训练数据集中的标签id，逗号分隔
  + `dataset_eval_labels:1` # 评估数据集中的标签id，逗号分隔
  + `aux_dataset_labels:2,3,4` # 辅助数据集中的标签id，逗号分隔
  + `zeroshot_dataset_labels:2,3,4` # zeroshot数据集中的标签id，逗号分隔
  + `pseudo_output_labels:1-4` # 伪标签输出的标签id，逗号分隔
  + 标签可以用'-'来表示范围，如`1,3-5,8`表示`1,3,4,5,8`

- 图像大小参数
  + `dataset_image_size: 214,214` # 数据集中图像的大小(只需要h,w)
  + 模型中的输入大小
    - `dataset_crop_size: 128,128` # for 3d (h,w),把输入crop到指定大小
    - `dataset_crop_size: 16,128,128` # for 3d (d,h,w)，把输入crop到指定大小, 随机裁切
    - `dataset_crop_deep_maxscale: 1` # for 3d，deep维度的裁切跨度大小，1表示连续裁切，2表示可以跳着1个像素裁切，以此类推
    - `dataset_valid_crop_size: 8,128,128` 
    - `dataset_valid_crop_deep_maxscale: 1`
    - `dataset_spacing:10,1.5,1.5` # 数据图像的分辨率，对应d/h/w方向，用于计算HD95距离

- 训练次数相关参数
  + `start_epoch:1` 开始训练的epoch
  + `sample_count:0` 训练样本采样的数量，0为使用全部的数据
  + `increase_epoch:None` 不管当前是哪个epoch，只增量训练指定的epoch个数,正数表示增加，负数表示重新训练最后几个，fix表示训练到最近整10的epoch

更多参数请见文件：run/baseline/model_arguments.py

## 保存的评估指标说明
- checkopint各epoch目录下，valid-metrics.pt & test-metrics.pt
  - 目前包含msd,hd,dice,iou,epoch_losses这几个指标的所有label平均值(bs)
  - 原始值在raw_msd, raw_hd, raw_dice, raw_iou里面，包含各指标具体的值(bs, label_count)
  - 为了方便t-test或者后续分析，这些值的顺序根据数据集中的顺序依次出现，并且顺序保持不变。

## 代码性能问题分析

1. 启用`--use_profile` 参数用于在代码运行时收集profile分析数据，运行5个epoch
2. `pip install torch_tb_profiler`
2. 在服务器上使用命令 `tensorboard --logdir .checkpoint/xxxx-xmodel-path --port 6003 --window_title $HOST ` 以打开`tensorboard`服务
3. 在chrome浏览器(不支持Safari)中打开 http://127.0.0.1:6003/#pytorch_profiler

注：
- 最好使用小数据集进行profile，否则后处理这些profile数据会很久，而且查看跟踪堆栈的时候网页很可能也会崩掉。 
- 或者搭配`--sample_count xx`进行使用，减少重复收集的数据数量。
- 另外profile会占用大量的存储空间，不建议长期启用。
- 通过profile界面的`Trace`和`Module`的时间可以找出最耗时的代码进行优化。
