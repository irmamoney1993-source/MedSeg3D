"""Train MONAI SegResNet with the 3D U-Net baseline protocol.

Fixed 387/97 split, 64^3 patches, Dice + CE, Adam 1e-4, 60 epochs.
RUN_MODE selects the synthetic smoke check or full training.
"""

# 导入库

from pathlib import Path

import sys
import json
import csv
import time
import math
import random

import numpy as np
import torch

from monai.data import (
    PersistentDataset,
    DataLoader,
)

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
)

from monai.losses import DiceLoss

from monai.metrics import DiceMetric

from monai.inferers import sliding_window_inference

from monai.networks.utils import one_hot

from monai.utils import set_determinism


# 项目根目录

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from monai.networks.nets import SegResNet


# 正式实验参数

# "smoke"：只测试一次模型训练和推理，不读取数据集、不创建权重。
# "train"：从Epoch 1正式训练到Epoch 60。
RUN_MODE = "smoke"

# 独立实验名称
RUN_NAME = "segresnet_baseline_60_fast"

# SegResNet结构对照实验从随机初始化训练60轮
NUM_EPOCHS = 60

# 首次正式训练，从随机初始化开始
RESUME_TRAINING = False

# 固定随机种子
SEED = 42

# 输入Patch大小
PATCH_SIZE = (64, 64, 64)

# 训练Batch Size
BATCH_SIZE = 1

# 每次病例采样一个Patch
NUM_SAMPLES = 1

# 模型基础通道数
INIT_FILTERS = 8

# Adam学习率
LEARNING_RATE = 1e-4

# 核心实验变量
# 本对照实验所有CE类别权重均为1。
#
# Class 0：背景
# Class 1：水肿
# Class 2：非增强肿瘤
# Class 3：增强肿瘤

CLASS_WEIGHTS = [
    1.0,
    1.0,
    1.0,
    1.0,
]

# Sliding Window参数

ROI_SIZE = (64, 64, 64)

# 每次并行预测4个窗口
SW_BATCH_SIZE = 4

OVERLAP = 0.25

# DataLoader性能优化

NUM_WORKERS = 2

PREFETCH_FACTOR = 2

# SSD缓存版本
# 只有预处理定义未变化时，
# 才复用已有的preprocess_v1缓存。

CACHE_VERSION = "preprocess_v1"


# 项目路径

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Task01_BrainTumour"
)

IMAGES_DIR = (
    DATASET_DIR
    / "imagesTr"
)

LABELS_DIR = (
    DATASET_DIR
    / "labelsTr"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

SPLIT_PATH = (
    OUTPUT_DIR
    / "splits"
    / "train_val_split.json"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / CACHE_VERSION
)

TRAIN_CACHE_DIR = (
    CACHE_DIR
    / "train"
)

VAL_CACHE_DIR = (
    CACHE_DIR
    / "val"
)


# 构建病例文件列表

def build_cases(names):

    cases = []

    for name in names:

        image_path = (
            IMAGES_DIR
            / name
        )

        label_path = (
            LABELS_DIR
            / name
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"找不到MRI文件：{image_path}"
            )

        if not label_path.exists():

            raise FileNotFoundError(
                f"找不到Label文件：{label_path}"
            )

        cases.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "name": name,
            }
        )

    return cases


# 构建训练/验证预处理
#
# 训练和验证共享相同的确定性变换：
#
# Load
# → EnsureChannelFirst
# → Orientation
# → Spacing
# → CropForeground
# → SpatialPad
# → NormalizeIntensity
#
# 只有训练阶段额外执行随机Patch采样。
#
# PersistentDataset将随机变换之前的
# 确定性预处理结果写入磁盘缓存。
#
# 随机Patch不会被固定在缓存中。

def build_transforms(training=True):

    transforms = [

        # 读取NIfTI

        LoadImaged(
            keys=[
                "image",
                "label",
            ]
        ),

        # MRI最后一维为4个模态
        # 统一变成[C, D, H, W]

        EnsureChannelFirstd(
            keys=["image"],
            channel_dim=-1
        ),

        # 标签没有原始通道维度

        EnsureChannelFirstd(
            keys=["label"],
            channel_dim="no_channel"
        ),

        # 统一空间方向为RAS

        Orientationd(
            keys=[
                "image",
                "label",
            ],

            axcodes="RAS",

            labels=(
                ("L", "R"),
                ("P", "A"),
                ("I", "S"),
            ),
        ),

        # 统一体素间距为1mm
        # MRI使用双线性/三线性空间插值
        # Label使用最近邻插值

        Spacingd(
            keys=[
                "image",
                "label",
            ],

            pixdim=(
                1.0,
                1.0,
                1.0,
            ),

            mode=(
                "bilinear",
                "nearest",
            ),
        ),

        # 裁剪MRI前景

        CropForegroundd(
            keys=[
                "image",
                "label",
            ],

            source_key="image"
        ),

        # 保证空间尺寸不小于64³

        SpatialPadd(
            keys=[
                "image",
                "label",
            ],

            spatial_size=PATCH_SIZE
        ),

        # 每个MRI模态分别Z-score标准化

        NormalizeIntensityd(
            keys=["image"],

            nonzero=True,

            channel_wise=True
        ),
    ]

    # 训练阶段：添加随机Patch采样

    if training:

        transforms.append(

            RandCropByPosNegLabeld(

                keys=[
                    "image",
                    "label",
                ],

                label_key="label",

                spatial_size=PATCH_SIZE,

                pos=1,

                neg=1,

                num_samples=NUM_SAMPLES
            )
        )

    return Compose(transforms)


# 构建DataLoader

def create_loader(dataset, training):

    options = {

        "dataset": dataset,

        "batch_size": (
            BATCH_SIZE
            if training
            else 1
        ),

        "shuffle": training,

        "num_workers": NUM_WORKERS,

        "pin_memory": torch.cuda.is_available(),
    }

    # Windows多进程优化

    if NUM_WORKERS > 0:

        options["persistent_workers"] = True

        options["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(**options)


# Loss单元测试
#
#
# 测试完成以后，main中重新固定随机种子，
# 避免随机测试数据影响正式模型初始化。

def test_loss_function(
    dice_loss,
    ce_loss,
    device
):

    # 创建随机Logits
    #
    # [Batch, Classes, D, H, W]

    test_logits = torch.randn(
        1,
        4,
        8,
        8,
        8,
        device=device
    )

    # 创建随机标签
    #
    # [Batch, 1, D, H, W]

    test_labels = torch.randint(
        low=0,
        high=4,

        size=(
            1,
            1,
            8,
            8,
            8,
        ),

        device=device
    )

    # 只检查Loss计算，不进行反向传播

    with torch.no_grad():

        test_dice = dice_loss(
            test_logits,
            test_labels
        )

        test_ce = ce_loss(
            test_logits,
            test_labels.squeeze(1).long()
        )

        test_total = (
            test_dice
            +
            test_ce
        )

    if not torch.isfinite(test_total).item():

        raise RuntimeError(
            "Loss单元测试失败：出现NaN或Inf"
        )

    print("\nLoss单元测试通过")

    print(
        "测试Dice Loss：",
        f"{test_dice.item():.6f}"
    )

    print(
        "测试CE：",
        f"{test_ce.item():.6f}"
    )

    print(
        "测试总Loss：",
        f"{test_total.item():.6f}"
    )


# 模型结构和本机显存检查

def build_segresnet_model():
    """保持独立、固定的SegResNet结构，避免误用自建UNet。"""
    return SegResNet(
        spatial_dims=3,
        init_filters=INIT_FILTERS,
        in_channels=4,
        out_channels=4,
        dropout_prob=None,
        norm=("GROUP", {"num_groups": 8}),
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        upsample_mode="nontrainable",
    )


def save_checkpoint_atomic(checkpoint_data, destination):
    """先写临时文件，完成后替换，降低中断导致Checkpoint损坏的风险。"""
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint_data, temp_path)
    temp_path.replace(destination)


def run_smoke_test():
    """合成数据单步训练+滑动窗口测试；不触碰真实数据或历史实验文件。"""
    print("=" * 62)
    print("SegResNet-60 烟雾测试（只测模型，不启动正式训练）")
    print("=" * 62)
    random.seed(SEED)
    np.random.seed(SEED)
    set_determinism(seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", device)
    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    model = build_segresnet_model().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print("模型：SegResNet")
    print("参数量：", parameters)
    print("输入Patch：", PATCH_SIZE)
    print("训练输入和标签：", (1, 4, *PATCH_SIZE), (1, 1, *PATCH_SIZE))

    # FP32、1个训练Patch，检查Forward + Loss + Backward + Adam。
    model.train()
    x = torch.randn((1, 4, *PATCH_SIZE), device=device)
    y = torch.randint(0, 4, (1, 1, *PATCH_SIZE), device=device)
    weights = torch.ones(4, dtype=torch.float32, device=device)
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    ce_loss = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    assert tuple(logits.shape) == (1, 4, *PATCH_SIZE), logits.shape
    loss = dice_loss(logits, y) + ce_loss(logits, y[:, 0])
    if not torch.isfinite(loss).item():
        raise RuntimeError("烟雾测试：Loss出现NaN/Inf")
    loss.backward()
    optimizer.step()
    print("训练单步通过，Loss：", f"{loss.item():.6f}")

    del x, y, logits, loss
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        # 112×112×64 会产生多个64³窗口，实际验证sw_batch_size=4。
        val_input = torch.zeros((1, 4, 112, 112, 64), device=device)
        val_output = sliding_window_inference(
            inputs=val_input,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
            sw_device=device,
            device=torch.device("cpu"),
        )
        assert tuple(val_output.shape) == (1, 4, 112, 112, 64), val_output.shape
    print("滑动窗口推理通过，输出形状：", tuple(val_output.shape))
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print("PyTorch峰值显存（GB）：", f"{peak:.2f}")
    print("=" * 62)
    print("烟雾测试通过；下一步把 RUN_MODE 改成 'train' 并重新运行。")
    print("=" * 62)


# 主程序

def main():

    # 8.1 随机种子

    random.seed(SEED)
    np.random.seed(SEED)
    set_determinism(seed=SEED)

    # 8.2 GPU / CPU

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "使用设备：",
        device
    )

    if device.type == "cuda":

        print(
            "GPU：",
            torch.cuda.get_device_name(0)
        )

    # 8.3 创建输出文件夹

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAIN_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VAL_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 8.4 模型保存位置

    best_path = (
        CHECKPOINT_DIR
        / f"{RUN_NAME}_best.pth"
    )

    last_path = (
        CHECKPOINT_DIR
        / f"{RUN_NAME}_last.pth"
    )

    log_path = (
        LOG_DIR
        / f"{RUN_NAME}_history.csv"
    )

    # 在耗时的数据加载之前检查文件冲突

    if not RESUME_TRAINING:

        if (
            best_path.exists()
            or last_path.exists()
            or log_path.exists()
        ):

            raise FileExistsError(
                "检测到同名实验文件。\n"
                "为了避免覆盖已有训练结果，程序已停止。\n"
                "如需新实验，请修改RUN_NAME。\n"
                "如需继续训练，请设置RESUME_TRAINING=True。"
            )

    # 固定387/97数据划分

    if not SPLIT_PATH.exists():

        raise FileNotFoundError(
            f"找不到划分文件：{SPLIT_PATH}"
        )

    with open(
        SPLIT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        split_data = json.load(f)

    train_names = split_data["train"]
    val_names = split_data["validation"]

    # 检查训练和验证病例是否重叠

    if set(train_names) & set(val_names):

        raise RuntimeError(
            "训练集与验证集存在重叠病例"
        )

    train_files = build_cases(train_names)
    val_files = build_cases(val_names)

    print("\n" + "=" * 55)
    print("SegResNet-60实验配置")
    print("=" * 55)

    print("实验名称：", RUN_NAME)
    print("Training病例：", len(train_files))
    print("Validation病例：", len(val_files))
    print("训练Epoch：", NUM_EPOCHS)

    print("Patch Size：", PATCH_SIZE)
    print("Batch Size：", BATCH_SIZE)

    print("Init Filters：", INIT_FILTERS)
    print("学习率：", LEARNING_RATE)

    print("CE类别权重：", CLASS_WEIGHTS)

    print("SW_BATCH_SIZE：", SW_BATCH_SIZE)
    print("NUM_WORKERS：", NUM_WORKERS)

    print("缓存位置：", CACHE_DIR)

    print("=" * 55)

    # 创建PersistentDataset
    #
    # 已有缓存可以复用。
    #
    # 首次缓存：
    #   执行预处理并写入SSD。
    #
    # 再次读取：
    #   从SSD读取已经预处理的数据。

    train_dataset = PersistentDataset(

        data=train_files,

        transform=build_transforms(
            training=True
        ),

        cache_dir=TRAIN_CACHE_DIR
    )

    val_dataset = PersistentDataset(

        data=val_files,

        transform=build_transforms(
            training=False
        ),

        cache_dir=VAL_CACHE_DIR
    )

    # 创建DataLoader

    train_loader = create_loader(
        train_dataset,
        training=True
    )

    val_loader = create_loader(
        val_dataset,
        training=False
    )

    # 定义未加权Dice+CE Loss

    weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=device
    )

    # Dice Loss
    # 与原Baseline一致，不进行额外类别加权

    dice_loss = DiceLoss(
        to_onehot_y=True,
        softmax=True
    )

    # Cross Entropy Loss
    # 四个类别权重均为1（未加权交叉熵）

    ce_loss = torch.nn.CrossEntropyLoss(
        weight=weights
    )

    # 测试Loss

    test_loss_function(
        dice_loss,
        ce_loss,
        device
    )

    # 恢复随机种子
    #
    # Loss测试使用了随机数。
    #
    # 正式初始化模型前重新固定种子，
    # 避免Loss测试改变模型初始化状态。
    #
    # 注意：
    # 多Worker下不能保证与Baseline-60拥有完全相同
    # 的逐步随机Patch序列。

    random.seed(SEED)
    np.random.seed(SEED)
    set_determinism(seed=SEED)

    print(
        "\n随机种子已重新固定：",
        SEED
    )

    # 创建MONAI SegResNet（随机初始化，不加载U-Net权重）

    model = build_segresnet_model()

    model = model.to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "SegResNet模型参数量：",
        parameter_count
    )

    # Adam优化器

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Dice评价
    #
    # include_background=False
    #
    # 只统计Class 1、Class 2、Class 3。

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch"
    )

    # 训练状态

    start_epoch = 1

    best_mean_dice = -1.0
    best_epoch = -1

    # 是否恢复训练

    if RESUME_TRAINING:

        if not last_path.exists():

            raise FileNotFoundError(
                f"找不到Last模型：{last_path}"
            )

        if not log_path.exists():

            raise FileNotFoundError(
                f"找不到历史日志：{log_path}"
            )

        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=False
        )

        # 检查实验名称

        if checkpoint["run_name"] != RUN_NAME:
            raise ValueError("Checkpoint实验名称不一致")
        if checkpoint.get("architecture") != "SegResNet":
            raise ValueError("Checkpoint网络架构不是SegResNet，禁止加载")
        if int(checkpoint.get("init_filters", -1)) != INIT_FILTERS:
            raise ValueError("Checkpoint初始通道数不匹配")

        # 检查类别权重

        if checkpoint["class_weights"] != CLASS_WEIGHTS:

            raise ValueError(
                "Checkpoint类别权重不一致"
            )

        # 续训前验证CSV最后一轮与Last Checkpoint一致。
        # 如果上次意外中断导致二者不一致，先停止，避免重复Epoch写入。
        with open(log_path, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows or int(rows[-1]["epoch"]) != int(checkpoint["epoch"]):
            raise RuntimeError(
                "CSV最后一轮与Last Checkpoint不一致，请不要直接续训；"
                "先检查是否在保存文件时意外中断。"
            )

        # 恢复模型参数

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # 恢复Adam优化器

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = int(
            checkpoint["epoch"]
        ) + 1

        best_mean_dice = float(
            checkpoint["best_mean_dice"]
        )

        best_epoch = int(
            checkpoint["best_epoch"]
        )

        print(
            "\n恢复训练，从Epoch",
            start_epoch,
            "开始"
        )

    # 初始化日志

    else:

        with open(
            log_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "dice_loss",
                "ce_loss",
                "class1_dice",
                "class2_dice",
                "class3_dice",
                "mean_dice",
                "train_minutes",
                "validation_minutes",
                "epoch_minutes",
                "peak_gpu_memory_gb",
            ])

    # 开始正式训练

    print("\n" + "=" * 55)
    print("开始正式SegResNet-60训练")
    print("=" * 55)

    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.perf_counter()

        if device.type == "cuda":

            torch.cuda.reset_peak_memory_stats()

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        print("-" * 55)

        # 21.1 训练模式

        model.train()

        epoch_total_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_ce_loss = 0.0

        step_count = 0

        # 21.2 遍历387例训练数据

        for step, batch in enumerate(
            train_loader,
            start=1
        ):

            # MRI传入GPU

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            # 标签传入GPU

            labels = (
                batch["label"]
                .long()
                .to(
                    device,
                    non_blocking=True
                )
            )

            # 清空梯度

            optimizer.zero_grad(
                set_to_none=True
            )

            # 前向传播

            outputs = model(images)

            # Dice Loss

            loss_dice = dice_loss(
                outputs,
                labels
            )

            # Cross Entropy (all class weights equal to 1)
            #
            # 标签维度：
            # [B, 1, D, H, W]
            #
            # CrossEntropy要求：
            # [B, D, H, W]

            targets = (
                labels
                .squeeze(1)
                .long()
            )

            loss_ce = ce_loss(
                outputs,
                targets
            )

            # 总损失

            loss = (
                loss_dice
                +
                loss_ce
            )

            # 检查数值是否正常

            if not torch.isfinite(loss).item():

                raise RuntimeError(
                    f"Epoch {epoch} Step {step} "
                    "Loss出现NaN或Inf"
                )

            # 反向传播

            loss.backward()

            # 更新模型参数

            optimizer.step()

            # 记录Loss

            epoch_total_loss += loss.item()
            epoch_dice_loss += loss_dice.item()
            epoch_ce_loss += loss_ce.item()

            step_count += 1

            # 每50 Step输出进度

            if (
                step % 50 == 0
                or step == len(train_loader)
            ):

                average_loss = (
                    epoch_total_loss
                    /
                    step_count
                )

                print(
                    f"Training {step}/{len(train_loader)}"
                    f" | Loss={average_loss:.6f}"
                )

        # 21.3 统计训练耗时

        if device.type == "cuda":

            torch.cuda.synchronize()

        training_end = time.perf_counter()

        train_minutes = (
            training_end
            -
            epoch_start
        ) / 60.0

        mean_train_loss = (
            epoch_total_loss
            /
            step_count
        )

        mean_dice_loss = (
            epoch_dice_loss
            /
            step_count
        )

        mean_ce_loss = (
            epoch_ce_loss
            /
            step_count
        )

        print("\n训练阶段完成")

        print(
            "Train Loss：",
            f"{mean_train_loss:.6f}"
        )

        print(
            "Dice Loss：",
            f"{mean_dice_loss:.6f}"
        )

        print(
            "CE Loss：",
            f"{mean_ce_loss:.6f}"
        )

        print(
            "训练耗时：",
            f"{train_minutes:.2f}分钟"
        )

        # 完整Validation

        model.eval()

        dice_metric.reset()

        with torch.inference_mode():

            for val_step, batch in enumerate(
                val_loader,
                start=1
            ):

                # MRI输入GPU

                val_images = batch["image"].to(
                    device,
                    non_blocking=True
                )

                # Label保持CPU

                val_labels = (
                    batch["label"]
                    .long()
                )

                # Sliding Window Inference
                #
                # roi_size=64³
                # sw_batch_size=4
                #
                # 使用FP32计算

                val_outputs = (
                    sliding_window_inference(

                        inputs=val_images,

                        roi_size=ROI_SIZE,

                        sw_batch_size=SW_BATCH_SIZE,

                        predictor=model,

                        overlap=OVERLAP,

                        sw_device=device,

                        device=torch.device("cpu")
                    )
                )

                # Logits → 预测类别

                val_predictions = torch.argmax(
                    val_outputs,
                    dim=1,
                    keepdim=True
                )

                # 转成One-hot格式

                pred_onehot = one_hot(
                    val_predictions,
                    num_classes=4
                )

                label_onehot = one_hot(
                    val_labels,
                    num_classes=4
                )

                # 累计Dice

                dice_metric(
                    y_pred=pred_onehot,
                    y=label_onehot
                )

                # 每20例输出进度

                if (
                    val_step % 20 == 0
                    or val_step == len(val_loader)
                ):

                    print(
                        f"Validation "
                        f"{val_step}/{len(val_loader)}"
                    )

                # 释放临时张量

                del val_images
                del val_labels
                del val_outputs
                del val_predictions
                del pred_onehot
                del label_onehot

        # 验证计时

        if device.type == "cuda":

            torch.cuda.synchronize()

        validation_end = time.perf_counter()

        validation_minutes = (
            validation_end
            -
            training_end
        ) / 60.0

        # 汇总Class Dice

        class_dice = dice_metric.aggregate()

        dice_metric.reset()

        class_scores = [
            float(class_dice[i].item())
            for i in range(3)
        ]

        mean_dice = float(
            torch.nanmean(class_dice).item()
        )

        if not math.isfinite(mean_dice):

            raise RuntimeError(
                "验证Mean Dice出现NaN或Inf"
            )

        # GPU显存统计

        if device.type == "cuda":

            peak_memory = (
                torch.cuda.max_memory_allocated()
                /
                (1024 ** 3)
            )

        else:

            peak_memory = 0.0

        # Epoch耗时

        epoch_minutes = (
            time.perf_counter()
            -
            epoch_start
        ) / 60.0

        # 输出验证结果

        print("\n" + "=" * 55)
        print(f"Epoch {epoch} 验证结果")
        print("=" * 55)

        print(
            "Class 1 Dice：",
            f"{class_scores[0]:.4f}"
        )

        print(
            "Class 2 Dice：",
            f"{class_scores[1]:.4f}"
        )

        print(
            "Class 3 Dice：",
            f"{class_scores[2]:.4f}"
        )

        print(
            "Mean Dice：",
            f"{mean_dice:.4f}"
        )

        print(
            "训练耗时：",
            f"{train_minutes:.2f}分钟"
        )

        print(
            "验证耗时：",
            f"{validation_minutes:.2f}分钟"
        )

        print(
            "Epoch总耗时：",
            f"{epoch_minutes:.2f}分钟"
        )

        print(
            "GPU峰值显存：",
            f"{peak_memory:.2f}GB"
        )

        # 保存CSV训练日志

        with open(
            log_path,
            "a",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                mean_train_loss,
                mean_dice_loss,
                mean_ce_loss,
                class_scores[0],
                class_scores[1],
                class_scores[2],
                mean_dice,
                train_minutes,
                validation_minutes,
                epoch_minutes,
                peak_memory,
            ])

        # 判断是否为Best模型

        improved = (
            mean_dice
            >
            best_mean_dice
        )

        if improved:

            best_mean_dice = mean_dice
            best_epoch = epoch

        # 保存Checkpoint内容

        checkpoint_data = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "mean_dice":
                mean_dice,

            "best_mean_dice":
                best_mean_dice,

            "best_epoch":
                best_epoch,

            "class1_dice":
                class_scores[0],

            "class2_dice":
                class_scores[1],

            "class3_dice":
                class_scores[2],

            "init_filters":
                INIT_FILTERS,

            "patch_size":
                PATCH_SIZE,

            "batch_size":
                BATCH_SIZE,

            "num_samples":
                NUM_SAMPLES,

            "learning_rate":
                LEARNING_RATE,

            "class_weights":
                CLASS_WEIGHTS,

            "run_name":
                RUN_NAME,

            "architecture": "SegResNet",
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 4,
            "blocks_down": (1, 2, 2, 4),
            "blocks_up": (1, 1, 1),

            "seed":
                SEED,

            "num_workers":
                NUM_WORKERS,

            "sw_batch_size":
                SW_BATCH_SIZE,

            "cache_version":
                CACHE_VERSION,
        }

        # 保存Best模型

        if improved:

            save_checkpoint_atomic(checkpoint_data, best_path)

            print(
                "\n新的Best模型已保存。"
            )

        # 保存Last模型

        save_checkpoint_atomic(checkpoint_data, last_path)

        print(
            "Last模型已保存。"
        )

        print(
            "当前Best Epoch：",
            best_epoch
        )

        print(
            "Best Mean Dice：",
            f"{best_mean_dice:.4f}"
        )

        print("=" * 55)


    print("\n" + "=" * 55)
    print("SegResNet-60训练完成")
    print("=" * 55)

    print(
        "最佳Epoch：",
        best_epoch
    )

    print(
        "最佳Mean Dice：",
        f"{best_mean_dice:.4f}"
    )

    print(
        "\nBest模型：",
        best_path
    )

    print(
        "\nLast模型：",
        last_path
    )

    print(
        "\n训练日志：",
        log_path
    )

    print("=" * 55)


# Windows多进程安全入口

if __name__ == "__main__":
    if RUN_MODE == "smoke":
        run_smoke_test()
    elif RUN_MODE == "train":
        main()
    else:
        raise ValueError("RUN_MODE只能为'smoke'或'train'")
