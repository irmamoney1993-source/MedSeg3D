
"""
compare_v1_v2_regions.py

用途：
    对比3D U-Net Baseline V1和Weighted CE V2。

评价数据：
    固定Validation Set，共97例。

模型：
    V1：unet_baseline_v1_best.pth
    V2：unet_weighted_ce_v2_fast_best.pth

评价指标：
    Class 1 / 2 / 3 Dice
    WT / TC / ET Dice
    Region Mean Dice

输出：
    1. v1_v2_per_case_comparison.csv
    2. v1_v2_summary.csv
    3. v1_v2_summary.json

说明：
    - 不训练模型
    - 不修改Checkpoint
    - 两个模型使用相同预处理和推理参数
    - 使用FP32进行评价
    - GT为空的区域Dice记为NaN
"""

# 0. 导入库

from pathlib import Path
import sys
import json
import csv
import time

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
)

from monai.inferers import sliding_window_inference


# 1. 项目路径

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from models.unet3d import UNet3D


# 2. 实验配置

MODELS = {
    "V1_Baseline": "unet_baseline_v1_best.pth",
    "V2_WeightedCE": "unet_weighted_ce_v2_fast_best.pth",
}

ROI_SIZE = (64, 64, 64)

SW_BATCH_SIZE = 4

OVERLAP = 0.25

NUM_WORKERS = 2


# 3. 数据和输出路径

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Task01_BrainTumour"
)

IMAGES_DIR = DATASET_DIR / "imagesTr"

LABELS_DIR = DATASET_DIR / "labelsTr"

SPLIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "splits"
    / "train_val_split.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

# 使用已建立的验证集预处理缓存
CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "preprocess_v1"
    / "val"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "v1_v2_epoch36_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PER_CASE_CSV = (
    OUTPUT_DIR
    / "v1_v2_per_case_comparison.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "v1_v2_summary.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "v1_v2_summary.json"
)


# 4. 定义完整验证预处理
# 保持与V1、V2的Validation阶段一致。

def build_val_transforms():

    return Compose([

        LoadImaged(
            keys=["image", "label"]
        ),

        EnsureChannelFirstd(
            keys=["image"],
            channel_dim=-1
        ),

        EnsureChannelFirstd(
            keys=["label"],
            channel_dim="no_channel"
        ),

        Orientationd(
            keys=["image", "label"],
            axcodes="RAS",
            labels=(
                ("L", "R"),
                ("P", "A"),
                ("I", "S"),
            ),
        ),

        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),

        CropForegroundd(
            keys=["image", "label"],
            source_key="image"
        ),

        SpatialPadd(
            keys=["image", "label"],
            spatial_size=ROI_SIZE
        ),

        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True
        ),
    ])


# 5. Dice计算函数
#
# Dice = 2 × |Prediction ∩ GT|
#        / (|Prediction| + |GT|)
#
# 如果GT不存在该区域：
#     返回NaN。
#
# 与之前的区域评价采用相同规则。

def calculate_dice(pred_mask, gt_mask):

    pred_mask = pred_mask.astype(bool)

    gt_mask = gt_mask.astype(bool)

    gt_count = np.count_nonzero(gt_mask)

    pred_count = np.count_nonzero(pred_mask)

    if gt_count == 0:
        return float("nan")

    intersection = np.count_nonzero(
        pred_mask & gt_mask
    )

    return float(
        2.0 * intersection
        / (gt_count + pred_count)
    )


# 6. 计算各类别及区域Dice

def calculate_all_metrics(prediction, label):

    # 原始类别

    c1 = calculate_dice(
        prediction == 1,
        label == 1
    )

    c2 = calculate_dice(
        prediction == 2,
        label == 2
    )

    c3 = calculate_dice(
        prediction == 3,
        label == 3
    )

    # BraTS复合区域
    #
    # 当前MSD Task01标签：
    # 1=水肿
    # 2=非增强肿瘤
    # 3=增强肿瘤

    wt = calculate_dice(
        prediction > 0,
        label > 0
    )

    tc = calculate_dice(
        (prediction == 2) | (prediction == 3),
        (label == 2) | (label == 3)
    )

    et = calculate_dice(
        prediction == 3,
        label == 3
    )

    return {
        "Class1": c1,
        "Class2": c2,
        "Class3": c3,
        "WT": wt,
        "TC": tc,
        "ET": et,
    }


# 7. 安全计算均值
# 对于GT不存在的病例，忽略NaN。
# 同时记录有效病例数量。

def calculate_mean(values):

    arr = np.asarray(values, dtype=np.float64)

    valid = arr[~np.isnan(arr)]

    if valid.size == 0:
        return float("nan"), 0

    return float(valid.mean()), int(valid.size)


# 8. 加载模型

def load_model(checkpoint_path, device):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到Checkpoint：{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    base_channels = checkpoint.get(
        "base_channels", 8
    )

    model = UNet3D(
        in_channels=4,
        num_classes=4,
        base_channels=base_channels
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# 9. 主程序

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("使用设备：", device)

    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(0))

    # 10. 读取固定Validation列表

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(SPLIT_PATH)

    with open(
        SPLIT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        split_data = json.load(f)

    val_names = split_data["validation"]

    val_files = []

    for name in val_names:

        image_path = IMAGES_DIR / name
        label_path = LABELS_DIR / name

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        val_files.append({
            "image": str(image_path),
            "label": str(label_path),
            "name": name,
        })

    print("\nValidation病例数量：", len(val_files))

    # 11. 创建Validation Dataset

    dataset = PersistentDataset(
        data=val_files,
        transform=build_val_transforms(),
        cache_dir=CACHE_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        **(
            {"prefetch_factor": 2}
            if NUM_WORKERS > 0
            else {}
        )
    )

    all_per_case = {}
    summaries = {}

    metric_names = [
        "Class1", "Class2", "Class3",
        "WT", "TC", "ET"
    ]

    # 12. 分别评估V1和V2

    for model_name, filename in MODELS.items():

        print("\n" + "=" * 55)
        print("正在评价：", model_name)
        print("=" * 55)

        checkpoint_path = (
            CHECKPOINT_DIR / filename
        )

        model, checkpoint = load_model(
            checkpoint_path,
            device
        )

        print(
            "Best Epoch：",
            checkpoint.get("epoch", "未知")
        )

        metrics_by_case = {}

        start_time = time.perf_counter()

        with torch.inference_mode():

            for index, batch in enumerate(loader):

                case_name = val_files[index]["name"]

                images = batch["image"].to(
                    device,
                    non_blocking=True
                )

                labels = batch["label"].long()

                # 两个模型都使用相同的FP32推理配置

                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=ROI_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=OVERLAP,
                    sw_device=device,
                    device=torch.device("cpu")
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                    keepdim=True
                )

                gt_np = (
                    labels[0, 0]
                    .cpu()
                    .numpy()
                )

                pred_np = (
                    predictions[0, 0]
                    .cpu()
                    .numpy()
                )

                metrics = calculate_all_metrics(
                    pred_np,
                    gt_np
                )

                metrics_by_case[case_name] = metrics

                if (
                    (index + 1) % 20 == 0
                    or index + 1 == len(loader)
                ):
                    print(
                        f"进度：{index + 1}/{len(loader)}"
                    )

                del images, labels, outputs, predictions

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        # 13. 计算总体均值

        means = {}
        valid_counts = {}

        for metric in metric_names:

            values = [
                case_metrics[metric]
                for case_metrics in metrics_by_case.values()
            ]

            mean_value, valid_count = calculate_mean(
                values
            )

            means[metric] = mean_value
            valid_counts[metric] = valid_count

        # 各类别均值的宏平均
        raw_mean = float(np.mean([
            means["Class1"],
            means["Class2"],
            means["Class3"],
        ]))

        # 三个BraTS区域均值的宏平均
        region_mean = float(np.mean([
            means["WT"],
            means["TC"],
            means["ET"],
        ]))

        summaries[model_name] = {
            **means,
            "RawMean": raw_mean,
            "RegionMean": region_mean,
            "valid_counts": valid_counts,
            "best_epoch": checkpoint.get("epoch"),
            "inference_seconds": elapsed_seconds,
        }

        all_per_case[model_name] = metrics_by_case

        print("\nRaw Label Dice：")

        for name in ["Class1", "Class2", "Class3"]:
            print(f"{name}：{means[name]:.4f}")

        print(f"Raw Mean Dice：{raw_mean:.4f}")

        print("\nBraTS Region Dice：")

        for name in ["WT", "TC", "ET"]:
            print(
                f"{name}：{means[name]:.4f} "
                f"(有效{valid_counts[name]}例)"
            )

        print(f"Region Mean Dice：{region_mean:.4f}")

        print(
            f"推理耗时：{elapsed_seconds:.2f}秒"
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 14. 输出V1/V2对比表

    v1 = summaries["V1_Baseline"]
    v2 = summaries["V2_WeightedCE"]

    comparison_metrics = [
        "Class1",
        "Class2",
        "Class3",
        "RawMean",
        "WT",
        "TC",
        "ET",
        "RegionMean",
    ]

    print("\n" + "=" * 70)
    print("V1 / V2最终对比")
    print("=" * 70)

    print(
        f"{'Metric':<15}"
        f"{'V1':>12}"
        f"{'V2':>12}"
        f"{'Delta':>12}"
    )

    for name in comparison_metrics:

        delta = v2[name] - v1[name]

        print(
            f"{name:<15}"
            f"{v1[name]:>12.4f}"
            f"{v2[name]:>12.4f}"
            f"{delta:>+12.4f}"
        )

    # 15. 保存总结CSV

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Metric",
            "V1_Baseline",
            "V2_WeightedCE",
            "Delta_V2_minus_V1",
        ])

        for name in comparison_metrics:

            writer.writerow([
                name,
                v1[name],
                v2[name],
                v2[name] - v1[name],
            ])

    # 16. 保存逐病例CSV

    with open(
        PER_CASE_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "case",
            *[f"V1_{m}" for m in metric_names],
            *[f"V2_{m}" for m in metric_names],
        ])

        for case_name in val_names:

            v1_case = all_per_case[
                "V1_Baseline"
            ][case_name]

            v2_case = all_per_case[
                "V2_WeightedCE"
            ][case_name]

            writer.writerow([
                case_name,
                *[v1_case[m] for m in metric_names],
                *[v2_case[m] for m in metric_names],
            ])

    # 17. 保存JSON

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summaries,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("比较完成")
    print("=" * 70)

    print("总结CSV：", SUMMARY_CSV)
    print("逐病例CSV：", PER_CASE_CSV)
    print("总结JSON：", SUMMARY_JSON)


# 18. Windows多进程安全入口

if __name__ == "__main__":
    main()
