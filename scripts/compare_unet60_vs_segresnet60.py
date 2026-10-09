"""Evaluate 3D U-Net and SegResNet on the fixed 97-case validation set.

Uses 3D sliding-window inference and per-case Class 1/2/3 and WT/TC/ET
Dice. GT-empty regions are excluded from Dice means; false positives are
reported separately. Best checkpoints are selected by validation Dice.
"""

from pathlib import Path
import csv
import json
import math
import sys
import time

import numpy as np
import torch
from monai.data import DataLoader, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
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

# 项目路径 / 实验配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.unet3d import UNet3D

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour"
IMAGES_DIR = DATASET_DIR / "imagesTr"
LABELS_DIR = DATASET_DIR / "labelsTr"
SPLIT_PATH = PROJECT_ROOT / "outputs" / "splits" / "train_val_split.json"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "preprocess_v1" / "val"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "unet60_vs_segresnet60"
PREFIX = "unet60_vs_segresnet60"

# 这里限定为当前已完成的两组 60 Epoch 实验；不是 Last，也不是续训后的新 Best。
MODEL_SETTINGS = {
    "UNet60": {
        "checkpoint": "unet_baseline_40_fast_best.pth",
        "run_name": "unet_baseline_40_fast",
        "expected_epoch": 53,
        "expected_params": 351484,
    },
    "SegResNet60": {
        "checkpoint": "segresnet_baseline_60_fast_best.pth",
        "run_name": "segresnet_baseline_60_fast",
        "expected_epoch": 59,
        "expected_params": 1176852,
    },
}

ROI_SIZE = (64, 64, 64)
SW_BATCH_SIZE = 4
OVERLAP = 0.25
NUM_WORKERS = 2
EXPECTED_VALIDATION_CASES = 97
EXPECTED_RAW_MEAN = {"UNet60": 0.6991, "SegResNet60": 0.7062}
REGION_METRICS = ("WT", "TC", "ET")
CLASS_METRICS = ("Class1", "Class2", "Class3")
ALL_METRICS = (*CLASS_METRICS, *REGION_METRICS)
SUMMARY_METRICS = (*CLASS_METRICS, "RawMean", *REGION_METRICS, "RegionMean")


# 与原实验相同的 Validation 预处理
def build_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(
            keys=["image", "label"],
            axcodes="RAS",
            labels=(("L", "R"), ("P", "A"), ("I", "S")),
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])


# Dice / 区域计算：与原 compare_v1_v2_regions.py 同一规则
def calculate_dice(pred_mask, gt_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    gt_count = int(np.count_nonzero(gt_mask))
    pred_count = int(np.count_nonzero(pred_mask))
    if gt_count == 0:
        return float("nan")
    intersection = int(np.count_nonzero(pred_mask & gt_mask))
    return float(2.0 * intersection / (gt_count + pred_count))


def calculate_all_metrics(prediction, label):
    # 原始类别：0 背景，1 水肿，2 非增强肿瘤，3 增强肿瘤。
    values = {
        "Class1": calculate_dice(prediction == 1, label == 1),
        "Class2": calculate_dice(prediction == 2, label == 2),
        "Class3": calculate_dice(prediction == 3, label == 3),
        "WT": calculate_dice(prediction > 0, label > 0),
        "TC": calculate_dice(
            (prediction == 2) | (prediction == 3),
            (label == 2) | (label == 3),
        ),
        "ET": calculate_dice(prediction == 3, label == 3),
    }
    # 额外记录 GT 阴性病例的假阳性，不改变任何既有 Dice 定义。
    values["ET_GT_empty"] = bool(np.count_nonzero(label == 3) == 0)
    values["ET_pred_voxels"] = int(np.count_nonzero(prediction == 3))
    values["RawMean_case"] = nanmean_safe([values[k] for k in CLASS_METRICS])
    values["RegionMean_case"] = nanmean_safe([values[k] for k in REGION_METRICS])
    return values


def calculate_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan"), 0
    return float(valid.mean()), int(valid.size)


def nanmean_safe(values):
    return calculate_mean(values)[0]


def compare_single_case(value_a, value_b):
    """Delta = B-A；任一数为 NaN 时 Delta 也为 NaN。"""
    if math.isnan(float(value_a)) or math.isnan(float(value_b)):
        return float("nan")
    return float(value_b - value_a)


# 严格检查 Best Checkpoint，分别重建两个网络
def load_model(model_name, device):
    cfg = MODEL_SETTINGS[model_name]
    path = CHECKPOINT_DIR / cfg["checkpoint"]
    if not path.is_file():
        raise FileNotFoundError(f"缺少 Best Checkpoint：{path}")

    # 首先加载到 CPU，避免初始化期间额外占用显存。
    # weights_only=False 仅可用于自己训练、可信来源的 checkpoint。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    for field, expected in (
        ("run_name", cfg["run_name"]),
        ("epoch", cfg["expected_epoch"]),
        ("class_weights", [1.0, 1.0, 1.0, 1.0]),
    ):
        actual = checkpoint.get(field)
        if actual != expected:
            raise RuntimeError(
                f"{model_name} 的 {field} 不匹配：实际 {actual!r}，预期 {expected!r}。\n"
                "请检查是否使用了 Last、覆盖了 60 Epoch Best，或选错模型文件。"
            )
    if checkpoint.get("best_epoch", checkpoint["epoch"]) != checkpoint["epoch"]:
        raise RuntimeError(f"{model_name} 的权重不是记录的 Best Epoch，请检查文件。")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} 中缺少 model_state_dict")

    if model_name == "UNet60":
        base_channels = int(checkpoint.get("base_channels", 8))
        if base_channels != 8:
            raise RuntimeError("UNet60 的 base_channels 不是预期的 8。")
        model = UNet3D(
            in_channels=4,
            num_classes=4,
            base_channels=base_channels,
        )
    elif model_name == "SegResNet60":
        init_filters = int(checkpoint.get("init_filters", -1))
        if init_filters != 8:
            raise RuntimeError("SegResNet60 的 init_filters 不是预期的 8。")
        if checkpoint.get("architecture") != "SegResNet":
            raise RuntimeError("SegResNet60 的 architecture 信息不匹配。")
        if tuple(checkpoint.get("blocks_down", ())) != (1, 2, 2, 4):
            raise RuntimeError("SegResNet60 的 blocks_down 信息不匹配。")
        if tuple(checkpoint.get("blocks_up", ())) != (1, 1, 1):
            raise RuntimeError("SegResNet60 的 blocks_up 信息不匹配。")
        # 与 train_segresnet_60.py 的 build_segresnet_model() 完全一致。
        model = SegResNet(
            spatial_dims=3,
            init_filters=8,
            in_channels=4,
            out_channels=4,
            dropout_prob=None,
            norm=("GROUP", {"num_groups": 8}),
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            upsample_mode="nontrainable",
        )
    else:
        raise ValueError("未知模型：" + str(model_name))

    num_params = sum(p.numel() for p in model.parameters())
    if num_params != cfg["expected_params"]:
        raise RuntimeError(
            f"{model_name} 参数量不匹配：实际 {num_params}，"
            f"预期 {cfg['expected_params']}；可能网络结构或 MONAI 版本不同。"
        )
    # strict=True，结构不一致立即报错，不静默跳过权重。
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    print(f"Checkpoint: {path.name}")
    print(f"Best Epoch：{checkpoint['epoch']}")
    print(f"模型参数量：{num_params}")
    return model, checkpoint, num_params


# 固定 97 例验证数据
def build_validation_loader(device):
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"找不到原始固定划分：{SPLIT_PATH}")
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    val_names = split_data["validation"]
    if len(val_names) != EXPECTED_VALIDATION_CASES:
        raise RuntimeError(
            f"验证集现在是 {len(val_names)} 例，预期 {EXPECTED_VALIDATION_CASES} 例。"
        )
    if len(set(val_names)) != len(val_names):
        raise RuntimeError("验证集名单中存在重复病例。")
    if set(val_names) & set(split_data.get("training", [])):
        raise RuntimeError("训练集与验证集名单有重叠，请停止实验。")

    val_files = []
    for name in val_names:
        image_path, label_path = IMAGES_DIR / name, LABELS_DIR / name
        if not image_path.is_file():
            raise FileNotFoundError(f"缺少 MRI：{image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"缺少标签：{label_path}")
        val_files.append({"image": str(image_path), "label": str(label_path), "name": name})

    dataset = PersistentDataset(
        data=val_files,
        transform=build_val_transforms(),
        cache_dir=CACHE_DIR,
    )
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": (device.type == "cuda"),
    }
    if NUM_WORKERS > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **loader_kwargs)
    return val_names, loader


# 全量推理 + 区域评价
def evaluate_model(model_name, device, loader, val_names):
    print("\n" + "=" * 66)
    print("正在评价：", model_name)
    print("=" * 66)
    model, checkpoint, num_params = load_model(model_name, device)
    per_case = {}
    start = time.perf_counter()

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            name = val_names[i]
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            labels = batch["label"].long()

            # 与以前的验证脚本保持相同的 FP32 / overlap / CPU 拼接配置。
            outputs = sliding_window_inference(
                inputs=images,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model,
                overlap=OVERLAP,
                sw_device=device,
                device=torch.device("cpu"),
            )
            predictions = torch.argmax(outputs, dim=1, keepdim=True)
            pred_np = predictions[0, 0].numpy()
            gt_np = labels[0, 0].cpu().numpy()

            if pred_np.shape != gt_np.shape:
                raise RuntimeError(f"病例 {name} 预测和 GT 的尺寸不一致。")
            per_case[name] = calculate_all_metrics(pred_np, gt_np)

            if (i + 1) % 20 == 0 or (i + 1) == len(val_names):
                print(f"进度：{i + 1}/{len(val_names)}")
            del images, labels, outputs, predictions

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start

    means = {}
    valid_counts = {}
    for metric in ALL_METRICS:
        means[metric], valid_counts[metric] = calculate_mean(
            [per_case[name][metric] for name in val_names]
        )
    # 与先前 compare_v1_v2_regions.py 的宏平均口径严格保持一致。
    raw_mean = float(np.mean([means[m] for m in CLASS_METRICS]))
    region_mean = float(np.mean([means[m] for m in REGION_METRICS]))

    et_empty = [name for name in val_names if per_case[name]["ET_GT_empty"]]
    et_fp = [name for name in et_empty if per_case[name]["ET_pred_voxels"] > 0]
    summary = {
        **means,
        "RawMean": raw_mean,
        "RegionMean": region_mean,
        "valid_counts": valid_counts,
        "best_epoch": int(checkpoint["epoch"]),
        "num_parameters": int(num_params),
        "inference_seconds": float(inference_seconds),
        "et_gt_empty_count": len(et_empty),
        "et_gt_empty_cases": et_empty,
        "et_false_positive_cases": et_fp,
        "et_false_positive_case_count": len(et_fp),
    }

    print("\n原始类别 Dice：")
    for m in CLASS_METRICS:
        print(f"{m}：{means[m]:.4f}（有效{valid_counts[m]}例）")
    print(f"Raw Mean Dice：{raw_mean:.4f}")
    print("\nBraTS 复合区域 Dice：")
    for m in REGION_METRICS:
        print(f"{m}：{means[m]:.4f}（有效{valid_counts[m]}例）")
    print(f"Region Mean Dice：{region_mean:.4f}")
    print(f"ET GT 阴性病例：{len(et_empty)}例，其中误报 ET：{len(et_fp)}例")
    print(f"完整验证集推理耗时：{inference_seconds:.2f} 秒")
    if abs(raw_mean - EXPECTED_RAW_MEAN[model_name]) > 0.003:
        print(
            f"[提醒] RawMean={raw_mean:.4f} 与原训练日志的 "
            f"{EXPECTED_RAW_MEAN[model_name]:.4f} 差异超过0.003，"
            "请核对预处理、MONAI版本和评价设置。"
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return per_case, summary


# 配对病例统计、CSV / JSON 报告
def build_paired_stats(all_cases, val_names):
    """统计每个指标：提升病例、退步病例、持平病例、无法评价病例。"""
    old = all_cases["UNet60"]
    new = all_cases["SegResNet60"]
    result = {}
    for metric in ALL_METRICS:
        pos = neg = tie = missing = 0
        for name in val_names:
            d = compare_single_case(old[name][metric], new[name][metric])
            if math.isnan(d):
                missing += 1
            elif d > 1e-8:
                pos += 1
            elif d < -1e-8:
                neg += 1
            else:
                tie += 1
        result[metric] = {
            "SegResNet_better": pos,
            "UNet_better": neg,
            "tie": tie,
            "GT_empty_excluded": missing,
        }
    return result


def save_results(all_cases, summaries, val_names, paired_stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = OUTPUT_DIR / (PREFIX + "_summary.csv")
    per_case_csv = OUTPUT_DIR / (PREFIX + "_per_case.csv")
    ranked_csv = OUTPUT_DIR / (PREFIX + "_tc_ranked.csv")
    summary_json = OUTPUT_DIR / (PREFIX + "_summary.json")

    old = summaries["UNet60"]
    new = summaries["SegResNet60"]
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "UNet60", "SegResNet60", "Delta_SegResNet_minus_UNet", "UNet_valid_n", "SegResNet_valid_n"])
        for m in SUMMARY_METRICS:
            writer.writerow([
                m, old[m], new[m], new[m] - old[m],
                old["valid_counts"].get(m, ""), new["valid_counts"].get(m, ""),
            ])

    case_rows = []
    for name in val_names:
        a = all_cases["UNet60"][name]
        b = all_cases["SegResNet60"][name]
        row = {"case": name}
        for m in ALL_METRICS + ("RawMean_case", "RegionMean_case"):
            row["UNet60_" + m] = a[m]
            row["SegResNet60_" + m] = b[m]
            row["Delta_SegResNet_minus_UNet_" + m] = compare_single_case(a[m], b[m])
        row["ET_GT_empty"] = a["ET_GT_empty"]
        row["UNet60_ET_pred_voxels"] = a["ET_pred_voxels"]
        row["SegResNet60_ET_pred_voxels"] = b["ET_pred_voxels"]
        case_rows.append(row)

    with open(per_case_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    # TC 是主要薄弱区域：按 TC 差异从大到小排列，便于找出改善和退步病例。
    # 缺少 TC 标签时 Delta 为 NaN，排在最后。
    tc_key = "Delta_SegResNet_minus_UNet_TC"
    tc_rows = sorted(
        case_rows,
        key=lambda row: (
            math.isnan(float(row[tc_key])),
            -float(row[tc_key]) if not math.isnan(float(row[tc_key])) else 0.0,
        ),
    )
    with open(ranked_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tc_rows)

    json_data = {
        "description": "Two architectures, fixed 60 Epoch training budget, 97-case validation",
        "Delta_definition": "SegResNet60 - UNet60",
        "validation_cases": len(val_names),
        "ROI_SIZE": list(ROI_SIZE),
        "SW_BATCH_SIZE": SW_BATCH_SIZE,
        "OVERLAP": OVERLAP,
        "ET_missing_rule": "When GT ET is empty, Dice=NaN and excluded from the mean",
        "models": summaries,
        "paired_win_counts": paired_stats,
        "TC_most_improved": [
            {"case": row["case"], "delta": row[tc_key]} for row in tc_rows[:5]
        ],
        "TC_most_declined": [
            {"case": row["case"], "delta": row[tc_key]} for row in tc_rows[-5:]
        ],
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, allow_nan=True)
    return summary_csv, per_case_csv, ranked_csv, summary_json, tc_rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)
    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(0))
    else:
        print("[提醒] 当前未检测到 CUDA，完整体积推理可能很慢。")
    print("实验：UNet60 (Epoch53) vs SegResNet60 (Epoch59)")
    print("比较指标：Class1/2/3、WT、TC、ET、RawMean、RegionMean")
    print("Delta = SegResNet60 - UNet60（正值代表 SegResNet60 更高）")

    val_names, loader = build_validation_loader(device)
    print(f"Validation病例数量：{len(val_names)}")

    all_cases, summaries = {}, {}
    for name in MODEL_SETTINGS:
        all_cases[name], summaries[name] = evaluate_model(name, device, loader, val_names)

    old = summaries["UNet60"]
    new = summaries["SegResNet60"]
    print("\n" + "=" * 76)
    print("UNet60 / SegResNet60 最终对比（Delta = SegResNet60 - UNet60）")
    print("=" * 76)
    print(f"{'Metric':<17}{'UNet60':>15}{'SegResNet60':>17}{'Delta':>15}")
    for m in SUMMARY_METRICS:
        print(f"{m:<17}{old[m]:>15.4f}{new[m]:>17.4f}{new[m]-old[m]:>+15.4f}")

    paired_stats = build_paired_stats(all_cases, val_names)
    print("\n逐病例优势统计（同一97例验证数据；GT为空的指标不计入）：")
    for m in REGION_METRICS:
        s = paired_stats[m]
        print(
            f"{m}: SegResNet更好 {s['SegResNet_better']}例，"
            f"UNet更好 {s['UNet_better']}例，"
            f"持平 {s['tie']}例，GT为空 {s['GT_empty_excluded']}例"
        )

    summary_csv, per_case_csv, ranked_csv, summary_json, ranked = save_results(
        all_cases, summaries, val_names, paired_stats
    )
    print("\nTC 改善最多的5例：")
    for row in ranked[:5]:
        print(f"{row['case']}: {row['Delta_SegResNet_minus_UNet_TC']:+.4f}")
    print("\nTC 退步最多的5例：")
    for row in sorted(ranked, key=lambda r: r['Delta_SegResNet_minus_UNet_TC'])[:5]:
        print(f"{row['case']}: {row['Delta_SegResNet_minus_UNet_TC']:+.4f}")

    print("\n" + "=" * 76)
    print("比较完成（没有训练，也没有更改模型权重）。")
    print("总结 CSV：", summary_csv)
    print("逐病例 CSV：", per_case_csv)
    print("TC 排序 CSV：", ranked_csv)
    print("总结 JSON：", summary_json)
    print("=" * 76)


# Windows + PyCharm：必须保留此入口，避免多进程 DataLoader 重复启动。
if __name__ == "__main__":
    main()
