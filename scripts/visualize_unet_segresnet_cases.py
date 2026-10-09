# -*- coding: utf-8 -*-
"""Render same-slice MRI/GT/prediction panels for selected validation cases.

Includes TC/ET false-positive and false-negative maps, orthogonal views,
and case-level statistics. Predictions are checked against the saved
97-case comparison table. Coordinates refer to the resampled/cropped grid.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# 后端必须在导入 pyplot 前设置，避免 PyCharm 弹窗阻塞批处理。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from monai.data import PersistentDataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, SpatialPadd, NormalizeIntensityd,
)


# 统一配置（如无特殊需要，不要修改）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.unet3d import UNet3D

CASE_MODE = "selected"       # "selected"：典型病例；"all"：固定验证集的全部 97 例
TOP_TC = 3                  # TC 提升前三
BOTTOM_TC = 3               # TC 退步前三
BOTTOM_ET = 2               # ET 退步前二 (GT ET 阳性)
INCLUDE_ALL_ET_NEGATIVE = True
INCLUDE_MEDIAN_TC = True     # 增加1例 TC 改变量接近零的参照病例

ROI_SIZE = (64, 64, 64)
SW_BATCH_SIZE = 4
OVERLAP = 0.25
EXPECTED_VAL_N = 97
METRIC_TOLERANCE = 0.002  # 新推理与原 CSV 不一致时停止，避免产生伪对齐图

RAW_DIR = ROOT / "data" / "raw" / "Task01_BrainTumour"
IMAGES_DIR = RAW_DIR / "imagesTr"
LABELS_DIR = RAW_DIR / "labelsTr"
SPLIT_PATH = ROOT / "outputs" / "splits" / "train_val_split.json"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CACHE_DIR = ROOT / "data" / "cache" / "preprocess_v1" / "val"
ANALYSIS_DIR = ROOT / "outputs" / "analysis" / "unet60_vs_segresnet60"
HISTORY_CSV = ANALYSIS_DIR / "unet60_vs_segresnet60_per_case.csv"
OUT_DIR = ANALYSIS_DIR / "visualization_v1"
FIG_DIR = OUT_DIR / "figures"

# 建议做基线对照时，这两个 Epoch 严格固定。如后续续训更新了 Best，应恢复备份。
MODELS = {
    "UNet60": {
        "filename": "unet_baseline_40_fast_best.pth",
        "run_name": "unet_baseline_40_fast", "epoch": 53, "params": 351484,
    },
    "SegResNet60": {
        "filename": "segresnet_baseline_60_fast_best.pth",
        "run_name": "segresnet_baseline_60_fast", "epoch": 59, "params": 1176852,
    },
}

# 与 MSD Task01_BrainTumour 的4通道顺序一致：FLAIR, T1, T1gd, T2。
MODALITIES = [(0, "FLAIR"), (2, "T1gd")]
LABEL_COLORS = {
    1: (0.00, 0.72, 0.69),   # Edema: teal
    2: (1.00, 0.65, 0.18),   # Non-enhancing tumor: amber
    3: (0.88, 0.24, 0.60),   # Enhancing tumor: magenta
}
ERR_COLORS = {
    1: (0.22, 0.77, 0.43),   # TP: green
    2: (0.96, 0.25, 0.25),   # FP: red
    3: (0.22, 0.65, 1.00),   # FN: blue
}


# 同历史评价完全一致的预处理 / 网络
def val_transform():
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
            keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])


def make_models(device):
    models = OrderedDict()
    for name, cfg in MODELS.items():
        ckpt_path = CKPT_DIR / cfg["filename"]
        if not ckpt_path.is_file():
            raise FileNotFoundError("缺少 Best Checkpoint：" + str(ckpt_path))
        # 只读取自己训练的可信 checkpoint，weights_only=False 不用于第三方文件。
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("run_name") != cfg["run_name"]:
            raise RuntimeError(f"{name} run_name 不一致：{ckpt.get('run_name')}")
        if int(ckpt.get("epoch", -1)) != cfg["epoch"]:
            raise RuntimeError(
                f"{name} Best Epoch 不是 {cfg['epoch']}；可能在续训中被覆盖。"
                "请恢复原 60 Epoch 的 Best 备份后再可视化。"
            )
        if ckpt.get("best_epoch", ckpt["epoch"]) != ckpt["epoch"]:
            raise RuntimeError(f"{name} checkpoint 不是 Best 权重")
        if ckpt.get("class_weights") != [1.0, 1.0, 1.0, 1.0]:
            raise RuntimeError(f"{name} CE 权重和未加权实验不一致")

        if name == "UNet60":
            if int(ckpt.get("base_channels", 8)) != 8:
                raise RuntimeError("UNet base_channels 不匹配")
            model = UNet3D(in_channels=4, num_classes=4, base_channels=8)
        else:
            if ckpt.get("architecture") != "SegResNet":
                raise RuntimeError("SegResNet architecture 不匹配")
            if int(ckpt.get("init_filters", -1)) != 8:
                raise RuntimeError("SegResNet init_filters 不匹配")
            if tuple(ckpt.get("blocks_down", ())) != (1, 2, 2, 4):
                raise RuntimeError("SegResNet blocks_down 不匹配")
            if tuple(ckpt.get("blocks_up", ())) != (1, 1, 1):
                raise RuntimeError("SegResNet blocks_up 不匹配")
            model = SegResNet(
                spatial_dims=3, init_filters=8, in_channels=4, out_channels=4,
                dropout_prob=None, norm=("GROUP", {"num_groups": 8}),
                blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                upsample_mode="nontrainable",
            )
        n_params = sum(int(p.numel()) for p in model.parameters())
        if n_params != cfg["params"]:
            raise RuntimeError(f"{name} 模型参数量 {n_params} != {cfg['params']}")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model = model.to(device=device, dtype=torch.float32).eval()
        models[name] = model
        print(f"加载 {name}：Epoch {cfg['epoch']}，{n_params:,} 参数")
    return models


# 历史 CSV / 病例选择 / Dice 复核
def read_csv_data():
    if not HISTORY_CSV.is_file():
        raise FileNotFoundError(
            f"缺少已完成对比实验的逐病例 CSV：{HISTORY_CSV}\n"
            "请先运行 compare_unet60_vs_segresnet60.py。"
        )
    with open(HISTORY_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_VAL_N:
        raise RuntimeError(f"对比 CSV 是 {len(rows)} 例，预期 97 例")
    essential = [
        "case", "Delta_SegResNet_minus_UNet_TC", "Delta_SegResNet_minus_UNet_ET",
        "ET_GT_empty", "UNet60_TC", "SegResNet60_TC",
    ]
    for key in essential:
        if key not in rows[0]:
            raise KeyError(f"原对比 CSV 缺少列：{key}")
    data = {row["case"]: row for row in rows}
    if len(data) != len(rows):
        raise RuntimeError("对比 CSV 的病例名不唯一")
    return data


def numeric(row, key):
    s = str(row.get(key, "")).strip()
    return float(s) if s and s.lower() not in {"nan", "none"} else float("nan")


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def selected_case_reasons(history, val_names):
    if CASE_MODE not in {"selected", "all"}:
        raise ValueError("CASE_MODE 只能取 'selected' 或 'all'")
    selection = OrderedDict()

    def add(name, reason):
        if name not in selection:
            selection[name] = []
        if reason not in selection[name]:
            selection[name].append(reason)

    def sorted_metric(metric, reverse=False, eligible=None):
        items = []
        for name in val_names:
            if eligible is not None and not eligible(history[name]):
                continue
            v = numeric(history[name], "Delta_SegResNet_minus_UNet_" + metric)
            if np.isfinite(v):
                items.append((v, name))
        return sorted(items, key=lambda x: (x[0], x[1]), reverse=reverse)

    if CASE_MODE == "all":
        for name in val_names:
            add(name, "fixed validation set")
        return selection

    for _, name in sorted_metric("TC", reverse=True)[:TOP_TC]:
        add(name, "Top TC improvement")
    for _, name in sorted_metric("TC")[:BOTTOM_TC]:
        add(name, "Largest TC decline")
    for _, name in sorted_metric(
        "ET", eligible=lambda r: not is_true(r["ET_GT_empty"])
    )[:BOTTOM_ET]:
        add(name, "Largest ET decline")
    if INCLUDE_ALL_ET_NEGATIVE:
        for name in val_names:
            if is_true(history[name]["ET_GT_empty"]):
                add(name, "ET GT-negative / FP audit")
    if INCLUDE_MEDIAN_TC:
        valid = sorted_metric("TC")
        if valid:
            _, name = min(valid, key=lambda x: abs(x[0]))
            add(name, "Near-zero TC delta reference")
    return selection


def dice(pred, gt):
    a = np.asarray(pred, dtype=bool)
    b = np.asarray(gt, dtype=bool)
    n_gt = int(np.count_nonzero(b))
    if n_gt == 0:
        return float("nan")
    return float(2 * np.count_nonzero(a & b) / (np.count_nonzero(a) + n_gt))


def regions(seg):
    return {"WT": seg > 0, "TC": (seg == 2) | (seg == 3), "ET": seg == 3}


def all_dice(pred, gt):
    res = {f"Class{c}": dice(pred == c, gt == c) for c in (1, 2, 3)}
    pm, gm = regions(pred), regions(gt)
    for reg in ("WT", "TC", "ET"):
        res[reg] = dice(pm[reg], gm[reg])
    return res


def check_historical_dice(case_name, pred, gt, row, model_name):
    current = all_dice(pred, gt)
    for metric, value in current.items():
        expected = numeric(row, f"{model_name}_{metric}")
        if np.isnan(value) and np.isnan(expected):
            continue
        if not (np.isfinite(value) and np.isfinite(expected)):
            raise RuntimeError(f"{case_name}: {model_name} {metric} NaN 规则与历史 CSV 不一致")
        if abs(value - expected) > METRIC_TOLERANCE:
            raise RuntimeError(
                f"{case_name}: {model_name} {metric} 重新推理 Dice={value:.5f}，"
                f"历史 CSV={expected:.5f}，不一致！\n"
                "已停止，避免生成错位/错误图；检查权重、预处理与 MONAI 版本。"
            )
    return current


def region_errors(pred, gt):
    pm, gm = regions(pred), regions(gt)
    ans = {}
    for reg in ("WT", "TC", "ET"):
        p, g = pm[reg], gm[reg]
        ans[reg] = {
            "gt": int(np.count_nonzero(g)),
            "pred": int(np.count_nonzero(p)),
            "TP": int(np.count_nonzero(p & g)),
            "FP": int(np.count_nonzero(p & ~g)),
            "FN": int(np.count_nonzero(~p & g)),
        }
    return ans


# 切片选择、颜色、成图（所有列共享同一 3D 数组 / 坐标）
def axial_peak(mask):
    # mask 的空间维度为 RAS 预处理之后的 (X, Y, Z)
    score = np.count_nonzero(mask, axis=(0, 1))
    return int(np.argmax(score)) if np.any(score) else mask.shape[2] // 2


def choose_slices(gt, unet_pred, seg_pred):
    g, u, s = regions(gt), regions(unet_pred), regions(seg_pred)
    tc_gt = axial_peak(g["TC"])
    e1 = g["TC"] ^ u["TC"]
    e2 = g["TC"] ^ s["TC"]
    diff = np.count_nonzero(e1, axis=(0, 1)) + np.count_nonzero(e2, axis=(0, 1))
    tc_err = int(np.argmax(diff)) if np.any(diff) else tc_gt
    if np.any(g["ET"]):
        et_focus = axial_peak(g["ET"])
        et_selection = "GT_ET_max"
    else:
        # GT 无 ET 时，突出两种模型合计误报最多的轴向层。
        et_fp = np.count_nonzero(u["ET"], axis=(0, 1)) + np.count_nonzero(s["ET"], axis=(0, 1))
        et_focus = int(np.argmax(et_fp)) if np.any(et_fp) else tc_gt
        et_selection = "ET_FP_max (GT_empty)" if np.any(et_fp) else "TC_GT_max (ET_empty)"
    center_points = np.argwhere(g["TC"])
    center = tuple(np.round(center_points.mean(axis=0)).astype(int)) if center_points.size else (
        gt.shape[0] // 2, gt.shape[1] // 2, tc_gt
    )
    return {"tc_gt_z": tc_gt, "tc_error_z": tc_err, "et_z": et_focus,
            "et_rule": et_selection, "center_xyz": tuple(int(x) for x in center)}


def plane(arr3d, view, index):
    if view == "axial":
        part = arr3d[:, :, index]
    elif view == "coronal":
        part = arr3d[:, index, :]
    elif view == "sagittal":
        part = arr3d[index, :, :]
    else:
        raise ValueError(view)
    return np.rot90(np.asarray(part))


def grayscale(img):
    arr = np.asarray(img, dtype=np.float32)
    vals = arr[np.isfinite(arr) & (np.abs(arr) > 1e-8)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    low, high = np.percentile(vals, [1, 99.5])
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def overlay_rgba(labels, colors, alpha=0.48):
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    for value, rgb in colors.items():
        mask = labels == value
        rgba[mask, :3] = rgb
        rgba[mask, 3] = alpha
    return rgba


def draw_background(ax, img, title):
    ax.imshow(grayscale(img), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_axis_off()


def overlay_seg(ax, label_slice):
    ax.imshow(overlay_rgba(label_slice, LABEL_COLORS), interpolation="nearest")


def plot_label_comparison(case_id, image, gt, pred_unet, pred_seg, view, index, out_path, subtitle):
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 9), dpi=135)
    for row, (channel, modality) in enumerate(MODALITIES):
        mri_slice = plane(image[channel], view, index)
        masks = [None, plane(gt, view, index), plane(pred_unet, view, index), plane(pred_seg, view, index)]
        for col, label in enumerate(("MRI", "Ground Truth", "3D U-Net", "SegResNet")):
            draw_background(axes[row, col], mri_slice, f"{modality} | {label}")
            if masks[col] is not None:
                overlay_seg(axes[row, col], masks[col])
    elements = [Patch(facecolor=LABEL_COLORS[x], label=y) for x, y in (
        (1, "1 Edema"), (2, "2 Non-enhancing"), (3, "3 Enhancing"))]
    fig.legend(handles=elements, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"{case_id} | {view.title()} slice={index} | {subtitle}", fontsize=13)
    fig.text(0.5, 0.036, "Preprocessed RAS / 1 mm / cropped voxel index (NOT native-image coordinates)",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.075, 1, 0.94), w_pad=0.5, h_pad=0.7)
    fig.savefig(out_path, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def error_labels(pred_mask, gt_mask):
    arr = np.zeros(gt_mask.shape, dtype=np.uint8)
    arr[pred_mask & gt_mask] = 1  # TP
    arr[pred_mask & ~gt_mask] = 2  # FP
    arr[~pred_mask & gt_mask] = 3  # FN
    return arr


def plot_error_comparison(case_id, image, gt, pred_unet, pred_seg, target, z, out_path, desc):
    g, u, s = regions(gt)[target], regions(pred_unet)[target], regions(pred_seg)[target]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.7), dpi=140)
    base = plane(image[0], "axial", z)
    draw_background(axes[0], base, f"FLAIR | GT {target} (outline)")
    gt_slice = plane(g, "axial", z).astype(np.uint8)
    # GT 只标轮廓：避免将 GT 覆盖在灰阶上面导致混淆。
    if np.any(gt_slice) and np.any(~gt_slice.astype(bool)):
        axes[0].contour(gt_slice, levels=[0.5], colors=["#fbbf24"], linewidths=1.3)
    for ax, prediction, title in ((axes[1], u, "3D U-Net"), (axes[2], s, "SegResNet")):
        draw_background(ax, base, title + f" | {target} TP/FP/FN")
        labels = error_labels(plane(prediction, "axial", z), plane(g, "axial", z))
        ax.imshow(overlay_rgba(labels, ERR_COLORS, alpha=0.67), interpolation="nearest")
    elements = [Patch(facecolor=ERR_COLORS[i], label=s) for i, s in (
        (1, "TP true positive"), (2, "FP false positive"), (3, "FN false negative"))]
    fig.legend(handles=elements, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{case_id} | {target} error analysis | axial z={z} | {desc}", fontsize=12)
    fig.text(0.5, 0.03, "All three panels use the EXACT SAME preprocessed slice; GT-empty ET: report FP, Dice=N/A.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(out_path, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_orthogonal(case_id, image, gt, pred_unet, pred_seg, xyz, out_path):
    fig, axes = plt.subplots(3, 4, figsize=(17, 13), dpi=140)
    for row, (view, index) in enumerate((
        ("axial", xyz[2]), ("coronal", xyz[1]), ("sagittal", xyz[0])
    )):
        base = plane(image[0], view, index)
        masks = (None, plane(gt, view, index), plane(pred_unet, view, index), plane(pred_seg, view, index))
        for col, title in enumerate(("FLAIR", "GT", "3D U-Net", "SegResNet")):
            draw_background(axes[row, col], base, f"{view.title()} {index} | {title}")
            if masks[col] is not None:
                overlay_seg(axes[row, col], masks[col])
    fig.legend(handles=[Patch(facecolor=LABEL_COLORS[i], label=lab) for i, lab in (
        (1, "1 Edema"), (2, "2 Non-enhancing"), (3, "3 Enhancing"))],
        loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{case_id} | orthogonal overview | preprocessed RAS center={xyz}", fontsize=13)
    fig.tight_layout(rect=(0, 0.047, 1, 0.955), h_pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 自动量化分析、可审核报告（不杜撰视觉现象）
def csv_save(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def nanmean_column(data, key):
    vals = np.array([numeric(row, key) for row in data], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return (float(vals.mean()), int(vals.size)) if vals.size else (float("nan"), 0)


def summary_metrics(history):
    rows = list(history.values())
    res = {}
    for mdl in MODELS:
        metrics = {}
        for metric in ("Class1", "Class2", "Class3", "WT", "TC", "ET"):
            metrics[metric] = nanmean_column(rows, mdl + "_" + metric)
        metrics["RawMean"] = (float(np.mean([metrics[k][0] for k in ("Class1", "Class2", "Class3")])), len(rows))
        metrics["RegionMean"] = (float(np.mean([metrics[k][0] for k in ("WT", "TC", "ET")])), len(rows))
        res[mdl] = metrics
    return res


def report_markdown(history, selected, results):
    sums = summary_metrics(history)
    a, b = sums["UNet60"], sums["SegResNet60"]
    lines = [
        "# 四模态脑肿瘤 MRI 分割：最终量化分析与可视化审核报告",
        "",
        "> 统计基于 97 例配对 Dice CSV 和本次推理掩膜。",
        "> 数值以三维体积为准；二维切片的形态判断仅供误差分析参考。",
        "",
        "## 1. 实验背景与可复现设置",
        "",
        "- 数据：MSD Task01_BrainTumour；训练 387 例、验证 97 例；四模态 FLAIR/T1/T1gd/T2。",
        "- 预处理：RAS 方向、1 mm 等距重采样、前景裁剪、64³ padding、分模态非零 Z-score。",
        "- 标签：0 背景；1 水肿；2 非增强肿瘤；3 增强肿瘤。WT=1+2+3；TC=2+3；ET=3。",
        "- 3D U-Net：351,484 参数，60 Epoch 中最佳 Epoch 53。",
        "- SegResNet：1,176,852 参数，60 Epoch 中最佳 Epoch 59。",
        "- FP32 完整体积滑动窗口：64³ / overlap 0.25 / SW batch 4；验证集用于选择模型，不是独立测试集。",
        "- Dice 口径：先对每个病例计算 Dice，再对 GT 非空病例取均值；GT 无 ET 时 ET Dice = NaN。",
        "",
        "## 2. 全部 97 例验证集对比",
        "",
        "| 指标 | 3D U-Net | SegResNet | 差值 (SegResNet - U-Net) |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("Class1", "Class2", "Class3", "RawMean", "WT", "TC", "ET", "RegionMean"):
        ua, sb = a[metric][0], b[metric][0]
        lines.append(f"| {metric} | {ua:.4f} | {sb:.4f} | {sb - ua:+.4f} |")
    lines += [
        "",
        f"- WT、TC 的有效病例数均为 97；ET 有效病例数为 {a['ET'][1]}。",
        "- **主要发现**：TC 在 SegResNet 下更好；ET 的差距小且方向相反，不能声称 SegResNet 全部区域均优。",
        "- 模型复杂度：SegResNet 的参数量约为 3D U-Net 的 3.35 倍；没有等参数量控制。",
        "- 单次模型评估不等同于多随机种子置信度，也不等同于外部独立测试。",
        "",
        "## 3. 可视化病例及真实掩膜误差统计",
        "",
        "以下 FP/FN/TP 数量均在同一 RAS + 1 mm 预处理体素坐标中统计。",
        "图像中的 Slice index 是预处理后的索引，**不是**原始患者的 NIfTI voxel index。",
        "",
        "| 病例 | 选择理由 | ΔTC | ΔET | U-Net TC FP/FN | SegResNet TC FP/FN | 图片 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        name = row["case"]
        stem = name.removesuffix(".nii.gz")
        delta_tc = row["delta_TC"]
        delta_et = row["delta_ET"]
        et_txt = "N/A (GT empty)" if not np.isfinite(delta_et) else f"{delta_et:+.4f}"
        fig_path = f"figures/{stem}_axial_TC_GT_peak.png"
        lines.append(
            f"| {name} | {row['reason']} | {delta_tc:+.4f} | {et_txt} | "
            f"{row['UNet_TC_FP']}/{row['UNet_TC_FN']} | {row['SegResNet_TC_FP']}/{row['SegResNet_TC_FN']} | "
            f"[四列对齐图]({fig_path}) |"
        )
    lines += [
        "",
        "## 4. 误差原因：可由图像进一步验证的假设（不是自动确认的事实）",
        "",
        "- 在 TC 提升较大的病例，查看 SegResNet 是否减少核心区 FN（漏分）或 FP（误报），以及是否改善病灶连通性。",
        "- 在 TC 退步病例，查看 SegResNet 是否出现更多区域外 FP、核心区 FN 或标签 1/2/3 混淆。",
        "- 在 ET 退步病例，使用 `*_axial_ET_errors.png` 查看 FP（红色）和 FN（蓝色）位于何处。",
        "- 对 GT 无 ET 的病例，**不要引用 ET Dice**；比较两模型的 ET FP 体素数和误报位置。",
        "- 如果单个选定切片与全体积错误统计看起来不一致，这是正常的：全体积 Dice/FP/FN 不是某一切片的结果。",
        "",
        "## 5. 项目结论（基于量化数据）",
        "",
        "在固定数据划分和 60 Epoch 训练预算下，SegResNet 相比自建 3D U-Net 在验证集 TC Dice 上有改善，",
        "Region Mean Dice 也有小幅增加，但 ET 略下降，且 SegResNet 参数量显著更多。",
        "因此将 SegResNet 作为区域均值较高的候选网络，同时保留 U-Net 作为轻量基线；不把优势描述为全面或显著。",
        "从工程角度，已完成统一四模态读取、3D 分割训练、全体积滑窗推理、复合区域评价、逐病例对比和可视化。",
        "",
        "## 6. 剩余必做审核与限制",
        "",
        "1. 人工逐张查看 `figures/`，确认 MRI 与 GT 空间对齐、两种预测颜色位置正确；记录具体案例的形态学判断。",
        "2. 如需独立泛化结论，另建测试集或开展重复随机种子/交叉验证；当前 97 例用于训练期间选优。",
        "3. ET GT 阴性病例原评价排除其 Dice，必须同时报告 ET FP 体素。",
        "4. 训练记录、源码和环境依赖整理进 README，记录 MONAI/PyTorch/CUDA 版本。",
        "5. 当前自动图像仅作为研究/项目可视化，不作临床诊断用途。",
        "",
    ]
    return "\n".join(lines) + "\n"


def html_escape(value):
    from html import escape
    return escape(str(value), quote=True)


def gallery_html(rows):
    cards = []
    for r in rows:
        stem = r["case"].removesuffix(".nii.gz")
        imgs = [
            ("FLAIR / T1gd segmentation", f"figures/{stem}_axial_TC_GT_peak.png"),
            ("TC errors: TP/FP/FN", f"figures/{stem}_axial_TC_errors.png"),
            ("ET errors: TP/FP/FN", f"figures/{stem}_axial_ET_errors.png"),
            ("Axial / coronal / sagittal", f"figures/{stem}_orthogonal.png"),
        ]
        figures = "".join(
            f'<figure><a href="{html_escape(path)}" target="_blank"><img loading="lazy" '
            f'src="{html_escape(path)}" alt="{html_escape(title)}"></a>'
            f'<figcaption>{html_escape(title)}</figcaption></figure>' for title, path in imgs
        )
        cards.append(
            f'<section><h2>{html_escape(r["case"])}</h2><p>{html_escape(r["reason"])} | '
            f'ΔTC = {float(r["delta_TC"]):+.4f}</p><div class="pics">{figures}</div></section>'
        )
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MedSeg3D | 97-case validation visualization</title>
<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0 auto;padding:20px;
max-width:1600px;background:#f8fafc;color:#1f2937}h1{font-size:27px}h2{font-size:19px}
section{background:white;padding:18px;margin:22px 0;border:1px solid #e5e7eb;border-radius:10px}
.pics{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
figure{margin:0}img{display:block;max-width:100%;border:1px solid #ddd;border-radius:5px}
figcaption{font-size:13px;color:#555;padding:6px}p{line-height:1.6}
</style></head><body><h1>MedSeg3D | Same-slice prediction comparison</h1>
<p>Green = TP, Red = FP, Blue = FN in error maps. ET GT-empty cases exclude ET Dice.
Images are in RAS resampled/cropped voxel space, not native NIfTI indices.</p>
""" + "\n".join(cards) + "</body></html>"


# 主程序：先核对数据/权重，再推理和生成图像
def main():
    print("=" * 68)
    print("脑肿瘤 MRI 三维分割：3D U-Net / SegResNet 可视化与误差分析")
    print("=" * 68)
    print("任务模式：", CASE_MODE)
    print("输出路径：", OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("运行设备：", device)
    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(0))
    else:
        warnings.warn("未检测到 CUDA，3D MRI 滑动窗口推理可能很慢")

    history = read_csv_data()
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError("找不到固定验证集名单：" + str(SPLIT_PATH))
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)
    val_names = split["validation"]
    if len(val_names) != EXPECTED_VAL_N or len(set(val_names)) != EXPECTED_VAL_N:
        raise RuntimeError("验证集数量或去重检查失败，预期 97 例")
    if set(split.get("training", [])) & set(val_names):
        raise RuntimeError("训练集与验证集名单重叠")
    if set(history) != set(val_names):
        raise RuntimeError("历史逐病例 CSV 与固定验证集名单不一致")
    cases = selected_case_reasons(history, val_names)
    print(f"本次选择 {len(cases)} / 97 例：")
    for case, reason in cases.items():
        print(f"  {case}: {', '.join(reason)}")

    data = []
    for name in cases:
        image_path, label_path = IMAGES_DIR / name, LABELS_DIR / name
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"病例 {name} 缺少影像或 GT：{image_path} / {label_path}")
        data.append({"image": str(image_path), "label": str(label_path)})

    # 对于选定病例直接逐一索引，避免 Windows 多进程 spawn 和无谓加载所有 97 例。
    dataset = PersistentDataset(data=data, transform=val_transform(), cache_dir=CACHE_DIR)
    models = make_models(device)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    result_rows = []

    for i, (case_name, reasons) in enumerate(cases.items(), start=1):
        print(f"\n[{i}/{len(cases)}] 开始：{case_name}")
        sample = dataset[i - 1]
        img = sample["image"].to(dtype=torch.float32).cpu().numpy()
        gt = sample["label"][0].to(dtype=torch.int64).cpu().numpy().astype(np.uint8)
        if img.ndim != 4 or img.shape[0] != 4 or img.shape[1:] != gt.shape:
            raise RuntimeError(f"{case_name} MRI、GT 维度不匹配：{img.shape} vs {gt.shape}")
        if not np.isin(np.unique(gt), [0, 1, 2, 3]).all():
            raise RuntimeError(f"{case_name} GT 标签并非 [0,1,2,3]")
        if not np.isfinite(img).all():
            raise RuntimeError(f"{case_name} MRI 出现 NaN / Inf")

        preds = {}
        for name, model in models.items():
            # 与之前同一输入、同一评价脚本参数；两个模型绝不分开裁剪或调整坐标。
            x = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                logits = sliding_window_inference(
                    inputs=x, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE,
                    predictor=model, overlap=OVERLAP, sw_device=device,
                    device=torch.device("cpu"),
                )
                pred = torch.argmax(logits, dim=1)[0].to(dtype=torch.uint8).numpy()
            del logits, x
            if pred.shape != gt.shape:
                raise RuntimeError(f"{case_name} {name} 预测/GT 空间尺寸不一致")
            check_historical_dice(case_name, pred, gt, history[case_name], name)
            preds[name] = pred
            print(f"  {name}: TC={dice(regions(pred)['TC'], regions(gt)['TC']):.4f}")

        u, s = preds["UNet60"], preds["SegResNet60"]
        chosen = choose_slices(gt, u, s)
        stem = case_name.removesuffix(".nii.gz")
        plot_label_comparison(
            case_name, img, gt, u, s, "axial", chosen["tc_gt_z"],
            FIG_DIR / f"{stem}_axial_TC_GT_peak.png", "max GT TC area",
        )
        plot_error_comparison(
            case_name, img, gt, u, s, "TC", chosen["tc_error_z"],
            FIG_DIR / f"{stem}_axial_TC_errors.png", "max combined TC errors",
        )
        plot_error_comparison(
            case_name, img, gt, u, s, "ET", chosen["et_z"],
            FIG_DIR / f"{stem}_axial_ET_errors.png", chosen["et_rule"],
        )
        plot_orthogonal(
            case_name, img, gt, u, s, chosen["center_xyz"],
            FIG_DIR / f"{stem}_orthogonal.png",
        )

        err_u, err_s = region_errors(u, gt), region_errors(s, gt)
        rec = {
            "case": case_name, "reason": "; ".join(reasons),
            "delta_TC": numeric(history[case_name], "Delta_SegResNet_minus_UNet_TC"),
            "delta_ET": numeric(history[case_name], "Delta_SegResNet_minus_UNet_ET"),
            "ET_GT_empty": bool(not np.any(gt == 3)),
            "shape_RAS_cropped": "x".join(map(str, gt.shape)),
            "slice_TC_GT_z": chosen["tc_gt_z"],
            "slice_TC_error_z": chosen["tc_error_z"],
            "slice_ET_z": chosen["et_z"],
            "ET_focus_rule": chosen["et_rule"],
            "center_RAS_crop_xyz": "x".join(map(str, chosen["center_xyz"])),
        }
        for region in ("WT", "TC", "ET"):
            rec[f"GT_{region}_voxels"] = err_u[region]["gt"]
            for name, stats in (("UNet", err_u), ("SegResNet", err_s)):
                for measure in ("pred", "TP", "FP", "FN"):
                    rec[f"{name}_{region}_{measure}"] = stats[region][measure]
        result_rows.append(rec)
        print(
            f"  图片 4 张已生成。TC [GT {err_u['TC']['gt']} vox]："
            f"U-Net FP/FN {err_u['TC']['FP']}/{err_u['TC']['FN']}，"
            f"SegResNet FP/FN {err_s['TC']['FP']}/{err_s['TC']['FN']}"
        )
        del img, gt, preds, u, s, sample

    csv_path = OUT_DIR / "selected_cases_error_stats.csv"
    csv_save(csv_path, result_rows)
    report_path = OUT_DIR / "最终项目分析.md"
    report_path.write_text(report_markdown(history, cases, result_rows), encoding="utf-8-sig")
    gallery_path = OUT_DIR / "image_gallery.html"
    gallery_path.write_text(gallery_html(result_rows), encoding="utf-8")
    manifest = {
        "case_mode": CASE_MODE, "case_count": len(result_rows), "fixed_validation_size": len(val_names),
        "model_epochs": {x: y["epoch"] for x, y in MODELS.items()},
        "model_checkpoints": {x: y["filename"] for x, y in MODELS.items()},
        "preprocess": "RAS 1mm / CropForeground / Pad64 / channel Z-score",
        "inference": {"FP32": True, "ROI_SIZE": ROI_SIZE, "SW_BATCH_SIZE": SW_BATCH_SIZE, "overlap": OVERLAP},
        "ET_empty_rule": "Dice=NaN, but FP voxels reported",
        "case_reasons": cases, "figures_per_case": 4,
        "notes": "Patient-native spatial orientation / coordinate mapping is not provided by these PNGs",
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 68)
    print(f"完成：{len(result_rows)} 例，生成 {len(result_rows) * 4} 张图片")
    print("图像目录：", FIG_DIR)
    print("浏览画廊：", gallery_path)
    print("误差体素表：", csv_path)
    print("项目结论与人工审核清单：", report_path)
    print("注意：自动报告不替代对实际图片进行人工审核。")
    print("=" * 68)


if __name__ == "__main__":
    main()
