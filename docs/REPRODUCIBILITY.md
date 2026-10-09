# 实验设置与复现

## 数据划分

采用 MSD `Task01_BrainTumour` 中 484 例带标签 MRI。训练 387 例、验证 97 例；名单和随机种子保存在 [`train_val_split.json`](../outputs/splits/train_val_split.json)。原始 MRI 不在仓库内；路径要求见 [data/README.md](../data/README.md)。

## 网络与训练设置

| 项目 | 设置 |
|---|---|
| 输入模态 | FLAIR、T1、T1gd、T2 |
| 预处理 | RAS、1 mm 重采样、前景裁剪、最小 64³ Pad、逐模态非零区 Z-score |
| Patch | 64³；正/负采样权重 1:1；每例每轮 1 个 Patch |
| Batch Size | 1 |
| 优化器与损失 | Adam，学习率 1e-4；Dice + CE，CE 类别权重全为 1 |
| 模型 | 3D U-Net（base_channels=8）；MONAI SegResNet（init_filters=8） |
| Epoch | 最多 60；验证集 Raw Mean Dice 选优 |
| 验证推理 | FP32 全体积滑动窗口；ROI 64³，overlap=0.25，窗口 batch=4 |

两模型训练预算与数据划分一致，但模型参数量和逐轮随机 Patch 并不严格匹配，因此属于网络架构对照，而不是等参数规模的消融实验。

## 实验版本

3D U-Net 在 `scripts/train_unet_baseline_40.py` 中最初按 40 Epoch 配置训练，随后恢复 Last 权重续训到 60 Epoch；最佳权重为 Epoch 53，文件名仍为 `unet_baseline_40_fast_best.pth`。该脚本保留了最初的 40 Epoch 参数记录。若从零训练 60 轮，需调整 `NUM_EPOCHS`，并使用新的 `RUN_NAME` 防止覆盖原权重。

SegResNet 使用 `scripts/train_segresnet_60.py`；从零训练时将 `RUN_MODE` 设为 `train`，`NUM_EPOCHS=60`，最佳权重为 Epoch 59。默认 `smoke` 模式仅用于模型检查。

旧版加权 CE 和其他对比实验保存在 `scripts/legacy/`。它们用于记录实验迭代，不作为本次 60-Epoch 网络比较的主要运行入口。

## 验证与结果复核

```bash
python scripts/check_public_release.py
python scripts/verify_results.py
```

上面两个脚本仅使用 Python 标准库，不依赖 GPU 或 MRI。原始数据和最佳权重准备好后，可按以下顺序运行：

```bash
python scripts/check_project.py
python scripts/compare_unet60_vs_segresnet60.py
python scripts/visualize_unet_segresnet_cases.py
```

推理脚本检查 Checkpoint 中的实验名称、最佳 Epoch 和网络参数量。模型权重存于 `outputs/checkpoints/`，MRI 存于 `data/raw/Task01_BrainTumour/`，生成的日志和分析结果写入 `outputs/`，不会随 Git 提交。

WT={1,2,3}、TC={2,3}、ET={3}。当 GT 不含目标区域时，Dice 记为 NaN，不参与宏平均，预测假阳性另作统计。原始类别及 WT、TC、ET 均按有效病例平均；Raw Mean 与 Region Mean 分别是三个类别均值和三个区域均值的算术平均。

## 复现范围

仓库保存了源码、固定划分和逐病例指标，可核验结果表的内部一致性。未提供原始数据、最佳权重及训练时精确依赖版本；原始 40+20 轮续训随机序列也未完整保存。因此不保证重训获得与已发布权重完全一致的数值。报告中的 97 例既参与最佳 Epoch 选择，也参与模型比较，不属于独立测试。
