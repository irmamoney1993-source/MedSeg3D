# 数据准备

数据集：[Medical Segmentation Decathlon — Task01_BrainTumour](https://medicaldecathlon.com/dataaws/)。按其发布条款下载并解压，目录结构为：

```text
data/raw/Task01_BrainTumour/
├── imagesTr/BRATS_XXX.nii.gz
└── labelsTr/BRATS_XXX.nii.gz
```

四通道顺序为 FLAIR、T1、T1gd、T2。标签 0/1/2/3 分别为背景、水肿、非增强肿瘤和增强肿瘤。训练和验证沿用 [`outputs/splits/train_val_split.json`](../outputs/splits/train_val_split.json) 中的固定划分。

不在 Git 仓库中存放原始 NIfTI、训练缓存及模型权重。
