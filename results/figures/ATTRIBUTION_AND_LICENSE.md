# MRI 结果图：来源与许可

图像来源于 [Medical Segmentation Decathlon (MSD)](https://medicaldecathlon.com/) 的 `Task01_BrainTumour` 四模态 MRI 与人工分割标签。

数据集引用：Antonelli M. et al., [*The Medical Segmentation Decathlon*](https://doi.org/10.1038/s41467-022-30695-9), *Nature Communications* 13, 4128 (2022)。

原始数据以 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 发布。本目录的衍生 MRI 对比图同样按照 CC BY-SA 4.0 共享，并保留来源和修改说明。

处理内容包括统一 RAS 方向、重采样到 1 mm、前景裁剪、强度显示映射，以及叠加 Ground Truth、3D U-Net 和 SegResNet 的预测。选定切片用于展示分割误差，并非随机抽样。

以上图像许可不自动授予仓库 Python 源码的使用许可。
