# 模型

`unet3d.py` 为实验中使用的轻量级 3D U-Net，采用三层下采样/上采样结构，卷积块使用 InstanceNorm3d 和 ReLU，通过跳跃连接拼接编码端与解码端特征。模型输入输出均为 `(N, 4, D, H, W)`，基础通道数为 8，总参数量 351,484。

SegResNet 使用 MONAI 的 `monai.networks.nets.SegResNet`，网络配置见 `scripts/train_segresnet_60.py`。
