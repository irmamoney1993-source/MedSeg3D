"""Lightweight 3D U-Net for four-channel brain MRI segmentation.
The model takes [B,4,D,H,W] and returns four class logits.
Architecture: three encoder/decoder stages with skip connections."""
import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm3d(out_channels, affine=True), nn.ReLU(inplace=True), nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm3d(out_channels, affine=True), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):

    def __init__(self, in_channels=4, num_classes=4, base_channels=8):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv3D(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)
        self.output_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        output = self.output_conv(d1)
        return output
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备：', device)
    model = UNet3D(in_channels=4, num_classes=4, base_channels=8).to(device)
    x = torch.randn(1, 4, 64, 64, 64).to(device)
    with torch.no_grad():
        y = model(x)
    print('\n输入形状：')
    print(x.shape)
    print('\n输出形状：')
    print(y.shape)
    print('\n3D U-Net前向传播测试成功。')
