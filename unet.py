# -*- coding: utf-8 -*-
"""
标准 U-Net（医学影像分割基线）

和 learning_start/Pytorch_Unet_01 里的 MiniUNet 的区别：
- MiniUNet 只有 3,649 个参数，是理解结构用的玩具；
- 这里是真正的 U-Net：4 层下采样 / 4 层上采样，每层双卷积，通道数 16->32->64->128，
  参数量约 2M 量级，属于医学影像分割常用的基线规模。

结构（和原论文一致）：
    Encoder:  conv -> pool -> conv -> pool -> conv -> pool -> conv
    Bottleneck
    Decoder:  upconv -> cat(skip) -> conv -> ... -> 1x1 conv -> logits

注意：输出是 logits（没有过 sigmoid），所以训练时必须配合
      nn.BCEWithLogitsLoss()，预测时再自己 sigmoid。
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv3x3 -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """4 层 U-Net。

    in_ch  : 输入通道，灰度 X 光填 1
    out_ch : 输出通道，二分类分割填 1
    base   : 第一层通道数，16 -> 约 2M 参数；32 -> 约 7.8M 参数
    """

    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        b = base

        # ---- Encoder ----
        self.enc1 = DoubleConv(in_ch, b)          # H,   W
        self.enc2 = DoubleConv(b, b * 2)          # H/2
        self.enc3 = DoubleConv(b * 2, b * 4)      # H/4
        self.enc4 = DoubleConv(b * 4, b * 8)      # H/8
        self.pool = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(b * 8, b * 16)   # H/16

        # ---- Decoder ----
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        self.head = nn.Conv2d(b, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        x = self.bottleneck(self.pool(s4))

        # Decoder（每层：上采样 -> 与对应 skip 拼接 -> 双卷积）
        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for base in (8, 16, 32):
        model = UNet(base=base)
        n = count_parameters(model)
        x = torch.randn(2, 1, 256, 256)
        y = model(x)
        ok = (x.shape[-2:] == y.shape[-2:]) and (y.shape[1] == 1)
        print(f"base={base:2d}  参数量={n:>9,}  输入{tuple(x.shape)} -> 输出{tuple(y.shape)}  尺寸对齐={ok}")

    model = UNet(base=16)
    print()
    print("默认 base=16 的结构：")
    print(model)
