"""
Watermark Decoder (Extractor).

Input:  image in [-1, 1] range, shape [B, 3, H, W]
Output: logits in [B, watermark_length]  (NO sigmoid inside — use BCEWithLogitsLoss)

The range follows the official PIMoG pipeline, where ScreenShooting output is
passed directly to the decoder after only a float cast.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Downsampling conv block: Conv2d -> BN -> SiLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class WatermarkDecoder(nn.Module):
    """
    Lightweight CNN decoder that extracts watermark bits from an image.

    Architecture: 4 downsampling conv blocks -> AdaptiveAvgPool2d(1) -> Linear
    """

    def __init__(self, watermark_length=64):
        super().__init__()
        self.watermark_length = watermark_length

        self.encoder = nn.Sequential(
            ConvBlock(3, 32),     # H/2
            ConvBlock(32, 64),    # H/4
            ConvBlock(64, 128),   # H/8
            ConvBlock(128, 256),  # H/16
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, watermark_length)

    def forward(self, x):
        # x: [B, 3, H, W] in [-1, 1]
        feat = self.encoder(x)          # [B, 256, 1, 1]
        feat = feat.flatten(1)          # [B, 256]
        logits = self.fc(feat)          # [B, watermark_length]
        return logits
