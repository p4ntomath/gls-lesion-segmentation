"""Baseline U-Net for binary GLS lesion segmentation.

Proposal ref: §4.2.7, Ronneberger et al. [10]
Config: configs/unet.yaml

Encoder-decoder with skip connections (Figure 3 in the proposal). This
implementation uses same padding so feature maps keep matching spatial sizes
at each depth.
"""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x2, spatial size preserved."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pool by 2, then DoubleConv. One encoder stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsample by 2, concat the skip connection, then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            height_diff = skip.shape[-2] - x.shape[-2]
            width_diff = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(
                x,
                [
                    width_diff // 2,
                    width_diff - width_diff // 2,
                    height_diff // 2,
                    height_diff - height_diff // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for binary segmentation returning raw logits."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        filters = [base_filters * (2**index) for index in range(depth + 1)]

        self.in_conv = DoubleConv(in_channels, filters[0])
        self.downs = nn.ModuleList([Down(filters[i], filters[i + 1]) for i in range(depth - 1)])
        self.bottleneck = Down(filters[depth - 1], filters[depth])
        self.ups = nn.ModuleList([Up(filters[i + 1], filters[i]) for i in reversed(range(depth))])
        self.out_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.in_conv(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.out_conv(x)


if __name__ == "__main__":
    # Quick self-test: verify output spatial size matches input size.
    for size in (256, 512):
        model = UNet(in_channels=3, out_channels=1, base_filters=64, depth=4)
        x = torch.randn(2, 3, size, size)
        y = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}  ({n_params:,} params)")
        assert y.shape == (2, 1, size, size), "output shape must match input H, W"
    print("OK")
