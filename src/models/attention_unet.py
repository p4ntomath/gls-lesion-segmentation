"""Attention U-Net for binary GLS lesion segmentation.

Proposal ref: §4.2.7, Oktay et al. [11]
Config: configs/attention_unet.yaml

Same encoder/backbone as unet.py (DoubleConv, Down, reused directly -- not
duplicated), with an attention gate inserted into every skip connection
(Figure 4 in the proposal). The gate learns to weight the encoder's skip
features before they're concatenated with the decoder's upsampled features,
suppressing background and emphasising lesion-relevant regions.

Where the gate signal comes from: in the original paper the gating signal
is the coarser (pre-upsample) decoder feature map, and the encoder skip is
downsampled to match it. Here, because AttentionUp already upsamples the
decoder feature to the skip's resolution via ConvTranspose2d (same as plain
Up in unet.py), both tensors are already at matching spatial size *and*
matching channel count (out_channels) by construction , so the gate can
attend directly, with no extra resampling step needed. This is the same
simplification used by most modern Attention U-Net implementations.
"""

from __future__ import annotations

import torch
from torch import nn

from .unet import DoubleConv, Down


class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al., Fig. 2). Learns a per-pixel
    attention coefficient in [0, 1] for the skip connection, conditioned on
    the decoder's gating signal.

    gate and skip must have the same spatial size. Their channel counts can
    differ (gate_channels vs skip_channels) -- both are projected to
    inter_channels before being combined.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True)
        self.phi_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attn = self.relu(self.theta_x(skip) + self.phi_g(gate))
        attn = self.psi(attn)  # (B, 1, H, W), values in [0, 1]
        return skip * attn


class AttentionUp(nn.Module):
    """Upsample, gate the skip connection through AttentionGate, concat,
    then DoubleConv. Drop-in replacement for unet.py's Up."""

    def __init__(self, in_channels: int, out_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attn = AttentionGate(
            gate_channels=out_channels, skip_channels=out_channels, inter_channels=inter_channels
        )
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

        skip = self.attn(gate=x, skip=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """Attention U-Net for binary segmentation, returning raw logits.

    Args mirror unet.UNet exactly, plus:
        attention_inter_channels: channel width inside each attention gate.
            If None (default), scales per level as out_channels // 2 (a
            common default). If an int is given (as in
            configs/attention_unet.yaml, currently 32), that fixed width is
            used at every level instead.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        attention_inter_channels: int | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        filters = [base_filters * (2**index) for index in range(depth + 1)]

        self.in_conv = DoubleConv(in_channels, filters[0])
        self.downs = nn.ModuleList([Down(filters[i], filters[i + 1]) for i in range(depth - 1)])
        self.bottleneck = Down(filters[depth - 1], filters[depth])

        self.ups = nn.ModuleList(
            [
                AttentionUp(
                    filters[i + 1],
                    filters[i],
                    inter_channels=attention_inter_channels or max(filters[i] // 2, 1),
                )
                for i in reversed(range(depth))
            ]
        )

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
    # Quick self-test, mirroring unet.py's.
    for size in (256,):
        model = AttentionUNet(
            in_channels=3, out_channels=1, base_filters=64, depth=4, attention_inter_channels=32
        )
        x = torch.randn(2, 3, size, size)
        y = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}  ({n_params:,} params)")
        assert y.shape == (2, 1, size, size), "output shape must match input H, W"
    print("OK")