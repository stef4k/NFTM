"""Tiny U-Net model for image inpainting.

This module defines a very small U-Net style architecture intended to match the
parameter count of the NFTM TinyController when ``base=10``. The model accepts
four-channel inputs (RGB plus mask) and produces RGB outputs in ``[-1, 1]`` via
``tanh`` activation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """A pair of 3x3 convolutions each followed by GELU activation."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connection and double convolution."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)
        self.double_conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Spatial dimensions should match, but guard against odd sizes.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x


class TinyUNet(nn.Module):
    """A capacity-matched tiny U-Net for image inpainting."""

    def __init__(self, in_ch: int = 4, out_ch: int = 3, base: int = 10) -> None:
        """Create the tiny U-Net model.

        Args:
            in_ch: Number of input channels (RGB + mask by default).
            out_ch: Number of output channels (RGB by default).
            base: Base channel count controlling model width.
        """
        super().__init__()
        if base <= 0:
            raise ValueError("base must be positive")

        enc_ch1 = base
        enc_ch2 = base * 2
        # Slightly reduce bottleneck width relative to 3 * base to hit parameter target.
        bottleneck = int(round(base * 2.8))

        self.enc1 = DoubleConv(in_ch, enc_ch1)
        self.down1 = nn.Conv2d(enc_ch1, enc_ch2, kernel_size=3, stride=2, padding=1, bias=True)

        self.enc2 = DoubleConv(enc_ch2, enc_ch2)
        self.down2 = nn.Conv2d(enc_ch2, bottleneck, kernel_size=3, stride=2, padding=1, bias=True)

        self.bottleneck = DoubleConv(bottleneck, bottleneck)

        self.up1 = UpBlock(bottleneck, enc_ch2, enc_ch2)
        self.up2 = UpBlock(enc_ch2, enc_ch1, enc_ch1)

        self.out_conv = nn.Conv2d(enc_ch1, out_ch, kernel_size=3, padding=1, bias=True)

        self.activation = nn.Tanh()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize convolution weights with Kaiming normal distribution."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            # ``torch.nn.init.kaiming_normal_`` does not support ``gelu`` directly, but
            # the recommended gain for GELU activations matches the ReLU gain
            # (``sqrt(2)``).  We therefore fall back to an explicit normal
            # initialisation using that gain.  This keeps the variance identical to
            # what Kaiming initialisation would produce for GELU while avoiding the
            # unsupported nonlinearity error raised by ``calculate_gain``.
            fan = nn.init._calculate_correct_fan(module.weight, mode="fan_in")
            gain = math.sqrt(2.0)
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                module.weight.normal_(0.0, std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the tiny U-Net to ``x``.

        Args:
            x: Input tensor of shape ``(B, in_ch, H, W)``.

        Returns:
            Tensor of shape ``(B, out_ch, H, W)`` with values in ``[-1, 1]``.
        """
        skip1 = self.enc1(x)
        down1 = self.down1(skip1)

        skip2 = self.enc2(down1)
        down2 = self.down2(skip2)

        bottleneck = self.bottleneck(down2)

        up1 = self.up1(bottleneck, skip2)
        up2 = self.up2(up1, skip1)

        out = self.out_conv(up2)
        out = self.activation(out)
        return out


def _smoke_test() -> None:
    """Run a simple smoke test to validate the implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(in_ch=4, out_ch=3, base=10).to(device)
    param_count = model.count_params()
    print(f"Parameter count: {param_count}")

    x = torch.randn(8, 4, 32, 32, device=device)
    y = model(x)
    assert y.shape == (8, 3, 32, 32)
    assert y.min().item() >= -1.0001 and y.max().item() <= 1.0001
    assert 0.95 * 46375 <= param_count <= 1.05 * 46375

    if torch.cuda.is_available():
        x_cuda = torch.randn(2, 4, 32, 32, device="cuda")
        y_cuda = model(x_cuda)
        assert y_cuda.shape == (2, 3, 32, 32)
        assert y_cuda.min().item() >= -1.0001 and y_cuda.max().item() <= 1.0001


if __name__ == "__main__":
    _smoke_test()
