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


def _valid_group_count(num_channels: int, max_groups: int = 8) -> int:
    """Return a GroupNorm group count that always divides ``num_channels``."""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class EmbedFC(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.shape[-1])
        return self.model(x)


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        if is_res and not self.same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = self.shortcut(x) + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    """Downsampling block for the larger diffusion UNet variant."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    """Upsampling block for the larger diffusion UNet variant."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), dim=1)
        return self.model(x)


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


class TinyUNetDiffusion(nn.Module):
    """Tiny U-Net with scalar time embedding for diffusion workflows."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 10) -> None:
        super().__init__()
        if base <= 0:
            raise ValueError("base must be positive")

        enc_ch1 = base
        enc_ch2 = base * 2
        bottleneck = int(round(base * 2.8))

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base = base
        self.enc_ch2 = enc_ch2
        self.bottleneck_ch = bottleneck

        self.enc1 = DoubleConv(in_ch, enc_ch1)
        self.down1 = nn.Conv2d(enc_ch1, enc_ch2, kernel_size=3, stride=2, padding=1, bias=True)

        self.enc2 = DoubleConv(enc_ch2, enc_ch2)
        self.down2 = nn.Conv2d(enc_ch2, bottleneck, kernel_size=3, stride=2, padding=1, bias=True)

        self.bottleneck = DoubleConv(bottleneck, bottleneck)

        self.timeembed1 = EmbedFC(1, bottleneck)
        self.timeembed2 = EmbedFC(1, enc_ch2)

        self.up1 = UpBlock(bottleneck, enc_ch2, enc_ch2)
        self.up2 = UpBlock(enc_ch2, enc_ch1, enc_ch1)

        self.out_conv = nn.Conv2d(enc_ch1, out_ch, kernel_size=3, padding=1, bias=True)
        # self.activation = nn.Tanh()

        self.apply(TinyUNet._init_weights)

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Apply tiny U-Net with diffusion timestep conditioning.

        Args:
            x: Input tensor of shape ``(B, in_ch, H, W)``.
            t: Diffusion timestep tensor of shape ``(B,)`` or ``(B, 1)``.

        Returns:
            Tensor of shape ``(B, out_ch, H, W)`` with values in ``[-1, 1]``.
        """
        if t is None:
            t = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.to(device=x.device, dtype=x.dtype)

        skip1 = self.enc1(x)
        down1 = self.down1(skip1)

        skip2 = self.enc2(down1)
        down2 = self.down2(skip2)

        bottleneck = self.bottleneck(down2)
        temb1 = self.timeembed1(t).view(-1, self.bottleneck_ch, 1, 1)
        bottleneck = bottleneck + temb1

        up1 = self.up1(bottleneck, skip2)
        temb2 = self.timeembed2(t).view(-1, self.enc_ch2, 1, 1)
        up1 = up1 + temb2

        up2 = self.up2(up1, skip1)

        out = self.out_conv(up2)
        #out = self.activation(out)
        return out


class BigUNetDiffusion(nn.Module):
    """Larger diffusion UNet variant with two-stage time embeddings."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 64,
        height: int = 64,
    ) -> None:
        super().__init__()
        if height % 4 != 0:
            raise ValueError("height must be divisible by 4")
        if n_feat <= 0:
            raise ValueError("n_feat must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        self.to_vec = nn.Sequential(nn.AvgPool2d(self.h // 4), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(_valid_group_count(2 * n_feat), 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(_valid_group_count(n_feat), n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.out_channels, 3, 1, 1),
        )

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = self.init_conv(x)
        down1 = self.down1(x0)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float()

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec + temb1)
        up2 = self.up1(up1, down2)
        up3 = self.up2(up2 + temb2, down1)
        return self.out(torch.cat((up3, x0), 1))


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
