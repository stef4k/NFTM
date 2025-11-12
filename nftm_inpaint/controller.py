#!/usr/bin/env python3
# controller.py

import torch, torch.nn as nn, torch.nn.functional as F
from nftm_inpaint.unet_model import TinyUNet

# -------------------------- Model --------------------------

class TinyController(nn.Module):
    """
    Inputs: concat(I_t (3ch), M (1ch)) -> conv stack
    Outputs:
      - dI: per-pixel correction (3ch), tanh-clamped
      - gate: per-pixel gate in (0,1) (1ch)
      - log_sigma: per-pixel per-channel log-std (3ch) (kept for compatibility)
    """

    def __init__(self, in_ch=4, width=48):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.head_dI   = nn.Conv2d(width, 3, 3, padding=1)
        self.head_gate = nn.Conv2d(width, 1, 3, padding=1)
        self.head_logS = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, I, M):
        x = torch.cat([I, M], dim=1)  # (B,4,H,W)
        h = self.body(x)
        dI   = self.head_dI(h).tanh()
        gate = torch.sigmoid(self.head_gate(h))
        logS = self.head_logS(h)
        return dI, gate, logS


class UNetController(nn.Module):
    def __init__(self, in_ch=4, base=10):
        super().__init__()
        self.unet = TinyUNet(in_ch=in_ch, out_ch=7, base=base)
        # For controller usage we need linear outputs; override the final activation.
        self.unet.activation = nn.Identity()

    def forward(self, I, M):
        x = torch.cat([I, M], dim=1)
        out = self.unet(x)
        dI_raw, gate_raw, logS = torch.split(out, [3, 1, 3], dim=1)
        dI = torch.tanh(dI_raw)
        gate = torch.sigmoid(gate_raw)
        return dI, gate, logS
