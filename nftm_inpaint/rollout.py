#!/usr/bin/env python3
# rollout.py

import os, random, argparse, json, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import math

# NEW: scale helpers
def downsample_like(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)

def upsample_like(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)

def downsample_mask_minpool(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Keep 'unknown' (0) dominant when shrinking: min-pool in practice via -maxpool(-mask)."""
    B, C, H, W = mask.shape  # C==1
    if H == size and W == size:
        return mask
    # compute stride/ks for power-of-two downs
    factor = H // size
    if (H % size) == 0 and (W % size) == 0 and factor >= 2:
        m = -F.max_pool2d(-mask, kernel_size=factor, stride=factor)
    else:
        # fallback: bilinear then threshold (keeps hard 0 where any 0 existed)
        m = F.interpolate(mask, size=(size, size), mode='bilinear', align_corners=False)
        m = (m > 0.999).float()
    return m

def parse_pyramid_arg(pyr: str, final_size: int):
    """Return an increasing list of sizes ending at final_size."""
    if not pyr:
        return [final_size]
    sizes = [int(s) for s in pyr.split(",") if s.strip()]
    sizes = sorted(set([s for s in sizes if 8 <= s <= final_size]))
    if not sizes or sizes[-1] != final_size:
        sizes.append(final_size)
    return sizes

def split_steps_eval(K_total: int, sizes, steps_arg: str | None = None):
    """Steps per level for eval (sum to K_total). If not provided, give 1 to each coarse, rest to finest."""
    if len(sizes) == 1:
        return [K_total]
    if steps_arg:
        steps = [int(s) for s in steps_arg.split(",") if s.strip()]
        assert sum(steps) == K_total and len(steps) == len(sizes), "pyr_steps must match pyramid and sum to K_eval"
        return steps
    L = len(sizes)
    steps = [1]*(L-1) + [K_total - (L-1)]
    steps[-1] = max(steps[-1], 1)
    return steps

def split_steps_train(K_curr: int, sizes, epoch: int):
    """Steps per level for training curriculum; a bit more coarse in very early epochs."""
    if len(sizes) == 1:
        return [K_curr]
    coarse_each = 2 if (epoch <= 2 and K_curr >= 3) else 1
    L = len(sizes)
    total_coarse = min(coarse_each*(L-1), K_curr-1)
    steps = [coarse_each]*(L-1) + [K_curr - total_coarse]
    # Clamp to >=1 and fix any rounding drift
    for i in range(L-1):
        if steps[i] < 1:
            steps[i] = 1
    steps[-1] = max(steps[-1], 1)
    diff = K_curr - sum(steps)
    steps[-1] += diff
    return steps

def corrupt_images(img, M, noise_std=0.3, noise_kind: str = "gaussian", gaussian_additive: bool = False, **noise_kwargs):
    """
    Corrupt ONLY the missing region (1-M). Known pixels remain clean.

    img: (B,3,H,W) in [-1,1]
    M:   (B,1,H,W) known-mask with 1=known, 0=missing
    Returns: corrupted image in [-1,1]

    noise_kind supported:
      - "gaussian": missing filled with N(0, noise_std)
      - "uniform": missing filled with U(-1, 1)
      - "saltpepper": missing filled with {-1, +1} with prob 'prob' else 0
      - "poisson": missing filled with Poisson sampled (requires 'peak', uses [0,1] mapping)
      - "speckle": missing filled with multiplicative noise around 0 (uses noise_std)
    """
    kind = (noise_kind or "gaussian").lower()
    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    if kind == "gaussian":
        # create tensor with zero-mean Gaussian noise with standard deviation noise_std
        eps = torch.randn_like(img) * float(noise_std)
        # pure noise if not gaussian_additive; else additive noise
        corrupt = (img + eps) if gaussian_additive else eps

    elif kind == "uniform":
        corrupt = (torch.rand_like(img) * 2.0 - 1.0)

    elif kind == "saltpepper":
        prob = float(noise_kwargs.get("prob", 0.10))
        u = torch.rand_like(img)
        corrupt = torch.zeros_like(img)
        corrupt = torch.where(u < (prob / 2.0), torch.ones_like(corrupt), corrupt)
        corrupt = torch.where(u > (1.0 - prob / 2.0), -torch.ones_like(corrupt), corrupt)

    elif kind == "poisson":
        # Shot-noise-like fill: sample Poisson on [0,1] then map back to [-1,1]
        peak = float(noise_kwargs.get("peak", 30.0))  # higher peak => less noise
        x01 = ((img.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
        lam = (x01 * peak).clamp_min(0.0)
        samp = torch.poisson(lam) / peak
        corrupt = (samp * 2.0 - 1.0).to(dtype=dtype)

    elif kind == "speckle":
        # multiplicative gaussian around 0 -> bounded
        n = torch.randn_like(img) * float(noise_std)
        corrupt = (img + img * n)

    else:
        raise ValueError(f"Unknown noise_kind: {noise_kind}")

    # clamp for safety (keeps controller input sane)
    corrupt = corrupt.clamp(-1.0, 1.0)

    # keep known pixels, corrupt missing pixels
    return M * img + (1 - M) * corrupt


def clamp_known(I, I_gt, M):
    # enforce known pixels to match ground truth (hard measurement)
    return I*(1-M) + I_gt*M

def psnr(x, y):
    # images in [-1,1]; peak-to-peak = 2
    mse = F.mse_loss(x, y)
    if mse.item() == 0: return torch.tensor(99.0, device=x.device)
    return 10 * torch.log10(4.0 / mse)

def tv_l1(x, weight=1.0):
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return weight*(dx + dy)

def energy(I, I_gt, tvw=0.01):
    data_term = F.mse_loss(I, I_gt)
    return data_term + tv_l1(I, tvw)

def nftm_step(I, I_gt, M, controller, beta: float | None = None, corr_clip=0.2, 
              clip_decay=1.0, use_gate: bool = False, return_gate=False):
    dI, gate, logS = controller(I.clamp(-1.0, 1.0), M)  # clamp controller input (important!)
    
    if corr_clip is not None and corr_clip > 0:
        dI = dI.clamp(-corr_clip * clip_decay, corr_clip * clip_decay)


    b = 1.0 if beta is None else float(beta)

    if use_gate:
        gate_use = gate.clamp(0.0, 1.0)
        I_prop = I + b * gate_use * dI
    else:
        I_prop = I + b * dI

    I_prop = I_prop.clamp(-1.0, 1.0) # clamp after update
    I_new = clamp_known(I_prop, I_gt, M)
    I_new = I_new.clamp(-1.0, 1.0) # keep state sane

    if return_gate:
        return I_new, logS, gate
    return I_new, logS

def nftm_step_guarded(
    I, I_gt, M, controller,
    corr_clip=0.2, tvw=0.01, beta: float | None = None,
    max_backtracks=3, shrink=0.5, clip_decay=1.0,
    use_gate: bool = False,
    reject_mode: str | None = None,  # None => auto: "keep_ste" in train, "keep" in eval
):
    """
    Fast guard (single-check):
      - propose ONE step
      - accept if energy decreases
      - otherwise reject

    Important:
      - In TRAINING, default reject_mode="keep_ste" so gradients still flow even if rejected.
      - In EVAL, default reject_mode="keep" (true rejection).
    """
    # auto mode: training needs grads, eval can hard-reject
    if reject_mode is None:
        reject_mode = "keep_ste" if controller.training else "keep"

    with torch.no_grad():
        E0 = energy(I, I_gt, tvw=tvw)

    used_beta = 1.0 if beta is None else float(beta)

    # propose ONE step (this must stay in grad mode!)
    I_prop, logS = nftm_step(
        I, I_gt, M, controller,
        beta=used_beta,
        corr_clip=corr_clip,
        clip_decay=clip_decay,
        use_gate=use_gate
    )

    with torch.no_grad():
        E1 = energy(I_prop, I_gt, tvw=tvw)

    dE = float(E1 - E0)

    if E1 <= E0:
        return I_prop, logS, used_beta, True, dE

    # reject
    if reject_mode == "keep":
        # strict reject: no update (EVAL-friendly, but can kill grads in TRAIN)
        return I, logS, used_beta, False, dE

    if reject_mode == "keep_ste":
        # forward: keep I (reject)
        # backward: behave like I_prop (lets gradients flow)
        I_out = I_prop + (I - I_prop).detach()
        return I_out, logS, used_beta, False, dE

    if reject_mode == "prop":
        # accept the proposal anyway (not really a guard)
        return I_prop, logS, used_beta, False, dE

    raise ValueError(f"Unknown reject_mode: {reject_mode}")

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def masked_psnr_metric(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=a.dtype)
    mask_exp = mask.expand_as(a)
    denom = mask_exp.sum().clamp_min(1e-8)
    mse = ((a - b) * mask_exp).pow(2).sum() / denom
    return 10.0 * torch.log10(4.0 / mse)

def masked_metric_mean(metric_fn, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=a.dtype)
    mask_exp = mask.expand_as(a)
    try:
        vals = metric_fn(a * mask_exp, b * mask_exp, reduction="none")
        frac = mask_exp.mean(dim=(1, 2, 3)).clamp_min(1e-6)
        return (vals / frac).mean()
    except TypeError:
        val = metric_fn(a * mask_exp, b * mask_exp)
        val_tensor = torch.as_tensor(val, device=a.device, dtype=a.dtype)
        frac_scalar = mask.mean().clamp_min(1e-6)
        scale = (1.0 / frac_scalar).to(device=val_tensor.device, dtype=val_tensor.dtype)
        return val_tensor * scale

def estimate_sigma_max_update(controller, I: torch.Tensor, I_gt: torch.Tensor, M: torch.Tensor, *,
    K: int = 1, corr_clip: float = 0.2, beta: float | None = None, 
    clip_decay: float = 1.0, use_gate: bool = False,
    power_iters: int = 8, eps: float = 1e-12, ) -> float:
    """
    Estimate sigma_max(J) of the map x -> missing-region state after K NFTM steps,
    where x parameterizes ONLY the missing region (known pixels are clamped to GT).

    Returns an estimate of the largest singular value (operator norm) of the Jacobian.

    Notes:
      - If sigma_max < 1 : locally contractive (good stability).
      - If sigma_max > 1 : can amplify perturbations (instability possible).
      - Uses power iteration on J^T J with autograd JVP/VJP.
    """
    controller_was_training = controller.training
    controller.eval()  # deterministic

    # We optimize/state only the unknown pixels; known pixels are clamped.
    # x has same shape as I, but we only "care" about missing region by multiplying (1-M).
    with torch.no_grad():
        miss = (1.0 - M).to(dtype=I.dtype)

    x0 = (I * miss).detach().requires_grad_(True)
    M_fix = M.detach()
    Igt_fix = I_gt.detach()

    used_beta = 1.0 if beta is None else float(beta)

    def f(x):
        # rebuild full image with known pixels clamped
        I_full = clamp_known(x, Igt_fix, M_fix).clamp(-1.0, 1.0)

        # K-step rollout map (compose the update K times)
        Icur = I_full
        for _ in range(int(K)):
            Icur, _ = nftm_step(
                Icur, Igt_fix, M_fix, controller,
                beta=used_beta,
                corr_clip=corr_clip,
                clip_decay=clip_decay,
                use_gate=use_gate
            )

        # return ONLY missing region, since known region is always clamped anyway
        return (Icur * miss)

    # init power vector
    v = torch.randn_like(x0)
    v = v / (v.norm() + eps)

    # Power iteration on J^T J
    for _ in range(int(power_iters)):
        # Jv
        _, Jv = torch.autograd.functional.jvp(
            f, (x0,), (v,), create_graph=False, strict=False
        )
        # J^T(Jv)
        _, vjp = torch.autograd.functional.vjp(
            f, (x0,), (Jv,), create_graph=False, strict=False
        )
        JTJv = vjp[0]
        v = JTJv / (JTJv.norm() + eps)

    # Rayleigh quotient to estimate largest eigenvalue of J^T J
    _, Jv = torch.autograd.functional.jvp(f, (x0,), (v,), create_graph=False, strict=False)
    _, vjp = torch.autograd.functional.vjp(f, (x0,), (Jv,), create_graph=False, strict=False)
    JTJv = vjp[0]

    num = (v * JTJv).sum()
    den = (v * v).sum().clamp_min(eps)
    lam = (num / den).abs()  # eigenvalue of JTJ (approx)
    sigma = torch.sqrt(lam).item()   # sigma_max = sqrt(lam_max)

    # restore mode
    controller.train(controller_was_training)
    return float(sigma)