#!/usr/bin/env python3
# image_inpainting.py
# NFTM-style iterative inpainting on CIFAR / CelebA-HQ + SIDD real-noise denoising.
#
# Key features:
# - Iterative controller (TinyController or UNetController)
# - Inpainting path (synthetic missing pixels + corruption) via nftm_inpaint.engine.train_epoch/eval_steps
# - SIDD path (paired noisy->gt) with denoising loops (mask all-zeros)
# - Optional per-epoch saved progress grids:
#     - CIFAR/CelebA: via eval_steps (existing)
#     - SIDD: saves MANY tiles with MANY step-columns (this file)
# - Optional metrics.json saving
# - Optional noise sweep for synthetic corruptions (NOT for SIDD)

import os, random, argparse, json, time, numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision as tv
import torchvision.transforms as T
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

# metrics
from nftm_inpaint.metrics import lpips_dist as _metric_lpips
from nftm_inpaint.metrics import psnr as _metric_psnr
from nftm_inpaint.metrics import ssim as _metric_ssim
from nftm_inpaint.metrics import (
    fid_init, fid_update, fid_compute,
    kid_init, kid_update, kid_compute,
)

# controllers / rollout helpers / engines
from nftm_inpaint.data_and_viz import set_seed, get_transform, ensure_dir, plot_metric_curve
from nftm_inpaint.rollout import parse_pyramid_arg, split_steps_eval, count_params
from nftm_inpaint.rollout import nftm_step, nftm_step_guarded, tv_l1, psnr
from nftm_inpaint.controller import TinyController, UNetController
from nftm_inpaint.engine import train_epoch, eval_steps, evaluate_metrics_full

# wandb is optional; only import if actually used
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

root_dir = os.path.dirname(os.path.abspath(__file__))
benchmarks_dir = os.path.join(root_dir, "benchmarks")

# -------------------------- SIDD utilities --------------------------

def _sidd_scene_group(scene_name: str) -> str:
    """
    Group scenes by the SIDD naming convention:
      <scene-instance-number>_<scene_number>_<smartphone-code>_...
    We use the second token (scene_number) to prevent leakage.
    """
    parts = scene_name.split("_")
    if len(parts) >= 2 and parts[1]:
        scene_num = parts[1]
        if scene_num.isdigit():
            return f"scene_{int(scene_num)}"
        return f"scene_{scene_num}"
    return scene_name

def _sidd_collect_pairs(scene_dir: Path):
    noisy = sorted(scene_dir.glob("*_NOISY_SRGB_*.PNG"))
    pairs = []
    for n in noisy:
        gt = Path(str(n).replace("_NOISY_SRGB_", "_GT_SRGB_"))
        if gt.exists():
            pairs.append((str(n), str(gt)))
    return pairs

def _sidd_make_or_load_index(sidd_root: str, index_dir: str, train_frac: float, seed: int):
    sidd_root = Path(sidd_root)
    data_dir = sidd_root / "Data"
    if not data_dir.exists():
        raise FileNotFoundError(f"[SIDD] Expected {data_dir} to exist")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    train_path = index_dir / "train_pairs.jsonl"
    test_path  = index_dir / "test_pairs.jsonl"
    meta_path  = index_dir / "meta.json"

    # reuse (important on cluster)
    if train_path.exists() and test_path.exists() and meta_path.exists():
        reuse = False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            reuse = (meta.get("scene_grouping") == "scene_number")
        except Exception:
            reuse = False
        if reuse:
            print(f"[SIDD] Reusing existing index in: {index_dir}")
            return str(train_path), str(test_path)
        print("[SIDD] Existing index missing scene_grouping=scene_number; rebuilding to avoid leakage.")

    scenes = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not scenes:
        raise RuntimeError("[SIDD] No scene folders found under Data/")

    grouped = {}
    for scene in scenes:
        key = _sidd_scene_group(scene.name)
        grouped.setdefault(key, []).append(scene)

    groups = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_train = int(len(groups) * float(train_frac))
    train_groups = groups[:n_train]
    test_groups  = groups[n_train:]

    train_scenes = [s for _, lst in train_groups for s in lst]
    test_scenes  = [s for _, lst in test_groups for s in lst]

    def build_rows(scene_list):
        rows = []
        for scene in scene_list:
            scene_group = _sidd_scene_group(scene.name)
            for noisy, gt in _sidd_collect_pairs(scene):
                rows.append({
                    "scene": scene.name,
                    "scene_group": scene_group,
                    "noisy": noisy,
                    "gt": gt,
                })
        return rows

    train_rows = build_rows(train_scenes)
    test_rows  = build_rows(test_scenes)

    meta = dict(
        sidd_root=str(sidd_root),
        num_scenes=len(scenes),
        num_scene_groups=len(groups),
        train_scenes=len(train_scenes),
        test_scenes=len(test_scenes),
        train_scene_groups=len(train_groups),
        test_scene_groups=len(test_groups),
        train_pairs=len(train_rows),
        test_pairs=len(test_rows),
        seed=int(seed),
        train_frac=float(train_frac),
        scene_grouping="scene_number",
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    train_path.write_text("\n".join(json.dumps(r) for r in train_rows) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(json.dumps(r) for r in test_rows) + "\n", encoding="utf-8")

    print(f"[SIDD] Built index: {index_dir}")
    print(f"[SIDD] train_pairs={len(train_rows)} | test_pairs={len(test_rows)}")
    return str(train_path), str(test_path)

def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _to_uint8_img(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W) in [-1,1]
    returns: uint8 HxWx3
    """
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).round().byte()
    return x.permute(1, 2, 0).numpy()

def _choose_step_indices(K_eval: int, max_cols: int):
    """
    Return list of step indices (1..K_eval) to visualize.
    If K_eval <= max_cols -> all steps.
    Else -> evenly spaced steps including 1 and K_eval.
    """
    K_eval = int(K_eval)
    max_cols = int(max_cols)
    if K_eval <= 0:
        return []
    if max_cols <= 0:
        return list(range(1, K_eval + 1))
    if K_eval <= max_cols:
        return list(range(1, K_eval + 1))
    xs = np.linspace(1, K_eval, num=max_cols)
    steps = sorted(set(int(round(v)) for v in xs))
    if steps[0] != 1:
        steps[0] = 1
    if steps[-1] != K_eval:
        steps[-1] = K_eval
    return steps

def _save_sidd_progress_grid(
    save_path: str,
    rows: list,
    col_names: list,
    dpi: int = 150,
):
    """
    rows: list of dicts; each dict maps col_name -> torch tensor (3,H,W) in [-1,1]
    col_names: list[str], consistent across rows
    """
    n_rows = len(rows)
    n_cols = len(col_names)
    if n_rows == 0 or n_cols == 0:
        return

    fig_w = 1.6 * n_cols
    fig_h = 1.6 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[a] for a in axes])

    for r in range(n_rows):
        for c, name in enumerate(col_names):
            ax = axes[r, c]
            ax.imshow(_to_uint8_img(rows[r][name]))
            ax.axis("off")
            if r == 0:
                ax.set_title(name, fontsize=7)

    plt.tight_layout(pad=0.15)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

class SIDDIndexTiles(torch.utils.data.Dataset):
    """
    Paired noisy->GT tiles from SIDD index JSONL.
    Returns (noisy, gt) in [-1,1], shape (3, patch, patch)
    """
    def __init__(self, index_jsonl: str, patch: int = 64, stride: int = 64, limit_tiles=None):
        self.patch = int(patch)
        self.stride = int(stride)
        assert self.patch > 0 and self.stride > 0

        rows = _read_jsonl(index_jsonl)
        if not rows:
            raise RuntimeError(f"[SIDD] Empty index file: {index_jsonl}")

        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize([0.5]*3, [0.5]*3)

        self.items = []
        for r in rows:
            noisy_path = r["noisy"]
            gt_path = r["gt"]

            with Image.open(noisy_path) as im:
                w, h = im.size

            for y in range(0, h - self.patch + 1, self.stride):
                for x in range(0, w - self.patch + 1, self.stride):
                    self.items.append((noisy_path, gt_path, x, y))
                    if limit_tiles is not None and len(self.items) >= limit_tiles:
                        break
                if limit_tiles is not None and len(self.items) >= limit_tiles:
                    break
            if limit_tiles is not None and len(self.items) >= limit_tiles:
                break

        if not self.items:
            raise RuntimeError("[SIDD] No tiles created. Check patch/stride and image sizes.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        noisy_path, gt_path, x, y = self.items[i]

        with Image.open(noisy_path) as imn:
            imn = imn.convert("RGB")
            noisy = imn.crop((x, y, x + self.patch, y + self.patch))

        with Image.open(gt_path) as img:
            img = img.convert("RGB")
            gt = img.crop((x, y, x + self.patch, y + self.patch))

        noisy = self.norm(self.to_tensor(noisy))
        gt = self.norm(self.to_tensor(gt))
        return noisy, gt


class SIDDPerEpochSubsetSampler(torch.utils.data.Sampler):
    """
    Draw a fresh random subset of tile indices each epoch.
    """
    def __init__(self, dataset_size: int, samples_per_epoch: int, seed: int = 0):
        self.dataset_size = int(dataset_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be > 0")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return min(self.samples_per_epoch, self.dataset_size)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        n = min(self.samples_per_epoch, self.dataset_size)
        perm = torch.randperm(self.dataset_size, generator=g).tolist()
        return iter(perm[:n])


def _sidd_tile_positions(size: int, patch: int, stride: int):
    if size <= patch:
        return [0]
    positions = list(range(0, size - patch + 1, stride))
    last = size - patch
    if positions[-1] != last:
        positions.append(last)
    return positions


def _sidd_blend_window(size: int, border: int) -> torch.Tensor:
    """
    2D separable blending window in [0,1], shape (1, size, size).
    """
    border = int(max(0, border))
    if border <= 0:
        return torch.ones((1, size, size), dtype=torch.float32)

    border = min(border, size // 2)
    if border <= 0:
        return torch.ones((1, size, size), dtype=torch.float32)

    axis = torch.ones(size, dtype=torch.float32)
    ramp = torch.linspace(0.0, 1.0, steps=border + 2, dtype=torch.float32)[1:-1]
    axis[:border] = ramp
    axis[-border:] = torch.flip(ramp, dims=[0])
    win = axis[:, None] * axis[None, :]
    return win.unsqueeze(0)


@torch.no_grad()
def _sidd_reconstruct_full_image(
    controller,
    noisy_full: torch.Tensor,
    gt_full: torch.Tensor,
    device,
    patch: int,
    stride: int,
    K_eval: int,
    beta: float,
    gate: bool,
    corr_clip: float,
    batch_size: int,
    tile_pad: int = 0,
):
    controller.eval()
    _, H, W = noisy_full.shape

    tile_pad = int(max(0, tile_pad))
    xs = _sidd_tile_positions(W, patch, stride)
    ys = _sidd_tile_positions(H, patch, stride)

    if tile_pad > 0:
        pad_mode = "reflect" if (H > tile_pad and W > tile_pad) else "replicate"
        noisy_src = F.pad(noisy_full.unsqueeze(0), (tile_pad, tile_pad, tile_pad, tile_pad), mode=pad_mode).squeeze(0)
        gt_src = F.pad(gt_full.unsqueeze(0), (tile_pad, tile_pad, tile_pad, tile_pad), mode=pad_mode).squeeze(0)
    else:
        noisy_src = noisy_full
        gt_src = gt_full

    patch_in = patch + 2 * tile_pad
    blend_win = _sidd_blend_window(patch, border=tile_pad)

    accum = torch.zeros((3, H, W), dtype=torch.float32)
    weight = torch.zeros((1, H, W), dtype=torch.float32)

    coords = [(x, y) for y in ys for x in xs]
    if batch_size <= 0:
        batch_size = len(coords)

    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start + batch_size]
        tiles_noisy = torch.stack(
            [noisy_src[:, y:y + patch_in, x:x + patch_in] for (x, y) in batch_coords],
            dim=0
        )
        tiles_gt = torch.stack(
            [gt_src[:, y:y + patch_in, x:x + patch_in] for (x, y) in batch_coords],
            dim=0
        )
        tiles_noisy = tiles_noisy.to(device, non_blocking=True)
        tiles_gt = tiles_gt.to(device, non_blocking=True)

        M = torch.zeros((tiles_noisy.size(0), 1, patch_in, patch_in),
                        device=device, dtype=tiles_noisy.dtype)

        I = tiles_noisy
        for s in range(int(K_eval)):
            I, _ = nftm_step(
                I, tiles_gt, M, controller,
                beta=min(float(beta), 0.9),
                use_gate=gate,
                corr_clip=corr_clip,
                clip_decay=(0.92 ** s)
            )

        preds = I.clamp(-1, 1).cpu()
        if tile_pad > 0:
            preds = preds[:, :, tile_pad:tile_pad + patch, tile_pad:tile_pad + patch]
        for idx, (x, y) in enumerate(batch_coords):
            accum[:, y:y + patch, x:x + patch] += preds[idx] * blend_win
            weight[:, y:y + patch, x:x + patch] += blend_win

    return accum / weight.clamp_min(1e-6)


@torch.no_grad()
def save_sidd_full_images(
    controller,
    index_jsonl: str,
    device,
    patch: int,
    stride: int,
    K_eval: int,
    beta: float,
    gate: bool,
    corr_clip: float,
    out_dir: str,
    max_images: int = 0,
    batch_size: int = 0,
    tile_pad: int = 0,
):
    rows = _read_jsonl(index_jsonl)
    if not rows:
        raise RuntimeError(f"[SIDD] Empty index file: {index_jsonl}")

    seen = set()
    images = []
    for r in rows:
        key = (r["noisy"], r["gt"])
        if key in seen:
            continue
        seen.add(key)
        images.append(r)

    if max_images and max_images > 0:
        images = images[:max_images]

    ensure_dir(out_dir)
    per_image = []

    to_tensor = T.ToTensor()

    for idx, r in enumerate(images):
        noisy_path = r["noisy"]
        gt_path = r["gt"]
        scene = r.get("scene", Path(noisy_path).parent.name)

        with Image.open(noisy_path) as imn:
            imn = imn.convert("RGB")
            noisy_t = to_tensor(imn)

        with Image.open(gt_path) as img:
            img = img.convert("RGB")
            gt_t = to_tensor(img)

        noisy_norm = (noisy_t - 0.5) / 0.5
        gt_norm = (gt_t - 0.5) / 0.5

        pred_norm = _sidd_reconstruct_full_image(
            controller=controller,
            noisy_full=noisy_norm,
            gt_full=gt_norm,
            device=device,
            patch=int(patch),
            stride=int(stride),
            K_eval=int(K_eval),
            beta=float(beta),
            gate=bool(gate),
            corr_clip=float(corr_clip),
            batch_size=int(batch_size),
            tile_pad=int(tile_pad),
        )

        pred_b = pred_norm.unsqueeze(0)
        gt_b = gt_norm.unsqueeze(0)
        psnr_val = float(_metric_psnr(pred_b, gt_b).item())
        ssim_val = float(_metric_ssim(pred_b, gt_b).item())

        stem = Path(noisy_path).stem
        base = f"{idx:04d}_{scene}_{stem}"

        Image.fromarray(_to_uint8_img(noisy_norm)).save(os.path.join(out_dir, f"{base}_noisy.png"))
        Image.fromarray(_to_uint8_img(pred_norm)).save(os.path.join(out_dir, f"{base}_pred.png"))
        Image.fromarray(_to_uint8_img(gt_norm)).save(os.path.join(out_dir, f"{base}_gt.png"))

        per_image.append({
            "scene": scene,
            "noisy": noisy_path,
            "gt": gt_path,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "file_base": base,
        })

        print(f"[SIDD] full {idx+1}/{len(images)} | PSNR {psnr_val:.2f} | SSIM {ssim_val:.4f}")

    summary = {
        "num_images": len(per_image),
        "mean_psnr": float(np.mean([m["psnr"] for m in per_image])) if per_image else None,
        "mean_ssim": float(np.mean([m["ssim"] for m in per_image])) if per_image else None,
        "per_image": per_image,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SIDD] full-image outputs saved to: {out_dir}")
    return summary

# -------------------------- SIDD denoising loops --------------------------

def train_epoch_sidd_denoise(
    controller, opt, loader, device, epoch,
    K_target=10, beta=0.4, beta_max=0.9, gate=False,
    tvw=0.0, corr_clip=0.2, guard_in_train=True, contract_w=1e-3,
    step_loss_mode: str = "final",
):
    controller.train()
    losses, psnrs = [], []
    accepted_steps, backtracks = 0, 0
    K_train = int(K_target)

    use_linear_steps = (step_loss_mode == "linear")
    if use_linear_steps:
        step_weights = torch.arange(1, K_train + 1, device=device, dtype=torch.float32)
        step_weights = step_weights / step_weights.sum()

    for bidx, (noisy, gt) in enumerate(loader):
        if (bidx % 20) == 0 and bidx > 0 and losses:
            print(f"[train] epoch={epoch} batch={bidx} loss={losses[-1]:.4f} psnr={psnrs[-1]:.2f}", flush=True)

        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        # denoising => no known pixels => mask is all zeros
        M = torch.zeros((noisy.size(0), 1, noisy.size(2), noisy.size(3)),
                        device=device, dtype=noisy.dtype)

        I = noisy.clone()
        I_prev_for_contract = I.detach()

        if use_linear_steps:
            data_loss_accum = 0.0
            step_idx = 0

        for s in range(K_train):
            clip_decay = (0.92 ** s)
            beta_s = min(float(beta), float(beta_max))

            if guard_in_train:
                I, _, _used_beta, ok, _dE = nftm_step_guarded(
                    I, gt, M, controller,
                    beta=beta_s, use_gate=gate,
                    corr_clip=corr_clip,
                    tvw=0.0, max_backtracks=2, shrink=0.5, clip_decay=clip_decay
                )
                accepted_steps += int(ok)
                backtracks += int(not ok)
            else:
                I, _ = nftm_step(
                    I, gt, M, controller,
                    beta=beta_s, use_gate=gate,
                    corr_clip=corr_clip, clip_decay=clip_decay
                )

            if use_linear_steps:
                step_idx += 1
                w_k = step_weights[step_idx - 1].to(dtype=gt.dtype)
                data_loss_accum = data_loss_accum + w_k * F.mse_loss(I, gt)

        data_loss = data_loss_accum if use_linear_steps else F.mse_loss(I, gt)
        loss_smooth = tv_l1(I, tvw)
        contract = contract_w * (I - I_prev_for_contract).pow(2).mean()
        loss = data_loss + loss_smooth + contract

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
        opt.step()

        with torch.no_grad():
            losses.append(float(loss.item()))
            psnrs.append(float(psnr(I, gt).item()))

    stats = dict(train_K=K_train, accepted=accepted_steps, backtracks=backtracks)
    return float(np.mean(losses)), float(np.mean(psnrs)), stats

@torch.no_grad()
def eval_steps_sidd_denoise(
    controller, loader, device,
    K_eval=10, beta=0.6, gate=False,
    corr_clip=0.2, tvw=0.0, descent_guard=False, max_batches: int = 0,
):
    """
    Returns step-wise curves (mean across evaluated batches).
    (Visualization is handled by a separate function to allow many samples + many step columns.)
    """
    controller.eval()
    psnrs_step, ssims_step, lpips_step = [], [], []

    for bidx, (noisy, gt) in enumerate(loader):
        if max_batches and bidx >= max_batches:
            break

        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        M = torch.zeros((noisy.size(0), 1, noisy.size(2), noisy.size(3)),
                        device=device, dtype=noisy.dtype)

        I = noisy.clone()
        step_psnrs, step_ssims, step_lpips = [], [], []

        for s in range(int(K_eval)):
            clip_decay = (0.92 ** s)
            beta_s = min(float(beta), 0.9)

            if descent_guard:
                I, _, _, _, _ = nftm_step_guarded(
                    I, gt, M, controller,
                    beta=beta_s, use_gate=gate,
                    corr_clip=corr_clip,
                    tvw=tvw, max_backtracks=3, shrink=0.5, clip_decay=clip_decay
                )
            else:
                I, _ = nftm_step(
                    I, gt, M, controller,
                    beta=beta_s, use_gate=gate,
                    corr_clip=corr_clip, clip_decay=clip_decay
                )

            I_metrics = I.clamp(-1.0, 1.0)
            step_psnrs.append(_metric_psnr(I_metrics, gt).item())
            step_ssims.append(_metric_ssim(I_metrics, gt).item())
            step_lpips.append(_metric_lpips(I_metrics, gt).item())

        psnrs_step.append(step_psnrs)
        ssims_step.append(step_ssims)
        lpips_step.append(step_lpips)

    curves = {
        "psnr": np.array(psnrs_step).mean(axis=0) if psnrs_step else np.array([]),
        "ssim": np.array(ssims_step).mean(axis=0) if ssims_step else np.array([]),
        "lpips": np.array(lpips_step).mean(axis=0) if lpips_step else np.array([]),
    }
    return curves

@torch.no_grad()
def dump_sidd_progress_grids(
    controller, loader, device,
    K_eval: int, beta: float, gate: bool, corr_clip: float,
    save_dir: str, epoch_tag: str,
    total_samples: int = 50,
    num_pngs: int = 10,
    max_step_cols: int = 24,
):
    """
    Save multiple PNGs, each containing multiple different tiles (rows),
    and many rollout step columns, like the inpainting visualizations.

    Columns: gt, noisy, step01..stepXX (chosen from 1..K_eval)
    Rows: different tiles (up to total_samples)
    """
    ensure_dir(save_dir)
    controller.eval()

    total_samples = int(total_samples)
    num_pngs = max(1, int(num_pngs))
    per_png = int(np.ceil(total_samples / num_pngs))

    step_ids = _choose_step_indices(int(K_eval), int(max_step_cols))
    col_names = ["gt", "noisy"] + [f"step{sid:02d}" for sid in step_ids]

    collected = 0
    png_idx = 0
    rows_for_png = []

    for bidx, (noisy, gt) in enumerate(loader):
        if collected >= total_samples:
            break

        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        B = noisy.size(0)

        # process each sample in the batch until we hit quota
        for i in range(B):
            if collected >= total_samples:
                break

            noisy_i = noisy[i:i+1]
            gt_i = gt[i:i+1]

            M = torch.zeros((1, 1, noisy.size(2), noisy.size(3)), device=device, dtype=noisy.dtype)
            I = noisy_i.clone()

            # store tensors for this row
            row = {"gt": gt_i[0].detach(), "noisy": noisy_i[0].detach()}

            # rollout and capture chosen steps
            step_ptr = 0
            want = step_ids[step_ptr] if step_ids else None

            for s in range(1, int(K_eval) + 1):
                I, _ = nftm_step(
                    I, gt_i, M, controller,
                    beta=min(float(beta), 0.9),
                    use_gate=gate,
                    corr_clip=corr_clip,
                    clip_decay=(0.92 ** (s - 1))
                )

                if want is not None and s == want:
                    row[f"step{s:02d}"] = I[0].detach()
                    step_ptr += 1
                    want = step_ids[step_ptr] if step_ptr < len(step_ids) else None

            # if for any reason some wanted steps weren't filled (shouldn't happen), fill with final
            if step_ids:
                final_img = I[0].detach()
                for sid in step_ids:
                    k = f"step{sid:02d}"
                    if k not in row:
                        row[k] = final_img

            rows_for_png.append(row)
            collected += 1

            # flush PNG if full
            if len(rows_for_png) >= per_png or (collected >= total_samples):
                save_path = os.path.join(save_dir, f"sidd_{epoch_tag}_grid_{png_idx:02d}.png")
                _save_sidd_progress_grid(save_path, rows_for_png, col_names, dpi=150)
                rows_for_png = []
                png_idx += 1

    if collected == 0:
        print("[SIDD] dump_sidd_progress_grids: collected 0 samples (loader empty?)")

@torch.no_grad()
def evaluate_metrics_sidd_denoise(
    controller, loader, device, K_eval,
    beta=0.6, gate=False, corr_clip=0.2, max_batches=0
):
    """
    Returns mean PSNR/SSIM/LPIPS across batches after K_eval steps.
    """
    controller.eval()
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    batches = 0

    for bidx, (noisy, gt) in enumerate(loader):
        if max_batches and bidx >= max_batches:
            break

        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        M = torch.zeros((noisy.size(0), 1, noisy.size(2), noisy.size(3)),
                        device=device, dtype=noisy.dtype)

        I = noisy.clone()
        for s in range(int(K_eval)):
            I, _ = nftm_step(
                I, gt, M, controller,
                beta=min(float(beta), 0.9),
                use_gate=gate,
                corr_clip=corr_clip,
                clip_decay=(0.92 ** s)
            )

        preds = I.clamp(-1, 1)
        totals["psnr"] += float(_metric_psnr(preds, gt).item())
        totals["ssim"] += float(_metric_ssim(preds, gt).item())
        totals["lpips"] += float(_metric_lpips(preds, gt).item())
        batches += 1

    return {k: v / max(1, batches) for k, v in totals.items()}

# -------------------------- main --------------------------

def main():
    parser = argparse.ArgumentParser(description="NFTM-style iterative inpainting + SIDD denoising")

    # train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # rollout
    parser.add_argument("--K_train", type=int, default=8)
    parser.add_argument("--K_eval", type=int, default=12)

    # beta
    parser.add_argument("--beta_start", type=float, default=None,
                        help="If set, enable beta. Otherwise beta off.")
    parser.add_argument("--beta_max", type=float, default=None)
    parser.add_argument("--beta_anneal", type=float, default=None)
    parser.add_argument("--beta_eval_bonus", type=float, default=None)

    # regularizers / step params
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--corr_clip", type=float, default=0.1)
    parser.add_argument("--contract_w", type=float, default=1e-3)
    parser.add_argument("--guard_in_train", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--step_loss", type=str, default="final", choices=["final", "linear"])

    # inpainting corruption
    parser.add_argument("--pmin", type=float, default=0.25)
    parser.add_argument("--pmax", type=float, default=0.5)
    parser.add_argument("--block_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--gaussian_additive", action="store_true",
                        help="Only for synthetic corruption (inpainting).")

    # model
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--controller", type=str, default="dense", choices=["dense", "unet"])
    parser.add_argument("--unet_base", type=int, default=10)

    # data
    parser.add_argument("--train_dataset", type=str, default="cifar", choices=["cifar", "celebahq", "sidd"])
    parser.add_argument("--benchmark", type=str, default="cifar", choices=["cifar", "set12", "cbsd68", "celebahq", "sidd"])
    parser.add_argument("--img_size", type=int, default=32, choices=[32, 64, 128])

    # SIDD
    parser.add_argument("--sidd_root", type=str, default="", help="Path to SIDD_Medium_Srgb (must contain Data/)")
    parser.add_argument("--sidd_index_dir", type=str, default="benchmarks/SIDD/index")
    parser.add_argument("--sidd_train_frac", type=float, default=0.8)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--patch_stride", type=int, default=64)
    parser.add_argument("--sidd_limit_tiles", type=int, default=0,
                        help="Debug limit for tiles. 0 = no limit.")
    parser.add_argument("--sidd_epoch_tiles", type=int, default=0,
                        help="Randomly sample this many SIDD train tiles per epoch (0 = use all available tiles).")

    # NEW: SIDD visualization controls (only affects saved PNG grids)
    parser.add_argument("--sidd_viz_samples", type=int, default=50,
                        help="How many different SIDD tiles to visualize per eval epoch.")
    parser.add_argument("--sidd_viz_pngs", type=int, default=10,
                        help="Split those samples into this many PNG files.")
    parser.add_argument("--sidd_viz_max_steps", type=int, default=24,
                        help="Max number of step-columns to show. If K_eval <= this, show all steps; else show evenly spaced steps.")

    # NEW: SIDD full-image reconstruction
    parser.add_argument("--sidd_save_full_images", action="store_true",
                        help="Reconstruct and save full SIDD test images by stitching denoised tiles.")
    parser.add_argument("--sidd_full_max_images", type=int, default=0,
                        help="Max number of full test images to reconstruct (0 = all).")
    parser.add_argument("--sidd_full_batch", type=int, default=0,
                        help="Batch size for full-image tile inference (0 = use --batch_size).")
    parser.add_argument("--sidd_full_tile_pad", type=int, default=0,
                        help="Extra reflected context around each full-image tile before denoising (pixels).")

    # eval cadence
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_max_batches", type=int, default=50)

    # pyramid (inpainting only)
    parser.add_argument("--pyramid", type=str, default="")
    parser.add_argument("--pyr_steps", type=str, default="")
    parser.add_argument("--viz_scale", type=float, default=1.0)

    # outputs
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--save_epoch_progress", action="store_true",
                        help="Save per-epoch visuals for eval.")
    parser.add_argument("--save_metrics", action="store_true",
                        help="Save metrics.json and curves.")

    # sweep (synthetic only)
    parser.add_argument("--eval_noise_sweep", action="store_true",
                        help="Synthetic corruption sweep (NOT for SIDD).")

    # misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device} | criterion=MSE")

    if args.use_wandb and wandb is None:
        raise RuntimeError("wandb requested but not installed/importable in this environment.")

    train_dataset_name = args.train_dataset.lower()
    benchmark = args.benchmark.lower()

    # SIDD consistency
    is_sidd = (train_dataset_name == "sidd" and benchmark == "sidd")
    if (train_dataset_name == "sidd") ^ (benchmark == "sidd"):
        raise ValueError("[SIDD] For denoising, set BOTH --train_dataset sidd and --benchmark sidd")

    # sizes
    img_size = int(args.img_size)
    if is_sidd:
        if not args.sidd_root:
            raise ValueError("[SIDD] Missing --sidd_root")
        if args.img_size != args.patch_size:
            print(f"[SIDD] Overriding img_size {args.img_size} -> patch_size {args.patch_size}")
        img_size = int(args.patch_size)

    # datasets
    sidd_test_jsonl = None
    if is_sidd:
        train_jsonl, test_jsonl = _sidd_make_or_load_index(
            sidd_root=args.sidd_root,
            index_dir=args.sidd_index_dir,
            train_frac=args.sidd_train_frac,
            seed=args.seed,
        )
        sidd_test_jsonl = test_jsonl
        limit = None if args.sidd_limit_tiles <= 0 else int(args.sidd_limit_tiles)
        train_set = SIDDIndexTiles(train_jsonl, patch=img_size, stride=int(args.patch_stride), limit_tiles=limit)
        test_set  = SIDDIndexTiles(test_jsonl,  patch=img_size, stride=int(args.patch_stride), limit_tiles=limit)
    else:
        if train_dataset_name == "cifar":
            transform_train = get_transform("cifar", img_size=img_size)
            train_set = tv.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        elif train_dataset_name == "celebahq":
            transform_train = get_transform("celebahq", img_size=img_size)
            train_set = ImageFolder(root=os.path.join(benchmarks_dir, "CelebAHQ"), transform=transform_train)
        else:
            raise ValueError(f"Unknown train dataset: {args.train_dataset}")

        transform_test = get_transform(benchmark, img_size=img_size)
        if benchmark == "cifar":
            test_set = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        elif benchmark == "set12":
            test_set = ImageFolder(root="./benchmarks/Set12", transform=transform_test)
        elif benchmark == "cbsd68":
            test_set = ImageFolder(root="./benchmarks/CBSD68", transform=transform_test)
        elif (benchmark == "celebahq") and train_dataset_name == "celebahq":
            train_size = int(0.8 * len(train_set))
            test_size = len(train_set) - train_size
            train_set, test_set = random_split(
                train_set, [train_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
        elif benchmark == "celebahq":
            test_set = ImageFolder(root=os.path.join(benchmarks_dir, "CelebAHQ"), transform=transform_test)
        else:
            raise ValueError(f"Unknown benchmark dataset: {args.benchmark}")

    # pyramids (inpainting only)
    if is_sidd:
        pyr_sizes = [img_size]
        pyr_steps_eval = [args.K_eval]
    else:
        pyr_sizes = parse_pyramid_arg(args.pyramid, img_size)
        pyr_steps_eval = split_steps_eval(args.K_eval, pyr_sizes, args.pyr_steps if args.pyr_steps else None)

    print(f"[Data] train_dataset={train_dataset_name}, benchmark={benchmark}, img_size={img_size}")
    print(f"[Data] train_set size={len(train_set)}, test_set size={len(test_set)}")

    # loaders
    use_cuda_pinning = (device.type == "cuda")

    if is_sidd:
        nw = min(2, os.cpu_count() or 2)
    else:
        nw = min(8, os.cpu_count() or 8)

    loader_kwargs = dict(pin_memory=use_cuda_pinning)
    if nw > 0:
        loader_kwargs.update(dict(
            num_workers=nw,
            persistent_workers=True,
            prefetch_factor=2 if is_sidd else 4
        ))
    else:
        loader_kwargs.update(dict(num_workers=0))

    train_sampler = None
    if is_sidd and int(args.sidd_epoch_tiles) > 0:
        train_sampler = SIDDPerEpochSubsetSampler(
            dataset_size=len(train_set),
            samples_per_epoch=int(args.sidd_epoch_tiles),
            seed=int(args.seed) + 1337,
        )
        print(f"[SIDD] per-epoch train subset: {len(train_sampler)}/{len(train_set)} tiles (reshuffled each epoch)")

    if train_sampler is not None:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, **loader_kwargs)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # model
    controller_info = {"name": args.controller}
    if args.controller == "dense":
        controller = TinyController(in_ch=4, width=args.width).to(device)
    else:
        target = count_params(TinyController(in_ch=4, width=args.width))
        base = args.unet_base
        best = None
        for b in range(6, 14):
            tmp = UNetController(in_ch=4, base=b)
            n = count_params(tmp)
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (b, n)
        if best is not None and abs(best[1] - target) <= 0.05 * target and best[0] != base:
            print(f"[controller] auto-adjust unet_base {base} -> {best[0]} (target params={target})")
            base = best[0]
        controller = UNetController(in_ch=4, base=base).to(device)
        controller_info["base"] = base

    param_total = count_params(controller)
    controller_info["params"] = param_total
    msg = f"[controller] {controller_info['name']}"
    if "base" in controller_info:
        msg += f" | base={controller_info['base']}"
    msg += f" | params={controller_info['params']}"
    print(msg)

    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # outputs
    ensure_dir(args.save_dir)
    steps_dir = os.path.join(args.save_dir, "steps") if args.save_epoch_progress else None
    if steps_dir:
        ensure_dir(steps_dir)

    log_path = os.path.join(args.save_dir, "train_log.txt")
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    print(f"[logging] writing epoch logs to {log_path}", flush=True)

    # beta schedule
    use_beta = (args.beta_start is not None)
    if use_beta:
        beta_start = float(args.beta_start)
        beta_anneal = float(args.beta_anneal) if args.beta_anneal is not None else 0.0
        beta_max = float(args.beta_max) if args.beta_max is not None else beta_start
        beta_eval_bonus = float(args.beta_eval_bonus) if args.beta_eval_bonus is not None else 0.0
    else:
        beta_start = beta_anneal = beta_max = beta_eval_bonus = None

    # wandb init
    if args.use_wandb:
        wandb.init(project="nftm_inpaint", config=vars(args))

    psnr_curve = ssim_curve = lpips_curve = None

    # ------------------------- training loop -------------------------
    train_loop_t0 = time.time()
    for ep in range(1, args.epochs + 1):
        ep_t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        beta = None
        beta_eval = None
        if use_beta:
            beta = min(beta_start + beta_anneal * (ep - 1), beta_max)
            beta_eval = min(beta + beta_eval_bonus, 0.9)

        if is_sidd:
            train_loss, train_psnr, stats = train_epoch_sidd_denoise(
                controller, opt, train_loader, device, epoch=ep,
                K_target=args.K_train,
                beta=beta if beta is not None else 0.35,
                beta_max=beta_max if beta_max is not None else 0.85,
                gate=args.gate,
                tvw=args.tv_weight,
                corr_clip=args.corr_clip,
                guard_in_train=args.guard_in_train,
                contract_w=args.contract_w,
                step_loss_mode=args.step_loss,
            )

            do_eval = (args.eval_every > 0) and ((ep % args.eval_every == 0) or (ep == args.epochs))
            if do_eval:
                curves = eval_steps_sidd_denoise(
                    controller, test_loader, device,
                    K_eval=args.K_eval,
                    beta=beta_eval if beta_eval is not None else 0.6,
                    gate=args.gate,
                    corr_clip=args.corr_clip,
                    tvw=0.0,
                    descent_guard=False,
                    max_batches=args.eval_max_batches,
                )

                # NEW: dump many tiles with many step-columns (like inpainting)
                if args.save_epoch_progress and steps_dir:
                    dump_sidd_progress_grids(
                        controller, test_loader, device,
                        K_eval=args.K_eval,
                        beta=beta_eval if beta_eval is not None else 0.6,
                        gate=args.gate,
                        corr_clip=args.corr_clip,
                        save_dir=steps_dir,
                        epoch_tag=f"ep{ep:03d}",
                        total_samples=args.sidd_viz_samples,
                        num_pngs=args.sidd_viz_pngs,
                        max_step_cols=args.sidd_viz_max_steps,
                    )
            else:
                curves = {"psnr": np.array([]), "ssim": np.array([]), "lpips": np.array([])}

        else:
            train_loss, train_psnr, stats = train_epoch(
                controller, opt, train_loader, device, epoch=ep,
                K_target=args.K_train, K_base=4,
                beta=beta, gate=args.gate, beta_max=beta_max,
                tvw=args.tv_weight,
                p_missing=(args.pmin, args.pmax),
                block_prob=args.block_prob,
                noise_std=args.noise_std,
                corr_clip=args.corr_clip,
                guard_in_train=args.guard_in_train,
                contract_w=args.contract_w,
                rollout_bias=True,
                pyramid_sizes=pyr_sizes,
                step_loss_mode=args.step_loss,
                gaussian_additive=args.gaussian_additive,
            )

            curves = eval_steps(
                controller, test_loader, device,
                K_eval=args.K_eval, beta=beta_eval, gate=args.gate,
                p_missing=(args.pmin, args.pmax),
                block_prob=args.block_prob,
                noise_std=args.noise_std,
                corr_clip=args.corr_clip,
                descent_guard=False, tvw=0.0,
                save_per_epoch_dir=steps_dir,
                epoch_tag=ep,
                pyramid_sizes=pyr_sizes,
                steps_split=pyr_steps_eval,
                viz_scale=max(1.0, float(args.viz_scale)),
                gaussian_additive=args.gaussian_additive,
            )

        psnr_curve = curves["psnr"]
        ssim_curve = curves["ssim"]
        lpips_curve = curves["lpips"]

        if psnr_curve.size > 0:
            head = psnr_curve[:min(5, len(psnr_curve))]
            curve_str = ", ".join(f"{v:.2f}" for v in head)
            tail_val = f"{psnr_curve[-1]:.2f}"
        else:
            curve_str = "skipped"
            tail_val = "skipped"

        beta_str = f"{beta:.3f}" if beta is not None else "off"
        line = (
            f"[ep {ep:02d}] β_train={beta_str} | K_train={stats['train_K']} | "
            f"loss {train_loss:.4f} | train PSNR {train_psnr:.2f} dB | "
            f"eval PSNR 1..{args.K_eval}: {curve_str} ... {tail_val} | ctrl={args.controller}"
        )
        if ssim_curve.size > 0 and lpips_curve.size > 0:
            line += f" | final SSIM {ssim_curve[-1]:.4f} | final LPIPS {lpips_curve[-1]:.4f}"
        line += f" | ep_time {time.time() - ep_t0:.1f}s"

        print(line)
        log_f.write(line + "\n")

        if args.guard_in_train:
            extra = f"         accepted steps: {stats['accepted']} | backtracks (approx): {stats['backtracks']}"
            print(extra)
            log_f.write(extra + "\n")

        if args.use_wandb:
            wandb.log({
                "epoch": ep,
                "train/loss": train_loss,
                "train/psnr": train_psnr,
                "eval/psnr_final": float(psnr_curve[-1]) if psnr_curve.size > 0 else None,
                "eval/ssim_final": float(ssim_curve[-1]) if ssim_curve.size > 0 else None,
                "eval/lpips_final": float(lpips_curve[-1]) if lpips_curve.size > 0 else None,
                "K_train": stats["train_K"],
                "controller": args.controller,
            })

    train_total_s = time.time() - train_loop_t0
    train_time_line = f"[timing] training loop wall-clock: {train_total_s:.1f}s"
    print(train_time_line)
    log_f.write(train_time_line + "\n")

    # save curves plots
    if psnr_curve is not None and psnr_curve.size > 0:
        plot_metric_curve(psnr_curve, os.path.join(args.save_dir, "psnr_curve.png"),
                          "PSNR (dB)", "Step-wise PSNR (eval)")
        np.save(os.path.join(args.save_dir, "psnr_curve.npy"), psnr_curve)
    if ssim_curve is not None and ssim_curve.size > 0:
        plot_metric_curve(ssim_curve, os.path.join(args.save_dir, "ssim_curve.png"),
                          "SSIM", "Step-wise SSIM (eval)")
        np.save(os.path.join(args.save_dir, "ssim_curve.npy"), ssim_curve)
    if lpips_curve is not None and lpips_curve.size > 0:
        plot_metric_curve(lpips_curve, os.path.join(args.save_dir, "lpips_curve.png"),
                          "LPIPS", "Step-wise LPIPS (eval)")
        np.save(os.path.join(args.save_dir, "lpips_curve.npy"), lpips_curve)

    # final beta_eval
    beta_eval_final = None
    if use_beta:
        beta_final = min(beta_start + beta_anneal * (args.epochs - 1), beta_max)
        beta_eval_final = min(beta_final + beta_eval_bonus, 0.9)

    print("[done] outputs under:", args.save_dir, f"| controller={args.controller}")

    # ------------------------- metrics.json -------------------------
    summary = {}
    if args.save_metrics:
        if is_sidd:
            metrics_full = evaluate_metrics_sidd_denoise(
                controller, test_loader, device,
                K_eval=args.K_eval,
                beta=beta_eval_final if beta_eval_final is not None else 0.6,
                gate=args.gate,
                corr_clip=args.corr_clip,
                max_batches=0
            )
        else:
            metrics_full = evaluate_metrics_full(
                controller,
                test_loader,
                device,
                K_eval=args.K_eval,
                beta=beta_eval_final,
                gate=args.gate,
                p_missing=(args.pmin, args.pmax),
                block_prob=args.block_prob,
                noise_std=args.noise_std,
                corr_clip=args.corr_clip,
                descent_guard=False,
                tvw=0.0,
                benchmark=benchmark,
                pyramid_sizes=pyr_sizes,
                steps_split=pyr_steps_eval,
                gaussian_additive=args.gaussian_additive
            )

        summary = dict(
            epochs=args.epochs,
            K_train=args.K_train,
            K_eval=args.K_eval,
            beta_start=args.beta_start,
            beta_max=args.beta_max,
            beta_anneal=args.beta_anneal,
            beta_eval_bonus=args.beta_eval_bonus,
            corr_clip=args.corr_clip,
            tv_weight=args.tv_weight,
            pmin=args.pmin, pmax=args.pmax, block_prob=args.block_prob,
            noise_std=args.noise_std,
            final_psnr=float(psnr_curve[-1]) if psnr_curve is not None and psnr_curve.size > 0 else None,
            params=float(param_total),
            seed=int(args.seed),
            controller=args.controller,
        )
        if args.controller == "unet":
            summary["unet_base"] = int(controller_info.get("base", args.unet_base))
        if ssim_curve is not None and ssim_curve.size > 0:
            summary["final_ssim"] = float(ssim_curve[-1])
        if lpips_curve is not None and lpips_curve.size > 0:
            summary["final_lpips"] = float(lpips_curve[-1])

        summary["curve_psnr"] = [float(x) for x in (psnr_curve.tolist() if psnr_curve is not None else [])]
        summary["curve_ssim"] = [float(x) for x in (ssim_curve.tolist() if ssim_curve is not None else [])]
        summary["curve_lpips"] = [float(x) for x in (lpips_curve.tolist() if lpips_curve is not None else [])]

        summary.update(metrics_full)

        with open(os.path.join(args.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[metrics] saved {os.path.join(args.save_dir, 'metrics.json')}")

    # ------------------------- SIDD full-image reconstruction -------------------------
    if is_sidd and args.sidd_save_full_images:
        if not sidd_test_jsonl:
            raise RuntimeError("[SIDD] Missing test index for full-image reconstruction.")
        full_dir = os.path.join(args.save_dir, "sidd_full")
        ensure_dir(full_dir)
        beta_full = beta_eval_final if beta_eval_final is not None else 0.6
        batch_full = int(args.sidd_full_batch) if args.sidd_full_batch > 0 else int(args.batch_size)
        save_sidd_full_images(
            controller=controller,
            index_jsonl=sidd_test_jsonl,
            device=device,
            patch=int(args.patch_size),
            stride=int(args.patch_stride),
            K_eval=int(args.K_eval),
            beta=float(beta_full),
            gate=bool(args.gate),
            corr_clip=float(args.corr_clip),
            out_dir=full_dir,
            max_images=int(args.sidd_full_max_images),
            batch_size=batch_full,
            tile_pad=int(args.sidd_full_tile_pad),
        )

    # ------------------------- synthetic noise sweep (NOT for SIDD) -------------------------
    if args.eval_noise_sweep:
        if is_sidd:
            print("[SIDD] eval_noise_sweep skipped: SIDD is real sensor noise (no synthetic corruption sweep).")
        else:
            sweep = [
                ("gaussian",   {"noise_std": args.noise_std}),
                ("uniform",    {}),
                ("saltpepper", {"prob": 0.10}),
                ("saltpepper", {"prob": 0.20}),
                ("poisson",    {"peak": 30.0}),
                ("poisson",    {"peak": 10.0}),
                ("speckle",    {"noise_std": 0.20}),
            ]

            sweep_dir = os.path.join(args.save_dir, "noise_sweep")
            ensure_dir(sweep_dir)

            beta_eval_sweep = beta_eval_final
            all_results = {}

            for noise_kind, kw in sweep:
                tag = "_".join([noise_kind] + [f"{k}{v}" for k, v in kw.items()])
                out_dir = os.path.join(sweep_dir, tag)
                ensure_dir(out_dir)

                noise_std_use = float(kw.get("noise_std", args.noise_std))
                noise_kwargs = dict(kw)
                noise_kwargs.pop("noise_std", None)

                _ = eval_steps(
                    controller, test_loader, device,
                    K_eval=args.K_eval, beta=beta_eval_sweep, gate=args.gate,
                    p_missing=(args.pmin, args.pmax),
                    block_prob=args.block_prob,
                    noise_std=noise_std_use,
                    corr_clip=args.corr_clip,
                    descent_guard=False, tvw=0.0,
                    save_per_epoch_dir=out_dir,
                    epoch_tag="final",
                    pyramid_sizes=pyr_sizes,
                    steps_split=pyr_steps_eval,
                    viz_scale=max(1.0, float(args.viz_scale)),
                    gaussian_additive=args.gaussian_additive
                )

                if args.save_metrics:
                    metrics_one = evaluate_metrics_full(
                        controller,
                        test_loader,
                        device,
                        K_eval=args.K_eval,
                        beta=beta_eval_sweep,
                        gate=args.gate,
                        p_missing=(args.pmin, args.pmax),
                        block_prob=args.block_prob,
                        noise_std=noise_std_use,
                        corr_clip=args.corr_clip,
                        descent_guard=False,
                        tvw=0.0,
                        benchmark=benchmark,
                        pyramid_sizes=pyr_sizes,
                        steps_split=pyr_steps_eval,
                        noise_kind=noise_kind,
                        noise_kwargs=noise_kwargs,
                        gaussian_additive=args.gaussian_additive
                    )
                    all_results[tag] = metrics_one
                    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(metrics_one, f, indent=2)

            if args.save_metrics:
                with open(os.path.join(sweep_dir, "noise_sweep.json"), "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)

            print("[noise-sweep] saved under:", sweep_dir)

    # wandb final
    if args.use_wandb:
        if summary:
            wandb.log({
                "final/psnr": summary.get("final_psnr"),
                "final/ssim": summary.get("final_ssim"),
                "final/lpips": summary.get("final_lpips"),
                "final/fid": summary.get("fid"),
                "final/kid": summary.get("kid"),
                "params": param_total,
            })
        wandb.finish()

    log_f.close()

if __name__ == "__main__":
    main()
