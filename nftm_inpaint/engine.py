#!/usr/bin/env python3
# engine.py

import os, random, argparse, json, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
import matplotlib.pyplot as plt
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency guard
    imageio = None

from nftm_inpaint.metrics import lpips_dist as _metric_lpips
from nftm_inpaint.metrics import psnr as _metric_psnr
from nftm_inpaint.metrics import ssim as _metric_ssim
from nftm_inpaint.metrics import (fid_init, fid_update, fid_compute,
                     kid_init, kid_update, kid_compute)

from nftm_inpaint.rollout import (
    nftm_step, nftm_step_guarded, energy, tv_l1, psnr,
    downsample_like, upsample_like, downsample_mask_minpool,
    clamp_known, corrupt_images, parse_pyramid_arg,
    split_steps_train, split_steps_eval, masked_psnr_metric,
    masked_metric_mean
)
from nftm_inpaint.data_and_viz import ensure_dir, upsample_for_viz
from nftm_inpaint.data_and_viz import random_mask  # data helper

def train_epoch(controller, opt, loader, device, epoch, K_target=10, K_base=4,
                beta: float | None = None, beta_max: float | None = None, gate: bool = False,
                tvw=0.01, p_missing=(0.25,0.5), block_prob=0.5, noise_std=0.3,
                corr_clip=0.2, guard_in_train=True, contract_w=1e-3, rollout_bias=True, pyramid_sizes=None,
                step_loss_mode: str = "final",  gaussian_additive: bool = False):
    controller.train()
    psnrs, losses = [], []
    accepted_steps, backtracks = 0, 0

    # curriculum on rollout depth: grow K_train with epochs
    # K_train = min(K_target, K_base + epoch)
    K_train = K_target
    sizes_default = pyramid_sizes if pyramid_sizes else None

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)  # ground truth in [-1,1]
        M = random_mask(imgs, p_missing=p_missing, block_prob=block_prob).to(device)
        I0 = corrupt_images(imgs, M, noise_std=noise_std, gaussian_additive=gaussian_additive)
        I = clamp_known(I0.clone(), imgs, M)

        # random rollout depth (bias early epochs to short rollouts)
        if rollout_bias:
            lengths = list(range(1, K_train+1))
            weights = torch.tensor([0.6*(0.7**(t-1)) for t in lengths])
            weights = (weights / weights.sum()).to(device='cpu')
            # K_curr = int(torch.multinomial(weights, 1).item() + 1)  # 1..K_train
            K_curr = K_target
        else:
            # K_curr = random.randint(1, K_train)
            K_curr = K_target

        sizes = sizes_default or [imgs.shape[-1]]
        steps_per = split_steps_train(K_curr, sizes, epoch=epoch)

        I = clamp_known(I0.clone(), imgs, M)
        I_prev_for_contract = I.clone().detach()

        # step-weighted loss
        use_linear_steps = (step_loss_mode == "linear")
        if use_linear_steps:
            # weights α_k ∝ k, normalized so sum(α_k) = 1
            # k = 1..K_curr (total number of NFTM steps in this rollout)
            step_weights = torch.arange(
                1, K_curr + 1,
                device=device,
                dtype=imgs.dtype
            )
            step_weights = step_weights / step_weights.sum()
            step_idx = 0
            data_loss_accum = 0.0  # tensor via first addition

        for lvl, (S, T) in enumerate(zip(sizes, steps_per)):
            gt_S = imgs if S == imgs.shape[-1] else downsample_like(imgs, S)
            M_S = M if S == imgs.shape[-1] else downsample_mask_minpool(M, S)
            I = I if I.shape[-1] == S else downsample_like(I, S)

            scale_fac = float(sizes[-1]) / float(S)
            corr_clip_S = corr_clip * scale_fac

            for s in range(T):
                clip_decay = (0.92 ** s)

                if beta is None:
                    beta_S = None  # nftm_step treats None as 1.0
                else:
                    cap = 0.9 if beta_max is None else float(beta_max)
                    beta_S = min(float(beta) * scale_fac, cap)

                if guard_in_train:
                    I, _, _used_beta, ok, _dE = nftm_step_guarded(
                        I, gt_S, M_S, controller,
                        beta=beta_S, use_gate=gate,
                        corr_clip=corr_clip_S,
                        tvw=0.0, max_backtracks=2, shrink=0.5, clip_decay=clip_decay
                    )
                    accepted_steps += int(ok)
                    backtracks += int(not ok)
                else:
                    I, _ = nftm_step(
                        I, gt_S, M_S, controller,
                        beta=beta_S, use_gate=gate,
                        corr_clip=corr_clip_S, clip_decay=clip_decay
                    )
                # ----- NEW: accumulate per-step MSE with linear weights -----
                if use_linear_steps:
                    step_idx += 1  # global step index across levels
                    w_k = step_weights[step_idx - 1]       # scalar tensor
                    # compare at current scale S against downsampled GT
                    step_mse = F.mse_loss(I, gt_S)
                    data_loss_accum = data_loss_accum + w_k * step_mse
                # -----------------------------------------------------------

            if S != sizes[-1]:
                nextS = sizes[lvl+1]
                I = upsample_like(I, nextS)
                gt_next = imgs if nextS == imgs.shape[-1] else downsample_like(imgs, nextS)
                M_next  = M if nextS == M.shape[-1] else downsample_mask_minpool(M, nextS)
                I = clamp_known(I, gt_next, M_next)

        # ----------------- data loss: choose mode -----------------
        if use_linear_steps:
            data_loss = data_loss_accum
        else:
            # original behavior: only final fine output
            data_loss = F.mse_loss(I, imgs)


        # Regularizers
        loss_smooth = tv_l1(I, tvw)
        contract = contract_w * (I - I_prev_for_contract).pow(2).mean()

        loss = data_loss + loss_smooth + contract

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
        opt.step()

        with torch.no_grad():
            losses.append(loss.item())
            psnrs.append(psnr(I, imgs).item())

    stats = dict(train_K=K_train, accepted=accepted_steps, backtracks=backtracks)
    return float(np.mean(losses)), float(np.mean(psnrs)), stats

@torch.no_grad()
def eval_steps(controller, loader, device, K_eval=10,
               beta: float | None = None, gate: bool = False,
               p_missing=(0.25,0.5), block_prob=0.5, noise_std=0.3,
               corr_clip=0.2, descent_guard=False, tvw=0.0,
               save_per_epoch_dir=None, epoch_tag=None, pyramid_sizes=None, 
               steps_split=None, viz_scale: float = 1.0,
               noise_kind: str = "gaussian", noise_kwargs: dict | None = None, gaussian_additive: bool = False):
    controller.eval()

    # same masks + noise every epoch
    EVAL_SEED = 40
    torch.manual_seed(EVAL_SEED)
    random.seed(EVAL_SEED)

    psnrs_step, ssims_step, lpips_step = [], [], []
    # optional per-epoch visualization of first batch progression
    save_seq = (save_per_epoch_dir is not None)
    if save_seq:
        ensure_dir(save_per_epoch_dir)

    use_gate_flag = bool(gate)

    for bidx, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        M = random_mask(imgs, p_missing=p_missing, block_prob=block_prob).to(device)
        I0 = corrupt_images(imgs, M, noise_std=noise_std, noise_kind=noise_kind, **(noise_kwargs or {}),
                            gaussian_additive=gaussian_additive)
        I = clamp_known(I0.clone(), imgs, M)
        I_metrics0 = I if I.shape[-1] == imgs.shape[-1] else upsample_like(I, imgs.shape[-1])
        I_metrics0 = I_metrics0.clamp(-1.0, 1.0)
        step_psnrs = [_metric_psnr(I_metrics0, imgs).item()]
        step_ssims = [_metric_ssim(I_metrics0, imgs).item()]
        step_lpips = [_metric_lpips(I_metrics0, imgs).item()]

        gif_frames = []
        make_gif_frame = None

        if save_seq and bidx == 0:
            vis_rows = min(6, imgs.size(0))
            cols = K_eval + 2
            rows = vis_rows
            plt.figure(figsize=(3*cols, 3*rows))

            def show_img(ax, x_tensor):
                vis = upsample_for_viz(x_tensor, viz_scale)
                ax.imshow(((vis.permute(1, 2, 0).cpu().numpy()+1)/2).clip(0,1))
                ax.axis('off')

            gif_cols = min(8, imgs.size(0))
            gif_cols = max(gif_cols, 1)
            gif_gt = imgs[:gif_cols].detach()

            def make_gif_frame(batch):
                if imageio is None:
                    return None
                current = batch[:gif_cols].detach().clamp(-1.0, 1.0)
                gt = gif_gt.clamp(-1.0, 1.0)
                up_gt = upsample_for_viz(gt, viz_scale)
                up_cur = upsample_for_viz(current, viz_scale)

                panels = [up_gt, up_cur]  # 2 rows: GT, Recon
                panel = torch.cat(panels, dim=0)
                panel = ((panel + 1.0) * 0.5).clamp(0.0, 1.0)
                grid = tv.utils.make_grid(panel, nrow=gif_cols, padding=2, pad_value=0.0)
                return (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            for r in range(vis_rows):
                ax = plt.subplot(rows, cols, r*cols + 1)
                show_img(ax, imgs[r])
                if r == 0: ax.set_title("GT")

                ax = plt.subplot(rows, cols, r*cols + 2)
                show_img(ax, I[r])
                if r == 0: ax.set_title("Init")

            if imageio is not None:
                init_frame = make_gif_frame(I.clamp(-1.0, 1.0))
                if init_frame is not None:
                    gif_frames.append(init_frame)   

        # BEFORE (conceptually):
        # for s in range(K_eval): step at native resolution

        # --- Multi-scale rollout (fills per-step curves) ---
        sizes = pyramid_sizes or [imgs.shape[-1]]
        steps_per = steps_split or split_steps_eval(K_eval, sizes)

        I = clamp_known(I0.clone(), imgs, M)
        total_steps = 0  # to ensure we produce exactly K_eval entries

        for lvl, (S, T) in enumerate(zip(sizes, steps_per)):
            gt_S = imgs if S == imgs.shape[-1] else downsample_like(imgs, S)
            M_S = M if S == imgs.shape[-1] else downsample_mask_minpool(M, S)
            I = I if I.shape[-1] == S else downsample_like(I, S)

            scale_fac = float(sizes[-1]) / float(S)
            corr_clip_S = corr_clip * scale_fac

            beta_S = None if beta is None else min(float(beta) * scale_fac, 0.9)

            for s in range(T):
                clip_decay = (0.92 ** s)
                if descent_guard:
                    I, _, _, _, _ = nftm_step_guarded(
                        I, gt_S, M_S, controller, corr_clip=corr_clip_S, beta=beta_S,
                        tvw=tvw, max_backtracks=3, shrink=0.5, clip_decay=clip_decay,
                        use_gate=use_gate_flag,
                    )
                    gate_map = None
                else:
                    if save_seq and bidx == 0 and use_gate_flag:
                        I, _, _ = nftm_step(
                            I, gt_S, M_S, controller, beta=beta_S, use_gate=True,
                            corr_clip=corr_clip_S, clip_decay=clip_decay, return_gate=True
                        )
                    else:
                        I, _ = nftm_step(
                            I, gt_S, M_S, controller, beta=beta_S, use_gate=use_gate_flag,
                            corr_clip=corr_clip_S, clip_decay=clip_decay
                        )
                        gate_map = None

                    
                if save_seq and bidx == 0:
                    vis_rows = min(6, imgs.size(0))
                    cols = K_eval + 2
                    show_tensor = I if I.shape[-1] == imgs.shape[-1] else upsample_like(I, imgs.shape[-1])
                    for r in range(vis_rows):
                        ax = plt.subplot(rows, cols, r*cols + (total_steps + 3))
                        show_img(ax, show_tensor[r])
                        if r == 0:
                            ax.set_title(f"step {total_steps+1}")

                # --- per-step metrics at native size ---
                I_metrics = I if I.shape[-1] == imgs.shape[-1] else upsample_like(I, imgs.shape[-1])
                I_metrics = I_metrics.clamp(-1.0, 1.0)
                step_psnrs.append(_metric_psnr(I_metrics, imgs).item())
                step_ssims.append(_metric_ssim(I_metrics, imgs).item())
                step_lpips.append(_metric_lpips(I_metrics, imgs).item())
                total_steps += 1

                if save_seq and bidx == 0 and imageio is not None and make_gif_frame is not None:
                    frame = make_gif_frame(I_metrics)
                    if frame is not None:
                        gif_frames.append(frame)

            if S != sizes[-1]:
                nextS = sizes[lvl+1]
                I = upsample_like(I, nextS)
                gt_next = imgs if nextS == imgs.shape[-1] else downsample_like(imgs, nextS)
                M_next  = M if nextS == M.shape[-1] else downsample_mask_minpool(M, nextS)
                I = clamp_known(I, gt_next, M_next)


        psnrs_step.append(step_psnrs)
        ssims_step.append(step_ssims)
        lpips_step.append(step_lpips)

        if save_seq and bidx == 0:
            plt.tight_layout()
            tag = f"epoch_{epoch_tag}" if epoch_tag is not None else "eval"
            out_path = os.path.join(save_per_epoch_dir, f"progress_{tag}.png")
            plt.savefig(out_path, dpi=140)
            print(f"[viz] saved per-epoch progression → {out_path}")
            plt.close()
            if imageio is not None and gif_frames:
                gif_path = os.path.join(save_per_epoch_dir, f"progress_{tag}.gif")
                imageio.mimsave(gif_path, gif_frames, duration=0.4)
                print(f"[gif] saved reconstruction GIF (GT top row, recon bottom) → {gif_path}")
            elif imageio is None:
                print("[gif] skipped GIF generation (imageio not installed)")

    curves = {
        "psnr": np.array(psnrs_step).mean(axis=0) if psnrs_step else np.array([]),
        "ssim": np.array(ssims_step).mean(axis=0) if ssims_step else np.array([]),
        "lpips": np.array(lpips_step).mean(axis=0) if lpips_step else np.array([]),
    }
    return curves


@torch.no_grad()
def evaluate_metrics_full(
    controller,
    loader,
    device,
    *,
    K_eval: int,
    beta: float | None = None,
    gate: bool = False,
    p_missing=(0.25, 0.5),
    block_prob=0.5,
    noise_std=0.3,
    corr_clip=0.2,
    descent_guard: bool = False,
    tvw: float = 0.0,
    benchmark=None,
    pyramid_sizes=None,
    steps_split=None,
    noise_kind: str = "gaussian",
    gaussian_additive: bool = False,
    noise_kwargs: dict | None = None,
):
    controller.eval()
    totals = {
        "psnr_all": 0.0,
        "psnr_miss": 0.0,
        "ssim_all": 0.0,
        "ssim_miss": 0.0,
        "lpips_all": 0.0,
        "lpips_miss": 0.0,
    }
    batches = 0
    fid_metric = fid_init(device)
    kid_metric = kid_init(device, benchmark=benchmark)

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        M = random_mask(imgs, p_missing=p_missing, block_prob=block_prob).to(device)
        I0 = corrupt_images(imgs, M, noise_std=noise_std, noise_kind=noise_kind, gaussian_additive=gaussian_additive,**(noise_kwargs or {}))


        # multi-scale rollout, same as eval_steps()
        sizes = pyramid_sizes or [imgs.shape[-1]]
        steps_per = steps_split or split_steps_eval(K_eval, sizes)

        I = clamp_known(I0.clone(), imgs, M)

        for lvl, (S, T) in enumerate(zip(sizes, steps_per)):
            gt_S = imgs if S == imgs.shape[-1] else downsample_like(imgs, S)
            M_S  = M if S == imgs.shape[-1] else downsample_mask_minpool(M, S)
            I    = I if I.shape[-1] == S else downsample_like(I, S)

            scale_fac = float(sizes[-1]) / float(S)
            corr_clip_S = corr_clip * scale_fac
            beta_S = None if beta is None else min(float(beta) * scale_fac, 0.9)

            for s in range(T):
                clip_decay = 0.92 ** s
                if descent_guard:
                    I, _, _, _, _ = nftm_step_guarded(I, gt_S, M_S, controller,
                                                      beta=beta_S, use_gate=gate,
                                                      corr_clip=corr_clip_S,
                                                      tvw=tvw, max_backtracks=3, shrink=0.5,
                                                      clip_decay=clip_decay)
                else:
                    I, _ = nftm_step(I, gt_S, M_S, controller, beta=beta_S, use_gate=gate,
                                     corr_clip=corr_clip_S, clip_decay=clip_decay)

            # upsample hand-off with size-matched clamp
            if S != sizes[-1]:
                nextS = sizes[lvl+1]
                I = upsample_like(I, nextS)
                gt_next = imgs if nextS == imgs.shape[-1] else downsample_like(imgs, nextS)
                M_next  = M if nextS == imgs.shape[-1] else downsample_mask_minpool(M, nextS)
                I = clamp_known(I, gt_next, M_next)

        preds = I.clamp(-1.0, 1.0)
        miss_mask = 1.0 - M

        totals["psnr_all"] += float(_metric_psnr(preds, imgs).item())
        totals["psnr_miss"] += float(masked_psnr_metric(preds, imgs, miss_mask).item())
        totals["ssim_all"] += float(_metric_ssim(preds, imgs).item())
        totals["ssim_miss"] += float(masked_metric_mean(_metric_ssim, preds, imgs, miss_mask).item())
        totals["lpips_all"] += float(_metric_lpips(preds, imgs).item())
        totals["lpips_miss"] += float(masked_metric_mean(_metric_lpips, preds, imgs, miss_mask).item())
        batches += 1
        # Accumulate FID/KID features
        fid_update(fid_metric, imgs, preds)
        kid_update(kid_metric, imgs, preds)


    if batches == 0:
        raise RuntimeError("Evaluation loader produced no batches for metric computation.")

    results = {k: totals[k] / batches for k in totals}
    # Compute final FID/KID scores over full test set
    results["fid"] = fid_compute(fid_metric)
    results["kid"] = kid_compute(kid_metric)

    return results
