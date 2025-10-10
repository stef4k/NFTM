#!/usr/bin/env python3
# image_inpainting.py
# NFTM-style iterative inpainting on CIFAR-10 with:
# - heteroscedastic OR homoscedastic data loss (switchable)
# - random rollouts + curriculum
# - descent guard (backtracking) at eval (optional at train)
# - damping (beta), per-step clip decay, contractive penalty
# - rich logging + saved plots + metrics.json

import os, math, random, argparse, json, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
import matplotlib.pyplot as plt

from metrics import lpips_dist as _metric_lpips
from metrics import psnr as _metric_psnr
from metrics import ssim as _metric_ssim
from unet_model import DoubleConv as UNetDoubleConv, UpBlock as UNetUpBlock, TinyUNet

# -------------------------- Utilities --------------------------

def set_seed(seed: int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def make_transforms():
    # Normalize to [-1,1]
    return T.Compose([T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)])

def random_mask(batch, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3):
    """Return mask M (1=known, 0=missing). Mix of random pixels and random square blocks."""
    B, C, H, W = batch.shape
    device = batch.device
    M = torch.ones((B, 1, H, W), device=device)
    # random pixels
    frac = torch.empty(B, 1, 1, 1, device=device).uniform_(*p_missing)
    pix_mask = (torch.rand(B, 1, H, W, device=device) > frac).float()
    M = M * pix_mask
    # random blocks
    for b in range(B):
        if random.random() < block_prob:
            for _ in range(random.randint(min_blocks, max_blocks)):
                sz = random.randint(H//8, H//3)
                y = random.randint(0, H - sz)
                x = random.randint(0, W - sz)
                M[b, :, y:y+sz, x:x+sz] = 0.0
    return M

def corrupt_images(img, M, noise_std=0.3):
    # keep known pixels, corrupt others with noise
    noise = torch.randn_like(img) * noise_std
    return M*img + (1-M)*noise

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


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def masked_psnr_metric(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=a.dtype)
    mask_exp = mask.expand_as(a)
    denom = mask_exp.sum().clamp_min(1e-8)
    mse = ((a - b) * mask_exp).pow(2).sum() / denom
    if mse <= 0:
        return torch.tensor(99.0, device=a.device, dtype=a.dtype)
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

def hetero_gauss_nll(pred, target, log_sigma):
    # Channel-wise Gaussian NLL; σ = softplus(exp(log_sigma)) to ensure positivity
    sigma = F.softplus(log_sigma) + 1e-3
    err = pred - target
    nll = 0.5*((err / sigma)**2 + 2*torch.log(sigma))
    return nll.mean(), sigma

def homoscedastic_loss(pred, target, sigma_const=0.1):
    # Gaussian NLL with fixed sigma (no predicted uncertainty)
    err = pred - target
    return 0.5 * ((err / sigma_const) ** 2 + 2 * math.log(sigma_const)).mean()

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

# -------------------------- Model --------------------------

class TinyController(nn.Module):
    """
    Inputs: concat(I_t (3ch), M (1ch)) -> conv stack
    Outputs:
      - dI: per-pixel correction (3ch), tanh-clamped
      - gate: per-pixel gate in (0,1) (1ch)
      - log_sigma: per-pixel per-channel noise log-std (3ch) [used only if hetero loss]
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
        c1, c2, c3 = base, base * 2, int(round(base * 2.8))
        self.enc1 = UNetDoubleConv(in_ch, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = UNetDoubleConv(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid = UNetDoubleConv(c3, c3)
        self.up1 = UNetUpBlock(c3, c2, c2)
        self.up2 = UNetUpBlock(c2, c1, c1)
        self.head_dI = nn.Conv2d(c1, 3, 3, padding=1)
        self.head_gate = nn.Conv2d(c1, 1, 3, padding=1)
        self.head_logS = nn.Conv2d(c1, 3, 3, padding=1)

        self.apply(TinyUNet._init_weights)

    def forward(self, I, M):
        x = torch.cat([I, M], dim=1)
        s1 = self.enc1(x)
        d1 = self.down1(s1)
        s2 = self.enc2(d1)
        d2 = self.down2(s2)
        h = self.mid(d2)
        u1 = self.up1(h, s2)
        u2 = self.up2(u1, s1)
        dI = self.head_dI(u2).tanh()
        gate = torch.sigmoid(self.head_gate(u2))
        logS = self.head_logS(u2)
        return dI, gate, logS

def nftm_step(I, I_gt, M, controller, beta=0.5, corr_clip=0.2, clip_decay=1.0):
    """One step without guard; returns (I_new, logS)."""
    dI, gate, logS = controller(I, M)
    dI = dI.clamp(-corr_clip*clip_decay, corr_clip*clip_decay)
    I_new = I + beta * gate * dI
    I_new = clamp_known(I_new, I_gt, M)
    return I_new, logS

# -------------------------- Energy & Guard --------------------------

def energy(I, I_gt, logS, tvw=0.01, sigma_prior=1e-4, loss_mode="hetero", homo_sigma=0.1):
    if loss_mode == "hetero":
        nll, sigma = hetero_gauss_nll(I, I_gt, logS)
        prior = sigma_prior * (torch.log(sigma + 1e-8)).pow(2).mean()
        data_term = nll + prior
    else:
        data_term = homoscedastic_loss(I, I_gt, sigma_const=homo_sigma)
    return data_term + tv_l1(I, tvw)

def nftm_step_guarded(I, I_gt, M, controller, beta, corr_clip=0.2, tvw=0.01,
                      max_backtracks=3, shrink=0.5, clip_decay=1.0,
                      sigma_prior=1e-4, loss_mode="hetero", homo_sigma=0.1):
    """Try a step; if energy ↑, shrink beta and retry."""
    with torch.no_grad():
        E0 = energy(I, I_gt, torch.zeros_like(I), tvw=tvw,
                    sigma_prior=sigma_prior, loss_mode=loss_mode, homo_sigma=homo_sigma)
    cur_beta = beta
    for _ in range(max_backtracks+1):
        I_prop, logS = nftm_step(I, I_gt, M, controller, beta=cur_beta,
                                 corr_clip=corr_clip, clip_decay=clip_decay)
        with torch.no_grad():
            E1 = energy(I_prop, I_gt, logS, tvw=tvw,
                        sigma_prior=sigma_prior, loss_mode=loss_mode, homo_sigma=homo_sigma)
        if E1 <= E0:
            return I_prop, logS, cur_beta, True, float(E1 - E0)
        cur_beta *= shrink
    # give up: return original (reject)
    return I, torch.zeros_like(I), cur_beta, False, 0.0

# -------------------------- Train / Eval --------------------------

def train_epoch(controller, opt, loader, device, epoch, K_target=10, K_base=4,
                beta=0.4, tvw=0.01, p_missing=(0.25,0.5), block_prob=0.5, noise_std=0.3,
                corr_clip=0.2, guard_in_train=False, sigma_prior=1e-4,
                contract_w=1e-3, rollout_bias=True,
                loss_mode="hetero", homo_sigma=0.1):
    controller.train()
    psnrs, losses = [], []
    accepted_steps, backtracks = 0, 0

    # curriculum on rollout depth: grow K_train with epochs
    K_train = min(K_target, K_base + epoch)

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)  # ground truth in [-1,1]
        M = random_mask(imgs, p_missing=p_missing, block_prob=block_prob).to(device)
        I0 = corrupt_images(imgs, M, noise_std=noise_std)
        I = clamp_known(I0.clone(), imgs, M)

        # random rollout depth (bias early epochs to short rollouts)
        if rollout_bias:
            lengths = list(range(1, K_train+1))
            weights = torch.tensor([0.6*(0.7**(t-1)) for t in lengths])
            weights = (weights / weights.sum()).to(device='cpu')
            t = int(torch.multinomial(weights, 1).item() + 1) - 0  # 1..K_train
        else:
            t = random.randint(1, K_train)

        logS_last = torch.zeros_like(imgs[:, :3, ...], device=device)
        I_prev_for_contract = I.clone().detach()

        for s in range(t):
            clip_decay = (0.92 ** s)  # slightly tighter clipping at later steps
            if guard_in_train:
                I, logS_last, used_beta, ok, dE = nftm_step_guarded(
                    I, imgs, M, controller, beta=beta, corr_clip=corr_clip,
                    tvw=0.0, max_backtracks=2, shrink=0.5, clip_decay=clip_decay,
                    sigma_prior=sigma_prior, loss_mode=loss_mode, homo_sigma=homo_sigma
                )
                accepted_steps += int(ok)
                backtracks += int(not ok)
            else:
                I, logS_last = nftm_step(I, imgs, M, controller, beta=beta,
                                         corr_clip=corr_clip, clip_decay=clip_decay)

        # Data loss
        if loss_mode == "hetero":
            nll, sigma = hetero_gauss_nll(I, imgs, logS_last)
            data_loss = nll
            sigma_reg = 1e-4 * (torch.log(F.softplus(logS_last)+1e-3 + 1e-8)).pow(2).mean()
        else:
            sigma = None
            data_loss = homoscedastic_loss(I, imgs, sigma_const=homo_sigma)
            sigma_reg = 0.0

        # Regularizers
        loss_smooth = tv_l1(I, tvw)
        contract = contract_w * (I - I_prev_for_contract).pow(2).mean()

        loss = data_loss + loss_smooth + contract + sigma_reg

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
def eval_steps(controller, loader, device, K_eval=10, beta=0.6,
               p_missing=(0.25,0.5), block_prob=0.5, noise_std=0.3,
               corr_clip=0.2, descent_guard=False, tvw=0.0,
               save_per_epoch_dir=None, epoch_tag=None, sigma_prior=1e-4,
               loss_mode="hetero", homo_sigma=0.1):
    controller.eval()
    psnrs_step = []
    # optional per-epoch visualization of first batch progression
    save_seq = (save_per_epoch_dir is not None)
    if save_seq:
        ensure_dir(save_per_epoch_dir)

    for bidx, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        M = random_mask(imgs, p_missing=p_missing, block_prob=block_prob).to(device)
        I0 = corrupt_images(imgs, M, noise_std=noise_std)
        I = clamp_known(I0.clone(), imgs, M)
        step_psnrs = []

        if save_seq and bidx == 0:
            vis_rows = min(6, imgs.size(0))
            cols = K_eval + 2
            plt.figure(figsize=(3*cols, 3*vis_rows))
            def show_img(ax, x):
                ax.imshow(((x.permute(1,2,0).cpu().numpy()+1)/2).clip(0,1))
                ax.axis('off')
            for r in range(vis_rows):
                ax = plt.subplot(vis_rows, cols, r*cols+1)
                show_img(ax, imgs[r]); 
                if r==0: ax.set_title("GT")
                ax = plt.subplot(vis_rows, cols, r*cols+2)
                show_img(ax, I[r]); 
                if r==0: ax.set_title("Init")

        for s in range(K_eval):
            clip_decay = (0.92 ** s)
            if descent_guard:
                I, _, used_beta, ok, dE = nftm_step_guarded(
                    I, imgs, M, controller, beta=beta, corr_clip=corr_clip,
                    tvw=tvw, max_backtracks=3, shrink=0.5,
                    clip_decay=clip_decay, sigma_prior=sigma_prior,
                    loss_mode=loss_mode, homo_sigma=homo_sigma
                )
            else:
                I, _ = nftm_step(I, imgs, M, controller, beta=beta,
                                 corr_clip=corr_clip, clip_decay=clip_decay)
            step_psnrs.append(psnr(I, imgs).item())

            if save_seq and bidx == 0:
                vis_rows = min(6, imgs.size(0))
                cols = K_eval + 2
                for r in range(vis_rows):
                    ax = plt.subplot(vis_rows, cols, r*cols + (s+3))
                    ax.imshow(((I[r].permute(1,2,0).cpu().numpy()+1)/2).clip(0,1))
                    ax.axis('off')
                    if r==0: ax.set_title(f"step {s+1}")

        psnrs_step.append(step_psnrs)

        if save_seq and bidx == 0:
            plt.tight_layout()
            tag = f"epoch_{epoch_tag}" if epoch_tag is not None else "eval"
            out_path = os.path.join(save_per_epoch_dir, f"progress_{tag}.png")
            plt.savefig(out_path, dpi=140)
            print(f"[viz] saved per-epoch progression → {out_path}")
            plt.close()

    return np.array(psnrs_step).mean(axis=0)  # mean PSNR per step


@torch.no_grad()
def evaluate_metrics_full(
    controller,
    loader,
    device,
    *,
    K_eval: int,
    beta: float,
    p_missing=(0.25, 0.5),
    block_prob=0.5,
    noise_std=0.3,
    corr_clip=0.2,
    descent_guard: bool = True,
    tvw: float = 0.0,
    sigma_prior: float = 1e-4,
    loss_mode: str = "hetero",
    homo_sigma: float = 0.1,
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

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        mask_known = random_mask(imgs, p_missing=p_missing, block_prob=block_prob).to(device)
        corrupted = corrupt_images(imgs, mask_known, noise_std=noise_std)
        I = clamp_known(corrupted.clone(), imgs, mask_known)

        for step in range(K_eval):
            clip_decay = 0.92 ** step
            if descent_guard:
                I, _, _, _, _ = nftm_step_guarded(
                    I,
                    imgs,
                    mask_known,
                    controller,
                    beta=beta,
                    corr_clip=corr_clip,
                    tvw=tvw,
                    max_backtracks=3,
                    shrink=0.5,
                    clip_decay=clip_decay,
                    sigma_prior=sigma_prior,
                    loss_mode=loss_mode,
                    homo_sigma=homo_sigma,
                )
            else:
                I, _ = nftm_step(
                    I,
                    imgs,
                    mask_known,
                    controller,
                    beta=beta,
                    corr_clip=corr_clip,
                    clip_decay=clip_decay,
                )

        preds = I.clamp(-1.0, 1.0)
        miss_mask = 1.0 - mask_known

        totals["psnr_all"] += float(_metric_psnr(preds, imgs).item())
        totals["psnr_miss"] += float(masked_psnr_metric(preds, imgs, miss_mask).item())
        totals["ssim_all"] += float(_metric_ssim(preds, imgs).item())
        totals["ssim_miss"] += float(masked_metric_mean(_metric_ssim, preds, imgs, miss_mask).item())
        totals["lpips_all"] += float(_metric_lpips(preds, imgs).item())
        totals["lpips_miss"] += float(masked_metric_mean(_metric_lpips, preds, imgs, miss_mask).item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Evaluation loader produced no batches for metric computation.")

    return {key: totals[key] / batches for key in totals}


def plot_psnr_curve(curve, save_path):
    ensure_dir(os.path.dirname(save_path) or ".")
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(curve)+1), curve, marker='o')
    plt.xlabel("NFTM step"); plt.ylabel("PSNR (dB)")
    plt.title("Step-wise PSNR (eval)")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=160)
    print(f"[plot] saved PSNR curve → {save_path}")
    plt.close()

# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser(description="NFTM-style iterative inpainting on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--K_train", type=int, default=8, help="max rollout steps for training curriculum")
    parser.add_argument("--K_eval", type=int, default=12, help="rollout steps for evaluation")
    parser.add_argument("--beta_start", type=float, default=0.28, help="initial beta (step size)")
    parser.add_argument("--beta_max", type=float, default=0.6, help="cap on beta during training")
    parser.add_argument("--beta_anneal", type=float, default=0.03, help="per-epoch beta increment")
    parser.add_argument("--beta_eval_bonus", type=float, default=0.05, help="extra beta for eval")
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--corr_clip", type=float, default=0.1, help="max per-step correction magnitude (base)")
    parser.add_argument("--pmin", type=float, default=0.25, help="min missing fraction")
    parser.add_argument("--pmax", type=float, default=0.5, help="max missing fraction")
    parser.add_argument("--block_prob", type=float, default=0.5, help="probability to add random occlusion blocks")
    parser.add_argument("--noise_std", type=float, default=0.3, help="corruption noise std for missing pixels")
    parser.add_argument("--width", type=int, default=48, help="controller width")
    parser.add_argument("--controller", type=str, default="dense", choices=["dense", "unet"],
                        help="controller architecture")
    parser.add_argument("--unet_base", type=int, default=10, help="base channels for UNet controller")
    parser.add_argument("--save_dir", type=str, default="out", help="directory to save plots/metrics")
    parser.add_argument("--save_epoch_progress", action="store_true", help="save per-epoch step grids for the first eval batch")
    parser.add_argument("--guard_in_train", action="store_true", help="enable descent guard during training (slower, more stable)")
    parser.add_argument("--contract_w", type=float, default=1e-3, help="contractive penalty weight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # NEW: loss mode switches + metrics dump
    parser.add_argument("--loss", type=str, default="hetero", choices=["hetero","homo"])
    parser.add_argument("--homo_sigma", type=float, default=0.1, help="σ for homoscedastic loss")
    parser.add_argument("--save_metrics", action="store_true", help="save metrics.json + psnr_curve.npy to save_dir")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device} | loss_mode={args.loss} | homo_sigma={args.homo_sigma if args.loss=='homo' else 'n/a'}")

    # Data
    transform = make_transforms()
    train_set = tv.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set  = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    use_cuda_pinning = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2,
                              pin_memory=use_cuda_pinning)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
                              pin_memory=use_cuda_pinning)

    # Model / Optim
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
    info_msg = f"[controller] {controller_info['name']}"
    if "base" in controller_info:
        info_msg += f" | base={controller_info['base']}"
    info_msg += f" | params={controller_info['params']}"
    print(info_msg)
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ensure_dir(args.save_dir)
    steps_dir = os.path.join(args.save_dir, "steps") if args.save_epoch_progress else None
    if steps_dir: ensure_dir(steps_dir)

    # Train
    for ep in range(1, args.epochs+1):
        beta = min(args.beta_start + args.beta_anneal * (ep-1), args.beta_max)
        beta_eval = min(beta + args.beta_eval_bonus, 0.9)

        train_loss, train_psnr, stats = train_epoch(
            controller, opt, train_loader, device, epoch=ep,
            K_target=args.K_train, K_base=4, beta=beta, tvw=args.tv_weight,
            p_missing=(args.pmin, args.pmax), block_prob=args.block_prob,
            noise_std=args.noise_std, corr_clip=args.corr_clip,
            guard_in_train=args.guard_in_train, sigma_prior=1e-4,
            contract_w=args.contract_w, rollout_bias=True,
            loss_mode=args.loss, homo_sigma=args.homo_sigma
        )

        curve = eval_steps(
            controller, test_loader, device,
            K_eval=args.K_eval, beta=beta_eval,
            p_missing=(args.pmin, args.pmax), block_prob=args.block_prob,
            noise_std=args.noise_std, corr_clip=args.corr_clip,
            descent_guard=False, tvw=0.0,
            save_per_epoch_dir=steps_dir, epoch_tag=ep, sigma_prior=1e-4,
            loss_mode=args.loss, homo_sigma=args.homo_sigma
        )

        curve_str = ", ".join(f"{v:.2f}" for v in curve[:min(5, len(curve))])
        print(f"[ep {ep:02d}] β_train={beta:.3f} K_train={stats['train_K']} | loss {train_loss:.4f} | "
              f"train PSNR {train_psnr:.2f} dB | eval PSNR 1..{args.K_eval}: {curve_str} ... {curve[-1]:.2f} | "
              f"ctrl={args.controller}")
        if args.guard_in_train:
            print(f"         accepted steps: {stats['accepted']} | backtracks (approx): {stats['backtracks']}")

    # Save final PSNR curve
    plot_psnr_curve(curve, os.path.join(args.save_dir, "psnr_curve.png"))

    # Save a final sample grid (progression of first test batch)
    final_beta = min(args.beta_start + args.beta_anneal * (args.epochs-1), args.beta_max)
    beta_eval_final = min(final_beta + args.beta_eval_bonus, 0.9)

    _ = eval_steps(
        controller, test_loader, device,
        K_eval=min(args.K_eval, 10), beta=beta_eval_final,
        p_missing=(args.pmin, args.pmax), block_prob=args.block_prob,
        noise_std=args.noise_std, corr_clip=args.corr_clip,
        descent_guard=True, tvw=0.0,
        save_per_epoch_dir=os.path.join(args.save_dir, "final"),
        epoch_tag="final", sigma_prior=1e-4,
        loss_mode=args.loss, homo_sigma=args.homo_sigma
    )
    print("[done] checkpoints and plots saved under:", args.save_dir, f"| controller={args.controller}")

    # Save metrics for the driver / comparisons
    if args.save_metrics:
        metrics_full = evaluate_metrics_full(
            controller,
            test_loader,
            device,
            K_eval=args.K_eval,
            beta=beta_eval_final,
            p_missing=(args.pmin, args.pmax),
            block_prob=args.block_prob,
            noise_std=args.noise_std,
            corr_clip=args.corr_clip,
            descent_guard=True,
            tvw=0.0,
            sigma_prior=1e-4,
            loss_mode=args.loss,
            homo_sigma=args.homo_sigma,
        )
        np.save(os.path.join(args.save_dir, "psnr_curve.npy"), curve)
        summary = dict(
            loss_mode=args.loss,
            homo_sigma=args.homo_sigma,
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
            final_psnr=float(curve[-1]),
            curve=[float(x) for x in curve]
        )
        summary.update(metrics_full)
        summary["params"] = float(param_total)
        summary["seed"] = int(args.seed)
        summary["controller"] = args.controller
        if args.controller == "unet":
            summary["unet_base"] = int(controller_info.get("base", args.unet_base))
        with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[metrics] saved metrics.json & psnr_curve.npy in {args.save_dir} | controller={args.controller}")

if __name__ == "__main__":
    main()
