# NFTM — Rule 110 with STE binarization + pretrained (frozen) stateless controller
# - Field f_t: scalar per cell in [0,1]
# - Read: local 3-cell neighborhood (radius 1) with differentiable boundary handling
# - Controller C: shared MLP, outputs LOGITS; trained to 100% on 8 triplets, then frozen
# - Write: overwrite with binarized controller output (via STE); state is binarized before reads
# - No per-step refits; pure read→compute→write execution

import math
import os
import argparse
from pathlib import Path
import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import product
from typing import Optional

# Optional deps for GIF writing
try:
    import imageio
except Exception:
    imageio = None
try:
    from PIL import Image
except Exception:
    Image = None

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Rule 110 utilities -------------------------------
RULE_TABLE = {
    (1, 1, 1): 0, (1, 1, 0): 1, (1, 0, 1): 1, (1, 0, 0): 0,
    (0, 1, 1): 1, (0, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): 0,
}

def rule110_next(bits_row, boundary="zeros"):
    N = len(bits_row)
    if boundary == "wrap":
        left, cent, right = np.roll(bits_row,1), bits_row, np.roll(bits_row,-1)
    elif boundary == "reflect":
        left  = np.concatenate([[bits_row[0]],  bits_row[:-1]])
        cent  = bits_row
        right = np.concatenate([bits_row[1:], [bits_row[-1]]])
    else:  # zeros
        pad = np.pad(bits_row, (1,1), mode="constant", constant_values=0)
        left, cent, right = pad[:-2], pad[1:-1], pad[2:]
    out = np.zeros_like(bits_row)
    for i in range(N):
        out[i] = RULE_TABLE[(int(left[i]), int(cent[i]), int(right[i]))]
    return out

def rule110_rollout(bits0, T, boundary="zeros"):
    rows = [bits0.astype(np.uint8)]
    cur = bits0
    for _ in range(1, T):
        cur = rule110_next(cur, boundary=boundary)
        rows.append(cur.astype(np.uint8))
    return np.stack(rows, axis=0)


class StatelessController(nn.Module):
    """Small MLP trained to emulate the Rule 110 truth table."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, neigh):  # neigh: (B,N,3)
        B, N, _ = neigh.shape
        logits = self.net(neigh.view(B * N, 3)).view(B, N)  # (B,N)
        return logits

def read_neighborhoods(field, boundary="zeros"):
    """Local read kernel (radius 1). field: (B,N) in [0,1] -> (B,N,3)."""
    B, N = field.shape
    if boundary == "wrap":
        left  = torch.roll(field, 1, dims=1);  cent = field;  right = torch.roll(field, -1, dims=1)
    elif boundary == "reflect":
        left  = torch.cat([field[:, :1], field[:, :-1]], dim=1)
        cent  = field
        right = torch.cat([field[:, 1:], field[:, -1:]], dim=1)
    else:  # zeros
        z = torch.zeros(B, 1, device=field.device, dtype=field.dtype)
        left  = torch.cat([z, field[:, :-1]], dim=1); cent = field; right = torch.cat([field[:, 1:], z], dim=1)
    return torch.stack([left, cent, right], dim=-1)

class BinarizeSTE(torch.autograd.Function):
    """Straight-Through Estimator for threshold at 0.5."""
    @staticmethod
    def forward(ctx, x):  # x in [0,1]
        return (x > 0.5).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # identity pass-through

def ste(x):  # helper
    return BinarizeSTE.apply(x)

def write_update(prev_field, write_bits, beta=1.0):
    """Local write (radius 0). Overwrite as special case of additive write."""
    return (1.0 - beta) * prev_field + beta * write_bits

class NFTMRule110(nn.Module):
    """NFTM executor: f_{t+1} = A_write( g(C( A_read( P(f_t) ))) )."""
    def __init__(self, controller, boundary="zeros", beta=1.0):
        super().__init__()
        self.controller = controller
        self.boundary = boundary
        self.beta = beta

    def forward(self, f0, T):
        """
        f0: (B,N) floats in [0,1]
        returns list of T fields (B,N)
        """
        fields = [f0]
        f = f0
        for _ in range(1, T):
            # P: binarize state for reads (radius-0 projection)
            f_read = ste(f)                                     # hard bits, STE for grads
            neigh  = read_neighborhoods(f_read, self.boundary)  # (B,N,3)
            logits = self.controller(neigh)                     # (B,N)
            w_prob = torch.sigmoid(logits)                      # in [0,1]
            w_bits = ste(w_prob)                                # binarize writes as well
            f = write_update(f, w_bits, beta=self.beta)         # local write
            fields.append(f)
        return fields

# ------------------------------- Controller pretrain -----------------------------
def train_controller_exact(controller, epochs=2000, lr=2e-3, verbose=True):
    """Train on all 8 triplets to 100% using BCEWithLogitsLoss."""
    patterns = list(product([0,1], repeat=3))
    X = torch.tensor(patterns, dtype=torch.float32, device=device)      # (8,3)
    y = torch.tensor([RULE_TABLE[p] for p in patterns], dtype=torch.float32,
                     device=device).unsqueeze(1)                         # (8,1)

    opt = optim.Adam(controller.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        logits = controller(X.view(1, 8, 3)).view(8,1)  # (8,1)
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

        if verbose and (ep % 200 == 0 or ep == 1):
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == y).float().mean().item()
            print(f"[Controller] ep {ep:4d} | loss {loss.item():.6f} | acc {acc*100:.1f}%")
            if acc == 1.0:
                break

    # final sanity
    with torch.no_grad():
        logits = controller(X.view(1, 8, 3)).view(8,1)
        pred = (torch.sigmoid(logits) > 0.5).long().view(-1).cpu().tolist()
    if verbose:
        print("\n[Controller] predictions on 8 cases:")
        for p, pb in zip(patterns, pred):
            print(f"  {p} -> {pb}  (rule110={RULE_TABLE[p]})")
    return True

# -------------------- Train controller from GT rollouts (sequence-derived) --------------------
def _np_read_neighborhoods(bits_row: np.ndarray, boundary: str) -> np.ndarray:
    N = bits_row.shape[0]
    if boundary == "wrap":
        left, cent, right = np.roll(bits_row,1), bits_row, np.roll(bits_row,-1)
    elif boundary == "reflect":
        left  = np.concatenate([[bits_row[0]],  bits_row[:-1]])
        cent  = bits_row
        right = np.concatenate([bits_row[1:], [bits_row[-1]]])
    else:  # zeros
        pad = np.pad(bits_row, (1,1), mode="constant", constant_values=0)
        left, cent, right = pad[:-2], pad[1:-1], pad[2:]
    return np.stack([left, cent, right], axis=-1).astype(np.float32)

def build_gt_dataset(num_inits: int, N: int, T: int, boundary: str, init_type: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for k in range(num_inits):
        if init_type == "single_one":
            init_np = np.zeros((N,), dtype=np.uint8); init_np[N//2] = 1
        elif init_type == "random_single_one":
            init_np = np.zeros((N,), dtype=np.uint8); init_np[rng.integers(0,N)] = 1
        else:  # random_bits
            init_np = rng.integers(0,2,size=(N,), dtype=np.uint8)
        gt = rule110_rollout(init_np, T, boundary=boundary)  # (T,N)
        for t in range(T-1):
            cur = gt[t]
            nxt = gt[t+1]
            neigh = _np_read_neighborhoods(cur, boundary)  # (N,3)
            Xs.append(neigh)
            ys.append(nxt.astype(np.float32)[:,None])
    X = np.concatenate(Xs, axis=0)  # (num_inits*(T-1)*N, 3)
    y = np.concatenate(ys, axis=0)  # (num_inits*(T-1)*N, 1)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    return X_t, y_t

def train_controller_from_rollouts(controller,
                                   num_inits: int = 32,
                                   N: int = 161,
                                   T: int = 100,
                                   boundary: str = "zeros",
                                   init_type: str = "random_bits",
                                   epochs: int = 5,
                                   lr: float = 2e-3,
                                   batch_size: int = 8192,
                                   seed: int = 0,
                                   verbose: bool = True):
    X, y = build_gt_dataset(num_inits, N, T, boundary, init_type, seed)
    dataset = torch.utils.data.TensorDataset(X.to(device), y.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = optim.Adam(controller.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        running = 0.0
        for xb, yb in loader:
            logits = controller(xb.view(1, -1, 3)).view(-1,1)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        if verbose:
            with torch.no_grad():
                logits = controller(X[:4096].view(1, -1, 3).to(device)).view(-1,1)
                lval = loss_fn(logits, y[:4096].to(device)).item()
            print(f"[Controller-GT] ep {ep:4d} | train_loss {running/len(loader):.6f} | val_loss {lval:.6f}")
    return True

# --------------------------------- Evaluation/plot -------------------------------
def eval_and_plot(model,
                  N: int = 161,
                  T: int = 100,
                  boundary: str = "zeros",
                  seed: int = 0,
                  init: str = "single_one",
                  save_png_path: Optional[Path] = None,
                  save_gif_path: Optional[Path] = None,
                  gif_fps: int = 10):
    rng = np.random.default_rng(seed)
    if init == "single_one":
        init_np = np.zeros((N,), dtype=np.uint8); init_np[N//2] = 1
    elif init == "random_single_one":
        init_np = np.zeros((N,), dtype=np.uint8); init_np[rng.integers(0,N)] = 1
    else:  # random_bits
        init_np = rng.integers(0,2,size=(N,), dtype=np.uint8)

    gt = rule110_rollout(init_np, T, boundary=boundary)              # (T,N)
    f0 = torch.tensor(init_np[None,:], dtype=torch.float32, device=device)

    with torch.no_grad():
        fields = model(f0, T=T)                                      # list of (1,N)
        arr = torch.stack(fields, 0).squeeze(1).cpu().numpy()        # (T,N) floats
    pred_bits = (arr > 0.5).astype(np.uint8)
    acc = (pred_bits == gt).mean()
    print(f"[Eval] rollout accuracy vs Rule110: {acc*100:.2f}%  (T={T}, N={N}, boundary={boundary})")

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(gt, cmap="binary", origin="upper", interpolation="nearest")
    ax1.set_title("Rule 110 (GT)")
    ax1.set_xlabel("position"); ax1.set_ylabel("time")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(pred_bits, cmap="binary", origin="upper", interpolation="nearest")
    ax2.set_title("NFTM prediction")
    ax2.set_xlabel("position"); ax2.set_ylabel("time")
    fig.tight_layout()

    # Save PNG if requested
    if save_png_path is not None:
        save_png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png_path, dpi=300, bbox_inches='tight')

    plt.show()

    # Save GIF if requested
    if save_gif_path is not None:
        save_gif_path.parent.mkdir(parents=True, exist_ok=True)

        # Build frames via in-memory PNG to avoid DPI/retina mismatches
        pil_frames = []
        np_frames = []
        for t in range(1, T + 1):
            f = plt.figure(figsize=(6, 6))
            ax = f.add_subplot(1,1,1)
            ax.imshow(pred_bits[:t, :], cmap="binary", origin="upper", interpolation="nearest")
            ax.set_title("NFTM rollout")
            ax.set_xlabel("position"); ax.set_ylabel("time")
            ax.set_xticks([]); ax.set_yticks([])
            f.tight_layout()

            buf = io.BytesIO()
            f.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            if Image is not None:
                pil_frames.append(Image.open(buf).convert('RGB'))
            elif imageio is not None:
                np_frames.append(imageio.v2.imread(buf))
            buf.close()
            plt.close(f)

        duration = 1.0 / max(1, gif_fps)
        if Image is not None and len(pil_frames) > 0:
            pil_frames[0].save(save_gif_path, save_all=True, append_images=pil_frames[1:], loop=0,
                               duration=int(1000 * duration))
        elif imageio is not None and len(np_frames) > 0:
            imageio.mimsave(save_gif_path, np_frames, duration=duration)
        else:
            print("[Warn] Cannot save GIF: neither PIL nor imageio are available.")

# --------------------------------------- Main -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFTM Rule 110 — train controller and visualize")
    parser.add_argument("--train_mode", type=str, choices=["truth_table", "gt_rollout"], default="truth_table",
                        help="How to train the controller")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs for controller")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate for controller")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size for controller MLP")

    parser.add_argument("--boundary", type=str, choices=["zeros", "wrap", "reflect"], default="zeros")
    parser.add_argument("--beta", type=float, default=1.0, help="Write blend for executor")
    parser.add_argument("--N", type=int, default=161)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    # Dataset for gt_rollout mode
    parser.add_argument("--dataset_inits", type=int, default=32, help="# of initializations for dataset")
    parser.add_argument("--dataset_T", type=int, default=100, help="Timesteps per initialization for dataset")
    parser.add_argument("--init_type", type=str, choices=["single_one", "random_single_one", "random_bits"],
                        default="random_bits")
    parser.add_argument("--batch_size", type=int, default=8192)

    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    boundary = args.boundary
    beta = args.beta
    N = args.N

    print("[Start] Training controller… mode=", args.train_mode)
    controller = StatelessController(hidden=args.hidden).to(device)
    if args.train_mode == "truth_table":
        train_controller_exact(controller, epochs=args.epochs, lr=args.lr, verbose=True)
    else:
        train_controller_from_rollouts(controller,
                                       num_inits=args.dataset_inits,
                                       N=N,
                                       T=args.dataset_T,
                                       boundary=boundary,
                                       init_type=args.init_type,
                                       epochs=max(1, args.epochs//20),  # fewer epochs needed on big dataset
                                       lr=args.lr,
                                       batch_size=args.batch_size,
                                       seed=args.seed,
                                       verbose=True)

    # Freeze controller (stateless executor)
    for p in controller.parameters():
        p.requires_grad = False

    print("\n[Run] Executing NFTM with STE binarization …")
    model = NFTMRule110(controller, boundary=boundary, beta=beta).to(device)

    # Output paths
    out_dir = Path("results_rule110")
    out_dir.mkdir(exist_ok=True)
    # Evaluation horizon: if trained from GT rollouts with T=train_T, evaluate at 2*train_T
    if args.train_mode == "gt_rollout":
        T_eval = int(2 * args.dataset_T)
        print(f"[Eval] Using doubled horizon: train_T={args.dataset_T} -> eval_T={T_eval}")
    else:
        T_eval = args.T
    tag = f"N{N}_TrainT{args.dataset_T if args.train_mode=='gt_rollout' else 'NA'}_EvalT{T_eval}_{boundary}_{args.train_mode}"
    png_path = out_dir / f"rule110_viz_{tag}.png"
    gif_path = out_dir / f"rule110_viz_{tag}.gif"

    eval_and_plot(
        model,
        N=N,
        T=T_eval,
        boundary=boundary,
        seed=args.seed,
        init="single_one",
        save_png_path=png_path,
        save_gif_path=gif_path,
        gif_fps=10,
    )
