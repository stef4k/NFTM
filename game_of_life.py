# NFTM — Conway's Game of Life with STE binarization + pretrained (frozen) stateless controller
# - Field f_t: binary grid in [0,1]
# - Read: 3x3 neighborhoods via F.pad + F.unfold (boundary: zeros|wrap|reflect)
# - Controller C: shared MLP, inputs 9 bits, outputs LOGIT; trained to 100% on 512 patterns, then frozen
# - Write: overwrite with binarized controller output (STE); state is binarized before reads
# - No per-step refits; pure read→compute→write execution

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import product

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------ Life utilities (NumPy) ---------------------------
def life_next_np(grid, boundary="wrap"):
    """grid: (H,W) uint8 in {0,1} -> next grid with specified boundary."""
    H, W = grid.shape

    if boundary == "wrap":
        # sum of 8 neighbor rolls
        s = (
            np.roll(np.roll(grid,  1, 0),  1, 1) +
            np.roll(np.roll(grid,  1, 0),  0, 1) +
            np.roll(np.roll(grid,  1, 0), -1, 1) +
            np.roll(np.roll(grid,  0, 0),  1, 1) +
            np.roll(np.roll(grid,  0, 0), -1, 1) +
            np.roll(np.roll(grid, -1, 0),  1, 1) +
            np.roll(np.roll(grid, -1, 0),  0, 1) +
            np.roll(np.roll(grid, -1, 0), -1, 1)
        )
        c = grid
    else:
        pad_mode = "edge" if boundary == "reflect" else "constant"
        p = np.pad(grid, 1, mode=pad_mode, constant_values=0)
        # 3x3 neighborhoods via slicing; exclude center for neighbor sum
        s = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2]               + p[1:-1, 2:] +
            p[2:,  :-2] + p[2:,  1:-1] + p[2:,  2:]
        )
        c = p[1:-1, 1:-1]

    # life rule
    next_grid = ((c == 1) & ((s == 2) | (s == 3))) | ((c == 0) & (s == 3))
    return next_grid.astype(np.uint8)

def life_rollout_np(grid0, T, boundary="wrap"):
    rows = [grid0.astype(np.uint8)]
    cur = grid0
    for _ in range(1, T):
        cur = life_next_np(cur, boundary)
        rows.append(cur)
    return np.stack(rows, 0)  # (T,H,W)

# ------------------------------- NFTM components --------------------------------
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def ste(x): return BinarizeSTE.apply(x)

def read_neighborhoods2d(field, boundary="wrap"):
    """
    field: (B,1,H,W) float in [0,1]
    returns: (B,H,W,9) neighborhoods flattened row-major (top-left -> bottom-right)
    """
    pad_mode = {"wrap":"circular", "reflect":"replicate", "zeros":"constant"}[boundary]
    field_p = F.pad(field, (1,1,1,1), mode=pad_mode, value=0.0 if pad_mode=="constant" else 0.0)
    patches = F.unfold(field_p, kernel_size=3, stride=1)  # (B, 9, H*W)
    B, _, L = patches.shape
    # reshape to (B,H,W,9)
    # H*W = L, with original H,W known from field shape
    H, W = field.shape[-2], field.shape[-1]
    neigh = patches.transpose(1, 2).contiguous().view(B, H, W, 9)
    return neigh

def write_update(prev_field, write_bits, beta=1.0):
    return (1.0 - beta) * prev_field + beta * write_bits

class StatelessController2D(nn.Module):
    """C: neighborhood (9,) -> logit (real)."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, neigh):  # neigh: (B,H,W,9)
        B, H, W, _ = neigh.shape
        logits = self.net(neigh.view(B*H*W, 9)).view(B, H, W)
        return logits

# ---------------------------- Controller pretraining -----------------------------
def life_label_from_patch9(p9):
    """p9: iterable of 9 bits in row-major; index 4 is center."""
    c = int(p9[4])
    s = sum(p9) - c
    return 1 if (c == 1 and (s == 2 or s == 3)) or (c == 0 and s == 3) else 0

def train_controller_exact(controller, epochs=4000, lr=2e-3, verbose=True):
    patterns = list(product([0,1], repeat=9))  # 512 cases
    X = torch.tensor(patterns, dtype=torch.float32, device=device).view(1, len(patterns), 1, 9)
    y = torch.tensor([life_label_from_patch9(p) for p in patterns],
                     dtype=torch.float32, device=device).view(1, len(patterns), 1)

    opt = optim.Adam(controller.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        logits = controller(X)                               # (1,512,1) via our forward’s reshape
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep % 200 == 0 or ep == 1):
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == y).float().mean().item()
            print(f"[Controller] ep {ep:4d} | loss {loss.item():.6f} | acc {acc*100:.1f}%")
            if acc == 1.0: break

    if verbose:
        with torch.no_grad():
            logits = controller(X)
            pred = (torch.sigmoid(logits) > 0.5).long().view(-1).tolist()
        print("\n[Controller] sample mapping checks (first 8 of 512):")
        for p, pb in list(zip(patterns, pred))[:8]:
            print(f"  {p} -> {pb}")
    return True

# -------------------------------- NFTM executor ---------------------------------
class NFTMGameOfLife(nn.Module):
    """NFTM executor: f_{t+1} = A_write( g(C( A_read( P(f_t) ))) )."""
    def __init__(self, controller, boundary="wrap", beta=1.0):
        super().__init__()
        self.controller = controller
        self.boundary = boundary
        self.beta = beta

    def forward(self, f0, T):
        """
        f0: (B,1,H,W) floats in [0,1]
        returns list of T fields (B,1,H,W)
        """
        fields = [f0]
        f = f0
        for _ in range(1, T):
            f_read = ste(f)                                        # binarize state before read
            neigh = read_neighborhoods2d(f_read, self.boundary)    # (B,H,W,9)
            logits = self.controller(neigh)                        # (B,H,W)
            w_prob = torch.sigmoid(logits)                         # in [0,1]
            w_bits = ste(w_prob).unsqueeze(1)                      # (B,1,H,W)
            f = write_update(f, w_bits, beta=self.beta)            # overwrite if beta=1
            fields.append(f)
        return fields

# --------------------------------- Evaluation/plot -------------------------------
def make_init(H, W, mode="random", p=0.30):
    g = np.zeros((H, W), dtype=np.uint8)
    if mode == "random":
        rng = np.random.default_rng(0)
        g = (rng.random((H, W)) < p).astype(np.uint8)
    elif mode == "glider":
        # place a glider near center
        r, c = H//2 - 1, W//2 - 1
        pts = [(r, c+1), (r+1, c+2), (r+2, c), (r+2, c+1), (r+2, c+2)]
        for y, x in pts:
            if 0 <= y < H and 0 <= x < W: g[y, x] = 1
    elif mode == "blinker":
        r, c = H//2, W//2
        pts = [(r, c-1), (r, c), (r, c+1)]
        for y, x in pts: g[y, x] = 1
    return g

def time_grid(arr, rows=5, cols=10):
    # arr: (T, H, W), uses min(T, rows*cols) frames
    T, H, W = arr.shape
    K = min(T, rows*cols)
    mats = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r*cols + c
            row_imgs.append(arr[idx] if idx < K else np.zeros((H, W), dtype=arr.dtype))
        mats.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(mats, axis=0)

def eval_and_plot(model, H=64, W=64, T=100, boundary="wrap", init_mode="glider"):
    init_np = make_init(H, W, mode=init_mode)
    gt = life_rollout_np(init_np, T, boundary=boundary)             # (T,H,W)

    f0 = torch.tensor(init_np[None,None,:,:], dtype=torch.float32, device=device)
    with torch.no_grad():
        fields = model(f0, T=T)                                     # list of (1,1,H,W)
        arr = torch.stack(fields, 0).squeeze(1).squeeze(1).cpu().numpy()  # (T,H,W)
    pred_bits = (arr > 0.5).astype(np.uint8)
    acc = (pred_bits == gt).mean()
    print(f"[Eval] rollout accuracy vs Life: {acc*100:.2f}%  (T={T}, H={H}, W={W}, boundary={boundary})")

    # --- sanity checks ---
    print("[Debug] gt ones per frame (first 5):", [int(gt[t].sum()) for t in range(min(T,5))])
    print("[Debug] pred ones per frame (first 5):", [int(pred_bits[t].sum()) for t in range(min(T,5))])

    # Stack first Tshow frames vertically
    Tshow = min(T, 50)
    tile_gt   = np.concatenate([gt[t]        for t in range(Tshow)], axis=0)   # (Tshow*H, W)
    tile_pred = np.concatenate([pred_bits[t] for t in range(Tshow)], axis=0)   # (Tshow*H, W)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # Stacked timelines
    axs[0,0].imshow(tile_gt, cmap="binary", vmin=0, vmax=1, interpolation="nearest", origin="upper")
    axs[0,0].set_title("GT Life (stacked time slices)")
    axs[0,0].axis('off')

    axs[1,0].imshow(tile_pred, cmap="binary", vmin=0, vmax=1, interpolation="nearest", origin="upper")
    axs[1,0].set_title("NFTM (stacked time slices)")
    axs[1,0].axis('off')

    # Single-frame zoom (last frame)
    axs[0,1].imshow(gt[Tshow-1], cmap="binary", vmin=0, vmax=1, interpolation="nearest", origin="upper")
    axs[0,1].set_title(f"GT frame t={Tshow-1}")
    axs[0,1].set_xticks([]); axs[0,1].set_yticks([])

    axs[1,1].imshow(pred_bits[Tshow-1], cmap="binary", vmin=0, vmax=1, interpolation="nearest", origin="upper")
    axs[1,1].set_title(f"NFTM frame t={Tshow-1}")
    axs[1,1].set_xticks([]); axs[1,1].set_yticks([])

    plt.tight_layout()
    plt.show()

import imageio.v2 as imageio

def rollout_bits(model, H, W, T, boundary, init_mode):
    """Run model and return binary frames as a numpy array (T, H, W) uint8."""
    init_np = make_init(H, W, mode=init_mode)
    f0 = torch.tensor(init_np[None,None,:,:], dtype=torch.float32, device=device)
    with torch.no_grad():
        fields = model(f0, T=T)                                      # list of (1,1,H,W)
        arr = torch.stack(fields, 0).squeeze(1).squeeze(1).cpu().numpy()  # (T,H,W), floats
    bits = (arr > 0.5).astype(np.uint8)
    return bits

def save_gif(frames_bits, out_path, scale=4, fps=20, invert=False):
    """
    frames_bits: (T, H, W) uint8 in {0,1}
    scale: integer upscaling for visibility
    fps: frames per second in the gif (controls speed)
    invert: if True, live cells are black; otherwise white
    """
    T, H, W = frames_bits.shape
    # map {0,1} -> {0,255} (white cells by default)
    if invert:
        gray = (1 - frames_bits) * 255
    else:
        gray = frames_bits * 255
    # upscale for visibility
    if scale > 1:
        gray = np.kron(gray, np.ones((scale, scale), dtype=np.uint8))

    # write gif
    # duration per frame (seconds) = 1/fps
    imageio.mimsave(out_path, [gray[t] for t in range(T)], duration=1.0 / fps, loop=0)
    print(f"[GIF] wrote {out_path}  | frames={T}  size={(gray.shape[1], gray.shape[2])}  fps={fps}")

def make_gif_for_mode(model, mode, H=64, W=64, T=1000, boundary="wrap",
                      scale=4, fps=20, invert=False):
    bits = rollout_bits(model, H, W, T, boundary, mode)
    save_gif(bits, f"life_{mode}.gif", scale=scale, fps=fps, invert=invert)

# ------------------------------- Main (GIF maker) -------------------------------
if __name__ == "__main__":
    boundary = "wrap"     # 'wrap' | 'zeros' | 'reflect'
    beta = 1.0
    H, W, T = 64, 64, 1000

    print("[Start] Pretraining controller on 512 neighborhoods…")
    controller = StatelessController2D(hidden=128).to(device)
    train_controller_exact(controller, epochs=4000, lr=2e-3, verbose=True)

    # Freeze controller
    for p in controller.parameters():
        p.requires_grad = False

    print("\n[Run] Executing NFTM Game of Life with STE …")
    model = NFTMGameOfLife(controller, boundary=boundary, beta=beta).to(device)

    # Produce GIFs for each mode
    for mode in ["glider", "blinker", "random"]:
        print(f"\n[GIF] Generating {mode} …")
        make_gif_for_mode(model, mode, H=H, W=W, T=T, boundary=boundary,
                          scale=6 if mode != "random" else 4,  # make small patterns bigger
                          fps=20, invert=True)  # invert=True -> live cells black
