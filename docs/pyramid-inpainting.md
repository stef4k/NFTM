## Pyramid multi-scale rollout for NFTM inpainting

We run the NFTM controller on a coarse to fine image pyramid. At each level we take a few NFTM steps at the current resolution, then upsample the prediction, clamp the known pixels, and continue at the next finer level. The two main benefits of this approach are faster convergence and more stable training, especially on higher resolutions.

Example usage:
```cmd
--pyramid 16,32,64 --pyr_steps 1,2,10
```

The above allocates 1 step at 16x16, 2 steps at 32x32, and 10 at 64x64 (total K_eval=13).

### Additions compared to standard NFTM inpainting
- New arguments: `--pyramid` and `--pyr_steps` and parsers
- Helpers: `downsample_like`, `upsample_like`, and `downsample_mask_minpool`
- Scale-aware schedules: `beta_S = min(beta * (finest/S), β_max)` and `corr_clip_S = corr_clip * (finest/S)`.
- Hand-off: after steps at size `S`, upsample to the next size and clamp known pixels with the GT (size matched) and mask.
- Per-step curves: we still compute PSNR/SSIM/LPIPS at native size after each (coarse or fine) step, so **curves have jumps at level transitions**.

### Training and evaluation

#### Training (curriculum + pyramid)
For each batch: 
1. Sample a random rollout length $\mathcal{K}_{curr} \in [1, K_{train}]$ (biased to short early rollouts).
2. Split $\mathcal{K}_{curr}$ across pyramid levels with `split_steps_train` (a couple more coarse steps only in very early epochs).
3. For each level `S` with `T` steps:
    - Downsample `(I, M, GT)` to `S` if needed.
    - Set `beta_S` and `corr_clip_S` using scale factor.
    - Do `T` NFTM steps (with optional descent guard if `--guard_in_train`).
4. If not at finest, upsample to next level and `clamp_known` to match GT on pixels.
5. Loss = MSE + TV-L1 (`tv_weight`) + contractive penalty (`contract_w`), then update.

`K_train` grows with epoch (min(`K_target`, `K_base+epoch`)), which is the standard rollout curriculum (the controller learns stable local corrections before being asked to do long rollouts).

#### Evaluation (fixed K_eval + pyramid)
For each batch:
1. Initialize corrupted input `I0 = M*img + (1−M)*noise`, then `I = clamp_known(I0, img, M)`.
2. Split total steps `K_eval` across pyramid levels with `split_steps_eval`.
3. For each level `S` with `T` steps:
    - Downsample `(I, M, GT)` to size `S` if needed.
    - Set `beta_S` and `corr_clip_S` scaled by `(finest/S)`.
    - Do `T` NFTM steps (optionally with --descent_guard).
    - After each step, upsample to native size and log PSNR/SSIM/LPIPS for curves.
    - If not at finest level, upsample to next size and `clamp_known` with GT.
4. After the final level, compute masked + unmasked PSNR/SSIM/LPIPS, and FID/KID on final preds.

### Important notes to consider
- The coarse results are upsampled for display. In coarser levels, the random missing-region noise is averaged out by the resampling.
- We take larger, more aggressive steps and clip less tightly at coarse scales (`beta_S`, `corr_clip_S`).
- At every level we operate with a downsampled mask (`downsample_mask_minpool`) so any unknown pixel at high res stays unknown. After finishing a level we upsample to the next `S` and immediately `clamp_known` with the size-matched GT and mask (removes any residual error on measured pixels and carries forward a clean coarse prediction for unknown ones).

> [!WARNING]
> Make sure `sum(--pyr_steps) == K_eval` and its length matches `--pyramid`.

> [!NOTE]  
> Very small levels (<8) are ignored by `parse_pyramid_arg`.

> [!TIP]
> If you see oscillations at coarse levels, reduce `beta_start` or `corr_clip`.