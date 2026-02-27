# GPU jobs (Slurm)

Submit a job:

```bash
sbatch gpu_jobs/celeb_inpaint_128_dense_linear_no_reg_baseline.slurm
```

Notes:
- Jobs assume the repo is located at `$WORKDIR/NFTM` on the cluster.
- Jobs assume a conda env named `nftm` is available (`conda activate nftm`).
- Logs are written to `logs/` and runs/artifacts to `runs/` (relative to the repo root).
