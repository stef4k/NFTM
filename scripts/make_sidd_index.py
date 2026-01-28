#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from pathlib import Path

def collect_pairs(scene_dir: Path):
    noisy = sorted(scene_dir.glob("*_NOISY_SRGB_*.PNG"))
    pairs = []
    for n in noisy:
        gt = Path(str(n).replace("_NOISY_SRGB_", "_GT_SRGB_"))
        if gt.exists():
            pairs.append((str(n), str(gt)))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidd_root", type=str, required=True,
                    help="Path to SIDD_Medium_Srgb (contains Data/)")
    ap.add_argument("--out_dir", type=str, default="benchmarks/SIDD/index")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sidd_root = Path(args.sidd_root)
    data_dir = sidd_root / "Data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected {data_dir}")

    scenes = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not scenes:
        raise RuntimeError("No scene folders found.")

    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    n_train = int(len(scenes) * args.train_frac)
    train_scenes = scenes[:n_train]
    test_scenes  = scenes[n_train:]

    def build_split(scenes_list):
        rows = []
        for scene in scenes_list:
            for noisy, gt in collect_pairs(scene):
                rows.append({"scene": scene.name, "noisy": noisy, "gt": gt})
        return rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_split(train_scenes)
    test_rows  = build_split(test_scenes)

    (out_dir / "meta.json").write_text(json.dumps({
        "sidd_root": str(sidd_root),
        "num_scenes": len(scenes),
        "train_scenes": len(train_scenes),
        "test_scenes": len(test_scenes),
        "train_pairs": len(train_rows),
        "test_pairs": len(test_rows),
        "seed": args.seed,
        "train_frac": args.train_frac,
    }, indent=2))

    (out_dir / "train_pairs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n"
    )
    (out_dir / "test_pairs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in test_rows) + "\n"
    )

    print("Wrote:")
    print(" ", out_dir / "meta.json")
    print(" ", out_dir / "train_pairs.jsonl")
    print(" ", out_dir / "test_pairs.jsonl")

if __name__ == "__main__":
    main()
