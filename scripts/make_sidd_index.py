#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from pathlib import Path


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

    grouped = {}
    for scene in scenes:
        key = _sidd_scene_group(scene.name)
        grouped.setdefault(key, []).append(scene)

    groups = list(grouped.items())
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n_train = int(len(groups) * args.train_frac)
    train_groups = groups[:n_train]
    test_groups = groups[n_train:]

    train_scenes = [s for _, lst in train_groups for s in lst]
    test_scenes  = [s for _, lst in test_groups for s in lst]

    def build_split(scenes_list):
        rows = []
        for scene in scenes_list:
            scene_group = _sidd_scene_group(scene.name)
            for noisy, gt in collect_pairs(scene):
                rows.append({
                    "scene": scene.name,
                    "scene_group": scene_group,
                    "noisy": noisy,
                    "gt": gt,
                })
        return rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_split(train_scenes)
    test_rows  = build_split(test_scenes)

    (out_dir / "meta.json").write_text(json.dumps({
        "sidd_root": str(sidd_root),
        "num_scenes": len(scenes),
        "num_scene_groups": len(groups),
        "train_scenes": len(train_scenes),
        "test_scenes": len(test_scenes),
        "train_scene_groups": len(train_groups),
        "test_scene_groups": len(test_groups),
        "train_pairs": len(train_rows),
        "test_pairs": len(test_rows),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "scene_grouping": "scene_number",
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
