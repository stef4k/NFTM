from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

@dataclass(frozen=True)
class TileItem:
    noisy: str
    gt: str
    x: int
    y: int

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

class SIDDIndexTiles(Dataset):
    """
    Reads train_pairs.jsonl/test_pairs.jsonl
    Produces non-overlapping tiles by default (stride=patch).
    """
    def __init__(self, index_jsonl: str, patch=64, stride=64, limit_tiles=None):
        self.patch = int(patch)
        self.stride = int(stride)
        self.rows = read_jsonl(index_jsonl)

        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize([0.5]*3, [0.5]*3)

        self.items: List[TileItem] = []
        for r in self.rows:
            noisy_path = r["noisy"]
            gt_path = r["gt"]

            with Image.open(noisy_path) as im:
                w, h = im.size

            for y in range(0, h - self.patch + 1, self.stride):
                for x in range(0, w - self.patch + 1, self.stride):
                    self.items.append(TileItem(noisy_path, gt_path, x, y))
                    if limit_tiles is not None and len(self.items) >= limit_tiles:
                        break
                if limit_tiles is not None and len(self.items) >= limit_tiles:
                    break
            if limit_tiles is not None and len(self.items) >= limit_tiles:
                break

        if not self.items:
            raise RuntimeError("No tiles created. Check index file / paths.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        with Image.open(it.noisy) as imn:
            imn = imn.convert("RGB")
            noisy = imn.crop((it.x, it.y, it.x+self.patch, it.y+self.patch))
        with Image.open(it.gt) as img:
            img = img.convert("RGB")
            gt = img.crop((it.x, it.y, it.x+self.patch, it.y+self.patch))

        noisy = self.norm(self.to_tensor(noisy))
        gt = self.norm(self.to_tensor(gt))
        return noisy, gt
