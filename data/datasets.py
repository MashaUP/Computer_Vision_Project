"""Datasets read from plain list files, so every model and every arm sees exactly the same samples.

List files (in ``data/splits/`` or under the data root) have one sample per line, paths relative to
the data root:

* unlabelled frames (LeJEPA pre-training, BatchNorm re-estimation): ``rgb_path``
* labelled pairs (fine-tuning, validation, test): ``rgb_path depth_path``

Depth maps are 16-bit PNGs; ``depth_scale`` converts them to metres (NYU from our extraction: 1000,
i.e. millimetres; KITTI: 256). A value of 0 means no measurement.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def read_list(path: str | Path) -> list[list[str]]:
    lines = [ln.split() for ln in Path(path).read_text().splitlines()]
    return [ln for ln in lines if ln and not ln[0].startswith("#")]


def _nyu_scene(tokens: list[str]) -> str | None:
    """Scene of a list line, or None. Raw frame / triplet: folder of the first path; labelled pair: 3rd column."""
    from data.download.nyu import scene_id

    first = Path(tokens[0])
    if first.name.startswith("r-") and first.suffix == ".ppm":
        return scene_id(first.parent.name)
    if len(tokens) >= 3 and "/" not in tokens[2]:
        return scene_id(tokens[2])
    return None


def check_list(path: str | Path, dataset: str, role: str) -> None:
    """Refuse a NYU list that contains test scenes (any role) or validation scenes (train, bn, pretrain).

    ``role``: train | val | bn | pretrain. Skips with a warning when the split files are missing or the lines
    do not name their scene. KITTI: not checked yet.
    """
    if dataset != "nyu":
        return
    from utils import paths

    splits = paths.data_root() / "splits"
    banned = {}
    for name, roles in (("nyu_test_scenes.txt", {"train", "val", "bn", "pretrain"}),
                        ("nyu_raw_val_scenes.txt", {"train", "bn", "pretrain"})):
        if role not in roles:
            continue
        f = splits / name
        if f.exists():
            banned.update({s: name for s in f.read_text().split()})
        else:
            print(f"[check_list] warning: {f} missing, {role} list {path} not checked against it")
    scenes = [_nyu_scene(t) for t in read_list(path)]
    if None in scenes:
        print(f"[check_list] warning: {scenes.count(None)} lines of {path} do not name their scene, not checked")
    bad = sorted({s for s in scenes if s in banned})
    if bad:
        raise ValueError(f"{path} ({role}) contains scenes from {sorted({banned[s] for s in bad})}: {bad[:5]}")


def load_rgb(path: Path) -> torch.Tensor:
    """(3, H, W) float in [0, 1]."""
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_depth(path: Path, depth_scale: float) -> torch.Tensor:
    """(1, H, W) depth in metres, 0 where invalid."""
    return torch.from_numpy(np.asarray(Image.open(path), dtype=np.float32) / depth_scale)[None]


def resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x[None], size=tuple(size), mode="bilinear", align_corners=False, antialias=True)[0]


class ImageList(Dataset):
    """Unlabelled RGB frames resized to ``size`` (H, W)."""

    def __init__(self, list_file, root, size):
        self.items = read_list(list_file)
        self.root, self.size = Path(root), tuple(size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return resize(load_rgb(self.root / self.items[i][0]), self.size)


class DepthPairs(Dataset):
    """RGB resized to the network input ``size``; depth kept at full resolution (metres).

    ``crop`` (H, W): random crop at full resolution before resizing (training on KITTI, whose
    images differ slightly in size); a centred bottom crop when ``train`` is False.
    ``train`` also enables horizontal flips. Returns ``(rgb, depth)``.
    """

    def __init__(self, list_file, root, size, depth_scale, crop=None, train=False):
        self.items = read_list(list_file)
        self.root, self.size, self.scale = Path(root), tuple(size), float(depth_scale)
        self.crop, self.train = (tuple(crop) if crop else None), train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rgb_path, depth_path = self.items[i][:2]
        rgb, depth = load_rgb(self.root / rgb_path), load_depth(self.root / depth_path, self.scale)
        if self.crop:
            ch, cw = self.crop
            h, w = depth.shape[-2:]
            top = random.randint(0, h - ch) if self.train else h - ch
            left = random.randint(0, w - cw) if self.train else (w - cw) // 2
            rgb, depth = rgb[:, top:top + ch, left:left + cw], depth[:, top:top + ch, left:left + cw]
        if self.train and random.random() < 0.5:
            rgb, depth = rgb.flip(-1), depth.flip(-1)
        return resize(rgb, self.size), depth
