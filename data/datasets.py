"""Datasets read from plain list files, so every model and every arm sees exactly the same samples.

List files (in ``data/splits/`` or under the data root) have one sample per line, paths relative to
the data root:

* unlabelled frames (LeJEPA pre-training, BatchNorm re-estimation): ``rgb_path``
* labelled pairs (fine-tuning, validation, test): ``rgb_path depth_path``
* frame triplets (photometric and ZipDepth fine-tuning): ``prev_path target_path next_path``

Depth maps are 16-bit PNGs; ``depth_scale`` converts them to metres (NYU from our extraction: 1000,
i.e. millimetres; KITTI: 256). A value of 0 means no measurement.

Pseudo-depth targets (``loss: zipdepth``) are written by ``scripts/make_pseudo_depth.py`` under
``<data root>/pseudo_depth/<teacher>/`` with the relative path of their frame (``.png``): 16-bit
relative inverse depth, divided by its per-image maximum. ``ZipDepthLoss`` normalises every target
by its median and mean absolute deviation, so this per-image scale does not change the loss.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter


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


_KITTI_DRIVE_RE = re.compile(r"\d{4}_\d{2}_\d{2}_drive_\d{4}_sync")


def _kitti_drive(tokens: list[str]) -> str | None:
    for tok in tokens:
        m = _KITTI_DRIVE_RE.search(tok)
        if m:
            return m.group(0)
    return None


def check_list(path: str | Path, dataset: str, role: str) -> None:
    """Refuse a NYU or KITTI list that contains test scenes/drives (any role) or validation ones (train, bn, pretrain).

    ``role``: train | val | bn | pretrain. Skips with a warning when the split files are missing or the lines
    do not name their scene/drive.
    """
    from utils import paths

    splits = paths.data_root() / "splits"
    if dataset == "nyu":
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
        return

    if dataset == "kitti":
        from data.download.download_kitti import EIGEN_TEST_DRIVES

        banned = {d: "EIGEN_TEST_DRIVES" for d in EIGEN_TEST_DRIVES}
        for name, roles in (("kitti_test_drives.txt", {"train", "val", "bn", "pretrain"}),
                            ("kitti_val_drives.txt", {"train", "bn", "pretrain"})):
            if role not in roles:
                continue
            f = splits / name
            if f.exists():
                banned.update({s: name for s in f.read_text().split()})
            elif name == "kitti_val_drives.txt":
                print(f"[check_list] warning: {f} missing, {role} list {path} not checked against it")
        drives = [_kitti_drive(t) for t in read_list(path)]
        if None in drives:
            print(f"[check_list] warning: {drives.count(None)} lines of {path} do not name their drive, not checked")
        bad = sorted({d for d in drives if d in banned})
        if bad:
            raise ValueError(f"{path} ({role}) contains drives from {sorted({banned[d] for d in bad})}: {bad[:5]}")
        return


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


NORM_INTRINSICS = {
    # NYU intrinsics after undistortion and 16 px border crop (448x608), matching scripts/train_flexdepth_nyu.py
    "nyu": (
        518.85790117450188 / 608.0,
        519.46961112127485 / 448.0,
        (325.58244941119034 - 16.0) / 608.0,
        (253.73616633400465 - 16.0) / 448.0,
    ),
    # Monodepth2 standard normalized KITTI intrinsics
    "kitti": (0.58, 1.92, 0.5, 0.5),
}


def frame_path(root: Path, rel: str, dataset: str) -> Path:
    """File to read for a list entry: on NYU, the rectified PNG of a raw frame when it exists
    (``nyu/rect/...``, written by ``scripts/train_flexdepth_nyu.py --make-png``), else the file itself."""
    root = Path(root)
    if dataset == "nyu":
        p = Path(rel)
        if len(p.parts) >= 2 and p.parts[0] == "nyu" and p.parts[1] == "raw":
            png = (root / "nyu" / "rect" / p.relative_to("nyu/raw")).with_suffix(".png")
            if png.exists():
                return png
    return root / rel


def target_frame(tokens: list[str]) -> str:
    """The frame a list line supervises: the middle one of a triplet, else the first column."""
    return tokens[1] if len(tokens) >= 3 else tokens[0]


def pseudo_depth_path(root: Path, teacher: str, rel: str) -> Path:
    """Where ``scripts/make_pseudo_depth.py`` writes the pseudo-depth of frame ``rel``."""
    return (Path(root) / "pseudo_depth" / teacher / rel).with_suffix(".png")


def load_pseudo_depth(path: Path) -> torch.Tensor:
    """(1, H, W) relative inverse depth in [0, 1] (per-image scale, see the module docstring)."""
    return torch.from_numpy(np.asarray(Image.open(path), dtype=np.float32) / 65535.0)[None]


def _color_jitter(jitter: ColorJitter, frames: torch.Tensor) -> torch.Tensor:
    """The same random colour jitter on a stack of frames (N, 3, H, W)."""
    from torchvision.transforms import functional as TF

    fn_idx, b, c, s, hue = ColorJitter.get_params(jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)
    for fn_id in fn_idx:
        if fn_id == 0 and b is not None:
            frames = TF.adjust_brightness(frames, b)
        elif fn_id == 1 and c is not None:
            frames = TF.adjust_contrast(frames, c)
        elif fn_id == 2 and s is not None:
            frames = TF.adjust_saturation(frames, s)
        elif fn_id == 3 and hue is not None:
            frames = TF.adjust_hue(frames, hue)
    return frames


class PseudoDepthList(Dataset):
    """Frames with their teacher pseudo-depth, for ZipDepth's own loss (``loss: zipdepth``).

    Reads the same list files as ``TripletList`` (the target frame of each triplet) or frame lists.
    RGB and pseudo-depth are resized to ``size``; ``train`` enables the same augmentation as
    ``TripletList``: horizontal flip (p = 0.5) and colour jitter on the input only (p = 0.5).
    Returns ``(rgb, pseudo_depth)``.
    """

    def __init__(self, list_file, root, size, dataset: str, teacher: str = "dav2_vitl", train: bool = True):
        self.items = read_list(list_file)
        self.root, self.size, self.dataset, self.teacher, self.train = Path(root), tuple(size), dataset, teacher, train
        self._jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        missing = [t for t in (target_frame(it) for it in self.items)
                   if not pseudo_depth_path(self.root, teacher, t).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(self.items)} frames of {list_file} have no {teacher} pseudo-depth "
                f"(first: {pseudo_depth_path(self.root, teacher, missing[0])}); run scripts/make_pseudo_depth.py")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel = target_frame(self.items[i])
        rgb = resize(load_rgb(frame_path(self.root, rel, self.dataset)), self.size)
        target = resize(load_pseudo_depth(pseudo_depth_path(self.root, self.teacher, rel)), self.size)
        if self.train and random.random() < 0.5:
            rgb, target = rgb.flip(-1), target.flip(-1)
        if self.train and random.random() < 0.5:
            rgb = _color_jitter(self._jitter, rgb[None])[0]
        return rgb, target


class TripletList(Dataset):
    """(prev, target, next) RGB triplets for self-supervised photometric fine-tuning."""

    def __init__(self, list_file, root, size, dataset: str, train: bool = True):
        if dataset not in NORM_INTRINSICS:
            raise ValueError(f"unknown dataset {dataset!r}")
        self.items = read_list(list_file)
        self.root, self.size, self.dataset, self.train = Path(root), tuple(size), dataset, train
        self._jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __len__(self):
        return len(self.items)

    def _load_frame(self, rel: str) -> torch.Tensor:
        return resize(load_rgb(frame_path(self.root, rel, self.dataset)), self.size)

    def __getitem__(self, i):
        prev_rel, target_rel, next_rel = self.items[i][:3]
        prev = self._load_frame(prev_rel)
        target = self._load_frame(target_rel)
        nxt = self._load_frame(next_rel)

        h, w = self.size
        fx, fy, cx, cy = NORM_INTRINSICS[self.dataset]
        K = torch.tensor(
            [[fx * w, 0.0, cx * w, 0.0],
             [0.0, fy * h, cy * h, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        if self.train and random.random() < 0.5:
            prev, target, nxt = prev.flip(-1), target.flip(-1), nxt.flip(-1)
            K[0, 2] = w - K[0, 2]

        if self.train and random.random() < 0.5:
            stacked = _color_jitter(self._jitter, torch.stack([prev, target, nxt], dim=0))
            prev_aug, target_aug, next_aug = stacked[0], stacked[1], stacked[2]
        else:
            prev_aug, target_aug, next_aug = prev, target, nxt

        inv_K = torch.linalg.inv(K)
        return {
            "prev_aug": prev_aug,
            "target_aug": target_aug,
            "next_aug": next_aug,
            "prev": prev,
            "target": target,
            "next": nxt,
            "K": K,
            "inv_K": inv_K,
        }
