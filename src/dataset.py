"""
dataset.py
==========
Datasets and DataLoaders for the METER + LeJEPA project.

Structure:
  1. Augmentation  — exact copy of the professor's augmentation.py (numpy, pre-resize)
  2. SSL transform — multi-crop for LeJEPA (used in train.py)
  3. IO helpers    — RGB / depth loading
  4. NYUv2 Dataset — official .mat, H5 files (Kaggle), image/depth pairs
  5. KITTI Dataset — Eigen split with sparse LiDAR GT
  6. DataLoader builder

Paper references:
  - Resize resolutions: Section IV-B (NYU 192×256, KITTI 192×636)
  - Augmentation policy: Section III-C (flip, mirror, crop, channel swap, shifting strategy)
  - Augmentation prob = 0.5: Section IV-A
  - Depth shift range ±10 cm (NYU) / ±10 dm (KITTI): Section III-C Eq. 8
"""

import os
import glob
import random
from itertools import product
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.config import (
    DATASET_NYU_PATH,
    DATASET_KITTI_PATH,
    DATASET_KITTI_PATH_GT,
    DEVICE,
    NYU_SIZE,
    KITTI_SIZE,
    AUG_PROB,
    DEPTH_SHIFT_RANGE,
)


# ==========================================
# 1. Professor's Augmentation
#    Exact copy of Meter_git/augmentation.py
#    Works on numpy (H,W,3) float 0-255 BEFORE resize
# ==========================================

def _pixel_shift(depth_img, shift):
    """Depth shift — professor's pixel_shift(), but restricted to valid depth values."""
    # Only shift valid pixels (depth > 0) to avoid turning invalid sparse points into valid ones.
    mask = depth_img > 0.0
    shifted = depth_img.copy()
    shifted[mask] += shift
    return shifted


def _random_crop_np(x, y, crop_size=(192, 256)):
    """
    Random crop.
    x: (H, W, 3) numpy
    y: (H, W, 1) numpy
    """
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    h, w, _ = x.shape
    crop_h, crop_w = crop_size
    rangew  = (w - crop_w) // 2 if w > crop_w else 0
    rangeh  = (h - crop_h) // 2 if h > crop_h else 0
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped_x = x[offseth:offseth + crop_h, offsetw:offsetw + crop_w, :]
    cropped_y = y[offseth:offseth + crop_h, offsetw:offsetw + crop_w, :]
    # Check that depth is not all zero after crop
    cropped_y_check = cropped_y[:, :, ~np.all(cropped_y == 0, axis=(0, 1))]
    if cropped_y_check.shape[-1] == 0:
        return x, y
    return cropped_x, cropped_y


class MeterAugmentation:
    """
    Exact copy of the professor's augmentation2D() (Meter_git/augmentation.py).

    Paper Section III-C:
      - Vertical flip (no-op in original code, kept for fidelity)
      - Horizontal mirroring
      - Channel swap (fixed to use permutations to keep 3 channels)
      - Random crop (fixed crop size dynamically)
      - Shifting strategy: C-shift (gamma + brightness + color) + D-shift (depth range)

    All applied with probability AUG_PROB = 0.5 (Section IV-A).

    Input/Output: numpy (H, W, 3) float64 for RGB, (H, W, 1) float32 for depth.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = str(dataset_name).lower().strip()
        # 0.5 probability for all transformations — Section IV-A
        self.flip_prob      = AUG_PROB
        self.mirror_prob    = AUG_PROB
        self.c_swap_prob    = AUG_PROB
        self.crop_prob      = AUG_PROB
        self.shifting_prob  = AUG_PROB

        # Set dataset-specific crop size
        if "nyu" in self.dataset_name:
            self.crop_size = (192, 256)
        else:
            self.crop_size = (192, 636)

    def __call__(self, img_np: np.ndarray, depth_np: np.ndarray):
        """
        Parameters
        ----------
        img_np   : (H, W, 3)  float64, range 0-255
        depth_np : (H, W, 1)  float32, centimeters (NYU) or decimeters (KITTI)

        Returns
        -------
        img_np, depth_np  augmented
        """
        # --- Vertical flip (::1 is no-op, kept for fidelity to the professor) ---
        if random.uniform(0, 1) <= self.flip_prob:
            img_np   = (img_np[...,   ::1, :]).copy()
            depth_np = (depth_np[..., ::1, :]).copy()

        # --- Horizontal mirroring (img[..., ::-1, :]) ---
        if random.uniform(0, 1) <= self.mirror_prob:
            img_np   = (img_np[...,   ::-1, :]).copy()
            depth_np = (depth_np[..., ::-1, :]).copy()

        # --- Channel swap ---
        if random.uniform(0, 1) <= self.c_swap_prob:
            import itertools
            indices    = list(itertools.permutations([0, 1, 2]))
            policy_idx = random.randint(0, len(indices) - 1)
            img_np     = img_np[..., list(indices[policy_idx])]

        # --- Random crop ---
        if random.random() <= self.crop_prob:
            img_np, depth_np = _random_crop_np(img_np, depth_np, crop_size=self.crop_size)

        # --- Shifting strategy (Section III-C, Eq. 6-8) ---
        if random.uniform(0, 1) <= self.shifting_prob:
            # C-shift: gamma + brightness (Eq. 6)
            gamma      = random.uniform(0.9, 1.1)
            # Correcting gamma bug: scale to [0, 1] before exponentiation, then scale back to [0, 255]
            img_np     = ((img_np / 255.0) ** gamma) * 255.0
            brightness = random.uniform(0.9, 1.1)
            img_np     = img_np * brightness
            # C-shift: color augmentation (Eq. 7)
            colors      = np.random.uniform(0.9, 1.1, size=3)
            white       = np.ones((img_np.shape[0], img_np.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
            img_np     *= color_image
            img_np      = np.clip(img_np, 0, 255)
            # D-shift (Eq. 8): ±10 cm (NYU) / ±10 dm (KITTI)
            random_shift = random.randint(-DEPTH_SHIFT_RANGE, DEPTH_SHIFT_RANGE)
            depth_np     = _pixel_shift(depth_np, random_shift)

        return img_np, depth_np


# ==========================================
# 2. SSL Transform for LeJEPA (used in train.py)
#    Multi-crop: 2 global views (256px) + 6 local views (128px)
# ==========================================

def get_ssl_transform(global_view: bool = False, debug: bool = False, dataset: str = "kitti"):
    """
    SSL multi-crop transforms (LeJEPA Sec. 6.1).

    Crop scales adapted per dataset:
    - NYU (indoor): wider scales because indoor scenes have larger spatially
      coherent regions (walls, furniture). Crops too tight (5-30%) capture
      textureless patches that hurt SSL learning of geometric features.
    - KITTI (outdoor): paper defaults work because the scene is rich in
      details at all scales (vehicles, vegetation, buildings).
    """
    is_nyu = "nyu" in dataset.lower()

    if debug:
        resize = v2.Resize((256, 256), antialias=True)
        blur   = v2.GaussianBlur(kernel_size=9)
    elif global_view:
        scale = (0.5, 1.0) if is_nyu else (0.3, 1.0)
        resize = v2.RandomResizedCrop(256, scale=scale, antialias=True)
        blur   = v2.GaussianBlur(kernel_size=23)
    else:
        scale = (0.15, 0.5) if is_nyu else (0.05, 0.3)
        resize = v2.RandomResizedCrop(128, scale=scale, antialias=True)
        blur   = v2.GaussianBlur(kernel_size=9)
    
    ops = [
        resize,
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([blur], p=0.5),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return v2.Compose(ops)


# ==========================================
# 3. Transformation constants
# ==========================================

_normalize         = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_nyu_resize_rgb    = v2.Resize(NYU_SIZE,   interpolation=InterpolationMode.BILINEAR, antialias=True)
_nyu_resize_depth  = v2.Resize(NYU_SIZE,   interpolation=InterpolationMode.NEAREST,  antialias=False)
_kitti_resize_rgb  = v2.Resize(KITTI_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True)
_kitti_resize_depth = v2.Resize(KITTI_SIZE, interpolation=InterpolationMode.NEAREST, antialias=False)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → float32 tensor [0, 1]."""
    t = v2.functional.to_image(img)
    return v2.functional.to_dtype(t, torch.float32, scale=True)


# ==========================================
# 4. IO Helpers
# ==========================================

def _load_rgb(path) -> Image.Image:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB not found: {path}")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _load_depth_png(path) -> Image.Image:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Depth not found: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(d)


def _load_rgb_np(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_depth_png_np(path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Depth not found: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return d


# ==========================================
# 5. NYU Depth v2 Dataset
# ==========================================

class NYUDataset(Dataset):
    """
    NYU Depth v2 — supports three formats:
      A) nyu_depth_v2_labeled.mat  (official, 1449 samples)
      B) .h5 / .hdf5 files per frame (Kaggle split, ~50K samples)
      C) (rgb, depth_png) pairs from filesystem

    Modes:
      - "pretrain"  → returns only normalized RGB tensor
      - "finetune"  → returns (rgb_tensor, depth_tensor_cm)
                      with MeterAugmentation applied before resize
    """

    def __init__(self, path, mode="finetune", split="all", kaggle=False, augment=None):
        """
        Parameters
        ----------
        path    : NYU root folder
        mode    : "pretrain" | "finetune"
        split   : "train" | "test" | "all"
                  used only with splits.mat or Kaggle structure
        kaggle  : True if data is in Kaggle format (H5 per frame)
        """
        self.mode    = mode
        self.split   = split
        self.kaggle  = kaggle
        self.samples = []
        self._format = None   # "mat" | "h5" | "pairs" | "rgb_only"

        path = Path(path)
        print(f"\n[INFO] NYUDataset — mode={mode}, split={split}, kaggle={kaggle}")
        self._init(path, split, kaggle)
        print(f"[INFO] NYUDataset — {len(self.samples)} samples  (format={self._format})")

        if augment is None:
            augment = False if split in ("test", "val") else True

        self.aug = MeterAugmentation("nyu") if (mode == "finetune" and augment) else None
    # ------------------------------------------------------------------
    def _init(self, path: Path, split: str, kaggle: bool):
        mat_path    = path / "nyu_depth_v2_labeled.mat"
        splits_path = path / "splits.mat"

        # ---- Kaggle H5 format ----
        if kaggle:
            dirs = {
                "train": [path / "train"],
                "test":  [path / "val"],
                "val":   [path / "val"],
                "all":   [path, path / "train", path / "val"],
            }.get(split, [path])
            h5_files = []
            for d in dirs:
                if d and d.exists():
                    h5_files.extend(sorted(d.rglob("*.h5")))
                    h5_files.extend(sorted(d.rglob("*.hdf5")))
            if not h5_files:
                h5_files = sorted(path.rglob("*.h5")) + sorted(path.rglob("*.hdf5"))
            if h5_files:
                self._format = "h5"
                self.samples = [str(p) for p in list(set(h5_files))]
                return

        # ---- pretrain: RGB images only ----
        if self.mode == "pretrain":
            # Respect the split folder if it exists (e.g. path / "train")
            search_dir = path / split if (path / split).exists() else path
            exts = ("*.png", "*.jpg", "*.jpeg", "*.ppm", "*.pgm")
            rgb_files = []
            for e in exts:
                for p in search_dir.rglob(e):
                    # Filter out depth maps which are located in 'depth_map' folders
                    rel_p = p.relative_to(path)
                    if "depth_map" in rel_p.parts or "depths" in rel_p.parts:
                        continue
                    rgb_files.append(str(p))
            self._format = "rgb_only"
            self.samples = sorted(list(set(rgb_files)))
            return

        # ---- test/val → official .mat ALWAYS (654 Silberman samples) ----
        if split in ("test", "val") and mat_path.exists():
            self._format  = "mat"
            self.mat_path = mat_path
            with h5py.File(mat_path, "r") as f:
                length = f["images"].shape[0]
            if splits_path.exists():
                sp  = sio.loadmat(splits_path)
                idx = np.array(sp["testNdxs"]).squeeze().astype(int) - 1
                self.samples = idx[(idx >= 0) & (idx < length)].tolist()
            else:
                self.samples = list(range(length))
            return

        # ---- train/all → jpg+png pairs (50K from download_nyu.py) ----
        # Priority on pairs BEFORE .mat for training
        pairs = []
        search_dir = path / split if (path / split).exists() else path
        for img_path in sorted(search_dir.rglob("*.jpg")):
            dm = img_path.parent / "depth_map" / (img_path.stem + ".png")
            if dm.exists():
                pairs.append((str(img_path), str(dm)))
        if pairs:
            self._format = "pairs"
            self.samples = pairs
            return

        # ---- fallback: official .mat for train (1449 samples) ----
        if mat_path.exists():
            self._format  = "mat"
            self.mat_path = mat_path
            with h5py.File(mat_path, "r") as f:
                length = f["images"].shape[0]
            if split == "all" or not splits_path.exists():
                self.samples = list(range(length))
            else:
                sp  = sio.loadmat(splits_path)
                key = "trainNdxs" if split == "train" else "testNdxs"
                idx = np.array(sp[key]).squeeze().astype(int) - 1
                self.samples = idx[(idx >= 0) & (idx < length)].tolist()
            return

        raise FileNotFoundError(
            f"NYU: no data found in {path}. "
            "Requires nyu_depth_v2_labeled.mat, .h5 files, or rgb/depth pairs."
        )

    # ------------------------------------------------------------------
    def _load_h5(self, h5_path: str):
        """Load a single H5 file (Kaggle format per frame)."""
        with h5py.File(h5_path, "r") as f:
            rgb_arr = np.array(f.get("rgb", f.get("image", f["images"]))).squeeze()
            if rgb_arr.shape[0] in (1, 3):
                rgb_arr = np.transpose(rgb_arr, (1, 2, 0))
            rgb_np = rgb_arr.astype(np.float64)   # (H, W, 3) 0-255

            depth_key = "depth" if "depth" in f else "depths"
            depth_arr = np.array(f[depth_key]).squeeze().astype(np.float32)
            depth_np  = depth_arr[:, :, np.newaxis] * 100.0   # m → cm
        return rgb_np, depth_np

    # ------------------------------------------------------------------
    def _to_tensors(self, rgb_np: np.ndarray, depth_np: np.ndarray):
        """
        Convert numpy → PyTorch tensors after augmentation.
        rgb_np   : (H, W, 3) float64 0-255
        depth_np : (H, W, 1) float32 cm
        Returns  : rgb_t (3, H, W) normalized, depth_t (1, H, W) cm >= 0
        """
        # Clamp depth: negative values from D-shift have no physical sense
        depth_np = np.clip(depth_np, 0.0, None)

        # Resize RGB with PIL
        rgb_pil = Image.fromarray(np.clip(rgb_np, 0, 255).astype(np.uint8))
        rgb_t   = _pil_to_tensor(_nyu_resize_rgb(rgb_pil))   # (3, H, W) [0,1]

        # Resize depth with nearest interpolation
        depth_t = torch.from_numpy(depth_np.squeeze()).unsqueeze(0).unsqueeze(0).float()
        depth_t = F.interpolate(depth_t, size=NYU_SIZE, mode="nearest").squeeze(0)  # (1, H, W)

        return _normalize(rgb_t), depth_t

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            return self._get(idx)
        except Exception as e:
            if isinstance(e, (TypeError, ValueError, AttributeError, FileNotFoundError)):
                raise
            # Skip corrupted sample — retry with a random one
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _get(self, idx):
        sample = self.samples[idx]

        # ---- pretrain: RGB only ----
        if self.mode == "pretrain":
            path   = sample[0] if isinstance(sample, (tuple, list)) else sample
            rgb_pil = _nyu_resize_rgb(_load_rgb(path))
            # Correcting double normalization bug for LeJEPA: return unnormalized [0, 1] tensor.
            # Normalization will be applied during SSL multi-crop views generation.
            return _pil_to_tensor(rgb_pil)

        # ---- H5 ----
        if self._format == "h5":
            rgb_np, depth_np = self._load_h5(sample)
            if self.aug:
                rgb_np, depth_np = self.aug(rgb_np, depth_np)
            return self._to_tensors(rgb_np, depth_np)

        # ---- filesystem pairs ----
        # download_nyu.py saves depth in MILLIMETERS (uint16 PNG).
        # We convert mm → cm by dividing by 10, so depth_np is in cm
        # consistently with .mat and H5 formats.
        if self._format == "pairs":
            rgb_path, depth_path = sample
            rgb_np   = _load_rgb_np(rgb_path).astype(np.float64)
            depth_np = _load_depth_png_np(depth_path).astype(np.float32)
            if depth_np.ndim == 2:
                depth_np = depth_np[:, :, np.newaxis]
            depth_np = depth_np / 10.0   # mm → cm
            if self.aug:
                rgb_np, depth_np = self.aug(rgb_np, depth_np)
            return self._to_tensors(rgb_np, depth_np)

        # ---- official .mat ----
        with h5py.File(self.mat_path, "r") as f:
            rgb_raw   = np.array(f["images"][idx]).transpose(2, 1, 0)   # (H, W, 3) uint8
            depth_raw = np.array(f["depths"][idx]).transpose(1, 0)      # (H, W) float m

        rgb_np   = rgb_raw.astype(np.float64)
        depth_np = depth_raw[:, :, np.newaxis] * 100.0   # m → cm (1 cm precision)

        if self.aug:
            rgb_np, depth_np = self.aug(rgb_np, depth_np)

        return self._to_tensors(rgb_np, depth_np)


# ==========================================
# 6. KITTI Dataset
# ==========================================

class KITTIDataset(Dataset):
    """
    KITTI — Eigen split with sparse LiDAR GT.

    Expected structure:
      kitti_root/
        <date>/
          <drive_sync>/
            image_02/data/*.png    (RGB)

      gt_root/
        data_depth_annotated/
          train|val/
            <drive_sync>/
              proj_depth/groundtruth/
                image_02/*.png     (GT depth, uint16, divide /256 → m)

    Modes:
      - "pretrain"  → RGB tensor only
      - "finetune"  → (rgb_tensor, depth_tensor_m) with MeterAugmentation
    """

    def __init__(self, path, gt_root=None, mode="finetune", cam="image_02", split="all", augment=None):
        """
        Parameters
        ----------
        path    : KITTI RGB root folder
        gt_root : KITTI GT depth root folder (required in finetune)
        mode    : "pretrain" | "finetune"
        cam     : camera to use (default "image_02")
        split   : "train" | "val" | "all"
        """
        self.mode    = mode
        self.cam     = cam
        self.samples = []

        path = Path(path)
        print(f"\n[INFO] KITTIDataset — mode={mode}, split={split}")

        if mode == "pretrain" or gt_root is None:
            # RGB images only (all .png available)
            self.samples = [str(p) for p in sorted(path.rglob("*.png"))]
        else:
            self._init_finetune(path, Path(gt_root), cam, split)

        print(f"[INFO] KITTIDataset — {len(self.samples)} samples")

        if augment is None:
            augment = False if split in ("val", "test") else True

        # Augmentation for KITTI: same policy as the professor, shift in dm
        self.aug = MeterAugmentation("kitti") if (mode == "finetune" and augment) else None

    # ------------------------------------------------------------------
    def _init_finetune(self, kitti_root: Path, gt_root: Path, cam: str, split: str):
        folders = ("train", "val") if split == "all" else (split,)
        for folder in folders:
            pattern  = os.path.join(
                gt_root, folder, "*_sync",
                "proj_depth", "groundtruth", cam, "*.png"
            )
            gt_files = sorted(glob.glob(pattern))
            for gt_path in gt_files:
                parts      = Path(gt_path).parts
                drive_dir  = parts[-5]
                frame_file = parts[-1]
                date_dir   = "_".join(drive_dir.split("_")[:3])
                rgb_path   = os.path.join(
                    kitti_root, date_dir, drive_dir, cam, "data", frame_file
                )
                if os.path.exists(rgb_path):
                    self.samples.append((rgb_path, gt_path))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            return self._get(idx)
        except Exception as e:
            if isinstance(e, (TypeError, ValueError, AttributeError, FileNotFoundError)):
                raise
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _get(self, idx):
        sample = self.samples[idx]

        # ---- pretrain: RGB only ----
        if self.mode == "pretrain":
            path    = sample[0] if isinstance(sample, (tuple, list)) else sample
            rgb_pil = _kitti_resize_rgb(_load_rgb(path))
            # Correcting double normalization bug for LeJEPA: return unnormalized [0, 1] tensor.
            # Normalization will be applied during SSL multi-crop views generation.
            return _pil_to_tensor(rgb_pil)

        # ---- finetune ----
        rgb_path, gt_path = sample

        # Load RGB as numpy for augmentation
        rgb_np = _load_rgb_np(rgb_path).astype(np.float64)   # (H, W, 3) 0-255

        # Load depth GT: uint16 PNG / 256 = meters (KITTI convention)
        depth_raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"KITTI GT not found: {gt_path}")
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        depth_np = (depth_raw.astype(np.float32) / 256.0)[:, :, np.newaxis]   # (H, W, 1) meters

        # MeterAugmentation on numpy BEFORE resize (like the professor)
        # Convert m → dm so the professor's ±10 pixel_shift acts as ±10 dm = ±1 m (paper Section III-C)
        if self.aug:
            depth_np = depth_np * 10.0                # m → dm
            rgb_np, depth_np = self.aug(rgb_np, depth_np)
            depth_np = depth_np / 10.0                # dm → m
            depth_np = np.clip(depth_np, 0.0, None)   # clamp negative depths from D-shift
        # Resize RGB
        rgb_pil = Image.fromarray(np.clip(rgb_np, 0, 255).astype(np.uint8))
        rgb_t   = _pil_to_tensor(_kitti_resize_rgb(rgb_pil))        # (3, H, W) [0,1]

        # Resize depth nearest
        depth_t = torch.from_numpy(depth_np.squeeze()).unsqueeze(0).unsqueeze(0).float()
        depth_t = F.interpolate(depth_t, size=KITTI_SIZE, mode="nearest").squeeze(0)  # (1, H, W) m

        return _normalize(rgb_t), depth_t


# ==========================================
# 7. DataLoader builder
# ==========================================

def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool   = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Builds a DataLoader with optimal settings for device.
    """
    num_workers = min(4, os.cpu_count() or 1)
    if DEVICE.type == "mps":
        # MPS does not support stable multiprocessing with many workers
        num_workers = min(2, num_workers)

    print(f"[INFO] DataLoader — batch={batch_size}  workers={num_workers}  "
          f"shuffle={shuffle}  pin_memory={torch.cuda.is_available()}")

    return DataLoader(
        dataset,
        batch_size       = batch_size,
        shuffle          = shuffle,
        drop_last        = drop_last,
        num_workers      = num_workers,
        persistent_workers = (num_workers > 0),
        pin_memory       = torch.cuda.is_available(),
        prefetch_factor  = 4 if num_workers > 0 else None,
    )


# ==========================================
# 8. Quick test
# ==========================================
if __name__ == "__main__":
    print("\n========== DATASET TEST ==========")
    # NYU — official mat
    try:
        ds = NYUDataset(DATASET_NYU_PATH, mode="finetune", split="all")
        rgb, depth = ds[0]
        print(f"[NYU] RGB: {rgb.shape}  Depth: {depth.shape}  "
              f"depth range [{depth.min():.1f}, {depth.max():.1f}] cm")
    except Exception as e:
        print(f"[NYU] {e}")

    # KITTI — finetune
    try:
        ds_k = KITTIDataset(
            path    = DATASET_KITTI_PATH / "rgb",
            gt_root = DATASET_KITTI_PATH_GT / "depth",
            mode    = "finetune",
            split   = "train"
        )
        rgb_k, depth_k = ds_k[0]
        print(f"[KITTI] RGB: {rgb_k.shape}  Depth: {depth_k.shape}  "
              f"depth range [{depth_k.min():.3f}, {depth_k.max():.3f}] m")
    except Exception as e:
        print(f"[KITTI] {e}")

    print("\n========== DONE ==========")