import glob
import h5py
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
from torchvision import transforms

from src.config import (
    DATASET_NYU_PATH,
    DATASET_KITTI_PATH,
    DATASET_KITTI_PATH_GT,
    IMG_SIZE,
    BATCH_SIZE,
    DEVICE,
)

# ==========================================
# Dataset download and verification
# ==========================================
NYU_KAGGLE_DATASET_SLUG = "artemmmtry/nyu-depth-v2"
KITTI_KAGGLE_DATASET_SLUG = "awsaf49/kitti-eigen-split-dataset"
KAGGLE_DATA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm", ".h5", ".mat",
}


def _kaggle_cli_prefix() -> list[str]:
    if shutil.which("kaggle") is not None:
        return ["kaggle"]
    return [sys.executable, "-m", "kaggle"]


def _count_kaggle_data_files(split_dir: Path) -> int:
    if not split_dir.exists():
        return 0

    preferred_roots = [
        split_dir / "image",
        split_dir / "images",
        split_dir / "depth_map",
        split_dir / "depth_maps",
    ]
    counts = []
    for root in preferred_roots:
        if root.exists():
            counts.append(
                sum(
                    1
                    for file_path in root.rglob("*")
                    if file_path.is_file()
                    and file_path.suffix.lower() in KAGGLE_DATA_EXTENSIONS
                )
            )

    if counts:
        return max(counts)

    return sum(
        1
        for file_path in split_dir.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in KAGGLE_DATA_EXTENSIONS
    )


def _download_kaggle_dataset(
    dataset_slug: str,
    output_dir: Path,
    unzip: bool = True,
    force: bool = False,
) -> bool:
    command = _kaggle_cli_prefix() + [
        "datasets", "download", "-d", dataset_slug, "-p", str(output_dir),
    ]
    if unzip:
        command.append("--unzip")
    if force:
        command.append("--force")

    print(f"[INFO] Kaggle download command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return True
    except FileNotFoundError:
        print("[ERROR] Kaggle CLI not found. Install the `kaggle` package and configure your API token.")
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Kaggle download failed with exit code {exc.returncode}")
    return False


def _download_and_verify_kaggle_dataset(
    output_dir: str | Path,
    dataset_slug: str,
    dataset_name: str,
    verify: bool = True,
    cleanup_archive: bool = True,
    force: bool = False,
) -> dict:
    """Download and lightly verify a Kaggle-hosted dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sum(
        1
        for file_path in output_dir.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in KAGGLE_DATA_EXTENSIONS
    )
    if force or existing_files == 0:
        print(f"[DL] Kaggle dataset: {dataset_slug}")
        if not _download_kaggle_dataset(dataset_slug, output_dir, unzip=True, force=force):
            return {
                "ok": False,
                "errors": ["kaggle download failed"],
                "warnings": [],
                "train_files": 0,
                "validation_files": 0,
                "total_files": 0,
            }
    else:
        print(f"[SKIP] Kaggle {dataset_name} data already present in {output_dir}")

    if cleanup_archive:
        for archive_path in output_dir.glob("*.zip"):
            archive_path.unlink(missing_ok=True)

    result = {
        "train_files": 0,
        "validation_files": 0,
        "total_files": 0,
        "errors": [],
        "warnings": [],
        "ok": False,
    }
    if not verify:
        result["ok"] = True
        return result

    print(f"\n[VERIFY] Checking Kaggle {dataset_name} dataset integrity...")
    train_dir = next((p for p in (output_dir / "train", output_dir / "training") if p.exists()), None)
    validation_dir = next(
        (
            p
            for p in (
                output_dir / "validation",
                output_dir / "val",
                output_dir / "valid",
                output_dir / "test",
            )
            if p.exists()
        ),
        None,
    )

    if train_dir is None and validation_dir is None:
        result["warnings"].append("No explicit train/validation split folders found")
        result["total_files"] = _count_kaggle_data_files(output_dir)
        result["ok"] = result["total_files"] > 0
    else:
        if train_dir is None:
            result["warnings"].append("Missing train split folder")
        else:
            result["train_files"] = _count_kaggle_data_files(train_dir)

        if validation_dir is None:
            result["warnings"].append("Missing validation split folder")
        else:
            result["validation_files"] = _count_kaggle_data_files(validation_dir)

        if result["train_files"] == 0 and result["validation_files"] == 0:
            result["errors"].append(f"No {dataset_name} data files were found after extraction")
        result["ok"] = len(result["errors"]) == 0
        result["total_files"] = result["train_files"] + result["validation_files"]

    print(f"  Train files            : {result['train_files']:,}")
    print(f"  Validation/test files : {result['validation_files']:,}")
    print(f"  Total data files       : {result['total_files']:,}")
    for warning in result["warnings"]:
        print(f"  [WARN] {warning}")
    for error in result["errors"]:
        print(f"  [ERROR] {error}")
    if result["ok"]:
        print(f"  [OK] Kaggle {dataset_name} dataset complete")
    return result


def download_nyu(
    output_dir: str | Path,
    dataset_slug: str = NYU_KAGGLE_DATASET_SLUG,
    verify: bool = True,
    cleanup_archive: bool = True,
    force: bool = False,
) -> dict:
    """Download NYU Depth V2 exclusively from Kaggle."""
    return _download_and_verify_kaggle_dataset(
        output_dir=output_dir,
        dataset_slug=dataset_slug,
        dataset_name="NYU Depth V2",
        verify=verify,
        cleanup_archive=cleanup_archive,
        force=force,
    )


def download_kitti(
    output_dir: str | Path,
    dataset_slug: str = KITTI_KAGGLE_DATASET_SLUG,
    verify: bool = True,
    cleanup_archive: bool = True,
    force: bool = False,
) -> dict:
    """Download KITTI Eigen Split exclusively from Kaggle."""
    return _download_and_verify_kaggle_dataset(
        output_dir=output_dir,
        dataset_slug=dataset_slug,
        dataset_name="KITTI Eigen Split",
        verify=verify,
        cleanup_archive=cleanup_archive,
        force=force,
    )

# ==========================================
# Transforms
# ==========================================
import torch.nn.functional as F


class SparseDepthResize:
    """
    Area-pooling resize that preserves only valid depth pixels (> 1e-3).
    Used to correctly downsample sparse KITTI depth maps without
    introducing bias from zero (invalid) pixels.
    """
    def __init__(self, target_size):
        self.target_size = target_size  # (H, W)

    def __call__(self, depth):
        # depth: Tensor [1, H_orig, W_orig]
        mask = (depth > 1e-3).float()

        depth_b = depth.unsqueeze(0)
        mask_b  = mask.unsqueeze(0)

        depth_valid = depth_b * mask_b
        sum_depth   = F.interpolate(depth_valid, size=self.target_size, mode='area')
        count_valid = F.interpolate(mask_b,      size=self.target_size, mode='area')

        output_depth = torch.zeros_like(sum_depth)
        valid_out_mask = (count_valid > 0)
        output_depth[valid_out_mask] = (
            sum_depth[valid_out_mask] / count_valid[valid_out_mask]
        )
        return output_depth.squeeze(0)


# ==========================================
# Low-level image loaders
# ==========================================

def _load_rgb_cv2(image_path):
    """Load an RGB image via OpenCV, return float32 tensor [3, H, W] in [0, 1]."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def _load_depth_cv2(depth_path):
    """Load a grayscale depth image via OpenCV, return float32 tensor [1, H, W]."""
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Unable to read depth image: {depth_path}")
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(depth).unsqueeze(0).float()


def _depth_read_kitti(filename):
    """
    Load a KITTI 16-bit PNG depth map.
    Returns a float32 numpy array in metres; invalid pixels are set to -1.
    Reference: KITTI depth completion readme.
    """
    depth_png = np.array(Image.open(filename), dtype=int)
    assert np.max(depth_png) > 255, (
        f"Expected 16-bit depth map, got max={np.max(depth_png)} in {filename}"
    )
    depth = depth_png.astype(float) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


def guided_filter(I, p, r, eps):
    if len(I.shape) == 3:
        I_gray = cv2.cvtColor((I * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        I_gray = I.astype(np.float32)
    p = p.astype(np.float32)
    mean_I = cv2.boxFilter(I_gray, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I_gray * p, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I_gray * I_gray, -1, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    q = mean_a * I_gray + mean_b
    return q


def densify_kitti_guided(depth, rgb, r=16, eps=0.01):
    h, w = depth.shape
    invalid_mask = (depth <= 0.0) | (depth == -1.0)
    rgb_norm = rgb.astype(np.float32) / 255.0
    
    depth_clean = depth.copy()
    depth_clean[invalid_mask] = 0.0
    max_val = depth_clean.max()
    if max_val == 0:
        return depth
        
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel_15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    
    # 1. Grow sparse points slightly to reduce gaps
    dilated = cv2.dilate(depth_clean, kernel_3)
    # 2. Apply morphological closing to bridge gaps between points
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_7)
    # 3. Apply a larger dilation to fill any remaining background pixels
    depth_base = cv2.dilate(closed, kernel_15)
    
    # 3.5. Downwards vertical propagation to fill bottom corners with road values (0.2 ms!)
    # This ensures that empty pixels in the bottom corners inherit nearby road depths.
    # The sky region (where there are no LiDAR points) remains 0.0 because there are no
    # non-zero pixels above it to propagate downwards.
    for y in range(1, h):
        mask = (depth_base[y] == 0) & (depth_base[y - 1] > 0)
        depth_base[y][mask] = depth_base[y - 1][mask]
        
    # 4. Smooth out morphological blockiness using a fast Gaussian Blur
    depth_base_blurred = cv2.GaussianBlur(depth_base, (15, 15), 0)
    
    # Run Kaiming He's Guided Filter using the smooth morphological base
    # Guided filter will perfectly transfer sharp boundaries and structures from the RGB image.
    depth_guided = guided_filter(rgb_norm, depth_base_blurred, r=r, eps=eps)
    
    final_depth = depth_guided.copy()
    valid_mask = ~invalid_mask
    final_depth[valid_mask] = depth[valid_mask]
    final_depth = np.clip(final_depth, 0.0, max_val)
    
    # Force the sky and any other unfilled regions to remain exactly 0.0
    final_depth[depth_base == 0.0] = 0.0
    
    return final_depth



# ==========================================
# SSL augmentation helper
# ==========================================

def get_ssl_transform(global_view: bool = False, debug: bool = False):
    """
    LeJEPA multi-crop augmentation pipeline (exact spec from the paper repo).

    Both global and local views share the **same** set of augmentations.
    The only difference is the crop scale and output resolution:

        Global views: 256×256, scale (0.3, 1.0)  → 30-100% of the image
        Local  views: 128×128, scale (0.05, 0.3) →  5-30%  of the image

    Pipeline (identical for both view types):
        RandomResizedCrop  → scale/resolution differ
        RandomHorizontalFlip  p=0.5
        ColorJitter           p=0.8  (brightness=0.4, contrast=0.4,
                                       saturation=0.2, hue=0.1)
        RandomGrayscale       p=0.2
        GaussianBlur          p=0.5
        RandomSolarize        p=0.2, threshold=128.0/255.0
        Normalize             ImageNet mean/std

    Note: resolution kept at 256/128 (instead of paper's 224/98) because
    NYU/KITTI images are larger than ImageNet and benefit from higher resolution.
    """
    if debug:
        resize = v2.Resize((256, 256), antialias=True)
    elif global_view:
        resize = v2.RandomResizedCrop(256, scale=(0.3, 1.0), antialias=True)
    else:
        resize = v2.RandomResizedCrop(128, scale=(0.05, 0.3), antialias=True)

    ops = [
        resize,
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=23 if global_view else 9)], p=0.5),
        v2.RandomSolarize(threshold=128.0/255.0, p=0.2),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return v2.Compose(ops)


def get_ssl_spatial_transform(global_view: bool = False):
    """
    Applies only spatial transformations (cropping and flipping).
    Keeps the operations fully on the GPU/MPS when inputting a tensor.
    """
    if global_view:
        resize = v2.RandomResizedCrop(256, scale=(0.3, 1.0), antialias=True)
    else:
        resize = v2.RandomResizedCrop(128, scale=(0.05, 0.3), antialias=True)
    return v2.Compose([
        resize,
        v2.RandomHorizontalFlip(p=0.5)
    ])


def get_ssl_color_transform(global_view: bool = False):
    """
    Applies color, noise, and normalization transforms to a cropped patch.
    """
    return v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=23 if global_view else 9)], p=0.5),
        v2.RandomSolarize(threshold=128.0/255.0, p=0.2),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ==========================================
# Dataset
# ==========================================

class DepthEstimationDataset(Dataset):
    """
    Unified dataset wrapper for:
      - NYU Depth V2  (Kaggle: artemmmtry/nyu-depth-v2)
      - KITTI Eigen   (Kaggle: awsaf49/kitti-eigen-split-dataset)

    NYU layout on disk
    ------------------
    <nyu_root>/
        data/
            nyu2_train/<scene>/<n>.jpg   # RGB
            nyu2_train/<scene>/<n>.png   # depth (same stem as .jpg)
            nyu2_test/<scene>/<n>.jpg
            nyu2_test/<scene>/<n>.png

    KITTI layout on disk  (Kaggle: awsaf49/kitti-eigen-split-dataset)
    --------------------
    <kitti_root>/
        train/
            <date>/<drive>/
                image_02/data/<frame>.png                       # RGB
                proj_depth/groundtruth/image_02/<frame>.png     # depth GT
        test/   # same structure
    """

    DATASETS = ("nyu", "kitti")

    def __init__(
        self,
        dataset: str,
        path,
        mode: str = "finetune",
        depth: bool = True,
        cam: str = "image_02",
        split: str = "all",
        cache_on_disk: bool = True,
    ):
        if dataset not in self.DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset!r}. Choose from {self.DATASETS}")

        self.dataset = dataset
        self.mode    = mode
        self.depth   = depth
        self.samples = []
        self.use_cache = cache_on_disk
        self.h5_handle = None

        print(f"\n[INFO] Initializing {dataset.upper()} dataset")
        print(f"[INFO] Mode: {mode}")

        if dataset == "nyu" and mode == "finetune":
            self.img_size = (192, 256)
        elif dataset == "kitti" and mode == "finetune":
            self.img_size = (192, 636)
        elif mode == "pretrain":
            self.img_size = (256, 256)

        cache_dir = Path(path) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_filename = f"{dataset}_{mode}_{split}_{self.img_size[0]}x{self.img_size[1]}_d{self.depth}_hybrid.h5"
        self.cache_path = cache_dir / cache_filename

        if self.use_cache and self.cache_path.exists():
            with h5py.File(self.cache_path, "r") as f:
                n_cached = len(f["rgb"])
            print(f"[INFO] Found pre-processed HDF5 cache: {self.cache_path} ({n_cached} samples)")
            
            # Populate self.samples with raw paths for scene-level/sequence-level split metadata
            if dataset == "nyu":
                self._init_nyu(Path(path), split)
            elif dataset == "kitti":
                self._init_kitti(kitti_root=Path(path), cam=cam, split=split)
                
            if len(self.samples) != n_cached:
                print(f"[WARNING] Cache size ({n_cached}) differs from raw file list ({len(self.samples)}). Forcing range(n_cached).")
                self.samples = range(n_cached)
            return

        if dataset == "nyu":
            self._init_nyu(Path(path), split)
        elif dataset == "kitti":
            self._init_kitti(
                kitti_root=Path(path),
                cam=cam,
                split=split,
            )

        self.raw_transform = transforms.Compose([
            transforms.Resize(self.img_size),
        ])
        self.rgb_transform = transforms.Compose([
            transforms.Resize(self.img_size),
        ])
        self.depth_transform = SparseDepthResize(self.img_size)

        print(f"[INFO] Target resolution (H, W): {self.img_size}")
        print(f"[INFO] Total samples: {len(self.samples)}")
        if len(self.samples) == 0:
            print("[WARNING] Dataset is empty")
            return

        if self.use_cache and not self.cache_path.exists():
            self._build_cache()

    # ------------------------------------------------------------------
    # NYU initialisation
    # ------------------------------------------------------------------

    def _init_nyu(self, path: Path, split: str):
        """
        Scan <path>/data/nyu2_train/ and/or <path>/data/nyu2_test/ for
        RGB (.jpg) / depth (.png) pairs that share the same filename stem.

        split values:
          'train' → nyu2_train only
          'test'  → nyu2_test  only
          'all'   → both folders
        """
        if split not in ("all", "train", "test"):
            raise ValueError(
                f"Unsupported NYU split: {split!r}. Choose 'all', 'train', or 'test'"
            )

        data_root = path / "data"
        if not data_root.exists():
            raise FileNotFoundError(
                f"NYU data root not found: {data_root}\n"
                "Expected layout: <path>/data/nyu2_train/<scene>/<n>.jpg + <n>.png"
            )

        # Determine which split folders to scan
        folders = []
        if split in ("train", "all"):
            folders.append(data_root / "nyu2_train")
        if split in ("test", "all"):
            folders.append(data_root / "nyu2_test")

        print(f"[INFO] NYU split: {split}")

        paired = []
        for folder in folders:
            if not folder.exists():
                print(f"[WARNING] NYU folder not found, skipping: {folder}")
                continue

            is_test = folder.name == "nyu2_test"

            if is_test:
                # test layout: <n>_colors.png (RGB) + <n>_depth.png (depth)
                for rgb_path in sorted(folder.rglob("*_colors.png")):
                    stem = rgb_path.name.replace("_colors.png", "")
                    depth_path = rgb_path.parent / f"{stem}_depth.png"
                    if depth_path.exists():
                        paired.append((str(rgb_path), str(depth_path)))
            else:
                # train layout: <n>.jpg (RGB) + <n>.png (depth)
                for rgb_path in sorted(folder.rglob("*.jpg")):
                    depth_path = rgb_path.with_suffix(".png")
                    if depth_path.exists():
                        paired.append((str(rgb_path), str(depth_path)))

        self.samples = paired
        print(f"[INFO] NYU RGB-depth pairs found: {len(self.samples)}")

    # ------------------------------------------------------------------
    # NYU item loading
    # ------------------------------------------------------------------

    def _get_nyu(self, idx):
        rgb_path, depth_path = self.samples[idx]

        rgb = _load_rgb_cv2(rgb_path)

        if self.mode == "pretrain":
            return self.raw_transform(rgb)

        rgb = self.rgb_transform(rgb)
        if not self.depth:
            return rgb

        depth = _load_depth_cv2(depth_path)
        if "nyu2_test" in depth_path:
            # Test depth PNGs are 16-bit uint16 storing millimetres -> divide by 1000
            depth = depth / 1000.0
        else:
            # Train depth PNGs are 8-bit uint8 scaled 0-255 for 0-10m -> scale by 10/255
            depth = depth * (10.0 / 255.0)

        depth = self.depth_transform(depth)
        return rgb, depth

    # ------------------------------------------------------------------
    # KITTI initialisation
    # ------------------------------------------------------------------

    def _init_kitti(self, kitti_root: Path, cam: str, split: str):
        """
        Build the list of (rgb_path, gt_path) pairs.

        Layout (both RGB and GT share the same drive directory):
          <kitti_root>/<split>/<date>/<drive>/image_02/data/<frame>.png        ← RGB
          <kitti_root>/<split>/<date>/<drive>/proj_depth/groundtruth/<cam>/<frame>.png  ← GT

        Same filename ⇒ same frame.
        """
        splits = ("train", "test") if split == "all" else (split,)
        print(f"[INFO] KITTI splits: {splits}")

        for split in splits:
            # Glob all GT depth files under this split
            pattern = str(
                kitti_root / split / "*" / "*_sync" / "proj_depth" / "groundtruth" / cam / "*.png"
            )
            gt_files = sorted(glob.glob(pattern))
            print(f"[INFO] GT files in split '{split}': {len(gt_files)}")

            for gt_path in gt_files:
                p = Path(gt_path)
                # p:  .../split/date/drive/proj_depth/groundtruth/image_02/frame.png
                #           [-7] [-6] [-5]    [-4]       [-3]       [-2]    [-1]
                drive_dir  = p.parts[-5]   # e.g. 2011_09_26_drive_0001_sync
                frame_file = p.parts[-1]   # e.g. 0000000005.png

                # RGB is a sibling of proj_depth inside the same drive folder
                rgb_path = str(
                    p.parents[3] / cam / "data" / frame_file
                )
                if os.path.exists(rgb_path):
                    self.samples.append((rgb_path, gt_path))

        print(f"[INFO] Valid KITTI RGB-depth pairs: {len(self.samples)}")

    # ------------------------------------------------------------------
    # KITTI item loading
    # ------------------------------------------------------------------

    def _get_kitti(self, idx):
        rgb_path, gt_path = self.samples[idx]
        rgb = _load_rgb_cv2(rgb_path)

        if self.mode == "pretrain":
            return self.raw_transform(rgb)

        rgb = self.rgb_transform(rgb)
        if not self.depth:
            return rgb

        # Load sparse depth and downsample it using SparseDepthResize (bias-free)
        depth_np = _depth_read_kitti(gt_path)
        depth_sparse = torch.from_numpy(depth_np).unsqueeze(0).float()
        depth_sparse = self.depth_transform(depth_sparse)

        # Convert downsampled rgb tensor [3, H, W] to numpy [H, W, 3] in [0, 255]
        rgb_np_u8 = (rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        # Convert downsampled sparse depth tensor [1, H, W] to numpy [H, W]
        depth_sparse_np = depth_sparse.squeeze(0).numpy()

        # Densify downsampled depth using RGB-guided filter
        depth_dense_np = densify_kitti_guided(depth_sparse_np, rgb_np_u8, r=16, eps=0.01)
        depth_dense = torch.from_numpy(depth_dense_np).unsqueeze(0).float()

        # Stack into 2-channel tensor: channel 0 = Dense, channel 1 = Sparse
        depth_stacked = torch.cat([depth_dense, depth_sparse], dim=0)
        return rgb, depth_stacked

    # ------------------------------------------------------------------
    # PyTorch Dataset API & Caching
    # ------------------------------------------------------------------

    def _build_cache(self):
        print(f"\n[INFO] Building HDF5 cache for {len(self.samples)} samples...")
        tmp_path = self.cache_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        N = len(self.samples)
        H, W = self.img_size

        with h5py.File(tmp_path, "w") as f:
            rgb_ds = f.create_dataset("rgb", shape=(N, 3, H, W), dtype=np.uint8, chunks=(1, 3, H, W))
            if self.depth and self.mode != "pretrain":
                # For KITTI we output 2 channels (dense, sparse), for NYU we output 1 channel
                c_out = 2 if self.dataset == "kitti" else 1
                depth_ds = f.create_dataset("depth", shape=(N, c_out, H, W), dtype=np.float32, chunks=(1, c_out, H, W))

            # Temporarily disable caching to load raw data on-the-fly in parallel
            self.use_cache = False
            
            # Force num_workers = 0 during cache generation to completely prevent HDF5 fork deadlocks.
            # Since the new morphological densification is ultra-fast (sub-millisecond),
            # single-threaded execution is extremely stable, safe from deadlocks, and very fast!
            num_workers = 0
            print(f"  [CACHE] Single-threaded CPU loader active for safe HDF5 cache generation...")
            
            # Use batch_size=64 to group disk operations and write contiguous blocks to HDF5
            batch_size = 64
            parallel_loader = DataLoader(
                self,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=False
            )
            
            idx = 0
            last_print_idx = 0
            for res in parallel_loader:
                if self.depth and self.mode != "pretrain":
                    rgb_batch, depth_batch = res
                    batch_len = rgb_batch.shape[0]
                    
                    rgb_np = (rgb_batch * 255.0).clamp(0, 255).byte().numpy()
                    depth_np = depth_batch.numpy()
                    
                    # Single contiguous disk write transaction
                    rgb_ds[idx : idx + batch_len] = rgb_np
                    depth_ds[idx : idx + batch_len] = depth_np
                    idx += batch_len
                else:
                    rgb_batch = res
                    batch_len = rgb_batch.shape[0]
                    
                    rgb_np = (rgb_batch * 255.0).clamp(0, 255).byte().numpy()
                    
                    # Single contiguous disk write transaction
                    rgb_ds[idx : idx + batch_len] = rgb_np
                    idx += batch_len

                # Print progress exactly every ~1000 samples (aligned to batch size) and at the very end
                if (idx - last_print_idx) >= 1000 or idx == N:
                    print(f"  [CACHE] Processed {idx}/{N} samples...")
                    last_print_idx = idx

            # Restore original caching flag
            self.use_cache = True

        tmp_path.rename(self.cache_path)
        print(f"[INFO] HDF5 cache successfully saved to {self.cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_cache and self.cache_path is not None and self.cache_path.exists():
            if self.h5_handle is None:
                self.h5_handle = h5py.File(self.cache_path, "r", swmr=True)
            rgb_np = self.h5_handle["rgb"][idx]
            rgb = torch.from_numpy(rgb_np).float() / 255.0
            if self.depth and self.mode != "pretrain":
                depth_np = self.h5_handle["depth"][idx]
                depth = torch.from_numpy(depth_np).float()
                return rgb, depth
            return rgb

        try:
            if self.dataset == "nyu":
                return self._get_nyu(idx)
            return self._get_kitti(idx)
        except Exception as e:
            import random
            new_idx = random.randint(0, len(self.samples) - 1)
            print(f"[WARNING] Failed to read sample {idx}: {e} — retrying with {new_idx}")
            return self.__getitem__(new_idx)


# ==========================================
# Dataset split helpers
# ==========================================

def random_subset(dataset, subset_size, seed=42):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    return Subset(dataset, indices)


def scene_level_split(dataset, train_ratio=0.8, seed=42):
    """
    Split data at sequence/scene level to reduce leakage from adjacent frames.
    Supports both Dataset and Subset inputs.
    """
    if isinstance(dataset, Subset):
        actual_indices = dataset.indices
        underlying_samples = dataset.dataset.samples
    else:
        actual_indices = list(range(len(dataset)))
        underlying_samples = dataset.samples

    if (
        isinstance(underlying_samples, range)
        or len(underlying_samples) == 0
        or isinstance(underlying_samples[0], int)
    ):
        print("\n[WARNING] File paths not available in samples; using deterministic frame-level split.")
        random.seed(seed)
        shuffled_indices = list(range(len(dataset)))
        random.shuffle(shuffled_indices)
        split_idx = int(len(dataset) * train_ratio)
        return (
            Subset(dataset, shuffled_indices[:split_idx]),
            Subset(dataset, shuffled_indices[split_idx:]),
        )

    curr = dataset
    while hasattr(curr, "dataset") and curr.__class__.__name__ != "DepthEstimationDataset":
        curr = curr.dataset

    dataset_name = None
    if curr.__class__.__name__ == "DepthEstimationDataset":
        dataset_name = getattr(curr, "dataset", None)

    if dataset_name is None and len(underlying_samples) > 0:
        first_sample = underlying_samples[0]
        if isinstance(first_sample, (list, tuple)) and len(first_sample) > 0:
            sample_str = str(first_sample[0]).lower()
            if "kitti" in sample_str or "2011_" in sample_str or "image_02" in sample_str:
                dataset_name = "kitti"
            elif "nyu" in sample_str:
                dataset_name = "nyu"

    is_kitti = dataset_name == "kitti"
    seq_to_indices = {}
    for subset_idx, original_idx in enumerate(actual_indices):
        _rgb_path, gt_path = underlying_samples[original_idx]
        path = Path(gt_path)

        if is_kitti:
            drive_parts = [part for part in path.parts if "_sync" in part or part.startswith("2011_")]
            scene_name = drive_parts[-1] if drive_parts else (path.parts[-5] if len(path.parts) >= 5 else "unknown")
        else:
            scene_name = path.parent.name

        seq_to_indices.setdefault(scene_name, []).append(subset_idx)

    unique_scenes = sorted(seq_to_indices.keys())
    random.seed(seed)
    random.shuffle(unique_scenes)

    split_idx = int(len(unique_scenes) * train_ratio)
    train_scenes = unique_scenes[:split_idx]
    val_scenes = unique_scenes[split_idx:]

    train_indices = []
    for scene in train_scenes:
        train_indices.extend(seq_to_indices[scene])

    val_indices = []
    for scene in val_scenes:
        val_indices.extend(seq_to_indices[scene])

    print("\n[INFO] Scene-level split completed:")
    print(f"  - Total scenes: {len(unique_scenes)}")
    print(f"  - Train scenes: {len(train_scenes)} | Val scenes: {len(val_scenes)}")
    print(f"  - Train frames: {len(train_indices)} | Val frames: {len(val_indices)}")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ==========================================
# DataLoader factory
# ==========================================

def build_dataloader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
):
    import sys
    if sys.platform == "darwin":
        # Using an alias ('torch_mp') prevents shadowing the global 'torch' module
        from torch import multiprocessing as torch_mp
        # Prevents "Broken pipe" and "Too many open files" on macOS
        torch_mp.set_sharing_strategy("file_system")

    num_workers = 4 if DEVICE.type == "mps" else min(4, os.cpu_count())

    print("\n[INFO] Building DataLoader")
    print(f"[INFO] Batch size:   {batch_size}")
    print(f"[INFO] Num workers:  {num_workers}")
    print(f"[INFO] Device:       {DEVICE}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=torch.cuda.is_available(),
    )
    print("[INFO] DataLoader ready")
    return loader


# ==========================================
# Visualization helper
# ==========================================

def visualize_depth(rgb_tensor, depth_tensor):
    """Quick visual debug: render RGB and normalised depth map side by side."""
    import matplotlib.pyplot as plt
    rgb = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(rgb, 0, 1)
    
    depth = depth_tensor.squeeze().cpu().numpy()
    valid = depth > 0
    depth_vis = np.zeros_like(depth)
    if valid.any():
        d_min = depth[valid].min()
        d_max = depth[valid].max()
        depth_vis[valid] = (depth[valid] - d_min) / (d_max - d_min + 1e-8)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(rgb)
    ax1.set_title("RGB")
    ax1.axis("off")
    
    im = ax2.imshow(depth_vis, cmap="magma")
    ax2.set_title("Depth Map")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()


# ==========================================
# Smoke test
# ==========================================

if __name__ == "__main__":
    print("\n========== DATASET SMOKE TEST ==========")

    nyu_dataset = DepthEstimationDataset(
        dataset="nyu",
        path=DATASET_NYU_PATH,
        mode="finetune",
        split="all",
    )
    print("\n[TEST] Loading NYU sample 0")
    rgb, depth = nyu_dataset[0]
    print(f"[TEST] RGB   shape: {rgb.shape}")
    print(f"[TEST] Depth shape: {depth.shape}")
    visualize_depth(rgb, depth)

    kitti_dataset = DepthEstimationDataset(
        dataset="kitti",
        path=DATASET_KITTI_PATH,
        split="train",
        mode="finetune",
    )
    print(f"\n[TEST] KITTI samples: {len(kitti_dataset)}")
    rgb_k, depth_k = kitti_dataset[0]
    print(f"[TEST] RGB   shape: {rgb_k.shape}")
    print(f"[TEST] Depth shape: {depth_k.shape}")
    visualize_depth(rgb_k, depth_k)

    print("\n========== DONE ==========")
