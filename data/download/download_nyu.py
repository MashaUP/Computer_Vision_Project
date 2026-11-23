"""
download_nyu.py
===============
Download and convert NYU Depth V2 for the METER + LeJEPA project.

Supports both automatic downloads through the Kaggle API and manually placing
the zip archive downloaded from a browser.

Quick start
-----------
  # If you downloaded the Kaggle zip manually, place it in ./data/raw/nyu/ and run:
  python download_nyu.py --output_dir ./data/raw/nyu

  # If you want to use the automatic Kaggle API flow (requires kaggle.json):
  python download_nyu.py --output_dir ./data/raw/nyu

Output layout
-------------
<output_dir>/
  nyu_depth_v2_labeled.mat          <- official Silberman 2012 dataset (test set)
  train/
    scene_00000/
      000000.jpg                    <- RGB, JPEG quality 95
      depth_map/
        000000.png                  <- uint16 depth in MILLIMETERS (train set)
  val/
    ...
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAT_URL       = (
    "http://horatio.cs.nyu.edu/mit/silberman/"
    "nyu_depth_v2/nyu_depth_v2_labeled.mat"
)
_KAGGLE_SLUG   = "artemmmtry/nyu-depth-v2"
_TMP_DIR_NAME  = "_tmp_h5"
_VAL_RATIO     = 0.05
_RANDOM_SEED   = 42
_FRAMES_PER_SCENE = 1000


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

class _SimpleProgress:
    """Fallback progress bar without tqdm."""
    def __init__(self, total=0, desc=""):
        self.total = total
        self.n     = 0
        self.desc  = desc
        print(f"{desc}: 0%", end="", flush=True)

    def update(self, n):
        self.n += n
        if self.total:
            pct = self.n / self.total * 100
            mb  = self.n / 1024 / 1024
            print(f"\r{self.desc}: {mb:.1f} MB ({pct:.1f}%)", end="", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        print()


def _progress(total=0, desc=""):
    if _HAS_TQDM:
        return tqdm(total=total, unit="B", unit_scale=True, desc=desc)
    return _SimpleProgress(total=total, desc=desc)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_stream(url: str, dest: Path,
                     retries: int = 3, chunk_mb: int = 4) -> bool:
    """Robust download with retries and progress output."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                chunk = chunk_mb * 1024 * 1024
                with open(tmp, "wb") as f:
                    with _progress(total=total, desc=dest.name) as pbar:
                        while True:
                            buf = resp.read(chunk)
                            if not buf:
                                break
                            f.write(buf)
                            pbar.update(len(buf))
            tmp.rename(dest)
            return True

        except Exception as exc:
            print(f"\n  [WARN] attempt {attempt}/{retries}: {exc}")
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(3 * attempt)

    print(f"  [ERROR] download failed: {url}")
    return False


# ---------------------------------------------------------------------------
# Kaggle API helpers
# ---------------------------------------------------------------------------

def _kaggle_cmd() -> list:
    if shutil.which("kaggle"):
        return ["kaggle"]
    return [sys.executable, "-m", "kaggle"]


def _check_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_ok = (
        bool(os.environ.get("KAGGLE_USERNAME"))
        and bool(os.environ.get("KAGGLE_KEY"))
    )
    if kaggle_json.exists() or env_ok:
        return True
    print("\n[ERROR] Missing Kaggle credentials for automatic download.")
    print("  If you downloaded the zip manually, rename it or extract it into '_tmp_h5'.")
    print("  Otherwise, create a Kaggle token and save it to ~/.kaggle/kaggle.json")
    return False


def _download_kaggle(tmp_dir: Path, force: bool = False) -> bool:
    """Download the Kaggle dataset into tmp_dir with --unzip."""
    cmd = _kaggle_cmd() + [
        "datasets", "download",
        "-d", _KAGGLE_SLUG,
        "-p", str(tmp_dir),
        "--unzip",
    ]
    if force:
        cmd.append("--force")

    print(f"\n  [CMD] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Kaggle download failed: {exc}")
        return False
    except FileNotFoundError:
        print("  [ERROR] 'kaggle' not found. Install it with: pip install kaggle")
        return False


# ---------------------------------------------------------------------------
# H5 discovery & processing
# ---------------------------------------------------------------------------

def _find_h5_files(root: Path) -> list:
    """Find all .h5 / .hdf5 files recursively."""
    files = []
    for ext in ("*.h5", "*.hdf5"):
        files.extend(sorted(root.rglob(ext)))
    seen = set()
    result = []
    for f in files:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def _load_h5(path: Path):
    with h5py.File(path, "r") as f:
        # --- RGB ---
        rgb = None
        for key in ("rgb", "image", "images"):
            if key in f:
                rgb = np.array(f[key])
                break
        if rgb is None:
            raise RuntimeError(f"No RGB key found in {path}. Keys: {list(f.keys())}")

        # --- Depth ---
        depth = None
        for key in ("depth", "depths"):
            if key in f:
                depth = np.array(f[key])
                break
        if depth is None:
            raise RuntimeError(f"No depth key found in {path}. Keys: {list(f.keys())}")

    rgb   = np.squeeze(rgb)
    depth = np.squeeze(depth)

    if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
        rgb = np.transpose(rgb, (1, 2, 0))

    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

    depth = depth.astype(np.float32)
    if depth.max() > 100.0:
        depth = depth / 1000.0

    return rgb, depth


def _save_pair(rgb: np.ndarray, depth: np.ndarray,
               scene_dir: Path, frame_id: int) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = scene_dir / "depth_map"
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_path   = scene_dir / f"{frame_id:06d}.jpg"
    depth_path = depth_dir / f"{frame_id:06d}.png"

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(rgb_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(depth_path), depth_mm)


def convert_h5_to_pairs(h5_files: list, output_dir: Path) -> dict:
    random.seed(_RANDOM_SEED)
    shuffled = list(h5_files)
    random.shuffle(shuffled)
    n_val   = max(1, int(len(shuffled) * _VAL_RATIO))
    val_set = set(str(f) for f in shuffled[:n_val])

    stats = {"train": 0, "val": 0, "errors": []}

    print(f"\n[CONVERT] Starting conversion of {len(h5_files)} H5 files...")
    iter_files = tqdm(h5_files, desc="Converting") if _HAS_TQDM else h5_files

    for h5_path in iter_files:
        try:
            rgb, depth = _load_h5(h5_path)
        except Exception as exc:
            stats["errors"].append(f"{h5_path.name}: {exc}")
            continue

        split    = "val" if str(h5_path) in val_set else "train"
        frame_id = stats[split]
        stats[split] += 1

        scene_idx = frame_id // _FRAMES_PER_SCENE
        scene_dir = output_dir / split / f"scene_{scene_idx:05d}"

        try:
            _save_pair(rgb, depth, scene_dir, frame_id)
        except Exception as exc:
            stats["errors"].append(f"save {h5_path.name}: {exc}")

    print(f"\n[OK] Conversion complete. Train: {stats['train']} frames, Val: {stats['val']} frames")
    if stats["errors"]:
        print(f"  [WARN] Found {len(stats['errors'])} errors during conversion.")
    return stats


def verify(output_dir: Path) -> None:
    print("\n[VERIFY] Checking folder integrity...")
    mat_path = output_dir / "nyu_depth_v2_labeled.mat"
    if not mat_path.exists():
        print("  [WARN] nyu_depth_v2_labeled.mat not found!")
    else:
        try:
            with h5py.File(mat_path, "r") as f:
                n = int(f["images"].shape[-1]) if "images" in f else 0
                print(f"  [OK] .mat detected correctly: {n} test pairs available.")
        except Exception as exc:
            print(f"  [ERROR] .mat file is corrupt or unreadable: {exc}")

    for split in ("train", "val"):
        split_dir = output_dir / split
        if not split_dir.exists():
            print(f"  [WARN] Folder {split}/ not found.")
            continue
        jpgs = list(split_dir.rglob("*.jpg"))
        pngs = list(split_dir.rglob("*.png"))
        print(f"  [{split}] {len(jpgs)} JPG images, {len(pngs)} depth PNG maps.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Downloader/Converter NYU Depth V2.")
    parser.add_argument("--output_dir", type=Path, default=Path("./data/raw/nyu"))
    parser.add_argument("--skip_kaggle", action="store_true", help="Skip the Kaggle download entirely.")
    parser.add_argument("--skip_mat", action="store_true", help="Skip download of the official .mat file.")
    parser.add_argument("--skip_convert", action="store_true", help="Skip H5 -> JPG/PNG conversion.")
    parser.add_argument("--force", action="store_true", help="Force download even if files already exist.")
    parser.add_argument("--dry_run", action="store_true", help="Simulation mode.")
    parser.add_argument("--no_verify", action="store_true", help="Skip the final verification.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / _TMP_DIR_NAME

    print("=" * 60)
    print("  NYU Depth V2 Downloader - METER + LeJEPA")
    print("=" * 60)
    print(f"  Output dir  : {output_dir.resolve()}")
    print("=" * 60)

    # 1. Check for local files (if the zip was downloaded manually)
    manual_zip = list(output_dir.glob("*.zip"))
    if manual_zip and not tmp_dir.exists() and not args.skip_kaggle:
        print(f"\n[INFO] Detected manually downloaded archive: {manual_zip[0].name}")
        print(f"  Extracting into {tmp_dir.name}...")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(manual_zip[0], "r") as zf:
            zf.extractall(tmp_dir)
        print("  [OK] Extraction complete.")

    # 2. Handle Kaggle download when no local files are available and it is not skipped
    h5_files_check = _find_h5_files(tmp_dir if tmp_dir.exists() else output_dir)
    
    if not h5_files_check and not args.skip_kaggle and not args.dry_run:
        if _check_kaggle_credentials():
            tmp_dir.mkdir(parents=True, exist_ok=True)
            ok = _download_kaggle(tmp_dir, force=args.force)
            if not ok:
                print("[ERROR] Automatic download failed.")
                return
    elif not args.skip_kaggle and h5_files_check:
        print(f"\n[SKIP] H5 files already detected locally. Skipping Kaggle download.")

    # 3. Convert H5 files into image/depth pairs
    if not args.skip_convert and not args.dry_run:
        h5_files = _find_h5_files(tmp_dir if tmp_dir.exists() else output_dir)
        if len(h5_files) == 0:
            print("\n[WARN] No H5 files found. If you downloaded the zip manually,")
            print("  make sure it was extracted inside a folder named '_tmp_h5'.")
        else:
            convert_h5_to_pairs(h5_files, output_dir)
            if tmp_dir.exists():
                print(f"\n[CLEANUP] Removing temporary H5 files from {tmp_dir.name}...")
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # 4. Check/download the official .mat file
    if not args.skip_mat:
        mat_path = output_dir / "nyu_depth_v2_labeled.mat"
        if mat_path.exists():
            print(f"\n[SKIP] Official .mat file already exists at the expected path.")
        elif args.dry_run:
            print(f"\n[DRY-RUN] Would download the .mat file from {_MAT_URL}")
        else:
            print(f"\n[STEP] Downloading official .mat file (~2.8 GB)...")
            _download_stream(_MAT_URL, mat_path)

    # 5. Verify the final structure
    if not args.no_verify and not args.dry_run:
        verify(output_dir)

    print("\n[DONE] Process complete.")

if __name__ == "__main__":
    main()
