"""
utils.py
========
Utility functions for the METER + LeJEPA project (Project 7).

Sections
--------
1.  Timer                  – wall-clock helper for training loops
2.  Checkpoints            – save / load model checkpoints
3.  CSV logging            – append per-epoch metrics to a .csv file
4.  JSON helpers           – save / load dicts as JSON
5.  Model helpers          – load METER pre-trained weights, random subset
6.  Dataset download       – NYU Depth V2  (labeled .mat + raw RGB for SSL)
                           – KITTI         (depth annotated + raw RGB)
"""

import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Subset

from src.config import MODEL_SAVE_PATH


# =============================================================================
# 1. Timer
# =============================================================================

@dataclass
class Timer:
    """Simple wall-clock timer for training / evaluation loops."""
    start_time: float = field(default_factory=time.time)

    def reset(self) -> None:
        self.start_time = time.time()

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def end(self) -> str:
        elapsed = self.elapsed_seconds()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m {secs}s"


# =============================================================================
# 2. Checkpoints
# =============================================================================

def _checkpoint_dir(sub: str = "") -> Path:
    """Return (and create) the checkpoint directory, optionally with a sub-folder."""
    path = Path(MODEL_SAVE_PATH) / sub if sub else Path(MODEL_SAVE_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    filename: str = "checkpoint.pt",
    sub: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Atomically save a model checkpoint.

    Parameters
    ----------
    model     : the PyTorch model
    optimizer : optimizer (pass None to skip)
    epoch     : current epoch index
    filename  : output file name
    sub       : sub-folder inside MODEL_SAVE_PATH (e.g. "ssl" or "finetune")
    extra     : optional dict of extra scalar info (loss, metrics …)

    Returns
    -------
    Path to the saved checkpoint file.
    """
    filepath = _checkpoint_dir(sub) / filename
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")

    state: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        state["extra"] = extra

    torch.save(state, tmp_path)
    tmp_path.replace(filepath)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Checkpoint saved : {filepath}")
    print(f"[INFO] Epoch: {epoch} | Total params: {total_params:,}")
    return filepath


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    filename: str = "checkpoint.pt",
    sub: str = "",
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    """
    Load a checkpoint into *model* (and optionally *optimizer*).

    Returns
    -------
    (start_epoch, extra_dict)
        start_epoch : epoch to resume from (saved_epoch + 1)
        extra_dict  : whatever was stored in state["extra"]
    """
    ckpt_path = _checkpoint_dir(sub) / filename
    if not ckpt_path.exists():
        print(f"[WARNING] Checkpoint not found: {ckpt_path}")
        return 0, {}

    checkpoint = torch.load(ckpt_path, map_location=map_location)

    # Unwrap torch.compile wrappers if present
    actual_model = getattr(model, "_orig_mod", model)
    msg = actual_model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    missing     = getattr(msg, "missing_keys", [])
    unexpected  = getattr(msg, "unexpected_keys", [])

    print(f"[INFO] Loaded checkpoint : {ckpt_path}")
    print(f"[INFO] Resuming from epoch: {start_epoch}")
    if not strict:
        print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    return start_epoch, checkpoint.get("extra", {})


# =============================================================================
# 3. CSV metric logging
# =============================================================================

def log_metrics_to_csv(
    filename: str,
    epoch: int,
    metrics: Dict[str, Any],
    root: Optional[Path] = None,
) -> Path:
    """
    Append one row (epoch + metrics) to a CSV file.

    Parameters
    ----------
    filename : CSV file name (e.g. "train_metrics.csv")
    epoch    : current epoch
    metrics  : dict of scalar values to log
    root     : directory for the CSV (defaults to MODEL_SAVE_PATH parent)

    Returns
    -------
    Path to the CSV file.
    """
    if root is None:
        root = Path(MODEL_SAVE_PATH).parent
    root.mkdir(parents=True, exist_ok=True)
    filepath = root / filename

    row: Dict[str, Any] = {"epoch": int(epoch)}
    row.update(metrics)
    fieldnames = list(row.keys())

    file_exists = filepath.exists()
    with filepath.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    if epoch == 0 or epoch % 5 == 0:
        print(f"[INFO] Metrics logged → {filepath}")
    return filepath


# =============================================================================
# 4. JSON helpers
# =============================================================================

def save_dict_to_json(data_dict: Dict[str, Any], file_path: "str | Path") -> None:
    """Serialize *data_dict* to a JSON file (tensors are converted to Python scalars)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in data_dict.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4)
    print(f"[INFO] JSON saved → {path}")


def load_dict_from_json(file_path: "str | Path") -> Dict[str, Any]:
    """Load a JSON file into a dict. Returns {} if the file does not exist."""
    path = Path(file_path)
    if not path.exists():
        print(f"[WARNING] JSON not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 5. Model helpers
# =============================================================================

def load_meter_pretrained_weights(
    model: torch.nn.Module,
    weight_path: "str | Path",
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load a METER (or LeJEPA-METER) checkpoint into the encoder of *model*.

    Handles:
    - ``model.encoder`` attribute lookup
    - ``torch.compile`` wrappers (``_orig_mod``)
    - Key remapping (strips "model.encoder.", "encoder.", "_orig_mod." prefixes)
    - Shape mismatches for depthwise / MV2 conv weights (reshape when safe)

    Parameters
    ----------
    model       : full METERModel (encoder + decoder)
    weight_path : path to the .pt / .ckpt file
    device      : map_location for torch.load

    Returns
    -------
    model with encoder weights loaded (decoder stays randomly initialised).
    """
    weight_path = Path(weight_path)
    if not weight_path.exists():
        print(f"[WARNING] Weight file not found: {weight_path}")
        return model

    ckpt = torch.load(weight_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Unwrap torch.compile if needed
    actual_model = getattr(model, "_orig_mod", model)
    model_state  = actual_model.encoder.state_dict()

    clean_dict: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        # Strip common prefixes
        new_key = (k
                   .replace("model.encoder.", "")
                   .replace("encoder.", "")
                   .replace("_orig_mod.", ""))

        if new_key not in model_state:
            continue  # ignore keys that don't belong to the encoder

        target_shape = model_state[new_key].shape
        if v.shape != target_shape:
            # Attempt safe reshape for depthwise / MV2 conv weights
            if v.numel() == math.prod(target_shape):
                v = v.reshape(target_shape)
            elif (
                "depthwise.weight" in new_key
                and len(target_shape) > 1
                and target_shape[1] == 1
                and v.shape[0] == v.shape[1]
            ):
                indices = torch.arange(v.shape[0])
                v = v[indices, indices].unsqueeze(1)
            else:
                # Shapes incompatible — skip
                continue

        clean_dict[new_key] = v

    msg = actual_model.encoder.load_state_dict(clean_dict, strict=False)
    print(f"[LOAD] {weight_path.name}")
    print(f"       Loaded           : {len(clean_dict)} layers")
    print(f"       Missing (rand)   : {len(msg.missing_keys)} layers")
    print(f"       Unexpected (skip): {len(msg.unexpected_keys)} layers")
    return model


def random_subset(dataset, subset_size: int, seed: int = 42) -> Subset:
    """Return a reproducible random subset of *dataset* of size *subset_size*."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    return Subset(dataset, indices)


# =============================================================================
# 6. Dataset download utilities
# =============================================================================

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _wget(url: str, dest: Path) -> bool:
    """Download *url* to *dest*. Returns True on success."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [DL]  {url}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:
        print(f"  [ERROR] Download failed: {exc}")
        return False


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    """Extract *zip_path* into *dest_dir* and remove the archive."""
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        zip_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"  [ERROR] Unzip failed: {exc}")


# ---------------------------------------------------------------------------
# NYU Depth V2
# ---------------------------------------------------------------------------

_NYU_BASE_URL    = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2"
_NYU_LABELED_URL = f"{_NYU_BASE_URL}/nyu_depth_v2_labeled.mat"

# Unlabelled room zips — used for LeJEPA SSL pre-training
_NYU_RAW_ZIPS: List[str] = [
    "bedrooms_part1", "bedrooms_part2", "bedrooms_part3",
    "living_rooms_part1", "living_rooms_part2",
    "kitchens_part1", "kitchens_part2",
    "offices_part1",
    "bathrooms_part1", "bathrooms_part2",
]

# Kaggle mirror (artemmmtry/nyu-depth-v2) — convenient for Colab
_NYU_KAGGLE_SLUG = "artemmmtry/nyu-depth-v2"
_NYU_KAGGLE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".h5"}


def _kaggle_cli() -> List[str]:
    return ["kaggle"] if shutil.which("kaggle") else [sys.executable, "-m", "kaggle"]


def _kaggle_download(slug: str, output_dir: Path, unzip: bool = True, force: bool = False) -> bool:
    cmd = _kaggle_cli() + ["datasets", "download", "-d", slug, "-p", str(output_dir)]
    if unzip:
        cmd.append("--unzip")
    if force:
        cmd.append("--force")
    print(f"[INFO] Kaggle command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        print("[ERROR] Kaggle CLI not found. Run: pip install kaggle and set API token.")
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Kaggle exited with code {exc.returncode}")
    return False


def _count_files(directory: Path, extensions: set) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def download_nyu(
    output_dir: "str | Path",
    skip_raw: bool = False,
    skip_labeled: bool = False,
    kaggle: bool = False,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Download NYU Depth V2.

    Layout after download
    ---------------------
    output_dir/
      raw/                          <- unlabelled RGB .ppm frames (for SSL)
        bedrooms_part1/ …
      nyu_depth_v2_labeled.mat      <- 1 449 annotated pairs (for fine-tuning)

    Parameters
    ----------
    skip_raw     : skip raw RGB download (already extracted)
    skip_labeled : skip labeled .mat download
    kaggle       : use the Kaggle mirror instead (requires kaggle CLI + token)
    verify       : run an integrity check after download

    Returns
    -------
    dict with keys: raw_frames, labeled_pairs, errors, warnings, ok
    """
    output_dir = Path(output_dir)

    # ── Kaggle shortcut ─────────────────────────────────────────────────────
    if kaggle:
        return _download_nyu_kaggle(output_dir, verify=verify)

    raw_dir   = output_dir / "rgb"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Raw RGB zips ─────────────────────────────────────────────────
    if not skip_raw:
        for i, name in enumerate(_NYU_RAW_ZIPS, 1):
            zip_name  = f"{name}.zip"
            dest_dir  = raw_dir / name
            print(f"[{i:02d}/{len(_NYU_RAW_ZIPS)}] {zip_name}")
            if dest_dir.exists():
                print("  [SKIP] already extracted")
                continue
            local_zip = raw_dir / zip_name
            if _wget(f"{_NYU_BASE_URL}/{zip_name}", local_zip):
                _unzip(local_zip, dest_dir)
                print("  [OK]")
            else:
                print(f"  [ERROR] {zip_name}")

    # ── Step 2: Labeled .mat ─────────────────────────────────────────────────
    if not skip_labeled:
        labeled_path = output_dir / "nyu_depth_v2_labeled.mat"
        if labeled_path.exists():
            print("[SKIP] nyu_depth_v2_labeled.mat already present")
        else:
            print("[DL] nyu_depth_v2_labeled.mat (~2.8 GB)…")
            if _wget(_NYU_LABELED_URL, labeled_path):
                print("[OK] labeled dataset ready")
            else:
                print("[ERROR] Failed to download nyu_depth_v2_labeled.mat")

    # ── Step 3: Verify ───────────────────────────────────────────────────────
    result: Dict[str, Any] = {
        "raw_frames": 0, "labeled_pairs": 0,
        "errors": [], "warnings": [], "ok": False,
    }
    if not verify:
        result["ok"] = True
        return result

    print("\n[VERIFY] NYU Depth V2…")
    for name in _NYU_RAW_ZIPS:
        dest_dir = raw_dir / name
        if not dest_dir.exists():
            result["errors"].append(f"Missing: {dest_dir}")
            continue
        n = len(list(dest_dir.rglob("r-*.ppm")))
        result["raw_frames"] += n
        if n == 0:
            result["warnings"].append(f"Empty dir: {dest_dir}")

    labeled_path = output_dir / "nyu_depth_v2_labeled.mat"
    if not labeled_path.exists():
        result["errors"].append(f"Missing: {labeled_path}")
    else:
        try:
            import h5py
            with h5py.File(labeled_path, "r") as f:
                result["labeled_pairs"] = f["images"].shape[-1]
        except Exception:
            result["warnings"].append("h5py not installed — cannot count labeled pairs")

    result["ok"] = len(result["errors"]) == 0
    print(f"  Raw RGB frames : {result['raw_frames']:,}")
    print(f"  Labeled pairs  : {result['labeled_pairs']}")
    _print_verify_summary(result)
    return result


def _download_nyu_kaggle(output_dir: Path, verify: bool = True) -> Dict[str, Any]:
    """Internal helper: download NYU via the Kaggle CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = _count_files(output_dir, _NYU_KAGGLE_EXTS)
    if existing == 0:
        print(f"[DL] Kaggle dataset: {_NYU_KAGGLE_SLUG}")
        if not _kaggle_download(_NYU_KAGGLE_SLUG, output_dir, unzip=True):
            return {"ok": False, "errors": ["kaggle download failed"],
                    "warnings": [], "train_files": 0, "validation_files": 0}
    else:
        print(f"[SKIP] Kaggle NYU already present ({existing} files)")

    # Clean up archives
    for archive in output_dir.glob("*.zip"):
        archive.unlink(missing_ok=True)

    result: Dict[str, Any] = {
        "train_files": 0, "validation_files": 0,
        "errors": [], "warnings": [], "ok": False,
    }
    if not verify:
        result["ok"] = True
        return result

    print("\n[VERIFY] Kaggle NYU Depth V2…")
    train_dir = next(
        (output_dir / d for d in ("train", "training") if (output_dir / d).exists()), None
    )
    val_dir = next(
        (output_dir / d for d in ("validation", "val", "valid") if (output_dir / d).exists()), None
    )

    if train_dir is None and val_dir is None:
        result["warnings"].append("No train/val split found — counting all files")
        result["train_files"] = _count_files(output_dir, _NYU_KAGGLE_EXTS)
    else:
        if train_dir:
            result["train_files"] = _count_files(train_dir, _NYU_KAGGLE_EXTS)
        else:
            result["warnings"].append("Missing train split folder")
        if val_dir:
            result["validation_files"] = _count_files(val_dir, _NYU_KAGGLE_EXTS)
        else:
            result["warnings"].append("Missing validation split folder")

        if result["train_files"] == 0 and result["validation_files"] == 0:
            result["errors"].append("No image files found after extraction")

    result["ok"] = len(result["errors"]) == 0
    print(f"  Train files      : {result['train_files']:,}")
    print(f"  Validation files : {result['validation_files']:,}")
    _print_verify_summary(result)
    return result


# ---------------------------------------------------------------------------
# KITTI
# ---------------------------------------------------------------------------

_KITTI_RAW_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
_KITTI_DEPTH_GT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
_KITTI_DRIVE_RE     = re.compile(r"(\d{4}_\d{2}_\d{2})_drive_(\d{4})_sync")


def _detect_kitti_sequences(depth_dir: Path) -> List[Tuple[str, str]]:
    """Scan data_depth_annotated/{train,val} for sequence folders."""
    sequences: List[Tuple[str, str]] = []
    for split in ("train", "val"):
        split_dir = depth_dir / split
        if not split_dir.exists():
            continue
        for entry in sorted(split_dir.iterdir()):
            m = _KITTI_DRIVE_RE.match(entry.name)
            if m:
                sequences.append((m.group(1), m.group(2)))
    return sequences


def download_kitti(
    output_dir: "str | Path",
    skip_depth: bool = False,
    skip_raw: bool = False,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Download KITTI raw RGB + depth-annotated GT.

    Layout after download
    ---------------------
    output_dir/
      rgb/                             <- raw RGB per date/drive/image_02/data
        2011_09_26/
          2011_09_26_drive_0001_sync/
            image_02/data/*.png
      depth/
        data_depth_annotated/
          train/ val/

    Parameters
    ----------
    skip_depth : skip depth_annotated download (already extracted)
    skip_raw   : skip raw RGB download
    verify     : run integrity check after download

    Returns
    -------
    dict with keys: rgb_frames, depth_frames, errors, warnings, ok
    """
    output_dir = Path(output_dir)
    raw_dir    = output_dir / "rgb"
    depth_dir  = output_dir / "depth" / "data_depth_annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Depth annotated ──────────────────────────────────────────────
    if not skip_depth:
        if depth_dir.exists():
            print("[SKIP] data_depth_annotated already present")
        else:
            zip_path = output_dir / "data_depth_annotated.zip"
            print("[DL] data_depth_annotated.zip…")
            if _wget(_KITTI_DEPTH_GT_URL, zip_path):
                print("[EXTRACT] data_depth_annotated…")
                _unzip(zip_path, depth_dir)
                print("[OK] depth annotated ready")
            else:
                return {"ok": False,
                        "errors": ["data_depth_annotated.zip download failed"],
                        "warnings": [], "rgb_frames": 0, "depth_frames": 0}

    # ── Step 2: Detect sequences ─────────────────────────────────────────────
    sequences = _detect_kitti_sequences(depth_dir)
    if not sequences:
        return {"ok": False,
                "errors": ["No sequences found in data_depth_annotated"],
                "warnings": [], "rgb_frames": 0, "depth_frames": 0}
    print(f"[INFO] {len(sequences)} sequences found in data_depth_annotated")

    # ── Step 3: Raw RGB per sequence ─────────────────────────────────────────
    if not skip_raw:
        downloaded_calib: set = set()
        failed: List[str] = []
        for i, (date, drive) in enumerate(sequences, 1):
            drive_name = f"{date}_drive_{drive}_sync"
            print(f"[{i:02d}/{len(sequences)}] {drive_name}")

            # Calibration files (once per date)
            if date not in downloaded_calib:
                downloaded_calib.add(date)
                calib_txt = raw_dir / date / "calib_cam_to_cam.txt"
                if not calib_txt.exists():
                    calib_zip = raw_dir / f"{date}_calib.zip"
                    print(f"  [CALIB] {date}_calib.zip")
                    _wget(f"{_KITTI_RAW_BASE_URL}/{date}_calib.zip", calib_zip)
                    _unzip(calib_zip, raw_dir)
                else:
                    print(f"  [SKIP CALIB] {date}")

            seq_path = raw_dir / date / drive_name
            if seq_path.exists():
                print("  [SKIP] already extracted")
                continue

            url = f"{_KITTI_RAW_BASE_URL}/{date}_drive_{drive}/{drive_name}.zip"
            zip_local = raw_dir / f"{drive_name}.zip"
            if _wget(url, zip_local):
                _unzip(zip_local, raw_dir)
                print("  [OK]")
            else:
                print(f"  [ERROR] {drive_name}")
                failed.append(drive_name)

        if failed:
            print(f"\n[WARN] {len(failed)} sequences failed to download: {failed}")

    # ── Step 4: Verify ───────────────────────────────────────────────────────
    result: Dict[str, Any] = {
        "rgb_frames": 0, "depth_frames": 0,
        "errors": [], "warnings": [], "ok": False,
    }
    if not verify:
        result["ok"] = True
        return result

    print("\n[VERIFY] KITTI…")
    for date, drive in sequences:
        drive_name = f"{date}_drive_{drive}_sync"

        # Calibration
        calib = raw_dir / date / "calib_cam_to_cam.txt"
        if not calib.exists():
            result["errors"].append(f"Missing calib: {calib}")

        # RGB
        rgb_dir = raw_dir / date / drive_name / "image_02" / "data"
        if not rgb_dir.exists():
            result["errors"].append(f"Missing RGB dir: {rgb_dir}")
        else:
            n = len(list(rgb_dir.glob("*.png")))
            result["rgb_frames"] += n
            if n == 0:
                result["warnings"].append(f"Empty RGB dir: {rgb_dir}")

        # Depth GT
        found_depth = False
        for split in ("train", "val"):
            d = depth_dir / split / drive_name / "proj_depth" / "groundtruth" / "image_02"
            if d.exists():
                n = len(list(d.glob("*.png")))
                result["depth_frames"] += n
                found_depth = True
                if n == 0:
                    result["warnings"].append(f"Empty depth dir: {d}")
                break
        if not found_depth:
            result["errors"].append(f"Missing depth GT: {drive_name}")

    result["ok"] = len(result["errors"]) == 0
    print(f"  RGB frames   : {result['rgb_frames']:,}")
    print(f"  Depth frames : {result['depth_frames']:,}")
    _print_verify_summary(result)
    return result


# ---------------------------------------------------------------------------
# Shared verify printer
# ---------------------------------------------------------------------------

def _print_verify_summary(result: Dict[str, Any]) -> None:
    if result["warnings"]:
        print(f"  Warnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"    - {w}")
    if result["errors"]:
        print(f"  Errors ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"    - {e}")
    else:
        print("  ✓ Dataset OK")