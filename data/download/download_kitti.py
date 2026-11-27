"""
download_kitti.py
=================
Download KITTI (depth GT + RGB) for the METER + LeJEPA project.

Quick start
-----------
  # Full download (depth GT ~1.8 GB + RGB ~45 GB):
  python -m data.download.download_kitti --output_dir ./data/raw/kitti

  # Depth GT only:
  python -m data.download.download_kitti --output_dir ./data/raw/kitti --skip_rgb

  # RGB only (depth GT already present):
  python -m data.download.download_kitti --output_dir ./data/raw/kitti --skip_depth

  # Dry run (print URLs without downloading):
  python -m data.download.download_kitti --output_dir ./data/raw/kitti --dry_run

  # Use the hardcoded sequence list instead of auto-detection:
  python -m data.download.download_kitti --output_dir ./data/raw/kitti --use_hardcoded

Output layout
-------------
<output_dir>/
  rgb/
    2011_09_26/
      calib_cam_to_cam.txt
      2011_09_26_drive_0001_sync/
        image_02/data/*.png
      ...
  depth/
    data_depth_annotated/
      train/
        2011_09_26_drive_0001_sync/
          proj_depth/groundtruth/image_02/*.png
      val/
        ...

KITTI URL notes
---------------
The correct raw file format is:
  https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/
    {date}_drive_{drive}_sync/
      {date}_drive_{drive}_sync_image_02.zip

Calibration by date:
  https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/
    {date}_calib.zip
"""

import argparse
import hashlib
import re
import zipfile
from pathlib import Path
from typing import List, Set, Tuple

from data.download.nyu import download
from utils import paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAW_BASE   = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
_DEPTH_URL  = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
_DRIVE_RE   = re.compile(r"(\d{4}_\d{2}_\d{2})_drive_(\d{4})_sync")
_REPO_ROOT  = Path(__file__).resolve().parents[2]

# Exact 28 drives in the KITTI Eigen benchmark test split (sota/flexdepth/splits/eigen_benchmark/test_files.txt).
# These drives are strictly excluded from KITTI pre-training, validation, BN recalibration, and fine-tuning.
EIGEN_TEST_DRIVES = frozenset({
    "2011_09_26_drive_0002_sync", "2011_09_26_drive_0009_sync", "2011_09_26_drive_0013_sync",
    "2011_09_26_drive_0020_sync", "2011_09_26_drive_0023_sync", "2011_09_26_drive_0027_sync",
    "2011_09_26_drive_0029_sync", "2011_09_26_drive_0036_sync", "2011_09_26_drive_0046_sync",
    "2011_09_26_drive_0048_sync", "2011_09_26_drive_0052_sync", "2011_09_26_drive_0056_sync",
    "2011_09_26_drive_0059_sync", "2011_09_26_drive_0064_sync", "2011_09_26_drive_0084_sync",
    "2011_09_26_drive_0086_sync", "2011_09_26_drive_0093_sync", "2011_09_26_drive_0096_sync",
    "2011_09_26_drive_0101_sync", "2011_09_26_drive_0106_sync", "2011_09_26_drive_0117_sync",
    "2011_09_28_drive_0002_sync", "2011_09_29_drive_0071_sync", "2011_09_30_drive_0016_sync",
    "2011_09_30_drive_0018_sync", "2011_09_30_drive_0027_sync", "2011_10_03_drive_0027_sync",
    "2011_10_03_drive_0047_sync",
})

# Eigen split sequences with LiDAR GT (used as fallback)
KNOWN_EIGEN_SEQUENCES: List[Tuple[str, str]] = [
    ("2011_09_26", "0001"), ("2011_09_26", "0002"), ("2011_09_26", "0005"),
    ("2011_09_26", "0009"), ("2011_09_26", "0011"), ("2011_09_26", "0013"),
    ("2011_09_26", "0014"), ("2011_09_26", "0015"), ("2011_09_26", "0017"),
    ("2011_09_26", "0018"), ("2011_09_26", "0019"), ("2011_09_26", "0020"),
    ("2011_09_26", "0022"), ("2011_09_26", "0023"), ("2011_09_26", "0027"),
    ("2011_09_26", "0028"), ("2011_09_26", "0029"), ("2011_09_26", "0032"),
    ("2011_09_26", "0035"), ("2011_09_26", "0036"), ("2011_09_26", "0039"),
    ("2011_09_26", "0046"), ("2011_09_26", "0048"), ("2011_09_26", "0051"),
    ("2011_09_26", "0052"), ("2011_09_26", "0056"), ("2011_09_26", "0057"),
    ("2011_09_26", "0059"), ("2011_09_26", "0060"), ("2011_09_26", "0061"),
    ("2011_09_26", "0064"), ("2011_09_26", "0070"), ("2011_09_26", "0079"),
    ("2011_09_26", "0084"), ("2011_09_26", "0086"), ("2011_09_26", "0087"), ("2011_09_26", "0091"),
    ("2011_09_26", "0093"), ("2011_09_26", "0095"), ("2011_09_26", "0096"),
    ("2011_09_26", "0101"), ("2011_09_26", "0104"), ("2011_09_26", "0106"),
    ("2011_09_26", "0113"), ("2011_09_26", "0117"), ("2011_09_26", "0119"),
    ("2011_09_28", "0001"), ("2011_09_28", "0002"),
    ("2011_09_29", "0004"), ("2011_09_29", "0026"), ("2011_09_29", "0071"),
    ("2011_09_30", "0016"), ("2011_09_30", "0018"), ("2011_09_30", "0020"),
    ("2011_09_30", "0027"), ("2011_09_30", "0028"), ("2011_09_30", "0033"),
    ("2011_09_30", "0034"),
    ("2011_10_03", "0027"), ("2011_10_03", "0034"), ("2011_10_03", "0042"),
    ("2011_10_03", "0047"),
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, dry_run: bool = False) -> bool:
    """Resumable, size-checked download (``nyu.download``). Return True on success."""
    if dry_run:
        print(f"    [DRY-RUN] {url}")
        return True
    try:
        download(url, dest)
        return True
    except Exception as exc:  # noqa: BLE001 - report and go on with the next file
        print(f"    [ERROR] download failed: {url}: {exc}")
        return False


def _unzip(zip_path: Path, dest_dir: Path, dry_run: bool = False) -> bool:
    """Extract zip_path into dest_dir and remove the archive. Return True on success."""
    if dry_run:
        return True
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"    [EXTRACT] {zip_path.name} -> {dest_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        # zip_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        print(f"    [ERROR] unzip {zip_path.name}: {exc}")
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Sequence detection from extracted depth GT
# ---------------------------------------------------------------------------

def detect_sequences(depth_annotated_dir: Path) -> List[Tuple[str, str]]:
    """
    Auto-detect sequences from data_depth_annotated/{train,val}.
    Recursively look for directories whose names match _drive_NNNN_sync.
    """
    sequences = []
    seen: Set[str] = set()

    for split in ("train", "val"):
        split_dir = depth_annotated_dir / split
        if not split_dir.exists():
            continue
        for entry in sorted(split_dir.iterdir()):
            if not entry.is_dir():
                continue
            m = _DRIVE_RE.match(entry.name)
            if m and entry.name not in seen:
                seen.add(entry.name)
                sequences.append((m.group(1), m.group(2)))

    return sequences


# ---------------------------------------------------------------------------
# Depth GT
# ---------------------------------------------------------------------------

def download_depth_gt(output_dir: Path, dry_run: bool) -> Path:
    """
    Download and extract data_depth_annotated.zip (~1.8 GB).
    Return the path to data_depth_annotated/.
    """
    depth_root = output_dir / "depth"
    depth_dir  = depth_root / "data_depth_annotated"

    # Check whether it has already been extracted correctly
    if depth_dir.exists():
        train_dir = depth_dir / "train"
        val_dir   = depth_dir / "val"
        if (train_dir.exists() and any(train_dir.iterdir())) or \
           (val_dir.exists()   and any(val_dir.iterdir())):
            print("[SKIP] data_depth_annotated already exists and is populated")
            return depth_dir

    zip_path = output_dir / "data_depth_annotated.zip"

    print(f"\n[STEP] Download depth GT (~1.8 GB)")
    print(f"  URL : {_DEPTH_URL}")

    # Download only if it is not already present
    if not zip_path.exists():
        ok = _download(_DEPTH_URL, zip_path, dry_run)
        if not ok:
            print("  [ERROR] Depth GT download failed")
            return depth_dir
    else:
        print(f"  [SKIP] zip already exists: {zip_path}")

    # Extract
    if not dry_run:
        # shortcut: assumes the zip root holds train/ and val/ (not checked); if it is wrapped in a
        # data_depth_annotated/ folder, extract into depth_root instead. One path for download,
        # detect_sequences and verify: output_dir/depth/data_depth_annotated.
        ok = _unzip(zip_path, depth_dir, dry_run)
        if ok:
            print("  [OK] depth GT extracted")
        else:
            print("  [ERROR] Extraction failed")
    return depth_dir


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def download_calibration(date: str, rgb_dir: Path,
                         downloaded_calib: Set[str],
                         dry_run: bool) -> None:
    """
    Download the calibration package for *date* once.

    URL: {_RAW_BASE}/{date}_calib.zip
    Extracted into: rgb_dir/{date}/
    """
    if date in downloaded_calib:
        return

    # Already extracted?
    calib_file = rgb_dir / date / "calib_cam_to_cam.txt"
    if calib_file.exists():
        downloaded_calib.add(date)
        return

    zip_name  = f"{date}_calib.zip"
    url       = f"{_RAW_BASE}/{zip_name}"
    local_zip = rgb_dir / zip_name

    print(f"  [CALIB] {zip_name}")
    if _download(url, local_zip, dry_run):
        _unzip(local_zip, rgb_dir, dry_run)
        downloaded_calib.add(date)
    else:
        print(f"  [WARN] Calibration not downloaded: {zip_name}")


# ---------------------------------------------------------------------------
# RGB sequences
# ---------------------------------------------------------------------------

def download_sequence_rgb(date: str, drive: str,
                          rgb_dir: Path, dry_run: bool) -> bool:
    """
    Download and extract RGB frames (image_02) for a single sequence.

    Correct KITTI URL:
      {_RAW_BASE}/{date}_drive_{drive}_sync/
        {date}_drive_{drive}_sync_image_02.zip

    Extracted into: rgb_dir/{date}/
    """
    drive_name = f"{date}_drive_{drive}_sync"
    seq_path = rgb_dir / date / drive_name       # where the zip extracts (same path as verify())

    if (seq_path / "image_02" / "data").exists():
        print(f"  [SKIP] {drive_name}")
        return True

    zip_name = f"{drive_name}.zip"
    url = f"{_RAW_BASE}/{date}_drive_{drive}/{zip_name}"
    local_zip = rgb_dir / date / zip_name

    print(f"  [DL]   {zip_name}")
    print(f"         {url}")
    print(f"  [DEBUG] Trying to download from: {url}")

    if _download(url, local_zip, dry_run):
        ok = _unzip(local_zip, rgb_dir, dry_run)
        if ok:
            n_png = len(list((rgb_dir / date / drive_name).rglob("*.png")))
            print(f"  [OK]   {drive_name} ({n_png} frames)")
            return True
        return False

    print(f"  [WARN] Failed: {zip_name}")
    return False


def download_all_rgb(sequences: List[Tuple[str, str]],
                     rgb_dir: Path, dry_run: bool) -> List[str]:
    """
    Download RGB (image_02) for all sequences.
    Return the list of failed drive_name values.
    """
    rgb_dir.mkdir(parents=True, exist_ok=True)
    downloaded_calib: Set[str] = set()
    failed: List[str] = []

    print(f"\n[STEP] Download RGB - {len(sequences)} sequences")

    for i, (date, drive) in enumerate(sequences, 1):
        drive_name = f"{date}_drive_{drive}_sync"
        print(f"\n[{i:02d}/{len(sequences)}] {drive_name}")
        download_calibration(date, rgb_dir, downloaded_calib, dry_run)
        if not download_sequence_rgb(date, drive, rgb_dir, dry_run):
            failed.append(drive_name)

    return failed


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify(sequences: List[Tuple[str, str]],
           rgb_dir: Path, depth_dir: Path) -> dict:
    """Verify RGB and depth GT for each sequence."""
    print("\n[VERIFY] Checking KITTI integrity...")
    result = {
        "rgb_frames":   0,
        "depth_frames": 0,
        "ok_sequences": 0,
        "errors":       [],
        "warnings":     [],
    }

    for date, drive in sequences:
        drive_name = f"{date}_drive_{drive}_sync"
        seq_ok = True

        # Calibration
        calib = rgb_dir / date / "calib_cam_to_cam.txt"
        if not calib.exists():
            result["errors"].append(f"Missing calibration: {calib}")
            seq_ok = False

        # RGB
        rgb_data = rgb_dir / date / drive_name / "image_02" / "data"
        if not rgb_data.exists():
            result["errors"].append(f"Missing RGB: {rgb_data}")
            seq_ok = False
        else:
            n = len(list(rgb_data.glob("*.png")))
            result["rgb_frames"] += n
            if n == 0:
                result["warnings"].append(f"Empty RGB: {rgb_data}")

        # Depth GT
        found_depth = False
        for split in ("train", "val"):
            d = (depth_dir / split / drive_name
                 / "proj_depth" / "groundtruth" / "image_02")
            if d.exists():
                n = len(list(d.glob("*.png")))
                result["depth_frames"] += n
                found_depth = True
                if n == 0:
                    result["warnings"].append(f"Empty depth: {d}")
                break
        if not found_depth:
            result["errors"].append(f"Missing depth GT: {drive_name}")
            seq_ok = False

        if seq_ok:
            result["ok_sequences"] += 1

    total = len(sequences)
    print(f"  OK sequences : {result['ok_sequences']}/{total}")
    print(f"  RGB frames   : {result['rgb_frames']:,}")
    print(f"  Depth frames : {result['depth_frames']:,}")

    for w in result["warnings"]:
        print(f"  [WARN]  {w}")
    for e in result["errors"][:20]:   # show up to 20 errors
        print(f"  [ERROR] {e}")
    if len(result["errors"]) > 20:
        print(f"  ... and {len(result['errors']) - 20} more errors")

    result["ok"] = len(result["errors"]) == 0
    if result["ok"]:
        print("  KITTI complete and verified")
    return result


# ---------------------------------------------------------------------------
# Split generation (anti-leak)
# ---------------------------------------------------------------------------

def is_val_drive(drive_name: str, fraction: float = 0.10) -> bool:
    """Deterministic hash-based holdout so validation drives never depend on what is on disk."""
    h = int.from_bytes(hashlib.sha256(drive_name.encode()).digest()[:8], "big")
    return (h % 10_000) < round(fraction * 10_000)


def _find_depth_file(depth_dir: Path, drive_name: str, frame_name: str) -> Path | None:
    for split in ("train", "val"):
        p = depth_dir / split / drive_name / "proj_depth" / "groundtruth" / "image_02" / frame_name
        if p.is_file():
            return p
    return None


def make_splits(data_root: Path, kitti_rel: str = "kitti", eigen_test_ref: Path | None = None) -> dict[str, int]:
    """Generate leak-free KITTI split lists under ``<data_root>/splits/``."""
    data_root = Path(data_root)
    kitti_dir = data_root / kitti_rel
    rgb_dir = kitti_dir / "rgb"
    depth_dir = kitti_dir / "depth" / "data_depth_annotated"
    splits_dir = data_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    ref = eigen_test_ref or (_REPO_ROOT / "sota" / "flexdepth" / "splits" / "eigen_benchmark" / "test_files.txt")

    drive_frames: dict[str, list[Path]] = {}
    if rgb_dir.is_dir():
        for date_dir in sorted(rgb_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for drive_dir in sorted(date_dir.iterdir()):
                if not drive_dir.is_dir() or not _DRIVE_RE.match(drive_dir.name):
                    continue
                if drive_dir.name in EIGEN_TEST_DRIVES:
                    continue
                img_dir = drive_dir / "image_02" / "data"
                if img_dir.is_dir():
                    frames = sorted(img_dir.glob("*.png"))
                    if frames:
                        drive_frames[drive_dir.name] = frames

    val_drives = sorted(d for d in drive_frames if is_val_drive(d))
    train_drives = sorted(d for d in drive_frames if d not in set(val_drives))

    raw_frames: list[str] = []
    train_triplets: list[str] = []
    val_triplets: list[str] = []
    val_pairs: list[str] = []

    for d in train_drives:
        frames = drive_frames[d]
        raw_frames.extend(str(f.relative_to(data_root)) for f in frames)
        for i in range(1, len(frames) - 1):
            if int(frames[i].stem) - int(frames[i - 1].stem) == 1 and int(frames[i + 1].stem) - int(frames[i].stem) == 1:
                train_triplets.append(
                    f"{frames[i - 1].relative_to(data_root)} {frames[i].relative_to(data_root)} {frames[i + 1].relative_to(data_root)}"
                )

    for d in val_drives:
        frames = drive_frames[d]
        for i in range(1, len(frames) - 1):
            if int(frames[i].stem) - int(frames[i - 1].stem) == 1 and int(frames[i + 1].stem) - int(frames[i].stem) == 1:
                val_triplets.append(
                    f"{frames[i - 1].relative_to(data_root)} {frames[i].relative_to(data_root)} {frames[i + 1].relative_to(data_root)}"
                )
        for f in frames:
            dp = _find_depth_file(depth_dir, d, f.name)
            if dp is not None:
                val_pairs.append(f"{f.relative_to(data_root)} {dp.relative_to(data_root)}")

    test_pairs: list[str] = []
    if ref.is_file():
        for ln in ref.read_text().splitlines():
            parts = ln.split()
            if len(parts) < 2:
                continue
            drive_rel, idx = parts[0], int(parts[1])
            drive_name = Path(drive_rel).name
            fname = f"{idx:010d}.png"
            rgb_p = rgb_dir / drive_rel / "image_02" / "data" / fname
            dp = _find_depth_file(depth_dir, drive_name, fname)
            if rgb_p.is_file() and dp is not None:
                test_pairs.append(f"{rgb_p.relative_to(data_root)} {dp.relative_to(data_root)}")

    files = {
        "kitti_test_drives.txt": sorted(EIGEN_TEST_DRIVES),
        "kitti_val_drives.txt": val_drives,
        "kitti_raw_frames.txt": raw_frames,
        "kitti_raw_triplets_train.txt": train_triplets,
        "kitti_raw_triplets_val.txt": val_triplets,
        "kitti_labeled_val.txt": val_pairs,
        "kitti_eigen_test.txt": test_pairs,
    }
    for name, lines in files.items():
        (splits_dir / name).write_text("\n".join(lines) + ("\n" if lines else ""))

    counts = {k: len(v) for k, v in files.items()}
    print(f"[splits] wrote KITTI splits to {splits_dir}: {counts}")
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download KITTI for METER + LeJEPA."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Root folder (default: <LG_DATA_ROOT>/kitti)"
    )
    parser.add_argument(
        "--data_root", type=Path, default=None,
        help="Data root for --make_splits (default: LG_DATA_ROOT)"
    )
    parser.add_argument(
        "--make_splits", action="store_true",
        help="Generate leak-free train/val/test split lists under <data_root>/splits/ and exit"
    )
    parser.add_argument(
        "--skip_depth", action="store_true",
        help="Skip depth GT download"
    )
    parser.add_argument(
        "--skip_rgb", action="store_true",
        help="Skip RGB download"
    )
    parser.add_argument(
        "--use_hardcoded", action="store_true",
        help="Use the hardcoded sequence list (ignore auto-detect)"
    )
    parser.add_argument(
        "--no_verify", action="store_true",
        help="Skip final verification"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print URLs without downloading"
    )
    args = parser.parse_args()

    if args.make_splits:
        make_splits(args.data_root or paths.data_root())
        return

    output_dir = args.output_dir or (paths.data_root() / "kitti")
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir   = output_dir / "rgb"
    depth_dir = output_dir / "depth" / "data_depth_annotated"

    print("=" * 60)
    print("  KITTI Downloader - METER + LeJEPA")
    print("=" * 60)
    print(f"  Output dir : {output_dir.resolve()}")
    print(f"  Skip depth : {args.skip_depth}")
    print(f"  Skip RGB   : {args.skip_rgb}")
    print(f"  Dry run    : {args.dry_run}")
    print("=" * 60)

    # -- Depth GT ----------------------------------------------------------------
    if not args.skip_depth:
        download_depth_gt(output_dir, args.dry_run)

    # -- Sequences ---------------------------------------------------------------
    if args.use_hardcoded:
        sequences = KNOWN_EIGEN_SEQUENCES
        print(f"\n[INFO] Hardcoded list: {len(sequences)} sequences")
    else:
        sequences = detect_sequences(depth_dir)
        if sequences:
            print(f"\n[INFO] Auto-detect: {len(sequences)} sequences found")
        else:
            print("\n[WARN] Auto-detect failed; using hardcoded list")
            sequences = KNOWN_EIGEN_SEQUENCES

    # -- RGB ---------------------------------------------------------------------
    if not args.skip_rgb:
        failed = download_all_rgb(sequences, rgb_dir, args.dry_run)
        if failed:
            print(f"\n[WARN] {len(failed)} RGB sequences failed:")
            for s in failed:
                print(f"  - {s}")
        else:
            print("\n[OK] All RGB sequences downloaded")

    # -- Verify ------------------------------------------------------------------
    if not args.no_verify and not args.dry_run:
        verify(sequences, rgb_dir, depth_dir)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
