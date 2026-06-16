"""
download_kitti.py
=================
Download KITTI (depth GT + RGB) for the METER + LeJEPA project.

Quick start
-----------
  # Full download (depth GT ~1.8 GB + RGB ~45 GB):
  python download_kitti.py --output_dir ./data/raw/kitti

  # Depth GT only:
  python download_kitti.py --output_dir ./data/raw/kitti --skip_rgb

  # RGB only (depth GT already present):
  python download_kitti.py --output_dir ./data/raw/kitti --skip_depth

  # Dry run (print URLs without downloading):
  python download_kitti.py --output_dir ./data/raw/kitti --dry_run

  # Use the hardcoded sequence list instead of auto-detection:
  python download_kitti.py --output_dir ./data/raw/kitti --use_hardcoded

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
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAW_BASE   = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
_DEPTH_URL  = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
_DRIVE_RE   = re.compile(r"(\d{4}_\d{2}_\d{2})_drive_(\d{4})_sync")

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
    ("2011_09_26", "0086"), ("2011_09_26", "0087"), ("2011_09_26", "0091"),
    ("2011_09_26", "0093"), ("2011_09_26", "0095"), ("2011_09_26", "0096"),
    ("2011_09_26", "0101"), ("2011_09_26", "0104"), ("2011_09_26", "0106"),
    ("2011_09_26", "0113"), ("2011_09_26", "0117"), ("2011_09_26", "0119"),
    ("2011_09_28", "0001"), ("2011_09_28", "0002"),
    ("2011_09_29", "0004"), ("2011_09_29", "0026"),
    ("2011_09_30", "0016"), ("2011_09_30", "0018"), ("2011_09_30", "0020"),
    ("2011_09_30", "0027"), ("2011_09_30", "0028"), ("2011_09_30", "0033"),
    ("2011_09_30", "0034"),
    ("2011_10_03", "0027"), ("2011_10_03", "0034"), ("2011_10_03", "0042"),
    ("2011_10_03", "0047"),
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, dry_run: bool = False,
              retries: int = 3, chunk_mb: int = 4) -> bool:
    """
    Download url to dest with retries and a text progress bar.
    Return True on success, False otherwise.
    """
    if dry_run:
        print(f"    [DRY-RUN] {url}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                chunk = chunk_mb * 1024 * 1024
                downloaded = 0
                with open(tmp, "wb") as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        if total:
                            pct = downloaded / total * 100
                            mb  = downloaded / 1024 / 1024
                            print(f"\r    {mb:7.1f} MB  ({pct:5.1f}%)", end="", flush=True)
            print()  # newline after progress
            tmp.rename(dest)
            return True

        except Exception as exc:
            print(f"\n    [WARN] attempt {attempt}/{retries}: {exc}")
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(3 * attempt)

    print(f"    [ERROR] download failed: {url}")
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
        ok = _unzip(zip_path, output_dir, dry_run)
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
    seq_path = rgb_dir / drive_name

    if (seq_path / "image_02" / "data").exists():
        print(f"  [SKIP] {drive_name}")
        return True

    n_png = len(list((seq_path / "image_02" / "data").rglob("*.png")))

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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download KITTI for METER + LeJEPA."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./data/raw/kitti"),
        help="Root folder (default: ./data/raw/kitti)"
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

    output_dir = args.output_dir
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
