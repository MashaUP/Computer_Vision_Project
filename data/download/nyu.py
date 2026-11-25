"""Download and prepare the complete NYU Depth V2 dataset (run on the lab machine, CPU only).

    python -m data.download.nyu download          # labeled .mat + official splits + all 37 raw parts (~431 GB)
    python -m data.download.nyu extract-labeled   # 1449 labelled frames -> PNGs + train (795) / test (654) lists
    python -m data.download.nyu extract-raw       # unzip the raw parts (add --delete-zip to free space)
    python -m data.download.nyu frames            # RGB frame list for LeJEPA, test + validation scenes excluded
    python -m data.download.nyu triplets          # validation scenes + frame triplets (FlexDepth self-supervised training)

Everything goes under ``<data root>/nyu`` (``LG_DATA_ROOT``, ``/mnt/hdd2/<user>/data`` on the lab):

    nyu/downloads/   nyu_depth_v2_labeled.mat, splits.mat, <part>.zip     (resumable, size-checked)
    nyu/raw/         <scene>/r-*.ppm (RGB), d-*.pgm (raw depth), a-*.dump, INDEX.txt
    nyu/labeled/     {train,test}/<index>_{rgb,depth}.png   (depth: uint16 millimetres)
    ../splits/       nyu_labeled_{train,test}.txt, nyu_test_scenes.txt, nyu_raw_frames_stride<k>.txt,
                     nyu_raw_val_scenes.txt, nyu_raw_triplets_{train,val}.txt

Downloads resume where they stopped (HTTP range requests; the NYU server supports them) and are
checked against the size the server reports, so the command can simply be re-run after an
interruption. Launch it in a CPU container on the lab machine:

    scripts/alcor/run.sh cpu nyu_download python3 -m data.download.nyu download
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import paths  # noqa: E402

BASE = "http://horatio.cs.nyu.edu/mit/silberman"
LABELED = f"{BASE}/nyu_depth_v2/nyu_depth_v2_labeled.mat"
SPLITS = f"{BASE}/indoor_seg_sup/splits.mat"
RAW_PARTS = [
    "basements", "bedrooms_part1", "bedrooms_part2", "bedrooms_part3", "bedrooms_part4", "bedrooms_part5",
    "bedrooms_part6", "bedrooms_part7", "home_offices", "misc_part1", "misc_part2", "bathrooms_part1",
    "bathrooms_part2", "bathrooms_part3", "bathrooms_part4", "bookstore_part1", "bookstore_part2",
    "bookstore_part3", "kitchens_part1", "kitchens_part2", "kitchens_part3", "office_kitchens", "offices_part1",
    "offices_part2", "libraries", "playrooms", "cafe", "living_rooms_part1", "living_rooms_part2",
    "living_rooms_part3", "living_rooms_part4", "reception_rooms", "studies", "dining_rooms_part1",
    "dining_rooms_part2", "study_rooms", "furniture_stores",
]
RAW_URLS = [f"{BASE}/nyu_depth_v2/{p}.zip" for p in RAW_PARTS]
CHUNK = 4 << 20
VAL_SCENE_FRACTION = 0.05   # share of the non-test scenes held out as the protocol's validation set (by id hash)
TRIPLET_STEP = 10           # raw frames between source and target: MonoIndoor subsamples the raw video 10x in time
MAX_GAP_S = 1.0             # drop triplets with a longer gap: at 20-30 FPS a 10-frame step is 0.33-0.5 s (our threshold)


def nyu_root() -> Path:
    return paths.data_root() / "nyu"


def splits_dir() -> Path:
    return paths.data_root() / "splits"


# --------------------------------------------------------------------------- download
def remote_size(url: str, timeout: float = 60) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers["Content-Length"])


def download(url: str, dest: Path, retries: int = 20, timeout: float = 120) -> Path:
    """Download ``url`` to ``dest``, resuming a partial ``dest.part``; verify the final size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url, timeout)
    if dest.exists() and dest.stat().st_size == total:
        print(f"[ok]   {dest.name} ({total / 2**30:.2f} GiB) already complete")
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        done = part.stat().st_size if part.exists() else 0
        if done > total:
            part.unlink()
            done = 0
        if done == total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, part.open("ab" if done else "wb") as f:
                if done and r.status != 206:      # server ignored the range: restart from scratch
                    f.truncate(0)
                    done = 0
                t0, last = time.time(), time.time()
                while chunk := r.read(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 30:
                        rate = done / max(time.time() - t0, 1e-6) / 2**20
                        print(f"[..]   {dest.name}: {done / 2**30:.2f}/{total / 2**30:.2f} GiB ({rate:.1f} MiB/s)", flush=True)
                        last = time.time()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            wait = min(60, 5 * attempt)
            print(f"[warn] {dest.name}: {type(e).__name__}: {e} (attempt {attempt}/{retries}, retry in {wait}s)", flush=True)
            time.sleep(wait)
            continue
        if done == total:
            break
    if not part.exists() or part.stat().st_size != total:
        raise RuntimeError(f"{url}: incomplete after {retries} attempts")
    part.replace(dest)
    print(f"[done] {dest.name} ({total / 2**30:.2f} GiB)", flush=True)
    return dest


def cmd_download(args) -> int:
    out = nyu_root() / "downloads"
    urls = []
    if "labeled" in args.what:
        urls += [LABELED, SPLITS]
    if "raw" in args.what:
        urls += RAW_URLS
    failed = []
    for url in urls:
        try:
            download(url, out / url.rsplit("/", 1)[1])
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            print(f"[fail] {url}: {e}", flush=True)
            failed.append(url)
    print(f"{len(urls) - len(failed)}/{len(urls)} files complete in {out}")
    for url in failed:
        print(f"  missing: {url}")
    return 1 if failed else 0


def cmd_sizes(_) -> int:
    total = 0
    for url in [LABELED, SPLITS, *RAW_URLS]:
        n = remote_size(url)
        total += n
        print(f"{n / 2**30:8.2f} GiB  {url}")
    print(f"{total / 2**30:8.2f} GiB  total")
    return 0


# --------------------------------------------------------------------------- labelled frames
def scene_id(name: str) -> str:
    """``bedroom_0001a`` / ``bedroom_0001`` -> ``bedroom_0001`` (raw folders carry a letter suffix)."""
    m = re.match(r"^(.*_\d{4})[a-z]?$", name)
    return m.group(1) if m else name


def read_labeled(mat_path: Path):
    """Yield ``(index, rgb uint8 HxWx3, depth float32 HxW in metres, scene)`` from the labelled .mat."""
    import h5py
    import numpy as np

    with h5py.File(mat_path, "r") as f:
        images, depths, scenes = f["images"], f["depths"], f["scenes"]
        for i in range(images.shape[0]):
            rgb = np.ascontiguousarray(np.asarray(images[i]).transpose(2, 1, 0))      # (3, W, H) -> (H, W, 3)
            depth = np.ascontiguousarray(np.asarray(depths[i]).T)                     # (W, H) -> (H, W)
            name = "".join(chr(c) for c in np.asarray(f[scenes[0][i]]).flatten())
            yield i, rgb, depth.astype("float32"), name


def cmd_extract_labeled(_) -> int:
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat

    dl, out = nyu_root() / "downloads", nyu_root() / "labeled"
    split = loadmat(dl / "splits.mat")
    test_idx = set(int(i) - 1 for i in np.asarray(split["testNdxs"]).flatten())   # MATLAB indices are 1-based
    lists = {"train": [], "test": []}
    test_scenes = set()
    for i, rgb, depth, scene in read_labeled(dl / "nyu_depth_v2_labeled.mat"):
        part = "test" if i in test_idx else "train"
        d = out / part
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(d / f"{i:04d}_rgb.png")
        Image.fromarray(np.clip(depth * 1000.0, 0, 65535).round().astype(np.uint16)).save(d / f"{i:04d}_depth.png")
        lists[part].append(f"nyu/labeled/{part}/{i:04d}_rgb.png nyu/labeled/{part}/{i:04d}_depth.png {scene}")
        if part == "test":
            test_scenes.add(scene_id(scene))
    shared = {scene_id(ln.split()[2]) for ln in lists["train"]} & test_scenes
    if shared:
        raise SystemExit(f"labelled train and test share scenes: {sorted(shared)}")
    splits_dir().mkdir(parents=True, exist_ok=True)
    for part, lines in lists.items():
        (splits_dir() / f"nyu_labeled_{part}.txt").write_text("\n".join(lines) + "\n")
    (splits_dir() / "nyu_test_scenes.txt").write_text("\n".join(sorted(test_scenes)) + "\n")
    print(f"train {len(lists['train'])}, test {len(lists['test'])} frames; {len(test_scenes)} test scenes")
    return 0 if (len(lists["train"]), len(lists["test"])) == (795, 654) else 1


# --------------------------------------------------------------------------- raw sequences
def cmd_extract_raw(args) -> int:
    dl, out = nyu_root() / "downloads", nyu_root() / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name in RAW_PARTS:
        z, marker = dl / f"{name}.zip", out / f".{name}.extracted"
        if marker.exists():
            print(f"[ok]   {name} already extracted")
            continue
        if not z.exists():
            print(f"[skip] {z.name} not downloaded")
            continue
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        marker.touch()
        print(f"[done] {name}", flush=True)
        if args.delete_zip:
            z.unlink()
    return 0


def is_val_scene(sid: str) -> bool:
    """Validation scene? Decided by the scene id alone (hash), so it never depends on which other scenes are on disk."""
    return int(hashlib.sha1(sid.encode()).hexdigest(), 16) % 100 < 100 * VAL_SCENE_FRACTION


def raw_scenes():
    """``(raw scene folders, test scene ids, validation scene ids)``; validation = the non-test scenes with
    ``is_val_scene``. Shared by ``frames`` and ``triplets``; also written to ``nyu_raw_val_scenes.txt``."""
    test_file = splits_dir() / "nyu_test_scenes.txt"
    if not test_file.exists():
        raise SystemExit("run extract-labeled first: it writes the test scenes to exclude")
    test = set(test_file.read_text().split())
    scenes = sorted(p for p in (nyu_root() / "raw").rglob("*") if p.is_dir() and any(p.glob("r-*.ppm")))
    val = {i for i in {scene_id(p.name) for p in scenes} - test if is_val_scene(i)}
    if not val:
        raise SystemExit("no scene falls in the validation set: the raw download looks incomplete")
    (splits_dir() / "nyu_raw_val_scenes.txt").write_text("\n".join(sorted(val)) + "\n")
    return scenes, test, val


def cmd_frames(args) -> int:
    """List RGB frames of the raw sequences, every ``stride``-th per scene, test and validation scenes excluded."""
    scenes, test, val = raw_scenes()
    lines, kept, dropped = [], 0, 0
    for scene in scenes:
        if scene_id(scene.name) in test | val:
            dropped += 1
            continue
        kept += 1
        frames = sorted(scene.glob("r-*.ppm"))[::args.stride]
        lines += [str(f.relative_to(paths.data_root())) for f in frames]
    out = splits_dir() / f"nyu_raw_frames_stride{args.stride}.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} frames from {kept} scenes ({dropped} test/val scenes excluded) -> {out}")
    return 0


def frame_time(path) -> float:
    """Timestamp (seconds) of ``r-<seconds.fraction>-<counter>.ppm``."""
    return float(re.match(r"r-(\d+(?:\.\d+)?)", Path(path).name).group(1))


def cmd_triplets(args) -> int:
    """Validation scenes + (previous, target, next) frame triplets for self-supervised training.

    Targets and sources are ``--step`` frames apart (the raw video subsampled in time) and never cross
    a scene or a timestamp jump. Test scenes are excluded from everything; ``VAL_SCENE_FRACTION`` of the
    remaining scenes (by id, seed 0) go to ``nyu_raw_triplets_val.txt``, the rest to ``..._train.txt``.
    """
    scenes, test, val = raw_scenes()
    lines = {"train": [], "val": []}
    for scene in scenes:
        sid = scene_id(scene.name)
        if sid in test:
            continue
        frames = sorted(scene.glob("r-*.ppm"), key=frame_time)
        t = [frame_time(f) for f in frames]
        rel = [str(f.relative_to(paths.data_root())) for f in frames]
        for i in range(args.step, len(frames) - args.step, args.step):
            if t[i] - t[i - args.step] <= args.max_gap and t[i + args.step] - t[i] <= args.max_gap:
                lines["val" if sid in val else "train"].append(f"{rel[i - args.step]} {rel[i]} {rel[i + args.step]}")
    splits_dir().mkdir(parents=True, exist_ok=True)
    for part, ls in lines.items():
        (splits_dir() / f"nyu_raw_triplets_{part}.txt").write_text("\n".join(ls) + ("\n" if ls else ""))
    print(f"{len(lines['train'])} train / {len(lines['val'])} val triplets; {len(val)} scenes held out for validation")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download")
    p.add_argument("--what", nargs="+", default=["labeled", "raw"], choices=["labeled", "raw"])
    p.set_defaults(func=cmd_download)
    sub.add_parser("sizes", help="print the size of every file on the server").set_defaults(func=cmd_sizes)
    sub.add_parser("extract-labeled").set_defaults(func=cmd_extract_labeled)
    p = sub.add_parser("extract-raw")
    p.add_argument("--delete-zip", action="store_true", help="delete each zip once extracted")
    p.set_defaults(func=cmd_extract_raw)
    p = sub.add_parser("frames")
    p.add_argument("--stride", type=int, default=10)
    p.set_defaults(func=cmd_frames)
    p = sub.add_parser("triplets", help="validation scenes and frame triplets for self-supervised training")
    p.add_argument("--step", type=int, default=TRIPLET_STEP)
    p.add_argument("--max-gap", type=float, default=MAX_GAP_S, help="seconds")
    p.set_defaults(func=cmd_triplets)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
