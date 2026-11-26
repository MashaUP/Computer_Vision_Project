"""Train FlexDepth on NYU Depth V2 raw with the ORIGINAL FlexDepth trainer (step 1 of the protocol).

    python scripts/train_flexdepth_nyu.py --make-png [--workers 6]     # once, before training: rectified PNG cache
    python scripts/train_flexdepth_nyu.py --name flexdepth_n_nyu [--scale n] [--seed 0] [-- <extra flexdepth options>]

FlexDepth is self-supervised (Monodepth2 photometric loss on (previous, target, next) frames, two stages);
the paper has no NYU recipe, so the KITTI recipe of ``sota/flexdepth/docs/train.md`` is applied to NYU, with the
indoor choices of the NYU self-supervised literature:

    value                      source
    -------------------------  ---------------------------------------------------------------------------
    NAdam, lr 1e-4, bs 12,     sota/flexdepth/docs/train.md (Flex-Nano, KITTI); epochs/schedulers: paper sec. 4.2
    30+20 epochs, StepLR 15    (arXiv 2607.00736); applied unchanged to NYU
    288x384 input              StructDepth / P2Net ``options.py`` defaults (height 288, width 384)
    depth range 0.1-10 m       StructDepth / P2Net ``options.py``
    frame_ids 0 -1 1           P2Net README (first stage: ``--frame_ids_to_train 0 -1 1``)
    raw video subsampled 10x   MonoIndoor (arXiv 2107.12429); triplet lists made by ``data.download.nyu triplets``
    undistort + 16 px crop     StructDepth / P2Net ``datasets/nyu_dataset.py`` (``cv2.undistort``, ``CROP = 16``)
    intrinsics, distortion     NYU toolbox ``camera_params.m`` values as used in StructDepth ``nyu_dataset.py``
                               (fx_rgb 518.858 matches the toolbox; the file itself was not read)
    COCO-pretrained YOLO11     sota/flexdepth/docs/train.md, "Prerequisites"

Nothing under ``sota/`` is modified. The original ``Trainer`` is imported and only its dataset hook is replaced
(``NYUTrainer.init_dataset``); the working directory is a cache folder holding ``cfg`` (symlink) and ``ckpt/`` because
FlexDepth reads both relative to the current directory. Outputs go to ``<LG_RUNS_ROOT>/<name>/mono_model/`` and the run
resumes automatically (``--resume``). Training is fp32 (no ``--amp``).

shortcut: single process only (no DDP); weight init of the original trainer uses ``init_seeds(1 + rank + seed)``.
shortcut: horizontal flip stays on as in the original dataset code although the NYU principal point is slightly
off-centre (cx 0.509, cy 0.528 after the crop), like StructDepth / P2Net.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets import check_list, read_list  # noqa: E402
from data.download.nyu import download  # noqa: E402
from utils import paths  # noqa: E402

os.environ.setdefault("WANDB_MODE", "disabled")                  # the trainer imports wandb; it is only used with --use_wb
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # FlexDepth torch.load()s a pickled ultralytics model

HEIGHT, WIDTH = 288, 384
CROP = 16                                                        # border pixels cut after undistortion
FULL_W, FULL_H = 640, 480
# NYU toolbox camera_params.m, RGB camera (values as in StructDepth datasets/nyu_dataset.py)
FX, FY, CX, CY = 5.1885790117450188e+02, 5.1946961112127485e+02, 3.2558244941119034e+02, 2.5373616633400465e+02
DIST = (2.0796615318809061e-01, -5.8613825163911781e-01, 7.2231363135888329e-04, 1.0479627195765181e-03,
        4.9856986684705107e-01)                                    # k1 k2 p1 p2 k3
CAM = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{scale}-seg.pt"   # n: 6 182 636 bytes
FLEXDEPTH_DIR = REPO_ROOT / "sota" / "flexdepth"

# options of the "Two-stage training in one command" recipe of docs/train.md (Flex-Nano), adapted to NYU
RECIPE = ["--use_var_net", "--use_step_2", "--num_epochs", "30", "--start_opt_epoch", "29", "--step_2_epoch", "20",
          "--resume", "--scales", "4", "--optim", "NAdam", "--learning_rate", "1e-4", "--batch_size", "12",
          "--height", str(HEIGHT), "--width", str(WIDTH), "--min_depth", "0.1", "--max_depth", "10",
          "--frame_ids", "0", "-1", "1", "--dy_mu"]


def rectify(ppm_path) -> np.ndarray:
    """RGB uint8 (448, 608, 3): the raw frame undistorted with the NYU intrinsics, borders cropped."""
    rgb = cv2.undistort(np.asarray(Image.open(ppm_path).convert("RGB")), CAM, np.array(DIST))
    return np.ascontiguousarray(rgb[CROP:FULL_H - CROP, CROP:FULL_W - CROP])


# shortcut: the PNGs do not record CAM / DIST / CROP; if those change, delete <data root>/nyu/rect/ and regenerate.
def png_path(data_root, rel) -> Path:
    """``nyu/raw/<scene>/r-*.ppm`` -> ``<data root>/nyu/rect/<scene>/r-*.png`` (the rectified, cropped frame)."""
    return (Path(data_root) / "nyu" / "rect" / Path(rel).relative_to("nyu/raw")).with_suffix(".png")


def _convert(job) -> None:
    data_root, rel = job
    dest = png_path(data_root, rel)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.stem + ".tmp.png")
        Image.fromarray(rectify(Path(data_root) / rel)).save(tmp, format="PNG")
        os.replace(tmp, dest)                          # never a half-written PNG
    except Exception as exc:
        raise RuntimeError(f"{rel}: {exc}") from exc


def make_png(lists, data_root, workers: int = 6) -> int:
    """Write the rectified PNG of every frame of ``lists`` (each frame once, existing PNGs skipped). Return the count."""
    rels = sorted({p for lst in lists for line in read_list(lst) for p in line})
    todo = [(str(data_root), r) for r in rels if not png_path(data_root, r).exists()]
    print(f"[make-png] {len(todo)} to write, {len(rels) - len(todo)} already done")
    if workers <= 1:
        for job in todo:
            _convert(job)
        return len(todo)
    with ProcessPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(_convert, todo, chunksize=16), 1):
            if i % 2000 == 0:
                print(f"[make-png] {i}/{len(todo)}", flush=True)
    return len(todo)


def prepare_workdir(scale: str, pretrained: bool) -> Path:
    """Directory used as cwd: FlexDepth opens ``./cfg/...`` and ``./ckpt/yolo11<scale>-seg.pt`` relative to it."""
    work = paths.cache_root() / "flexdepth_workdir"
    (work / "ckpt").mkdir(parents=True, exist_ok=True)
    cfg = work / "cfg"
    if not cfg.exists():
        cfg.symlink_to(FLEXDEPTH_DIR / "cfg")
    weights = work / "ckpt" / f"yolo11{scale}-seg.pt"
    if pretrained and not weights.exists():
        download(WEIGHTS_URL.format(scale=scale), weights)
    return work


@contextlib.contextmanager
def build(argv: list[str]):
    """Import the original FlexDepth trainer; yields ``(trainer module, NYUTrainer class, parsed options)``.

    FlexDepth imports its own modules lazily while building models, so the isolated import stays open for the
    whole life of the trainer: construct and run it inside the ``with`` block."""
    from sota.loader import isolated_import

    with isolated_import("flexdepth"):
        import torch
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.LANCZOS   # MonoDataset prints it; removed in Pillow 10 (FlexDepth pins 9.5, no py3.12 wheel)
        import trainer as flex_trainer
        from datasets.mono_dataset import MonoDataset
        from options import FlexdepthOptions

        old_argv, sys.argv = sys.argv, ["train.py", *argv]
        try:
            opts = FlexdepthOptions().parse()
        finally:
            sys.argv = old_argv

        class NYURawTriplets(MonoDataset):
            """NYU raw frames as (previous, target, next) triplets; ``frame_ids`` must be ``0 -1 1``."""

            def __init__(self, data_path, filenames, *args, **kwargs):
                super().__init__(data_path, [ln.split() for ln in filenames], *args, **kwargs)
                assert set(self.frame_idxs) == {0, -1, 1}, "NYU triplets only provide frame_ids 0 -1 1"
                w, h = FULL_W - 2 * CROP, FULL_H - 2 * CROP
                # normalised by the cropped size (MonoDataset multiplies by the network input size)
                self.K = np.array([[FX / w, 0, (CX - CROP) / w, 0], [0, FY / h, (CY - CROP) / h, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

            def check_depth(self):
                return False

            def index_to_folder_and_frame_idx(self, index):
                return index, 0, None            # get_color receives (triplet index, -1 | 0 | 1)

            def get_color(self, folder, frame_index, side, do_flip):
                rel = self.filenames[folder][frame_index + 1]
                png = png_path(self.data_path, rel)
                rgb = np.asarray(Image.open(png).convert("RGB")) if png.exists() else rectify(Path(self.data_path) / rel)
                if do_flip:
                    rgb = rgb[:, ::-1]
                return Image.fromarray(np.ascontiguousarray(rgb))

        class NYUTrainer(flex_trainer.Trainer):
            def __init__(self, options, train_list, val_list):
                self.train_list, self.val_list = train_list, val_list
                super().__init__(options)

            def init_dataset(self):
                # Trimmed copy of Trainer.init_dataset (sota/flexdepth/trainer.py:254-340): it hard-codes the dataset
                # classes and reads the split files from sota/flexdepth/splits. Single process, no DDP / prefetcher.
                assert self.rank == -1, "single-process training only"
                read = lambda p: [ln for ln in Path(p).read_text().splitlines() if ln.strip()]
                train_files, val_files = read(self.train_list), read(self.val_list)
                o = self.opt
                epochs = o.num_epochs + (o.step_2_epoch if o.use_step_2 else 0)
                self.num_total_steps = len(train_files) // o.batch_size * (o.step_2_epoch if o.use_step_2 and o.skip_step1
                                                                           else epochs)
                workers = min(os.cpu_count() or 1, o.batch_size if o.batch_size > 1 else 0, o.num_workers)
                self.train_dataset = NYURawTriplets(o.data_path, train_files, o.height, o.width, o.frame_ids, 4,
                                                    is_train=True, img_ext=".ppm")
                val_dataset = NYURawTriplets(o.data_path, val_files, o.height, o.width, o.frame_ids, 4,
                                             is_train=False, img_ext=".ppm")
                self.use_dataprefetcher = False
                self.train_loader = torch.utils.data.DataLoader(
                    self.train_dataset, o.batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
                self.val_loader = torch.utils.data.DataLoader(
                    val_dataset, o.batch_size, True, num_workers=workers, pin_memory=True, drop_last=True)
                self.val_iter = iter(self.val_loader)
                print(f"There are {len(self.train_dataset)} training items and {len(val_dataset)} validation items\n")

        yield flex_trainer, NYUTrainer, opts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="run name (required for training): outputs go to <LG_RUNS_ROOT>/<name>/mono_model/")
    ap.add_argument("--make-png", action="store_true", help="write the rectified PNG cache of the train/val lists, then exit")
    ap.add_argument("--workers", type=int, default=6, help="--make-png processes")
    ap.add_argument("--scale", default="n", choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-list", default=None, help="default <data root>/splits/nyu_raw_triplets_train.txt")
    ap.add_argument("--val-list", default=None, help="default <data root>/splits/nyu_raw_triplets_val.txt")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after `--`: further FlexDepth options (they win)")
    args = ap.parse_args(argv)
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    splits = paths.data_root() / "splits"
    train_list = args.train_list or splits / "nyu_raw_triplets_train.txt"
    val_list = args.val_list or splits / "nyu_raw_triplets_val.txt"
    if args.make_png:
        make_png([train_list, val_list], paths.data_root(), args.workers)
        return 0
    if not args.name:
        ap.error("--name is required for training")
    check_list(train_list, "nyu", "train")
    check_list(val_list, "nyu", "val")
    scratch = "--weights_init" in extra and extra[extra.index("--weights_init") + 1] == "scratch"
    work = prepare_workdir(args.scale, pretrained=not scratch)
    os.chdir(work)

    flex_argv = [*RECIPE, "--encoder_model_type", f"yolo11{args.scale}-seg", "--decoder_model_type", f"flex{args.scale}",
                 "--data_path", str(paths.data_root()), "--log_dir", str(paths.runs_root() / args.name),
                 "--model_name", "mono_model", *extra]
    with build(flex_argv) as (flex_trainer, NYUTrainer, opts):
        original_init_seeds = flex_trainer.init_seeds
        # the original trainer fixes its seed (1 + rank): one line makes it follow --seed
        flex_trainer.init_seeds = lambda seed, **kw: original_init_seeds(seed + args.seed, **kw)
        NYUTrainer(opts, train_list, val_list).train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
