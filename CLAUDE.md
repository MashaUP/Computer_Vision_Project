# Lean Geometry

LeJEPA (SIGReg) self-supervised pre-training vs random init for lightweight depth encoders
(FlexDepth, ZipDepth, BoRe-Depth, METER), on NYU Depth V2 and KITTI. Overview in `README.md`,
experimental protocol in `docs/protocol.md`, checklist in `docs/experimental_plan.md` (tick items as they are done).

## Setup
- Local: `.venv` with Python 3.12; `pip install torch torchvision` from PyPI, then `pip install '.[test]'`.
- Third-party code lives in submodules: `git submodule update --init --recursive`.
- Lab machine (ALCOR, Podman): see `docs/alcor.md`.

## Commands
- `python -m pytest -q` (CPU only, ~3 min)
- `python main.py check [--device cuda|mps] [--models ...]`: smoke test of encoders + both LeJEPA recipes
- `python main.py pretrain|finetune|evaluate --config configs/runs/<file>.yaml --set key=value`
- `python main.py eval-public --model flexdepth|zipdepth|bore --dataset nyu|kitti --weights <path> --test-list splits/nyu_labeled_test.txt`
- FlexDepth on NYU (original trainer, needs CUDA + `pip install '.[flexdepth-train]'`):
  `python scripts/train_flexdepth_nyu.py --name flexdepth_n_nyu [-- <extra FlexDepth options>]`
- ZipDepth pseudo-depth for `loss: zipdepth` (Depth Anything V2 Large, GPU, resumable; `--limit 200` to time it; NYU ~2 frames/s on a 1080 Ti):
  `python scripts/make_pseudo_depth.py --dataset nyu|kitti --list splits/<dataset>_raw_triplets_train.txt`

## Hard rules
- `sota/` is pinned third-party code: NEVER modify it. Adapt in `models/baselines.py` or the callers of `sota/loader.py`.
- `models/meter.py` is METER from phase 1: do not change it without asking.
- Precision: fp32 on the lab GPUs (Pascal) and in every test; fp16 autocast only on newer GPUs (`utils/device.py`). Never bf16 (SIGReg diverges).
- Step-3 loss (decided 28/11): the original loss where the model has one. `loss: photometric` for FlexDepth and
  BoRe-Depth, `loss: zipdepth` for ZipDepth (`ZipDepthLoss` on Depth Anything V2 Large pseudo-depth, as in its
  original distillation). `loss: silog` (ground truth) is only a reference. The trainer refuses `zipdepth` on other models.
- Pre-training and fine-tuning use the full NYU raw set, not only the 795 labeled frames.
- NYU validation = ~5% of the non-test raw scenes (chosen by a hash of the scene id, so it never depends on which scenes are on disk; `data.datasets.check_list` rejects lists containing test or val scenes), excluded from LeJEPA pre-training (`frames`), train (`triplets`) and the 654 test scenes:
  `python -m data.download.nyu triplets` writes `nyu_raw_val_scenes.txt` and the train/val frame triplets.

## Conventions
- Run configs: YAML in `configs/runs/`. LeJEPA recipes: `configs/lejepa/`. Unknown keys are an error in both
  (`RunConfig`, recipes); resume refuses a config that differs from the stored one.
- Data, runs and caches stay outside the repo: `LG_DATA_ROOT`, `LG_RUNS_ROOT`, `LG_CACHE_ROOT`.
- `to_depth` uses min depth 0.1 m (Monodepth2 convention); `MIN_DEPTH=1e-3` is only for metrics and clamping.
- Tests must run on CPU; add a test for any risky logic.

## Status (28/11)
- Split lists live in the data root (`<LG_DATA_ROOT>/splits/`, `/work/data/splits/` in the container), not in the repo;
  `--list splits/...` is resolved against the data root.
- FlexDepth-Nano retrained on NYU with `train_flexdepth_nyu.py` on the lab (50 epochs, `weights/flexdepth_n_nyu/`).
- Step 2 (reference weights, CPU, 28/11): FlexDepth NYU, BoRe-Depth NYU / KITTI, ZipDepth NYU / KITTI evaluated, JSON +
  `summary.csv` in `results/step2_public/` (protocol, weight SHA-256 and README comparison in its README). BoRe NYU
  matches its README; KITTI numbers are on the 652 improved-GT split, not comparable with raw-LiDAR tables.
  Released BoRe checkpoints are on the lab in `cache/public_weights/bore/` (not in git).
- Depth Anything V2 Large on the lab (1080 Ti, fp32): released weights load strictly; NYU 2.0 frames/s, 2.8 GB peak,
  ~53 MB of PNG per 200 frames. NYU full run (24,863 triplets) launched 28/11; KITTI not measured yet.
- KITTI: layout on ALCOR verified (151 drives, 47,889 frames, annotated GT on the same drives); `--make_splits` run on
  28/11: 109 train / 14 val / 28 Eigen test drives, 27,154 train triplets, 652 test pairs; no test or val drive in any
  path of the train lists.
- NYU: 24,863 train triplets. `data/splits/` also holds `.bak` copies from 26/11 with 6 more lines, probably the lists
  from before the `ppm_ok` fix (c68b153, truncated raw frames skipped): to confirm, then remove.

## Open items
- Step-3 loss: `photometric` (FlexDepth, BoRe) and `zipdepth` (ZipDepth) implemented; ZipDepth `cross_scale.low_to_high`
  is trained with the decoder in both. Not yet run as a training job on GPU.
- Step 2: FlexDepth-Nano KITTI still to evaluate (released weights from HuggingFace `StarNew/flexdepth`); FlexDepth NYU
  has no published reference, compare with Luca's step-1 evaluation.
- `make_views` takes ~65% of a training step on CPU; not measured on GPU yet.
