# Lean Geometry

LeJEPA (SIGReg) self-supervised pre-training vs random init for lightweight depth encoders
(FlexDepth, ZipDepth, BoRe-Depth, METER), on NYU Depth V2 and KITTI. Overview in `README.md`,
experimental protocol in `docs/protocol.md`.

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
- ZipDepth pseudo-depth for `loss: zipdepth` (Depth Anything V2 Large, GPU, resumable; `--limit 200` to time it):
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

## Open items
- `train_flexdepth_nyu.py`: GPU run and the COCO-pretrained weights path not yet exercised; the lab needs network for the
  YOLO11 and ImageNet ResNet-18 (pose net) weights.
- Step-3 loss: `photometric` (FlexDepth, BoRe) and `zipdepth` (ZipDepth) implemented; ZipDepth `cross_scale.low_to_high`
  is trained with the decoder in both. Not yet run on GPU: strict loading of the released Depth Anything V2 weights,
  memory and frames/s of `make_pseudo_depth.py` on the 1080 / 1080 Ti.
- KITTI preparation: `data/download/download_kitti.py --make_splits` and `data.datasets.check_list` leak guards are implemented; zip layout on first ALCOR download still unverified.
- `make_views` takes ~65% of a training step on CPU; not measured on GPU yet.
