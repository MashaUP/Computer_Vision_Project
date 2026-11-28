# Reference Weights

Curated inference-only checkpoints, tracked in git via the `!weights/` exception in `.gitignore`.
Only **our own** small checkpoints go here; released third-party weights are not mirrored.

## In git
- `weights/flexdepth_n_nyu/`: `encoder.pth` and `depth.pth` (FlexDepth-Nano retrained on NYU raw in step 1,
  epoch 50 = `weights_49`; do **not** include `adam.pth`, `pose*.pth` or `var.pth`).

## Released weights (not in git)
They stay on the lab machine, outside the repository; origin, licence and SHA-256 of each file are in
[`results/step2_public/README.md`](../results/step2_public/README.md).

| Model | Where (lab) | `--weights` in the container |
|---|---|---|
| FlexDepth-Nano KITTI (HuggingFace `StarNew/flexdepth`, `kitti/flex_n`) | `cache/public_weights/flexdepth_n_kitti/` | `/work/cache/public_weights/flexdepth_n_kitti` |
| BoRe-Depth NYU / KITTI (official Google Drive) | `cache/public_weights/bore/{nyu,kitti}.ckpt` | `/work/cache/public_weights/bore/<dataset>.ckpt` |
| ZipDepth-Base | in its submodule | `sota/zipdepth/checkpoints/zipdepth_base.pth` |

Evaluate any of them (CPU is enough, no GPU booking):
```bash
scripts/alcor/run.sh cpu <job> python3 main.py eval-public --model <model> --dataset <nyu|kitti> --weights <path> \
    --test-list splits/<test list> [--image-size H W] --device cpu --out /work/runs/eval_public/<model>_<dataset>.json
```
All loaders are strict and use `torch.load(weights_only=True)` only (`models/public_weights.py`).
