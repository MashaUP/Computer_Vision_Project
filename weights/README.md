# Reference Weights

Curated inference-only checkpoints for step 1 (reference baselines), tracked in git via the `!weights/` exception in `.gitignore`.

## Naming convention
- `weights/flexdepth_n_nyu/` — `encoder.pth` and `depth.pth` (FlexDepth-Nano retrained on NYU raw, 50 epochs; do **not** include `adam.pth`, `pose*.pth`, or `var.pth`).
- `weights/flexdepth_n_kitti/` — `encoder.pth` and `depth.pth` (official released FlexDepth-Nano KITTI weights, if mirrored).
- `weights/zipdepth_base.pth` — ZipDepth-Base zero-shot checkpoint.
- `weights/bore_<dataset>.ckpt` — BoRe-Depth reference checkpoint (`nyu`, `kitti`).

Evaluate any entry with:
```bash
python main.py eval-public --model <model> --dataset <nyu|kitti> --weights weights/<name> --test-list splits/<test>.txt
```
