# Step 2: reference weights on the unified protocol

Released (FlexDepth-Nano KITTI, BoRe-Depth, ZipDepth) and step-1 (FlexDepth-Nano NYU) weights, evaluated with no
retraining by `python main.py eval-public` on the ALCOR lab machine, CPU only, on 28/11/2026. Lab repository at
`fa15b1e`, and at `afcff69` (PR #9, profiler buffers dropped by the FlexDepth loader) for FlexDepth-Nano KITTI.

| File | Contents |
|---|---|
| `<model>_<dataset>.json` | output of `eval-public --out`, copied unchanged from `/mnt/hdd2/masciarelli/runs/eval_public/` (SHA-256 checked after the copy) |
| `summary.csv` | one row per run and alignment, generated from the JSON files: `python scripts/summarize_eval_public.py results/step2_public` |

## Protocol (`evaluation/metrics.py`, `Trainer.evaluate`)
- **Test sets:** NYU Depth V2, the 654 labelled test images (`splits/nyu_labeled_test.txt`). KITTI, the 652-frame
  Eigen *benchmark* split with the improved annotated ground truth (`data_depth_annotated`,
  `splits/kitti_eigen_test.txt`, built from `sota/flexdepth/splits/eigen_benchmark/test_files.txt`), not the
  697-frame Eigen split with raw LiDAR.
- **Input:** the whole RGB image resized (bilinear, antialiased) to the resolution below. No crop, and the aspect
  ratio is not preserved when the target size has a different one. The prediction is upsampled (bilinear) to the
  ground-truth size and converted to depth.
- **Valid pixels:** Eigen crop on NYU, Garg crop on KITTI; depth caps 10 m (NYU) and 80 m (KITTI), minimum 1e-3 m;
  predictions clamped to the same range after alignment.
- **Alignment, per image:** `none` (prediction as is); `median` (depth multiplied by median(gt) / median(pred),
  Monodepth2 convention); `scale_shift` (least-squares scale and shift fitted on *inverse* depth, MiDaS convention,
  then converted back to depth). Metrics are averaged over the images (each image weighs the same).
- **Which rows to read:** none of the three models predicts metric depth, so `none` is not meaningful.
  FlexDepth-Nano and BoRe-Depth are self-supervised and scale-ambiguous: `median` is their original protocol,
  `scale_shift` is reported for comparison. ZipDepth outputs affine-invariant relative inverse depth: only
  `scale_shift` is meaningful, and its `median` row is not.
- **Precision:** fp32.

| Model | NYU (H×W) | KITTI (H×W) | Source of the resolution |
|---|---|---|---|
| FlexDepth-Nano | 288×384 | 192×640 | `height`/`width` stored in each `encoder.pth` (NYU: step-1 training resolution; KITTI: released checkpoint, as in the official `evaluate_depth.py`) |
| BoRe-Depth | 256×320 | 256×832 | `sota/bore_depth/configs/configs.py::get_size` |
| ZipDepth-Base | 384×512 | 384×1280 | `sota/zipdepth/scripts/eval.py` (`--input-size 384`: shorter side 384, aspect kept, multiple of 32); KITTI ~375×1242 → 384×1280 fixed |

## Weights

| Model | Weights | Origin | Licence | SHA-256 |
|---|---|---|---|---|
| FlexDepth-Nano NYU | `weights/flexdepth_n_nyu/` | step 1: FlexDepth-Nano retrained on NYU raw with the original recipe, epoch 50 (`weights_49`) | code AGPL-3.0 (`sota/flexdepth/LICENSE`) | `encoder.pth` `027bf552ec658cb7300ddfd10d025a1481ed35efc1fb43dfb555c7bc4c7292b0`<br>`depth.pth` `41aa693a0fadae27bb9d07214af80c7102cf3558ad977a5f9aeab9d399f1a539` |
| FlexDepth-Nano KITTI | `encoder.pth`, `depth.pth` (lab: `cache/public_weights/flexdepth_n_kitti/`, not in git) | [huggingface.co/StarNew/flexdepth](https://huggingface.co/StarNew/flexdepth), `kitti/flex_n/` (linked in `sota/flexdepth/README.md`); size and SHA-256 equal to the LFS values published by HuggingFace | AGPL-3.0 (`sota/flexdepth/LICENSE`) | `encoder.pth` `68cb2bdbe39368daf3dba092b92c1adcbb3885424a5c93872c47f8f07e5e4717`<br>`depth.pth` `0d5525197123a4a257cc0b992345925ce3d192622dbcbb9958dd8c656ec7ba09` |
| ZipDepth-Base | `sota/zipdepth/checkpoints/zipdepth_base.pth` (submodule at `91f3fd2`) | [github.com/fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base.pth), zero-shot | MIT (`sota/zipdepth/LICENSE`) | `a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65` |
| BoRe-Depth NYU | `nyu.ckpt` (lab: `cache/public_weights/bore/`, not in git) | [official Google Drive folder](https://drive.google.com/drive/folders/11XxOXqWKp3bXe2Sv2ED90__S3GwkgMRI) linked in `sota/bore_depth/README.md` | MIT (repository licence, `sota/bore_depth/LICENSE`; the Drive folder states no separate licence) | `fc134fdc54c00c52c77ef44d7b78d97b459a88a68c332d36c73e59e87395c256` |
| BoRe-Depth KITTI | `kitti.ckpt` (as above) | as above | as above | `7dfc659ff5529e0c85ba6c6a930190a921f6fa526dd4e94867271860e06b60a7` |

Loading (`models/public_weights.py`, strict, `weights_only=True`):
- **BoRe-Depth:** both `.ckpt` files hold only `state_dict` and `pytorch-lightning_version` (1.9.0), with no
  `hyper_parameters`. Of their 1,083 tensors, the 953 of `depth_net` are loaded strictly and the 130 of `pose_net`
  (training only) are dropped. The runs used the loader from before PR #7, which still had the unsafe fallback, but
  `weights_only=True` succeeded on the first try, so PR #7 does not change these numbers.
- **ZipDepth-Base:** loaded strictly with `weights_only=True`; only the `mean`/`std` normalisation buffers are ignored.
- **FlexDepth-Nano KITTI:** both files load with `weights_only=True`. `encoder.pth` stores only `height` 192,
  `width` 640 and `use_stereo` False, with no `decoder_model_type` or decoder flags. The released files also carry
  the scalar `*.total_ops` / `*.total_params` buffers of the `thop` profiler (72 in the encoder, 32 in the decoder,
  all float64 zeros), which the official `evaluate_depth.py` drops with a non-strict load. Since PR #9 our loader
  drops exactly these buffers, and the other 240 encoder and 116 decoder tensors are loaded strictly (same names
  and shapes as our `flexn` model). The decoder options are the ones the official `docs/eval.md` command gets from
  `--decoder_model_type flexn` (`upsample dysample_pl`, HEB, no `c3k`, `all_dysample` or `dysample_group8`),
  which is how `models/baselines.py` builds the model.

## Comparison with the original READMEs
- **BoRe-Depth NYU:** matches. README: AbsRel 0.101, RMSE 0.429, δ1 0.883 / δ2 0.971 / δ3 0.993. Ours (`median`):
  0.103, 0.433, 0.881 / 0.970 / 0.992. Their `test.py` uses the same protocol: median scaling, Eigen crop,
  cap 10 m (min 0.1 m instead of our 1e-3 m).
- **BoRe-Depth KITTI:** does not match numerically. README: AbsRel 0.103, RMSE 4.323, δ1 0.889. Ours (`median`):
  0.080, 3.669, 0.927. The alignment, the Garg crop and the 80 m cap are the same as their `test.py`. Their README
  does not say which KITTI test set it uses, and our 652-frame benchmark split with improved ground truth gives lower
  errors than raw LiDAR, so the two numbers are not directly comparable.
- **FlexDepth-Nano KITTI:** does not match numerically, for the same reason as BoRe-Depth KITTI. README (KITTI
  Eigen split, cap 80 m, median scaling): AbsRel 0.110, SqRel 0.794, RMSE 4.678, δ1 0.878. Ours (`median`):
  0.081, 0.451, 3.698, 0.924. The README table is on the Eigen split; its benchmark-split (652 frames) table only
  covers Flex-X-Large with least-squares alignment on dense GT.
- **FlexDepth-Nano NYU:** no reference. The FlexDepth README only reports KITTI and Cityscapes, and these weights are
  our own step-1 retraining.

## Run times (CPU)
`--device cpu` in a Podman container limited to 6 threads on `alcor-gtx`, while `dav2_nyu` ran on GPU 2. Wall-clock
from the creation of the job's `stdout.log` to the JSON write, so model building and loading are included.

| Run | Images | Time |
|---|---|---|
| flexdepth_nyu | 654 | 110 s |
| flexdepth_kitti | 652 | 106 s |
| bore_nyu | 654 | 276 s |
| bore_kitti | 652 | 654 s |
| zipdepth_nyu | 654 | 151 s |
| zipdepth_kitti | 652 | 300 s |

## Commands
```bash
scripts/alcor/run.sh cpu eval_flex_nyu   python3 main.py eval-public --model flexdepth --dataset nyu   --weights weights/flexdepth_n_nyu --test-list splits/nyu_labeled_test.txt --device cpu --out /work/runs/eval_public/flexdepth_nyu.json
scripts/alcor/run.sh cpu eval_flex_kitti python3 main.py eval-public --model flexdepth --dataset kitti --weights /work/cache/public_weights/flexdepth_n_kitti --test-list splits/kitti_eigen_test.txt --device cpu --out /work/runs/eval_public/flexdepth_kitti.json
scripts/alcor/run.sh cpu eval_bore_nyu   python3 main.py eval-public --model bore      --dataset nyu   --weights /work/cache/public_weights/bore/nyu.ckpt   --test-list splits/nyu_labeled_test.txt --image-size 256 320  --device cpu --out /work/runs/eval_public/bore_nyu.json
scripts/alcor/run.sh cpu eval_bore_kitti python3 main.py eval-public --model bore      --dataset kitti --weights /work/cache/public_weights/bore/kitti.ckpt --test-list splits/kitti_eigen_test.txt --image-size 256 832 --device cpu --out /work/runs/eval_public/bore_kitti.json
scripts/alcor/run.sh cpu eval_zip_nyu    python3 main.py eval-public --model zipdepth  --dataset nyu   --weights sota/zipdepth/checkpoints/zipdepth_base.pth --test-list splits/nyu_labeled_test.txt --image-size 384 512  --device cpu --out /work/runs/eval_public/zipdepth_nyu.json
scripts/alcor/run.sh cpu eval_zip_kitti  python3 main.py eval-public --model zipdepth  --dataset kitti --weights sota/zipdepth/checkpoints/zipdepth_base.pth --test-list splits/kitti_eigen_test.txt --image-size 384 1280 --device cpu --out /work/runs/eval_public/zipdepth_kitti.json
```
