# References

## Papers
- LeJEPA: Balestriero and LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*,
  [arXiv:2511.08544](https://arxiv.org/abs/2511.08544). Code: <https://github.com/galilai-group/lejepa> (we pin `c293d29`).
- METER: Papa, Russo, Amerini, *METER: a mobile vision transformer architecture for monocular depth estimation*,
  [arXiv:2403.08368](https://arxiv.org/abs/2403.08368).
- V-JEPA 2.1: Mur-Labadia et al., *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*,
  [arXiv:2603.14482](https://arxiv.org/abs/2603.14482).

## Baselines (git submodules in `sota/`)
| Model | Repository | Pinned commit |
|---|---|---|
| FlexDepth | <https://github.com/startnew/flexdepth> | `b04ae3a` |
| ZipDepth | <https://github.com/fabiotosi92/ZipDepth> | `91f3fd2` |
| BoRe-Depth | <https://github.com/liangxiansheng093/BoRe-Depth> | `db801e2` |
| Depth Anything V2 (ZipDepth's teacher, pseudo-depth only) | <https://github.com/DepthAnything/Depth-Anything-V2> | `a561b84` |

Depth Anything V2 Large weights: <https://huggingface.co/depth-anything/Depth-Anything-V2-Large>
(`depth_anything_v2_vitl.pth`, 1.34 GB, CC-BY-NC-4.0), downloaded by `scripts/make_pseudo_depth.py`
to `<cache root>/dav2/`.

## Data
- NYU Depth V2: <https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html>
- NYU training split with frame sequences: StructDepth, <https://github.com/SJTU-ViSYS/StructDepth> (`1747827`),
  `splits/nyu_train_0_10_20_30_40_21483-exceptfailed-21465.txt` (no licence file: downloaded, not copied).
- KITTI raw and depth benchmark: <https://www.cvlibs.net/datasets/kitti/>

## Phase 1
The course assignment (`Project_7.pdf`), the paper PDFs and the presentation are in the original repository,
<https://github.com/Zunza5/lejepa-monocular-depth-estimation> (full history; this repository restarted its history to stay light).
