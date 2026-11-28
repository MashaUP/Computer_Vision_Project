# Experimental plan: frozen LeJEPA encoders vs original models

FlexDepth-Nano, ZipDepth-Base, BoRe-Depth. Evaluation on NYU Depth V2 (654) and KITTI (652).

Editable version of the former `piano_sperimentale_lejepa.pdf` (Luca, 27/11), updated on 28/11.
Tick a box in the same commit that completes the item. Protocol details: [`protocol.md`](protocol.md).

## 0. Prerequisites, decisions and common setup
- [x] **Step-3 decoder loss agreed** (28/11; it replaces option A, "photometric for all", and option B, "GT depth on the labelled frames only")
  - [x] FlexDepth-Nano and BoRe-Depth: photometric self-supervised loss on NYU raw / KITTI raw triplets (`loss: photometric`)
  - [x] ZipDepth-Base: its own `ZipDepthLoss` on Depth Anything V2 Large pseudo-depth, as in its original distillation (`loss: zipdepth`, PR #4)
  - [x] Ground-truth `silog` kept as a reference only
- [x] **Test sets and unified evaluation protocol**
  - [x] NYU test set (654 images) and KITTI test set (652 Eigen pairs, `kitti_eigen_test.txt`, generated and checked on 28/11), same crops and depth caps
  - [x] Metrics after *median scaling* and after *scale-and-shift*
- [x] **Training data lists** (in `<data root>/splits/`)
  - [x] NYU raw: 24,863 train triplets, ~5% of the non-test scenes held out for validation
  - [x] KITTI raw: 27,154 train triplets (109 drives), 14 validation drives, 28 Eigen test drives excluded; leak-checked on 28/11
- [x] **Common trainer** (based on FlexDepth)
  - [x] Remove the method-specific heuristics
  - [x] Reproducible seed handling
  - [x] Encoder BatchNorm kept in `eval`

## 1. Retraining FlexDepth on NYU
- [x] Download the NYU raw sequences (~400 GB)
- [x] Retrain **FlexDepth-Nano** on NYU with the original recipe (50 epochs)
- [x] Save and check the NYU reference checkpoint (`weights/flexdepth_n_nyu/`)

## 2. Evaluation of the reference weights (no retraining)
- [ ] **FlexDepth-Nano** (1.52 M)
  - [ ] Evaluate on KITTI (652) with the released weights
  - [x] Evaluate on NYU (654) with the step-1 checkpoint
- [ ] **ZipDepth-Base** (6.79 M)
  - [ ] Load the released zero-shot weights (check 278/278 tensors)
  - [ ] Evaluate on NYU (654) and KITTI (652) at their own resolutions
- [ ] **BoRe-Depth** (8.66 M)
  - [ ] Download the NYU and KITTI weights from Drive and load them with *strict key checking*
  - [ ] Evaluate on NYU (654) and KITTI (652)
- [ ] Save all step-2 numbers (FlexDepth NYU included) in `results/` as JSON / CSV

## 3. LeJEPA SSL pre-training and decoder fine-tuning (3 seeds)

### 3.0 Pseudo-depth targets for ZipDepth (Depth Anything V2 Large)
- [x] Weights cached on the lab and loaded strictly; 200-frame timing on NYU (1080 Ti, fp32: 2.0 frames/s, 2.8 GB)
- [ ] NYU, 24,863 frames (running on 28/11, ~3.4 h)
- [ ] KITTI, 27,154 frames (time it first with `--limit 200`)

### 3.1 SSL pre-training of the encoders (hook on the level-5 feature map)
- [ ] **FlexDepth-Nano (YOLO11-N)**
  - [ ] **LeJEPA classic** pre-training (pooled embedding)
  - [ ] **LeJEPA dense** pre-training (4×4 grid)
- [ ] **ZipDepth-Base (QARep CNN)**
  - [ ] **LeJEPA classic** pre-training (pooled embedding)
  - [ ] **LeJEPA dense** pre-training (4×4 grid)
- [ ] **BoRe-Depth (MPViT-Tiny)** (note: ~15 img/GPU in `fp32`)
  - [ ] **LeJEPA classic** pre-training (pooled embedding)
  - [ ] **LeJEPA dense** pre-training (4×4 grid)

One pre-training per dataset (NYU raw, KITTI raw): 12 in total.

### 3.2 Decoder-only fine-tuning with a frozen encoder (3 seeds per arm)
For every model and dataset: same trainer, frozen encoder with BN in `eval`; for ZipDepth-Base the fusion
layer (0.3%, `cross_scale.low_to_high`) is trained together with the decoder. Loss as decided in section 0.

- [ ] **FlexDepth-Nano** (NYU and KITTI)
  - [ ] **Random** (3 seeds), with the BatchNorm statistics re-estimated on unlabelled frames
  - [ ] **LeJEPA classic** (3 seeds)
  - [ ] **LeJEPA dense** (3 seeds)
- [ ] **ZipDepth-Base** (NYU and KITTI)
  - [ ] **Random** (3 seeds), with the BatchNorm statistics re-estimated on unlabelled frames
  - [ ] **LeJEPA classic** (3 seeds)
  - [ ] **LeJEPA dense** (3 seeds)
- [ ] **BoRe-Depth** (NYU and KITTI)
  - [ ] **Random** (3 seeds), with the BatchNorm statistics re-estimated on unlabelled frames
  - [ ] **LeJEPA classic** (3 seeds)
  - [ ] **LeJEPA dense** (3 seeds)

54 decoder trainings in total.

## 4. Final evaluation and comparative analysis
- [ ] Evaluate every step-3 checkpoint on NYU (654) and KITTI (652) with the unified protocol
- [ ] Aggregate the step-3 results as **mean ± std over 3 seeds** (*median scaling* and *scale-and-shift*)
- [ ] Compare results **within each model** (no cross-model comparison: the decoders differ by ~20×)
  - [ ] Place **LeJEPA classic** and **LeJEPA dense** between **Random** (lower bound) and **released weights / steps 1–2** (reference)
  - [ ] **Classic vs dense** per architecture and per domain (NYU vs KITTI)
