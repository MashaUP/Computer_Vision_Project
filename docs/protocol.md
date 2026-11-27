# Comparison protocol

## Question
Does in-domain LeJEPA pre-training of a lightweight encoder improve monocular depth estimation over
random initialisation, when the encoder is then frozen and only the decoder is trained?

## Models
| Model | Encoder | Decoder | Pyramid (channels, finest to deepest) | Native output |
|---|---|---|---|---|
| FlexDepth-Nano | YOLO11-N, 1.37 M | 0.16 M | 16 / 64 / 128 / 128 / 256 | sigmoid disparity (4 scales) |
| ZipDepth-Base | QARep CNN, 6.63 M (unfused) | 0.15 M | 24 / 48 / 96 / 192 / 384 | affine-invariant inverse depth (ReLU) |
| BoRe-Depth | MPViT-Tiny, 5.63 M | 3.04 M | 64 / 96 / 176 / 216 / 216 | depth = 1 / (10 sigmoid + 0.01) |
| METER (xxs / xs / s) | MobileViT, 0.64 / 1.20 / 2.99 M | 0.08 / 0.25 / 0.30 M | 5 levels | metric depth (1/4 resolution) |

Numbers measured on the pinned commits (`tests/test_baselines.py`). Decoders differ by about 20x,
so LeJEPA gains are compared **within** each model, not across models.

## Steps (per model, per dataset)
1. **Public weights**, evaluated on the test sets without retraining (FlexDepth on NYU: retrained by us).
2. **LeJEPA pre-training** of the encoder on the deepest map (stride 32), on unlabelled frames of
   NYU raw / KITTI raw with the test scenes excluded, one pre-training per dataset:
   - classic: official recipe, [`configs/lejepa/classic.yaml`](../configs/lejepa/classic.yaml);
   - dense: METER recipe (lambda 0.02, lr 5e-3, 4x4 grid), [`configs/lejepa/dense.yaml`](../configs/lejepa/dense.yaml);
   - `classic_phase1.yaml` / `dense_phase1.yaml` reproduce the course-project code exactly: same views,
     loss and gradients for the same seed (`scripts/verify_phase1_equivalence.py`). The protocol `dense`
     differs from `dense_phase1` only in drawing crops and colour augmentation per image (phase 1 drew
     one set of parameters for the whole batch).
3. **Frozen encoder, decoder-only training**, identical procedure for the three initialisations
   (random, LeJEPA classic, LeJEPA dense), 3 seeds each (`training/trainer.py`, `task: finetune`).
   - Chosen: (A) photometric self-supervised loss (`training/photometric.py`, `loss: photometric`);
     (B) ground-truth `silog` is also implemented as reference.

Totals: 12 pre-trainings (3 models x 2 variants x 2 datasets) and 54 decoder trainings
(3 x 3 initialisations x 3 seeds x 2 datasets), plus FlexDepth on NYU and the public-weight evaluation.

## Fairness rules
- Same test sets (NYU 654, KITTI 652 annotated), same crops (Eigen / Garg) and depth caps (10 m / 80 m).
- Metrics after median scaling and after scale-and-shift (on inverse depth): [`evaluation/metrics.py`](../evaluation/metrics.py).
- Mean and standard deviation over 3 seeds; same training budget for every arm.
- Frozen encoder always in eval mode (`DepthBaseline.freeze_encoder()`), so BatchNorm statistics never change.
- Random arm: BatchNorm statistics re-estimated on unlabelled frames before freezing.

## Numerics
- Pascal GPUs train in fp32, newer GPUs in fp16 autocast, never bf16 (SIGReg diverged in bf16 in phase 1):
  [`utils/device.py`](../utils/device.py).
- SIGReg is not compatible with gradient accumulation: the batch is what fits on the GPU.

## Known pitfalls of the original code (handled here)
- FlexDepth ignores `--seed` (`init_seeds(1 + rank)` is hard-coded) and its `--freeze_encoder` leaves
  BatchNorm in train mode: we only import its architecture.
- ZipDepth: `cross_scale.low_to_high` (18,432 parameters) is not reached from the deepest map, so it is
  trained with the decoder (DDP would need `find_unused_parameters`).
- BoRe-Depth: its MPViT `train()` returns `None` (never chain `.eval()`); mmcv is replaced by plain PyTorch
  for norm layers and checkpoint loading; its public weights must be loaded with strict key checking
  (the original code uses `strict=False`).
- METER needs view sizes divisible by 32 (phase 1 used 256 / 128 px views).
