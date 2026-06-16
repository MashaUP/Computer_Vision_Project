# METER + LeJEPA: Self-Supervised Geometry Probing & Depth Estimation

**Project 7: "Lean Geometry: Boosting Mobile MDE with LeJEPA"**
Computer Vision course (Prof. Irene Amerini), Sapienza University of Rome, Spring 2026.

> [!NOTE]
> This repository contains two projects: the main setup (running via [`main_final.ipynb`](main_final.ipynb)) and an experimental setup located in the [`experimental_setup`](experimental_setup) directory.
> The experimental setup features its own isolated code structure and configuration documented in its internal [`README.md`](experimental_setup/README.md), and its main execution files are [`main.ipynb`](experimental_setup/main.ipynb) and [`evaluation.ipynb`](experimental_setup/evaluation.ipynb).

This repository implements **METER** (MobileViT-based encoder-decoder for Monocular Depth Estimation) coupled with **LeJEPA** (Joint-Embedding Predictive Architecture for self-supervised pre-training). The goal is a **controlled comparative study**: the same lightweight METER architecture is fine-tuned for depth estimation starting either from **LeJEPA self-supervised pre-training** or from **random initialization**, isolating the contribution of the SSL objective on depth accuracy, convergence speed, and feature interpretability.

---

## Table of Contents
1. [Project Overview](#-project-overview)
2. [Project Structure](#-project-structure)
3. [Prerequisites & Installation](#-prerequisites--installation)
4. [Dataset Setup](#-dataset-setup)
5. [Running the Project](#-running-the-project)
6. [Results Summary](#-results-summary)
7. [Code Design & Reorganization Pattern](#-code-design--reorganization-pattern)
8. [Methodology & Architecture Details](#-methodology--architecture-details)

---

## Project Overview

The project covers the four objectives of Project 7:

1. **Lightweight LeJEPA Integration**: LeJEPA pre-training (invariance loss + **SIGReg**, *Sketched Isotropic Gaussian Regularization*) is implemented on the MobileViT-lite encoder used by METER, trained on unlabeled raw frames on a single consumer GPU.
2. **Zero-Shot Geometric Probing**: before attaching the depth decoder, the SSL latent space is analyzed via PCA to verify whether the emerging "free" semantic segments align with depth discontinuities (foreground/background separation).
3. **MDE Downstream Performance**: the LeJEPA-pretrained encoder is fine-tuned for depth estimation and compared against a randomly-initialized baseline trained under identical settings, in terms of **RMSE, AbsRel, δ₁** and convergence speed.
4. **Efficiency–Accuracy Trade-off**: latency/FPS benchmarking and training-cost vs. accuracy analysis (see `extra/analysis.ipynb`).

> [!NOTE]
> The encoder architecture supports three scale variants: **METER-XXS** (0.71M params), **METER-XS** (1.45M params), and **METER-S** (3.29M params), utilizing MobileViT blocks in the bottleneck.

> [!NOTE]
> **Baseline choice.** The original METER initializes its encoder with ImageNet-supervised weights; no public ImageNet checkpoint exists for this custom MobileViT-lite encoder, so the controlled comparison is *LeJEPA pre-training vs. random initialization* on the same architecture, optimizer, schedule, and data.

---

## Project Structure

The project separates notebook execution and pipeline checks from the modular core source code:

- [`main_final.ipynb`](main_final.ipynb): **the main deliverable notebook**, full pre-training, fine-tuning, baseline, and evaluation pipeline, structured into the 7 standard code sections.
- [`main.ipynb`](main.ipynb): the original development version of the training pipeline.
- [`extra/analysis.ipynb`](extra/analysis.ipynb): comparative graphs, qualitative depth predictions, convergence curves, and efficiency benchmarks (Objectives 3–4).
- [`extra/test.ipynb`](extra/test.ipynb): 16 sanity checks and unit tests.
- `src/`: core Python modules.
  - [`config.py`](src/config.py): device settings, paths, epochs, learning rates, and target resolutions.
  - [`dataset.py`](src/dataset.py): NYU and KITTI dataset classes, dataloader builders, and training augmentations.
  - [`network.py`](src/network.py): PyTorch modules for Separable Convolutions, MobileNetV2 blocks, Transformer layers, MobileViT encoders, and depth decoders.
  - [`train.py`](src/train.py): loss functions (Sobel, SSIM, balanced MDE loss), LightningModules (`LeJEPAMeter`, `METERFinetune`), and callbacks.
  - [`evaluation.py`](src/evaluation.py): test metrics evaluation, pixel-weighted evaluation, PCA zero-shot geometric probing, prediction plotting, and model latency/FPS benchmarking.
  - [`utils.py`](src/utils.py): timers, checkpoint managers, CSV logging utilities, and dataset downloader helpers.
- `architecture.py`, `loss.py`, `augmentation.py` (project root): **original course-provided reference files (unmodified)**. The corresponding functionality used by the pipeline lives in `src/`, which preserves the reference formulas from the METER paper.

---

## Prerequisites & Installation

The project is built with **Python 3.12** and dependencies are managed using **PDM**.

### 1. Install PDM
```bash
pip install pdm
```

### 2. Install Project Dependencies
```bash
pdm install
```

This sets up a virtual environment in `.venv` containing PyTorch with CUDA, PyTorch Lightning, timm, OpenCV, and the custom local self-supervised `lejepa` dependency.

---

## Dataset Setup

Download scripts are integrated into the utility files. By default, dataloaders look for raw files in `data/raw/`.

### NYU Depth V2
1. **Supervised fine-tuning (train)**: the ~50K RGB-depth pairs split (Kaggle H5 / jpg-png pairs format), matching METER paper Section IV-B. A random 10K subset is sampled for training.
2. **Test**: the 654 official test pairs from `nyu_depth_v2_labeled.mat`.
3. **Unlabeled pre-training (SSL)**: bedroom, kitchen, and bathroom PPM raw sequences.

To automatically download the NYU dataset:
```python
from src.utils import download_nyu
download_nyu("data/raw/nyu", skip_raw=False, skip_labeled=False)
```

### KITTI Dataset
1. **GT depth**: Eigen split annotated depth sequences (`data_depth_annotated.zip`).
2. **Raw RGB**: matching sequence archives from the KITTI Raw database (also used, unlabeled, for SSL pre-training, 10K-frame random subset).

To download the KITTI sequences:
```python
from src.utils import download_kitti
download_kitti("data/raw/kitti", skip_depth=False, skip_raw=False)
```

---

## Running the Project

### 1. Run Pipeline Sanity Checks
Open and run [`extra/test.ipynb`](extra/test.ipynb) for the pre-flight checks:
- Verifies CUDA acceleration and GPU VRAM precision.
- Validates NYU and KITTI file paths and batch shapes.
- Tests loss calculations, backpropagation, gradient flow, and SSL steps.
- Verifies model architecture instantiation and checkpoint loading.

### 2. Run the Main Pipeline
Open and execute [`main_final.ipynb`](main_final.ipynb) to:
- Build dataloaders for pre-training and fine-tuning (select the dataset via `DATASET_NAME` in the Data section: `"nyu"` or `"kitti"`).
- Execute LeJEPA SSL pre-training for 50 epochs.
- Perform zero-shot PCA geometric probing.
- Fine-tune encoder + decoder in supervised mode for 60 epochs (differential learning rate: encoder ×0.1 when SSL-pretrained).
- Train the from-scratch baseline under identical settings.
- Evaluate both models on the target MDE test set and print a side-by-side comparison.

### 3. Generate Comparative Analyses
Open [`extra/analysis.ipynb`](extra/analysis.ipynb) to:
- Compare LeJEPA-pretrained METER models with the randomly-initialized baseline controls.
- Generate metrics bar charts and unified comparison tables.
- Visualize side-by-side predicted depth maps against ground truth.
- Plot convergence curves (LeJEPA pre-training vs. scratch initialization) and the efficiency–accuracy trade-off.

---

## Results Summary

Headline finding (METER-XXS, controlled comparison): **LeJEPA pre-training helps on KITTI (~−11.6% RMSE vs. random initialization) but does not improve NYU at this model scale**, which we attribute to scene complexity, SSL data volume, and capacity saturation of the 0.71M-parameter variant. Full tables, convergence curves, and qualitative maps are produced in [`extra/analysis.ipynb`](extra/analysis.ipynb).

---

## Code Design & Reorganization Pattern

All notebooks follow the strict structure standard required by the course:
- **Imports**: environment configuration (CWD, Colab drive setup) and library/module imports.
- **Globals**: device specifications, precision variables, seeds, and global training hyperparameter overrides.
- **Utils**: code support functions (e.g. best-checkpoint selection) plus helpers imported from `src/utils.py`.
- **Data**: target dataset selection, dataloader definitions, and sanity checks.
- **Network**: network instantiation (`METERModel`), parameter count prints, and forward-pass verification.
- **Train**: self-supervised pre-training, transfer-learning fine-tuning, and from-scratch baseline training.
- **Evaluation**: quantitative metrics (RMSE, AbsRel, δ₁) and qualitative geometric probing.

---

## Methodology & Architecture Details

### LeJEPA Pre-Training
- Multi-crop strategy: **2 global views (256px) + 6 local views (128px)** per image (8 forward passes per step).
- Objective: invariance (ℓ₂ between view embeddings) + **SIGReg** (*Sketched Isotropic Gaussian Regularization*) with λ = 0.05 and 256 slices, single trade-off hyperparameter, no teacher–student, no stop-gradient.
- Training: AdamW, lr 5e-4, linear warm-up + cosine annealing, `fp16-mixed` precision with gradient clipping 1.0.

### Composite Loss Function (Fine-Tuning)
The MDE model is fine-tuned under a composite loss (METER paper, Section IV-A):

$$\mathcal{L} = \mathcal{L}_{\text{depth}} + \lambda_1 \mathcal{L}_{\text{grad}} + \lambda_2 \mathcal{L}_{\text{norm}} + \lambda_3 \mathcal{L}_{\text{SSIM}}$$

- **$\mathcal{L}_{\text{depth}}$**: mean absolute error (L1 loss).
- **$\mathcal{L}_{\text{grad}}$**: Sobel gradient matching ($\lambda_1 = 0.5$).
- **$\mathcal{L}_{\text{norm}}$**: normal-vector cosine similarity ($\lambda_2 = 100$ on NYU, $1$ on KITTI unit-dependent: cm vs. m).
- **$\mathcal{L}_{\text{SSIM}}$**: Structural Similarity loss ($\lambda_3 = 100$ on NYU, $1$ on KITTI).

### Data Augmentation
Downstream MDE training applies each augmentation with probability $0.5$:
- Horizontal mirroring and vertical flip.
- C-shift: brightness, gamma, and color-channel permutation shifts.
- D-shift: depth shifting of **±10 cm on NYU** and **±10 dm (= ±1 m) on KITTI** (METER paper Section III-C, Eq. 8 the shift unit follows the dataset's native depth unit).

Augmentations are applied to the training split only; validation and test data are never augmented.
