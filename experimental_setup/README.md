# Experimental Setup

This directory is part of the same Project 7 repository, not a separate
project. It contains the more exploratory implementation developed alongside
the main root setup for:

**Lean Geometry - Boosting Mobile MDE with LeJEPA**

The overall research question is unchanged: evaluate whether LeJEPA
self-supervised pretraining can improve a lightweight METER/MobileViT model for
monocular depth estimation on NYU Depth V2 and KITTI.

## Role in the Project

The project includes two complementary implementations:

- **Root setup**: the main deliverable, centered on `../main_final.ipynb`. This
  version is the cleaner and more LeJEPA-faithful pipeline, and should be read
  as the reference implementation of the project.
- **Experimental setup**: this folder. It keeps an isolated version of the code
  used to test more exploratory choices, ablations, intermediate training
  configurations, and additional analysis outputs without changing the main
  root pipeline.

Both setups follow the same Project 7 objective and compare a LeJEPA-pretrained
METER model against a supervised baseline using RMSE, AbsRel, Delta1, PCA-based
geometric probing, convergence analysis, and efficiency-accuracy plots.

## Main Implementation Differences

The most important difference between the two implementations is the
self-supervised pretraining stage.

In the **root setup**, LeJEPA is implemented as a direct adaptation of the
original objective to the METER/MobileViT encoder. The encoder feature map is
globally averaged before the projection head, so each augmented image view
produces one embedding vector. The pretraining loss aligns the projected
embeddings from 2 global crops and 6 local crops, then applies SIGReg to the
flattened view/batch distribution. This version is simpler, closer to the
standard LeJEPA formulation, and is the main reference pipeline used for the
project.

In the **experimental setup**, the pretraining was modified to test a denser
variant of the idea. Instead of projecting only one global embedding per image,
the encoder feature map is pooled to a fixed 4x4 token grid and the projector is
applied to every spatial token. The loss combines two terms:

1. same-crop dense invariance, where two color-augmented versions of the same
   crop are aligned token by token;
2. global cross-crop invariance, where token projections are averaged spatially
   and aligned across different crops of the same image.

SIGReg is then applied over the flattened view, batch, and spatial-token
distribution. This makes the experimental pretraining less faithful to the
minimal LeJEPA formulation, but more explicitly targeted at dense prediction:
the goal was to test whether forcing spatial tokens to remain informative could
help monocular depth estimation and PCA-based geometric probing.

Other differences are mostly practical: the experimental setup keeps separate
checkpoints, logs, and outputs, and includes additional diagnostics such as
throughput, memory, FLOPs, and alternative comparison plots. These experiments
support the final discussion, while the root implementation remains the cleaner
project pipeline.

## Contents

```text
experimental_setup/
+-- main.ipynb              # Experimental training pipeline
+-- evaluation.ipynb        # Experimental evaluation and analysis
+-- src/
|   +-- config.py           # Paths, device, image size, and hyperparameters
|   +-- dataset.py          # NYU/KITTI dataset handling and dataloaders
|   +-- network.py          # METER / MobileViT architecture
|   +-- train.py            # SSL pretraining, fine-tuning, baseline training
|   +-- evaluation.py       # Metrics, plots, PCA probing, benchmarking
|   +-- utils.py            # Checkpoint and support utilities
+-- models/checkpoints/     # Experimental checkpoints
+-- logs/                   # Lightning logs
+-- output/
    +-- csv/                # Experimental tables
    +-- images/             # Experimental plots
    +-- json/               # Experimental metric summaries
```

The notebooks keep the same conceptual structure required by the course:
Imports, Globals, Utils, Data, Network, Train, and Evaluation. Most reusable
logic is kept in `experimental_setup/src/`, while the notebooks orchestrate the
experiment.

## Dependency Installation

The experimental setup can reuse the same Python environment created for the
root project. This is the simplest option when running both implementations in
the same VS Code workspace.

If the experimental folder is run in isolation, install its dependencies from
inside `experimental_setup/`:

```bash
cd experimental_setup
pdm install
pdm run python -m ipykernel install --user --name cv-project-25-26-exp
```

Then select the `cv-project-25-26-exp` kernel in VS Code.

The equivalent setup with `venv` and `pip` is:

```bash
cd experimental_setup
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision lightning numpy scipy einops datasets h5py matplotlib scikit-learn timm opencv-python ipykernel ipywidgets wandb kaggle
python -m pip install git+https://github.com/galilai-group/lejepa.git
python -m ipykernel install --user --name cv-project-25-26-exp
```

Dataset downloads use Kaggle, so the Kaggle API token must be configured before
running the download helpers.

## How to Use This Setup

Open the repository in VS Code, then open:

1. `experimental_setup/main.ipynb`
2. `experimental_setup/evaluation.ipynb`

The Jupyter extension opens the notebooks automatically inside VS Code. Select
the root project kernel or the isolated `cv-project-25-26-exp` kernel described
above.

Run `main.ipynb` first to execute:

1. LeJEPA SSL pretraining.
2. METER fine-tuning from the pretrained encoder.
3. Supervised METER training from scratch as the baseline.

Then run `evaluation.ipynb` to generate metrics, qualitative comparisons, PCA
probing figures, convergence plots, and efficiency-accuracy summaries.

## Dataset and Outputs

The experimental setup uses Kaggle-hosted copies of NYU Depth V2 and KITTI:

- NYU Depth V2: [artemmmtry/nyu-depth-v2](https://www.kaggle.com/datasets/artemmmtry/nyu-depth-v2)
- KITTI Eigen Split: [awsaf49/kitti-eigen-split-dataset](https://www.kaggle.com/datasets/awsaf49/kitti-eigen-split-dataset)

Download the datasets from Kaggle, extract them, and place the extracted
contents inside the paths expected by `experimental_setup/src/config.py`.
When running locally, those paths are:

```text
experimental_setup/data/raw/nyu/
experimental_setup/data/raw/kitti/
```

After extraction, `DepthEstimationDataset` expects the following layouts.

```text
experimental_setup/data/raw/nyu/
+-- data/
    +-- nyu2_train/
    |   +-- <scene>/
    |       +-- <frame>.jpg
    |       +-- <frame>.png
    +-- nyu2_test/
        +-- <scene>/
            +-- <frame>_colors.png
            +-- <frame>_depth.png
```

```text
experimental_setup/data/raw/kitti/
+-- train/
|   +-- <date>/
|       +-- <drive>_sync/
|           +-- image_02/data/<frame>.png
|           +-- proj_depth/groundtruth/image_02/<frame>.png
+-- test/
    +-- <date>/
        +-- <drive>_sync/
            +-- image_02/data/<frame>.png
            +-- proj_depth/groundtruth/image_02/<frame>.png
```

The experimental outputs are kept inside this folder under:

```text
experimental_setup/output/json/
experimental_setup/output/csv/
experimental_setup/output/images/
```

These artifacts should be interpreted as supporting evidence for the project
discussion, while the root setup remains the main reference implementation.
