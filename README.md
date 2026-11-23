# Lean Geometry: LeJEPA pre-training for lightweight monocular depth

Does in-domain **LeJEPA** self-supervised pre-training give a lightweight depth encoder better
features than random initialisation, at zero inference cost? We answer it with a controlled
comparison on three recent encoder-decoders (**FlexDepth**, **ZipDepth**, **BoRe-Depth**) plus
**METER** from the course project, on **NYU Depth V2** and **KITTI**.

Sapienza University of Rome, ALCOR Lab. Supervision: Claudio Schiavella and Prof. Irene Amerini.
It continues Project 7 of the Computer Vision course ("Lean Geometry: Boosting Mobile MDE with LeJEPA").

## Protocol in one paragraph
For every model and dataset: (1) the public weights are evaluated on the test sets as they are
(FlexDepth on NYU is retrained by us); (2) the encoder is pre-trained with LeJEPA in two variants,
**classic** (official recipe) and **dense** (METER recipe), on unlabelled NYU raw / KITTI raw frames
with the test scenes excluded; (3) the encoder is **frozen** and only the decoder is trained, with
the same procedure for three initialisations (random, LeJEPA classic, LeJEPA dense) and three seeds.
Only the architectures are kept: method-specific heuristics are dropped. Details: [`docs/protocol.md`](docs/protocol.md).

## Repository layout
The top level follows the ALCOR thesis-repository structure (`models/`, `data/`, `sota/`, `utils/`, `main.py`).

```
main.py            single entry point: python main.py <command>
models/            METER (meter.py), common interface of every model (baselines.py), LeJEPA projectors
data/              data code only (views for LeJEPA, download scripts, split lists); data lives outside the repo
sota/              third-party code, never modified: baselines as pinned submodules, course reference files
utils/             device/precision policy, seeding, paths, result I/O
training/          LeJEPA objective (classic | dense), METER supervised loss
evaluation/        protocol metrics (crops, caps, median / scale-shift alignment), phase-1 METER utilities
configs/           recipes (configs/lejepa/classic.yaml, dense.yaml)
docker/            Dockerfile (cu126 for the lab GPUs, cu130 for an RTX 5070)
scripts/alcor/     build, run and clean up containers on the ALCOR machines
tests/             CPU tests (pytest)
results/           small final tables only; phase-1 results in results/phase1_meter/
docs/              protocol, how to work on the lab machines, references
```

## Quick start
```bash
git clone --recurse-submodules https://github.com/MashaUP/Computer_Vision_Project.git
cd Computer_Vision_Project
# local (any machine with PyTorch): `pip install .` installs the dependencies only (code runs from the repo root)
pip install . && python -m pytest -q && python main.py check
```
On the ALCOR machines everything runs in Podman: see [`docs/alcor.md`](docs/alcor.md).

## Status
| Step | State |
|---|---|
| Repository restructure, common model interface, LeJEPA classic/dense objective, protocol metrics, container | done |
| Data scripts (NYU raw frames via HTTP range requests, KITTI raw + annotated, scene-level splits) | next |
| Evaluation of the public weights (step 1) | next |
| LeJEPA pre-training (step 2) and frozen-encoder decoder training (step 3) | after the step-3 loss decision |

## Phase 1 (course project)
The exam version (two METER setups, "classic" at the root and "experimental" dense, with notebooks,
checkpoints and outputs) is commit `52aa283` (tag `v1.0-course-project`). Its results are in
[`results/phase1_meter/`](results/phase1_meter/); METER itself lives on in `models/meter.py`.
