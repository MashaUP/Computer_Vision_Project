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
training/          LeJEPA objective (classic | dense), step-3 loss, trainer with resume
evaluation/        protocol metrics (crops, caps, median / scale-shift alignment)
configs/           recipes (configs/lejepa/classic.yaml, dense.yaml)
docker/            Dockerfile (cu118 for the lab GPUs, cu130 for an RTX 5070)
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
pip install '.[test]' && python -m pytest -q && python main.py check
```
On the ALCOR machines everything runs in Podman: see [`docs/alcor.md`](docs/alcor.md).

## Status
| Step | State |
|---|---|
| Repository restructure, common model interface, protocol metrics, container and lab scripts | done |
| LeJEPA classic / dense in one code path; phase-1 recipes reproduce the exam code exactly (`scripts/verify_phase1_equivalence.py`) | done |
| One trainer for pre-training and frozen-encoder fine-tuning, with checkpoint + resume | done (step-3 loss: photometric for FlexDepth/BoRe, ZipDepthLoss on Depth Anything V2 pseudo-depth for ZipDepth) |
| NYU complete download and labelled-frame extraction (`data/download/nyu.py`) | done, to run on the lab |
| FlexDepth on NYU, original trainer (`scripts/train_flexdepth_nyu.py`, lists from `data.download.nyu triplets`) | script + CPU test done; GPU run on the lab pending |
| Public-weight loaders (step 1), KITTI preparation, NYU raw depth alignment | next |

## Running
```bash
python main.py pretrain --config configs/runs/example_pretrain.yaml          # LeJEPA classic | dense
python main.py finetune --config configs/runs/example_finetune.yaml          # frozen encoder, decoder only
python main.py evaluate --config configs/runs/example_finetune.yaml --test-list splits/nyu_labeled_test.txt
```
Every run writes to `<runs root>/<name>/` (`config.yaml`, `metrics.csv`, `last.ckpt`) and resumes
by itself when the same command is run again.

## Phase 1 (course project)
The exam version (two METER setups, "classic" at the root and "experimental" dense, with notebooks,
checkpoints and outputs, with full history) is the original repository
<https://github.com/Zunza5/lejepa-monocular-depth-estimation>. Its results are in
[`results/phase1_meter/`](results/phase1_meter/); METER itself lives on in `models/meter.py`.
