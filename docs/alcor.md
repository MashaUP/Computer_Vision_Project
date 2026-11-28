# Working on the ALCOR machines

Full lab rules: <https://alcor.dpdns.org/guides/>. What matters for this repository:

- **Machine:** Tower GTX `alcor-gtx`, 192.168.110.41 (VPN DIAG or Sapienza network, SSH only).
  GPU 0 is a GTX 1080 (8 GB), GPUs 1 and 2 are GTX 1080 Ti (11 GB). 24 threads, 15 GB of RAM.
- **Files:** everything large under `/mnt/hdd2/<user>/`, never in the home (it gets deleted).
- **Software:** nothing installed on the host; everything runs in our Podman image.
- **GPUs:** book them on the portal before using them (thesis students: 6 credits = 6 GPU-days).
  Unbooked jobs are killed. Release the GPUs you no longer use.
- **No debugging on the GPUs:** run `pytest` and `main.py check` on CPU (here or locally) first.
- **Restarts:** every 3 months from January. Every training run saves checkpoints and resumes.

## Folder layout on the lab machine
```
/mnt/hdd2/<user>/
├── Computer_Vision_Project/   this repository       -> /work/project in the container
├── data/                      datasets              -> /work/data
├── runs/                      checkpoints and logs  -> /work/runs
└── cache/                     pip, torch hub, matplotlib -> /work/cache
```

## First time
```bash
cd /mnt/hdd2/$USER
git clone --recurse-submodules https://github.com/MashaUP/Computer_Vision_Project.git
cd Computer_Vision_Project
podman login docker.io -u <dockerhub-user>     # once
PUSH=1 scripts/alcor/build.sh                  # builds docker.io/masciarelli/masciarelli_schiavella:latest (cu118)
scripts/alcor/run.sh cpu check python3 main.py check     # CPU: no booking needed
podman logs -f check
```

## Running a job
```bash
scripts/alcor/run.sh <gpu> <job-name> python3 main.py <command> ...   # GPU must be booked for today
podman logs -f <job-name>        # or: tail -f /mnt/hdd2/$USER/runs/<job-name>/stdout.log
```
`run.sh` refuses a GPU that already runs processes, passes exactly one GPU (`--device nvidia.com/gpu=N`,
seen as `cuda:0` inside), uses `--ipc host` and `--cpus=6`, and mounts the four folders above.
Jobs run detached, so closing the SSH session does not stop them (tmux is still handy for watching).

## ZipDepth pseudo-depth (before its step-3 runs)
`loss: zipdepth` needs the Depth Anything V2 Large targets of the training frames, written once per dataset
to `data/pseudo_depth/dav2_vitl/` (fp32, resumable: re-run the same command after an interruption).
```bash
# time it first on 200 frames, then launch the full list (GPU booked for today)
scripts/alcor/run.sh 2 dav2_nyu_time python3 scripts/make_pseudo_depth.py --dataset nyu --list splits/nyu_raw_triplets_train.txt --limit 200
scripts/alcor/run.sh 2 dav2_nyu python3 scripts/make_pseudo_depth.py --dataset nyu --list splits/nyu_raw_triplets_train.txt
```
The weights (1.34 GB) are downloaded on first use to `cache/dav2/depth_anything_v2_vitl.pth`. If Hugging Face
is not reachable from the lab, download them elsewhere and copy them there with `scp`.
`--list splits/...` is relative to the data root: the lists live in `/mnt/hdd2/<user>/data/splits/`.

Measured on the Tower GTX (28/11, NYU, GPU 2 = 1080 Ti, fp32): Hugging Face reachable, 2.0 frames/s,
2.8 GB GPU memory, ~53 MB of PNG per 200 frames, so ~3.4 h and ~6.6 GB for the 24,863 NYU triplets.
KITTI frames are wider (375 x 1242): time it with `--limit 200` before the full run (27,154 triplets).

## KITTI split lists (CPU, no booking)
```bash
scripts/alcor/run.sh cpu kitti_splits python3 -m data.download.download_kitti --make_splits
```
Writes the seven `kitti_*` lists to `/mnt/hdd2/<user>/data/splits/` (it overwrites existing ones). Run on 28/11: 109 train /
14 val / 28 Eigen test drives, 27,154 train triplets, 652 test pairs.

## Reference weights and CPU evaluation (step 2)
Released weights stay outside the repository, in `cache/public_weights/` (see `weights/README.md`); download
large files elsewhere and copy them with `scp` when the lab cannot reach the source (Google Drive). Test-set
evaluation of a reference model fits on CPU (2-11 min per model and dataset), so it needs no booking:
the exact commands of step 2 are in `results/step2_public/README.md`.

## After a campaign
```bash
scripts/alcor/cleanup.sh          # stop/remove our containers, drop dangling layers
ALL=1 scripts/alcor/cleanup.sh    # also remove the image (pull it again from Docker Hub when needed)
```
Release the GPU booking on the portal if you finished early.
