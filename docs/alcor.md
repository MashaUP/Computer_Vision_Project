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

## After a campaign
```bash
scripts/alcor/cleanup.sh          # stop/remove our containers, drop dangling layers
ALL=1 scripts/alcor/cleanup.sh    # also remove the image (pull it again from Docker Hub when needed)
```
Release the GPU booking on the portal if you finished early.
