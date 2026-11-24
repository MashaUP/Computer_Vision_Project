#!/usr/bin/env bash
# Run one job in a detached container on ONE booked GPU of the lab machine, following the ALCOR rules.
#
#   scripts/alcor/run.sh <gpu-index|cpu> <job-name> [command ...]
#
#   scripts/alcor/run.sh cpu check python3 main.py check            # CPU only: no booking needed
#   scripts/alcor/run.sh 2 lejepa_flexdepth_nyu python3 main.py ...  # GPU 2 must be booked for today
#
# Mounts (personal folder under /mnt, never the home):
#   $BASE/Computer_Vision_Project -> /work/project    $BASE/data  -> /work/data
#   $BASE/runs                    -> /work/runs       $BASE/cache -> /work/cache
# Output is also written to $BASE/runs/<job-name>/stdout.log; follow it with `podman logs -f <job-name>`.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,13p' "$0"; exit 1
fi
GPU="$1"; NAME="$2"; shift 2
[[ $# -gt 0 ]] || set -- python3 main.py env

BASE="${BASE:-/mnt/hdd2/${USER}}"
IMAGE="${IMAGE:-docker.io/masciarelli/masciarelli_schiavella:latest}"
CPUS="${CPUS:-6}"   # 24 threads shared by 3 GPUs
mkdir -p "$BASE/data" "$BASE/runs/$NAME" "$BASE/cache"

DEVICE_ARGS=()
if [[ "$GPU" != cpu ]]; then
    # refuse a GPU that already has compute processes (someone else's job, or one of ours)
    uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")"
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then
        echo "GPU $GPU is busy:"; nvidia-smi -i "$GPU"; exit 1
    fi
    echo "Using GPU $GPU: it must be booked on the ALCOR portal for today (unbooked jobs are killed)."
    DEVICE_ARGS=(--device "nvidia.com/gpu=$GPU")   # inside the container it is cuda:0
fi

podman run -d --rm --name "$NAME" \
    "${DEVICE_ARGS[@]}" --ipc host --cpus="$CPUS" \
    -v "$BASE/Computer_Vision_Project:/work/project" \
    -v "$BASE/data:/work/data" -v "$BASE/runs:/work/runs" -v "$BASE/cache:/work/cache" \
    -e LG_DATA_ROOT=/work/data -e LG_RUNS_ROOT=/work/runs -e LG_CACHE_ROOT=/work/cache \
    -w /work/project "$IMAGE" \
    bash -c 'set -o pipefail; "$@" 2>&1 | tee -a "/work/runs/'"$NAME"'/stdout.log"' bash "$@"

echo "started $NAME | logs: podman logs -f $NAME | file: $BASE/runs/$NAME/stdout.log"
