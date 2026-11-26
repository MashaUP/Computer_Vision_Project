#!/usr/bin/env bash
# Build the project image on the lab machine (no GPU needed, no booking) and optionally push it.
#
#   scripts/alcor/build.sh                 # cu118 (lab, torch 2.7.1) -> docker.io/masciarelli/masciarelli_schiavella:latest
#   PUSH=1 scripts/alcor/build.sh          # ... and push it to Docker Hub (needs `podman login docker.io`)
#   CUDA=cu130 scripts/alcor/build.sh      # RTX 5070 build (torch 2.14.*) -> ...:cu130
set -euo pipefail
cd "$(dirname "$0")/../.."

CUDA="${CUDA:-cu118}"
# torch depends on the CUDA build: cu118 wheels stop at 2.7.1, the RTX 5070 (cu130) needs a recent one
TORCH="${TORCH:-$([[ "$CUDA" == cu118 ]] && echo 2.7.1 || echo '2.14.*')}"
IMAGE="${IMAGE:-docker.io/masciarelli/masciarelli_schiavella}"
TAG="${TAG:-$([[ "$CUDA" == cu118 ]] && echo latest || echo "$CUDA")}"

podman build --build-arg CUDA="$CUDA" --build-arg TORCH="$TORCH" -t "$IMAGE:$TAG" -f docker/Dockerfile .
echo "built $IMAGE:$TAG"
if [[ "${PUSH:-0}" == 1 ]]; then
    podman push "$IMAGE:$TAG"
fi
