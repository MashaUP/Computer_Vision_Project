#!/usr/bin/env bash
# Build the project image on the lab machine (no GPU needed, no booking) and optionally push it.
#
#   scripts/alcor/build.sh                 # cu126 (Pascal) -> docker.io/masciarelli/masciarelli_schiavella:latest
#   PUSH=1 scripts/alcor/build.sh          # ... and push it to Docker Hub (needs `podman login docker.io`)
#   CUDA=cu130 scripts/alcor/build.sh      # RTX 5070 build -> ...:cu130
set -euo pipefail
cd "$(dirname "$0")/../.."

CUDA="${CUDA:-cu126}"
IMAGE="${IMAGE:-docker.io/masciarelli/masciarelli_schiavella}"
TAG="${TAG:-$([[ "$CUDA" == cu126 ]] && echo latest || echo "$CUDA")}"

podman build --build-arg CUDA="$CUDA" -t "$IMAGE:$TAG" -f docker/Dockerfile .
echo "built $IMAGE:$TAG"
if [[ "${PUSH:-0}" == 1 ]]; then
    podman push "$IMAGE:$TAG"
fi
