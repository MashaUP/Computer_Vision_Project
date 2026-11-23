#!/usr/bin/env bash
# Free shared disk space on the lab machine: stop and remove our containers, drop dangling layers.
#
#   scripts/alcor/cleanup.sh            # containers of our image + dangling layers
#   ALL=1 scripts/alcor/cleanup.sh      # also remove the image itself (rebuild or pull it afterwards)
set -euo pipefail
IMAGE="${IMAGE:-docker.io/masciarelli/masciarelli_schiavella}"

for tag in $(podman images "$IMAGE" --format '{{.Tag}}'); do
    ids="$(podman ps -a --filter ancestor="$IMAGE:$tag" -q)"
    if [[ -n "$ids" ]]; then
        echo "$ids" | xargs podman stop >/dev/null
        echo "$ids" | xargs podman rm -f >/dev/null || true
    fi
done
podman image prune -f >/dev/null
if [[ "${ALL:-0}" == 1 ]]; then
    podman images "$IMAGE" -q | sort -u | xargs -r podman image rm -f
fi
podman images
podman ps -a
