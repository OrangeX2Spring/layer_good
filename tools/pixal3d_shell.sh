#!/bin/bash
# Start the Pixal3D container on an allocated node.
#
#   bash tools/pixal3d_shell.sh                    # base image, for the build
#   bash tools/pixal3d_shell.sh localhost/pixal3d # after podman commit
#
# The GPU device is optional so dependency compilation and image maintenance can
# run on the data partition without holding an idle GPU.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"

PODMAN_ARGS=(
  -v /mnt:/mnt:rw
  -v /tmp:/tmp:rw
  -w "$REPO"
  -e ATTN_BACKEND=sdpa
  -e SPARSE_ATTN_BACKEND=sdpa
  -e HF_HOME=/mnt/projects/gr/3DRecon/.hf_cache
  -e HF_XET_HIGH_PERFORMANCE=1
  -e TORCH_HOME=/mnt/projects/gr/3DRecon/.torch_cache
)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  PODMAN_ARGS+=(--device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES")
fi
if [ -n "${HF_TOKEN:-}" ]; then
  PODMAN_ARGS+=(-e HF_TOKEN)
fi

podman run "${PODMAN_ARGS[@]}" \
  --name=pixal3d --network=host -it --replace "$IMAGE"
