#!/bin/bash
# Start (or re-enter) the GenRecon container on an allocated GPU node.
#
#   bash tools/genrecon_shell.sh                      # base image, for the build
#   bash tools/genrecon_shell.sh localhost/genrecon   # after podman commit
#
# Exists as a script because this command is long enough to be corrupted by
# terminal line-wrapping when pasted (docs/opt-pose-cluster.md, gotcha 2).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"

podman run -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" \
  --name=genrecon --network=host -it --replace "$IMAGE"
