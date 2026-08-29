#!/bin/bash
# Start (or re-enter) the GenRecon container on an allocated GPU node.
#
#   bash tools/genrecon_shell.sh                      # base image, for the build
#   bash tools/genrecon_shell.sh localhost/genrecon   # after podman commit
#
# Exists as a script because this command is long enough to be corrupted by
# terminal line-wrapping when pasted (docs/opt-pose-cluster.md, gotcha 2).
#
# Works with or without a GPU. On a `data`-partition allocation CUDA_VISIBLE_DEVICES
# is unset and the --device flag is dropped, which is how CPU-only maintenance (pip
# installs, podman commit, podman save) is done without tripping the 24g idle-GPU
# auto-stop rule that killed job 21016.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"

GPU_ARG=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_ARG=(--device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES")
fi

podman run -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  "${GPU_ARG[@]+"${GPU_ARG[@]}"}" \
  --name=genrecon --network=host -it --replace "$IMAGE"
