#!/bin/bash
# Start (or re-enter) the OPT-Pose container on an allocated GPU node.
#
#   bash tools/opt_pose_shell.sh                  # base image, for the build
#   bash tools/opt_pose_shell.sh localhost/optpose  # after podman commit
#
# Exists as a script because this command is long enough to be corrupted by
# terminal line-wrapping when pasted (docs/opt-pose-cluster.md, gotcha 2).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-docker.io/pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel}"

podman run -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" \
  --name=optpose --network=host -it --replace "$IMAGE"
