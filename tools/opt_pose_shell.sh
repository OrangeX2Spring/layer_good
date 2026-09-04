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
HF_HOME="${HF_HOME:-/mnt/projects/gr/3DRecon/.hf_cache}"
# sonata.load does NOT honour HF_HOME -- measured 2026-09-04, it writes 414 MB to
# ~/.cache/sonata, and rootless podman puts $HOME at /root, so the download is
# re-paid once per container. Bind-mounting that path persists it across
# allocations; HF_HOME above still covers everything that does go through the hub.
SONATA_CACHE=/mnt/projects/gr/3DRecon/.sonata_cache
mkdir -p "$SONATA_CACHE"

podman run -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" \
  -e HF_HOME="$HF_HOME" \
  -v "$SONATA_CACHE":/root/.cache/sonata:rw \
  --name=optpose --network=host -it --replace "$IMAGE"
