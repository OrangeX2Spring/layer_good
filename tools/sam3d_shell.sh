#!/bin/bash
# Start (or re-enter) the SAM 3D Objects container on an allocated node.
#
#   bash tools/sam3d_shell.sh                   # base image, for the build
#   bash tools/sam3d_shell.sh localhost/sam3d   # after podman commit
#
# Exists as a script because this command is long enough to be corrupted by
# terminal line-wrapping when pasted (docs/opt-pose-cluster.md, gotcha 2).
#
# Works with or without a GPU. On a `data`-partition allocation CUDA_VISIBLE_DEVICES
# is unset and the --device flag is dropped, which is how CPU-only maintenance (pip
# installs, podman commit, podman save) is done without tripping the 24g idle-GPU
# auto-stop rule. Note the BUILD itself needs a GPU attached: pytorch3d otherwise
# compiles without GPU support and fails later.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-docker.io/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel}"

GPU_ARG=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_ARG=(--device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES")
fi

# Two runtime variables this image needs, both consequences of not using conda:
#   CONDA_PREFIX - notebook/inference.py:5 does
#     os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
#     and raises KeyError without it. Upstream got the CUDA toolkit from conda; here
#     it is at /usr/local/cuda, which is what CUDA_HOME should point at anyway.
#   HF_HOME - the pipeline fetches MoGe (Ruicheng/moge-vitl) at construction time.
#     Pointing at the project cache avoids re-downloading 1.26 GB into the container,
#     where it would die with the allocation.
podman run -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  "${GPU_ARG[@]+"${GPU_ARG[@]}"}" \
  -e CONDA_PREFIX=/usr/local/cuda \
  -e HF_HOME="${HF_HOME:-/mnt/projects/gr/3DRecon/.hf_cache}" \
  --name=sam3d --network=host -it --replace "$IMAGE"
