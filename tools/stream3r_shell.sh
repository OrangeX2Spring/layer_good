#!/bin/bash
# Start (or re-enter) a container for STream3R work on an allocated node.
#
#   bash tools/stream3r_shell.sh                          # borrows localhost/optpose
#   bash tools/stream3r_shell.sh localhost/some-other     # override
#
# STream3R has NO image of its own, deliberately. Its inference import closure is
# torch, torchvision, numpy, PIL and huggingface_hub, and optpose.tar already
# carries every one (opt_pose/requirements.txt), because OPT-Pose is VGGT plus
# pose heads and STream3R is VGGT plus causal attention. Borrowing it read-only
# costs 0 GB against a 250 GB quota that is already at 183 GB, and touches
# nothing the OPT-Pose line depends on - no commit, no save.
#
# optpose.tar is also the only image built with 7.5 in TORCH_CUDA_ARCH_LIST, so
# it is the only one that runs on 12g. That matters: STream3R sits at 5.5-9 GB,
# under the 24g rule-(c) 12 GB floor.
#
# PYTHONPATH rather than `pip install -e .`: setup.py declares no
# install_requires, so the checkout is importable as-is, and this leaves the
# borrowed image unmodified.
#
# Exists as a script because the podman line is long enough to be corrupted by
# terminal wrapping when pasted (docs/opt-pose-cluster.md, gotcha 2).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-localhost/optpose}"

GPU_ARG=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_ARG=(--device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES")
fi

podman run -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  "${GPU_ARG[@]+"${GPU_ARG[@]}"}" \
  -e PYTHONPATH="$REPO/stream3r" \
  -e HF_HOME=/mnt/projects/gr/3DRecon/.hf_cache \
  --name=stream3r --network=host -it --replace "$IMAGE"
