#!/bin/bash
# Bounded pilot only. Select resources at submission; bootstrap pull in allocation.
#SBATCH --job-name=dynamic-history
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/dynamic_history-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
test "$(hostname -s)" != head
REPO=/mnt/projects/gr/3DRecon/layer_good
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
unset CACHE_TUM_ZIP CACHE_ARCTIC_SCENE CACHE_INPUT_TAR CACHE_MANIFEST
export CACHE_HOST=streamvggt CACHE_WIDTH=308
export CACHE_SWEEP="$REPO/tools/stream_cache_dynamic.json"
export CACHE_IMAGE=localhost/optpose
export CACHE_IMAGE_TAR=/mnt/projects/gr/3DRecon/optpose.tar
export CACHE_CHECKPOINT=/mnt/projects/gr/3DRecon/streamvggt_ckpt/checkpoints.pth
bash "$REPO/tools/stream_cache.sbatch"
