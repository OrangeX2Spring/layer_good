#!/bin/bash
#SBATCH --job-name=optpose-tracking
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/optpose_tracking_slurm-%j.log
set -euo pipefail
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
MODE="${1:-camera}"
QUERIES="${2:-50}"
SEQUENCES="${3:-1}"
hostname
nvidia-smi
ulimit -m
bash "$REPO/tools/opt_pose_extract_housecat.sh" /tmp/data/housecat6d test_scene1
export HF_HOME="$ROOT/.hf_cache"
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$ROOT/.sonata_cache"
podman load -i "$ROOT/optpose.tar"
podman run --rm -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" --network=host \
  -e HF_HOME -e HF_XET_HIGH_PERFORMANCE -e SLURM_JOB_ID \
  -v "$ROOT/.sonata_cache":/root/.cache/sonata:rw \
  localhost/optpose bash tools/opt_pose_tracking.sh \
    "$MODE" "$QUERIES" "$SEQUENCES" /tmp/data/housecat6d bf16
echo "JOB OK"
