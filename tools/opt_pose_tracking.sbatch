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
# Viewpoint controls, forwarded into the container. Defaults reproduce the old
# behaviour; set them deliberately -- see the header of opt_pose_tracking.sh.
export OPT_SEQ_NAMES="${OPT_SEQ_NAMES:-}"
export OPT_REF_STRIDE="${OPT_REF_STRIDE:-1}"
export OPT_STRIDE="${OPT_STRIDE:-1}"
export OPT_START="${OPT_START:-0}"
export OPT_NUM_REF="${OPT_NUM_REF:-3}"
export OPT_TRACKING_PROVENANCE
OPT_TRACKING_PROVENANCE=$(mktemp -d /tmp/optpose_tracking_provenance.XXXXXX)
git -C "$REPO" rev-parse HEAD > "$OPT_TRACKING_PROVENANCE/superrepo_commit.txt"
git -C "$REPO/opt_pose" rev-parse HEAD > "$OPT_TRACKING_PROVENANCE/opt_commit.txt"
git -C "$REPO" diff --binary -- tools/opt_pose_tracking.sh tools/opt_pose_tracking.sbatch tools/opt_pose_tracking_report.py tools/OPT_TRACKING.md > "$OPT_TRACKING_PROVENANCE/superrepo.patch"
git -C "$REPO/opt_pose" diff --binary > "$OPT_TRACKING_PROVENANCE/opt_pose.patch"
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
  -e HF_HOME -e HF_XET_HIGH_PERFORMANCE -e SLURM_JOB_ID -e OPT_TRACKING_PROVENANCE \
  -e OPT_SEQ_NAMES -e OPT_REF_STRIDE -e OPT_STRIDE -e OPT_START -e OPT_NUM_REF \
  -v "$ROOT/.sonata_cache":/root/.cache/sonata:rw \
  localhost/optpose bash tools/opt_pose_tracking.sh \
    "$MODE" "$QUERIES" "$SEQUENCES" /tmp/data/housecat6d bf16
echo "JOB OK"
