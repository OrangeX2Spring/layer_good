#!/bin/bash
# Two submissions: kvt and streaming. All experiments are short diagnostics.
#SBATCH --job-name=cache-sensitivity
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/cache_sensitivity-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
test "$(hostname -s)" != head
MODE=${1:?Choose kvt or streaming}
case "$MODE" in kvt|streaming) ;; *) echo "Unknown group: $MODE" >&2; exit 2 ;; esac
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
if [ "$MODE" = kvt ]; then
    bash "$REPO/tools/kvt_tum.sbatch" sensitivity
else
    unset CACHE_TUM_ZIP CACHE_ARCTIC_SCENE CACHE_INPUT_TAR CACHE_MANIFEST
    export CACHE_SWEEP="$REPO/tools/cache_sensitivity.json"
    export CACHE_IMAGE=localhost/optpose CACHE_IMAGE_TAR="$ROOT/optpose.tar"
    export CACHE_WIDTH=308
    for CACHE_HOST in longstream streamvggt; do
        export CACHE_HOST
        if [ "$CACHE_HOST" = longstream ]; then
            export CACHE_CHECKPOINT="$ROOT/longstream_ckpt/50_longstream.pt"
        else
            export CACHE_CHECKPOINT="$ROOT/streamvggt_ckpt/checkpoints.pth"
        fi
        bash "$REPO/tools/stream_cache.sbatch"
    done
fi
echo "OVERNIGHT SENSITIVITY OK group=$MODE"
