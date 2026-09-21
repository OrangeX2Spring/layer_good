#!/bin/bash
# One host per array, one full TUM sequence per task. Submit --array=0-2%1.
# Usage: sbatch tools/stream_cache_long.sbatch streamvggt|longstream|stream3r
# StreamVGGT calibration: sbatch --array=0 tools/stream_cache_long.sbatch streamvggt calibrate
#SBATCH --job-name=cache-long
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --array=0-2%1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/stream_cache_long-%A_%a.log
set -euo pipefail
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Submit with --array=0-2%1}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
export CACHE_HOST=${1:?Choose streamvggt, longstream or stream3r}
STAGE=${2:-full}
case "$STAGE" in full|calibrate) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
case "$CACHE_HOST" in
    streamvggt) export CACHE_CHECKPOINT="$ROOT/streamvggt_ckpt/checkpoints.pth" ;;
    longstream) export CACHE_CHECKPOINT="$ROOT/longstream_ckpt/50_longstream.pt" ;;
    stream3r) export CACHE_CHECKPOINT="$ROOT/stream3r_ckpt" ;;
    *) echo "Unknown host: $CACHE_HOST" >&2; exit 2 ;;
esac
case "$SLURM_ARRAY_TASK_ID" in
    0) SCENE=freiburg3_long_office_household; COUNT=2585 ;;
    1) SCENE=freiburg2_large_with_loop; COUNT=5182 ;;
    2) SCENE=freiburg2_large_no_loop; COUNT=3359 ;;
    *) echo "Expected array task 0, 1 or 2" >&2; exit 2 ;;
esac
export CACHE_SWEEP="$REPO/tools/stream_cache_long_bounded.json"
if [ "$CACHE_HOST" = streamvggt ]; then
    export CACHE_SWEEP="$REPO/tools/stream_cache_streamvggt_reduced.json"
fi
if [ "$CACHE_HOST" = longstream ]; then
    export CACHE_SWEEP="$REPO/tools/stream_cache_long_longstream.json"
fi
if [ "$STAGE" = calibrate ]; then
    test "$CACHE_HOST" = streamvggt
    test "$SLURM_ARRAY_TASK_ID" = 0
    COUNT=256
fi
export CACHE_TUM_ZIP="/mnt/datasets/tum-rgbd/rgbd_dataset_${SCENE}.zip"
export CACHE_TUM_COUNT="$COUNT" CACHE_TUM_START=0 CACHE_TUM_STRIDE=1
export CACHE_WIDTH=308 CACHE_MAX_PAIR_GAP=0.1
export CACHE_IMAGE=localhost/optpose CACHE_IMAGE_TAR="$ROOT/optpose.tar"
unset CACHE_INPUT_TAR CACHE_MANIFEST
echo "SWEEP stage=$STAGE host=$CACHE_HOST scene=$SCENE frames=$COUNT config=$CACHE_SWEEP"
exec bash "$REPO/tools/stream_cache.sbatch"
