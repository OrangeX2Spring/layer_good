#!/bin/bash
# One host per array, one full TUM sequence per task. Submit --array=0-2%1.
# Usage: sbatch tools/stream_cache_long.sbatch streamvggt|longstream|stream3r
# StreamVGGT calibration: sbatch --array=0 tools/stream_cache_long.sbatch streamvggt calibrate
# Combined policy: submit --array=0. combined-pilot takes one or more comma
# separated hosts and runs the office pilot for each; combined runs all three
# sequences for each host inside this one job. Slurm's MaxSubmitPU counts
# submissions, not loops, so several runs per job stay within the QoS limit.
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
HOSTS=${1:?Choose streamvggt, longstream or stream3r}
STAGE=${2:-full}
if [[ "$HOSTS" == *,* && $# -lt 2 ]]; then
    echo "Multi-host runs require an explicit stage; use correspondence-gates for all correspondence gates" >&2
    exit 2
fi
if [ "$HOSTS" = correspondence-gates ]; then
    HOSTS=stream3r,streamvggt,longstream
    STAGE=correspondence-gates
fi
case "$STAGE" in full|calibrate|coverage|combined|combined-pilot|correspondence-gates) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
echo "WRAPPER stage=$STAGE hosts=$HOSTS commit=$(git -C "$REPO" rev-parse --short HEAD)"
case "$SLURM_ARRAY_TASK_ID" in
    0) ARRAY_SCENE=freiburg3_long_office_household; ARRAY_COUNT=2585 ;;
    1) ARRAY_SCENE=freiburg2_large_with_loop; ARRAY_COUNT=5182 ;;
    2) ARRAY_SCENE=freiburg2_large_no_loop; ARRAY_COUNT=3359 ;;
    *) echo "Expected array task 0, 1 or 2" >&2; exit 2 ;;
esac

# Each scene is a separate preparation, run and archive; stream_cache.sbatch
# names its own work directory and archive, so sequential calls do not collide.
run_scene() {
    SCENE=$1
    COUNT=$2
    export CACHE_TUM_ZIP="/mnt/datasets/tum-rgbd/rgbd_dataset_${SCENE}.zip"
    export CACHE_TUM_COUNT="$COUNT" CACHE_TUM_START=0 CACHE_TUM_STRIDE=1
    export CACHE_WIDTH=308 CACHE_MAX_PAIR_GAP=0.1
    export CACHE_IMAGE=localhost/optpose CACHE_IMAGE_TAR="$ROOT/optpose.tar"
    unset CACHE_INPUT_TAR CACHE_MANIFEST CACHE_ARCTIC_SCENE CACHE_ARCTIC_COUNT
    echo "SWEEP stage=$STAGE host=$CACHE_HOST scene=$SCENE frames=$COUNT config=$CACHE_SWEEP"
    bash "$REPO/tools/stream_cache.sbatch"
}
run_object() {
    export CACHE_ARCTIC_SCENE=box_grab_01 CACHE_ARCTIC_COUNT=121
    export CACHE_WIDTH=308 CACHE_IMAGE=localhost/optpose CACHE_IMAGE_TAR="$ROOT/optpose.tar"
    unset CACHE_TUM_ZIP CACHE_TUM_COUNT CACHE_INPUT_TAR CACHE_MANIFEST
    echo "SWEEP stage=$STAGE host=$CACHE_HOST scene=box_grab_01 frames=121 config=$CACHE_SWEEP"
    bash "$REPO/tools/stream_cache.sbatch"
}

FAILURES=0

for CACHE_HOST in ${HOSTS//,/ }; do
    export CACHE_HOST
    case "$CACHE_HOST" in
        streamvggt)
            export CACHE_CHECKPOINT="$ROOT/streamvggt_ckpt/checkpoints.pth"
            export CACHE_SWEEP="$REPO/tools/stream_cache_streamvggt_reduced.json" ;;
        longstream)
            export CACHE_CHECKPOINT="$ROOT/longstream_ckpt/50_longstream.pt"
            export CACHE_SWEEP="$REPO/tools/stream_cache_long_longstream.json" ;;
        stream3r)
            export CACHE_CHECKPOINT="$ROOT/stream3r_ckpt"
            export CACHE_SWEEP="$REPO/tools/stream_cache_long_bounded.json" ;;
        *) echo "Unknown host: $CACHE_HOST" >&2; exit 2 ;;
    esac
    if [ "$STAGE" = correspondence-gates ]; then
        test "$SLURM_ARRAY_TASK_ID" = 0
        export CACHE_SWEEP="$REPO/tools/stream_cache_correspondence_object_gate.json"
        if ! run_object; then
            echo "GATE FAILED host=$CACHE_HOST task=object" >&2
            FAILURES=$((FAILURES + 1))
        fi
        export CACHE_SWEEP="$REPO/tools/stream_cache_correspondence_scene_gate.json"
        if ! run_scene freiburg3_long_office_household 128; then
            echo "GATE FAILED host=$CACHE_HOST task=scene" >&2
            FAILURES=$((FAILURES + 1))
        fi
    elif [ "$STAGE" = calibrate ] || [ "$STAGE" = coverage ]; then
        test "$CACHE_HOST" = streamvggt
        test "$SLURM_ARRAY_TASK_ID" = 0
        if [ "$STAGE" = coverage ]; then
            export CACHE_SWEEP="$REPO/tools/stream_cache_streamvggt_coverage.json"
        fi
        run_scene "$ARRAY_SCENE" 256
    elif [ "$STAGE" = combined ] || [ "$STAGE" = combined-pilot ]; then
        test "$CACHE_HOST" = streamvggt || test "$CACHE_HOST" = longstream
        test "$SLURM_ARRAY_TASK_ID" = 0
        export CACHE_SWEEP="$REPO/tools/stream_cache_combined.json"
        if [ "$STAGE" = combined-pilot ]; then
            run_scene freiburg3_long_office_household 256
        else
            run_scene freiburg3_long_office_household 2585
            run_scene freiburg2_large_with_loop 5182
            run_scene freiburg2_large_no_loop 3359
        fi
    else
        run_scene "$ARRAY_SCENE" "$ARRAY_COUNT"
    fi
done
test "$FAILURES" -eq 0
if [ "$STAGE" = correspondence-gates ]; then
    echo "ALL CORRESPONDENCE GATES OK"
fi
