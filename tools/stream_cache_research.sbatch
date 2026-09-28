#!/bin/bash
# Bounded pilots only. User submits one experiment at a time and reviews its archive.
# Usage: sbatch ... tools/stream_cache_research.sbatch context|context_diagnostic|context_mass|temporal|refresh [scene]
#SBATCH --job-name=cache-research
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/cache_research-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
test "$(hostname -s)" != head
REPO=/mnt/projects/gr/3DRecon/layer_good
METHOD=${1:?Choose context, context_diagnostic, context_mass, temporal or refresh}
case "$METHOD" in context|context_diagnostic|context_mass|temporal|refresh) ;; *) exit 2 ;; esac
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
unset CACHE_TUM_ZIP CACHE_ARCTIC_SCENE CACHE_INPUT_TAR CACHE_MANIFEST
export CACHE_HOST=streamvggt CACHE_WIDTH=308
export CACHE_SWEEP="$REPO/tools/stream_cache_research_${METHOD}.json"
export CACHE_IMAGE=localhost/optpose
export CACHE_IMAGE_TAR=/mnt/projects/gr/3DRecon/optpose.tar
export CACHE_CHECKPOINT=/mnt/projects/gr/3DRecon/streamvggt_ckpt/checkpoints.pth
if [[ "$METHOD" == context* ]]; then
    export CACHE_ARCTIC_SCENE=${2:-espressomachine_grab_01}
    case "$CACHE_ARCTIC_SCENE" in box_grab_01|ketchup_grab_01|espressomachine_grab_01) ;; *) exit 2 ;; esac
    export CACHE_ARCTIC_COUNT=128 CACHE_WIDTH=518 CACHE_ARCTIC_OBJECT_GT=1
else
    SCENE=${2:-freiburg3_long_office_household}
    case "$SCENE" in freiburg3_long_office_household|freiburg2_large_with_loop|freiburg2_large_no_loop) ;; *) exit 2 ;; esac
    export CACHE_TUM_ZIP="/mnt/datasets/tum-rgbd/rgbd_dataset_${SCENE}.zip"
    export CACHE_TUM_COUNT=256 CACHE_TUM_START=0 CACHE_TUM_STRIDE=1
    export CACHE_MAX_PAIR_GAP=0.1
fi
bash "$REPO/tools/stream_cache.sbatch"
