#!/bin/bash
# Submit twice, once with a3 and once with b. Select partition/node at submission.
#SBATCH --job-name=cache-pilots
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/cache_pilots-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
test "$(hostname -s)" != head
MODE=${1:?Choose a3 or b}
case "$MODE" in a3|b) ;; *) echo "Unknown pilot group: $MODE" >&2; exit 2 ;; esac
REPO=/mnt/projects/gr/3DRecon/layer_good
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
export SLURM_ARRAY_TASK_ID=0
if [ "$MODE" = a3 ]; then
    bash "$REPO/tools/stream_cache_long.sbatch" streamvggt,longstream precision-pilot
else
    bash "$REPO/tools/stream_cache_long.sbatch" streamvggt structure-pilot
fi
echo "OVERNIGHT PILOTS OK group=$MODE"
