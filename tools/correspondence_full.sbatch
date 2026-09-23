#!/bin/bash
# Full correspondence methods only: four models x three ARCTIC + three TUM scenes.
# Submit after publishing and reviewing the correspondence gate evidence.
#SBATCH --job-name=correspondence-full
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/correspondence_full-%j.log
set -euo pipefail
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Submit through sbatch}"
REPO=/mnt/projects/gr/3DRecon/layer_good
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
echo "CORRESPONDENCE FULL models=4 method_runs=36 sam3=deferred"
echo "COMMIT $(git -C "$REPO" rev-parse HEAD)"
# Streaming wrapper requires a scene-array index; this mode loops over all scenes.
export SLURM_ARRAY_TASK_ID=0
bash "$REPO/tools/stream_cache_long.sbatch" correspondence-full
for scene in box_grab_01 ketchup_grab_01 espressomachine_grab_01; do
    bash "$REPO/tools/kvt_correspondence.sbatch" full "$scene"
done
SCENES=freiburg3_long_office_household,freiburg2_large_with_loop,freiburg2_large_no_loop
bash "$REPO/tools/kvt_tum.sbatch" correspondence "$SCENES" full
echo "ALL FOUR MODEL CORRESPONDENCE FULL RUNS OK"
