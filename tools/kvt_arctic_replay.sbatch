#!/bin/bash
# Stage 1 of the keyframe-selection experiment: replay fixed keyframe sets through
# KV-Tracker's existing keyframe_indices argument. kv_tracker itself is unmodified.
#
#   sbatch tools/kvt_arctic_replay.sbatch
#
# Batch rather than interactive because `srun --pty` has to build a thread tree on
# head, which is where it keeps failing with
#   _start_msg_tree_internal: pthread_create error Resource temporarily unavailable
# `sbatch` only queues a script. See docs/camp-cluster-field-notes.md, "`srun` itself
# fails when head is out of process slots".
#
# Pinned to muenchen deliberately: the baseline ATEs this gate must reproduce were
# measured on its A5000 under bf16 autocast, so a different card is a confound.
#SBATCH --job-name=kvt-arctic-replay
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/kvt_arctic_replay_slurm-%j.log
set -euo pipefail
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
BASELINE="${1:-$ROOT/kvt_arctic_out/arctic_r518_20260915T152911Z.tar}"
SETS=/tmp/kf_sets.json
UNPACKED=/tmp/kf_baseline_run

hostname
nvidia-smi
ulimit -m

# The pull belongs here, not on head: head's memory cgroup kills a real fetch with a
# misleading "corrupt pack", and a separate pull job would contend for the one GPU
# students_normal allows. docs/camp-cluster-field-notes.md, "`git pull` belongs
# inside the allocation". Submodules are pinned and not needed by this job.
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
git -C "$REPO" rev-parse HEAD

# The sets are derived from the baseline run's own traj/kf_idx/keyframes, so the
# archive it produced is the input. /tmp is per node, hence the unpack here.
rm -rf "$UNPACKED"
mkdir -p "$UNPACKED"
tar -xf "$BASELINE" -C "$UNPACKED"
bash "$REPO/tools/kvt_run.sh" keyframe-sets \
  --run-root "$UNPACKED" --arc 25 --out "$SETS"

# Fidelity gate. These are the keyframes the baseline already chose, so this run has
# to reproduce it: box 0.183, ketchup 0.294, espresso 0.150. If it does not, replay
# is not faithful and every policy below it is void.
bash "$REPO/tools/kvt_run.sh" arctic-run --results kf_baseline --no-viz \
  --keyframes-from "$SETS" --policy baseline

# arc25 is the matched-count corrected geometry; uniform and random are the
# matched-budget nulls that say whether *which* frames you keep matters at all.
for POLICY in arc25 uniform_match_baseline \
              random_match_baseline_s0 random_match_baseline_s1 random_match_baseline_s2; do
  bash "$REPO/tools/kvt_run.sh" arctic-run --results "kf_$POLICY" --no-viz \
    --keyframes-from "$SETS" --policy "$POLICY"
done

echo "JOB OK"
