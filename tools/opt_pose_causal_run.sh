#!/bin/bash
# Inference half of the causal-readout Step 0. Runs INSIDE localhost/optpose.
# Invoked by tools/opt_pose_causal.sbatch; not meant to be run by hand.
#
#   bash tools/opt_pose_causal_run.sh <num_ref> <num_seqs>
#
# The dataset is already extracted by the sbatch wrapper, on the node and not in
# here. Results go straight to /mnt so they survive the allocation.
#
# num_ref is the cache size. Keep num_ref + 1 <= 4: the checkpoint was trained
# with `img_nums: [2, 4]` (training/config/housecat_default.yaml), so a longer
# window adds a frame-count distribution shift on top of the one-directional
# attention shift and the two become impossible to separate.
#
# The script exits non-zero when the SELFCHECK exceeds its tolerance. That is
# the go/no-go: it means a cross-frame path is still ungated, and the DRIFT
# numbers from the same run say nothing about the research question.
set -euxo pipefail

NUM_REF="${1:?usage: opt_pose_causal_run.sh <num_ref> <num_seqs>}"
NUM_SEQS="${2:?usage: opt_pose_causal_run.sh <num_ref> <num_seqs>}"
ROOT=/mnt/projects/gr/3DRecon
OUT="$ROOT/optpose_causal_out"

cd "$(dirname "$0")/../opt_pose"

python test_causal_housecat6d.py \
  --data_root  "$ROOT/housecat6d" \
  --checkpoint "$ROOT/opt_pose_ckpt/abs_pose_housecat.pt" \
  --num_ref    "$NUM_REF" \
  --num_seqs   "$NUM_SEQS" \
  --out        "$OUT/causal_ref${NUM_REF}_seqs${NUM_SEQS}.json"

echo "RUN OK"
