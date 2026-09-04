#!/bin/bash
# Inference half of the causal-readout Step 0. Runs INSIDE localhost/optpose.
# Invoked by tools/opt_pose_causal.sbatch; not meant to be run by hand.
#
#   bash tools/opt_pose_causal_run.sh <num_ref> <num_seqs> [data_root] [query_gap] [start] [dtype]
#
# The dataset is already extracted by the sbatch wrapper, on the node and not in
# here. Results go straight to /mnt so they survive the allocation.
#
# data_root defaults to job-local /tmp, the standing convention across every
# line in this repo: the annotation cache under <data_root>/cache rebuilds in
# about a second, and the file count on /mnt is the binding quota. It was
# hardcoded to /mnt until now, which is why the 2026-09-03 session bypassed this
# script and called python directly.
#
# query_gap is the distance from the last reference frame to the query frame
# (1 = adjacent, what Step 0 measured), and start shifts the whole window --
# together they sweep temporal separation and cache size without confounding
# either. The output name carries all four, so sweep cells cannot overwrite
# each other.
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

NUM_REF="${1:?usage: opt_pose_causal_run.sh <num_ref> <num_seqs> [data_root] [query_gap] [start]}"
NUM_SEQS="${2:?usage: opt_pose_causal_run.sh <num_ref> <num_seqs> [data_root] [query_gap] [start]}"
DATA_ROOT="${3:-/tmp/data/housecat6d}"
QUERY_GAP="${4:-1}"
START="${5:-0}"
DTYPE="${6:-fp32}"
ROOT=/mnt/projects/gr/3DRecon
OUT="$ROOT/optpose_causal_out"

cd "$(dirname "$0")/../opt_pose"

python test_causal_housecat6d.py \
  --data_root  "$DATA_ROOT" \
  --checkpoint "$ROOT/opt_pose_ckpt/abs_pose_housecat.pt" \
  --num_ref    "$NUM_REF" \
  --num_seqs   "$NUM_SEQS" \
  --query_gap  "$QUERY_GAP" \
  --start      "$START" \
  --dtype      "$DTYPE" \
  --out        "$OUT/causal_ref${NUM_REF}_gap${QUERY_GAP}_start${START}_seqs${NUM_SEQS}_${DTYPE}.json"

echo "RUN OK"
