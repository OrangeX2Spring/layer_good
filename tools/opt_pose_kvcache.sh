#!/bin/bash
# Step 1a of the causal-readout line: the KV cache. Runs INSIDE localhost/optpose.
#
#   bash tools/opt_pose_kvcache.sh <num_ref> <num_seqs> [data_root] [query_gap] [start] [cache_only] [dtype]
#
# dtype is fp32 (default), tf32 or bf16, and the last two need sm_80+ -- so 24g,
# not the 12g Turing nodes, where bf16 is emulated and tf32 does not exist. bf16
# halves the cache as well as the matmuls: 128.8 MiB per reference frame instead
# of 257.6, which is what decides whether a 70-frame cache fits 24 GB.
#
# A non-empty sixth argument passes --cache_only: no readout baseline, no
# fidelity, no controls, just what the cached path costs. That is the only way
# the large-n cells run at all -- the baseline holds 24 intermediates of
# [B, S, P, 2C], which is 2.4 GB at n=8 and 8.9 GB at n=32 on top of 5.4 GB of
# weights, so it runs out of memory long before the cache does.
#
# Same shape as tools/opt_pose_causal_run.sh, and the same defaults: data_root
# is job-local /tmp, query_gap 1, start 0.
#
# num_ref here is NOT capped at 3 the way the causal sweep is. Accuracy past the
# checkpoint's `img_nums: [2, 4]` window is uninterpretable, but this script
# measures fidelity of the cached path against the readout, plus time and cache
# bytes -- none of which care what the checkpoint was trained on. The cost curve
# out to n = 8, 16, 32 is the point: dense per-patch K/V is roughly 270 MB per
# reference frame in fp32, so a KV-Tracker-scale cache does not fit a 16 GB card,
# and that number is the redundancy objection stated quantitatively.
#
# Expect an OOM somewhere in that sweep. Where it lands IS a result; record it
# rather than working around it.
set -euxo pipefail

NUM_REF="${1:?usage: opt_pose_kvcache.sh <num_ref> <num_seqs> [data_root] [query_gap] [start]}"
NUM_SEQS="${2:?usage: opt_pose_kvcache.sh <num_ref> <num_seqs> [data_root] [query_gap] [start]}"
DATA_ROOT="${3:-/tmp/data/housecat6d}"
QUERY_GAP="${4:-1}"
START="${5:-0}"
CACHE_ONLY="${6:-}"
DTYPE="${7:-fp32}"
ROOT=/mnt/projects/gr/3DRecon
OUT="$ROOT/optpose_kvcache_out"

mkdir -p "$OUT"
cd "$(dirname "$0")/../opt_pose"

SUFFIX="_$DTYPE"
EXTRA=(--dtype "$DTYPE")
if [ -n "$CACHE_ONLY" ]; then
  SUFFIX="${SUFFIX}_cacheonly"
  EXTRA+=(--cache_only)
fi

python test_kvcache_housecat6d.py \
  --data_root  "$DATA_ROOT" \
  --checkpoint "$ROOT/opt_pose_ckpt/abs_pose_housecat.pt" \
  --num_ref    "$NUM_REF" \
  --num_seqs   "$NUM_SEQS" \
  --query_gap  "$QUERY_GAP" \
  --start      "$START" \
  --out        "$OUT/kvcache_ref${NUM_REF}_gap${QUERY_GAP}_start${START}_seqs${NUM_SEQS}${SUFFIX}.json" \
  "${EXTRA[@]}"

echo "RUN OK"
