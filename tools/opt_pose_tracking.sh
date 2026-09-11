#!/bin/bash
# Inside localhost/optpose, on an allocated Linux GPU. See OPT_TRACKING.md.
# bash tools/opt_pose_tracking.sh [camera|geometry] [queries] [sequences] [data_root] [dtype]
# Viewpoint controls are environment variables, since the defaults hid a fault:
# consecutive references over a stride-1 clip gave a cache holding ONE viewpoint
# (0.36 deg spread) and queries reaching only 16.5 deg. Set these deliberately.
#   OPT_SEQ_NAMES  object(s) by name, e.g. "teapot-white_floral" (default: first sorted)
#   OPT_REF_STRIDE spacing between reference frames      (default 1 = the old fault)
#   OPT_STRIDE     spacing between query frames          (default 1)
#   OPT_START      first reference frame                 (default 0)
#   OPT_NUM_REF    reference count, 1-3                  (default 3)
set -euo pipefail
MODE="${1:-camera}"
QUERIES="${2:-50}"
SEQUENCES="${3:-1}"
DATA_ROOT="${4:-/tmp/data/housecat6d}"
DTYPE="${5:-bf16}"
STRIDE="${OPT_STRIDE:-1}"
REF_STRIDE="${OPT_REF_STRIDE:-1}"
START="${OPT_START:-0}"
NUM_REF="${OPT_NUM_REF:-3}"
SEQ_NAMES="${OPT_SEQ_NAMES:-}"
TAG="$(printf %s "${SEQ_NAMES:-default}" | tr -c 'A-Za-z0-9_.-' '_')"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
NAME="optpose_tracking_${SLURM_JOB_ID:-manual}_${MODE}_${TAG}_r${REF_STRIDE}_s${STRIDE}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
WORK="/tmp/$NAME"
RUN="$WORK/run"
mkdir -p "$WORK" "$ROOT/optpose_tracking_out"
# Persist exact prepared inputs and partial outputs even on a normal command error.
# SIGKILL/node loss cannot run a shell trap; Slurm's persistent log remains available.
trap 'STATUS=$?
  printf "%s\n" "$STATUS" > "$WORK/exit_status.txt"
  tar -cf "$ROOT/optpose_tracking_out/$NAME.tar.partial" -C /tmp "$NAME"
  mv "$ROOT/optpose_tracking_out/$NAME.tar.partial" "$ROOT/optpose_tracking_out/$NAME.tar"
  printf "Artifact: %s\n" "$ROOT/optpose_tracking_out/$NAME.tar"
  exit "$STATUS"' EXIT
cd "$REPO"
cp -r "${OPT_TRACKING_PROVENANCE:?Run tools/opt_pose_tracking.sbatch outside the container}" "$WORK/provenance"
{
  hostname
  date -u
  ulimit -m
  nvidia-smi
  cat "$WORK/provenance/superrepo_commit.txt" "$WORK/provenance/opt_commit.txt"
} > "$WORK/environment.txt"
tar -czf "$WORK/source.tar.gz" \
  tools/opt_pose_tracking.sh tools/opt_pose_tracking.sbatch tools/opt_pose_tracking_report.py tools/OPT_TRACKING.md \
  opt_pose/opt opt_pose/test_tracking_housecat6d.py opt_pose/test_kvcache_housecat6d.py \
  opt_pose/test_causal_housecat6d.py opt_pose/test_abs_housecat6d.py \
  opt_pose/training/config opt_pose/training/data/datasets/housecat.py
cd "$REPO/opt_pose"
COMMON=(--run "$RUN" --mode "$MODE" --queries "$QUERIES" --num_seqs "$SEQUENCES"
        --num_ref "$NUM_REF" --data_root "$DATA_ROOT" --dtype "$DTYPE"
        --start "$START" --stride "$STRIDE" --ref_stride "$REF_STRIDE"
        --opt_commit "$(cat "$WORK/provenance/opt_commit.txt")")
if [ -n "$SEQ_NAMES" ]; then
  # Deliberately unquoted: several names may be given, space separated.
  COMMON+=(--seq_names $SEQ_NAMES)
fi
for METHOD in prepare verify cached original readout; do
  python -u test_tracking_housecat6d.py --method "$METHOD" "${COMMON[@]}" \
    2>&1 | tee "$WORK/$METHOD.log"
done
python "$REPO/tools/opt_pose_tracking_report.py" "$RUN" 2>&1 | tee "$WORK/report.log"
echo "TRACKING COMPARISON OK"
