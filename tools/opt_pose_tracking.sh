#!/bin/bash
# Inside localhost/optpose, on an allocated Linux GPU. See OPT_TRACKING.md.
# bash tools/opt_pose_tracking.sh [camera|geometry] [queries] [sequences] [data_root] [dtype]
set -euo pipefail
MODE="${1:-camera}"
QUERIES="${2:-50}"
SEQUENCES="${3:-1}"
DATA_ROOT="${4:-/tmp/data/housecat6d}"
DTYPE="${5:-bf16}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
NAME="optpose_tracking_${SLURM_JOB_ID:-manual}_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
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
        --num_ref 3 --data_root "$DATA_ROOT" --dtype "$DTYPE"
        --opt_commit "$(cat "$WORK/provenance/opt_commit.txt")")
for METHOD in prepare verify cached original readout; do
  python -u test_tracking_housecat6d.py --method "$METHOD" "${COMMON[@]}" \
    2>&1 | tee "$WORK/$METHOD.log"
done
python "$REPO/tools/opt_pose_tracking_report.py" "$RUN" 2>&1 | tee "$WORK/report.log"
echo "TRACKING COMPARISON OK"
