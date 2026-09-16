#!/bin/bash
# CPU-only evaluation inside an existing CAMP allocation; no model inference.
set -euo pipefail
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d /tmp/stream3r_eval_XXXXXX)"
REPORT="$ROOT/stream3r_out/stream3r_arctic_eval_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}.json"
podman run --rm -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO" \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 localhost/optpose \
    python tools/stream3r_arctic_eval.py \
    --archives \
    "$ROOT/stream3r_out/stream3r_arctic_20260916T190534Z_25621_yALo7o.tar" \
    "$ROOT/stream3r_out/stream3r_arctic_20260916T190931Z_25621_eKETrF.tar" \
    --kvt "$ROOT/kvt_arctic_out/arctic_r518_20260915T152911Z.tar" \
    --out "$WORK/report.json"
cp "$WORK/report.json" "$REPORT"
echo "SAVED $REPORT"
