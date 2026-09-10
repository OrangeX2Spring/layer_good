#!/bin/bash
# Run on an allocated Linux CPU node; uses an existing tracking archive and image.
set -euo pipefail
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
ARCHIVE="${1:?usage: opt_pose_tracking_accuracy.sh /path/to/tracking.tar}"
NAME="optpose_accuracy_$(date -u +%Y%m%dT%H%M%SZ)_$$"
WORK="/tmp/$NAME"
mkdir -p "$WORK" "$ROOT/optpose_tracking_out"
trap 'STATUS=$?; echo "$STATUS" > "$WORK/exit_status.txt"; tar -cf "$ROOT/optpose_tracking_out/$NAME.tar" -C /tmp "$NAME"; echo "Artifact: $ROOT/optpose_tracking_out/$NAME.tar"; exit "$STATUS"' EXIT
cp "$REPO/tools/opt_pose_tracking_accuracy.py" "$WORK/"
cp "$REPO/tools/opt_pose_tracking_accuracy.sh" "$WORK/"
tar -xf "$ARCHIVE" -C "$WORK"
RUNS=("$WORK"/optpose_tracking_*/run)
[ "${#RUNS[@]}" -eq 1 ]
RUN="${RUNS[0]}"
podman load -i "$ROOT/optpose.tar"
podman run --rm -v /tmp:/tmp:rw localhost/optpose \
  python "$WORK/opt_pose_tracking_accuracy.py" --self-check 2>&1 | tee "$WORK/self_check.log"
podman run --rm -v /tmp:/tmp:rw localhost/optpose \
  python "$WORK/opt_pose_tracking_accuracy.py" "$RUN" 2>&1 | tee "$WORK/accuracy.log"
