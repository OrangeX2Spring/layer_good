#!/bin/bash
# Run inside a CAMP allocation. No installs, downloads, or image mutations.
# bash tools/kvt_run.sh capture --input /mnt/.../input.json --out /mnt/.../run
# bash tools/kvt_run.sh artifacts --capture /mnt/.../run --evaluation /mnt/.../evaluation.json --out /mnt/.../artifacts
set -euo pipefail
test "$(uname -s)" = Linux
test "$(hostname)" != head
test -n "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
ROOT=/mnt/projects/gr/3DRecon/layer_good
STORE=/mnt/projects/gr/3DRecon
MODE="${1:?Expected prepare, capture, artifacts, or check}"
shift
GPU_ARGS=()
case "$MODE" in
  prepare) SCRIPT=kvt_prepare_housecat.py ;;
  capture)
    SCRIPT=kvt_capture.py
    GPU_ARGS=(--device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}")
    test -f "$HOME/.hf_env"
    source "$HOME/.hf_env"
    ;;
  artifacts) SCRIPT=kvt_artifacts.py ;;
  check) SCRIPT=test_kvt_artifacts.py ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
if ! podman image exists localhost/kvt; then
  podman load -i "$STORE/kvt.tar"
fi
podman run --rm --network=host \
  -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$ROOT/kv_tracker" \
  "${GPU_ARGS[@]}" \
  -e PYTHONPATH="$ROOT/kv_tracker/thirdparty/Pi3:$ROOT/kv_tracker" \
  -e HF_HOME="$STORE/.hf_cache" -e HF_TOKEN -e HF_XET_HIGH_PERFORMANCE=1 \
  localhost/kvt python "$ROOT/tools/$SCRIPT" "$@"
