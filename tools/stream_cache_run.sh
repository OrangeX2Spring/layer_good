#!/bin/bash
# Inside an allocated GPU node, with the chosen compatible Podman image loaded.
# Arguments: host checkpoint prepared-inputs sweep-json [LongStream-model-yaml]
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a GPU allocation}"
: "${CUDA_VISIBLE_DEVICES:?No allocated GPU}"
: "${CACHE_IMAGE:?Set CACHE_IMAGE to the verified local container image}"
test "$(hostname -s)" != head
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT=/mnt/projects/gr/3DRecon
HOST=$1
CKPT=$2
INPUTS=$3
SWEEP=$4
MODEL_CONFIG=${5:-}
WORK="$(mktemp -d "/tmp/cache_${HOST}_${SLURM_JOB_ID}_XXXXXX")"
mkdir -p "$ROOT/stream_cache_out"
ARCHIVE="$ROOT/stream_cache_out/$(basename "$WORK").tar"
archive_run() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$WORK/exit_status.txt"
    tar -cf "$ARCHIVE.part" -C "$WORK" .
    mv "$ARCHIVE.part" "$ARCHIVE"
    echo "ARCHIVED $ARCHIVE (exit status $status)"
    exit "$status"
}
trap archive_run EXIT
exec > >(tee "$WORK/run.log") 2>&1
podman image inspect "$CACHE_IMAGE" > "$WORK/container.json"
nvidia-smi > "$WORK/nvidia-smi.txt"
ulimit -m > "$WORK/memory_grant_kb.txt"
cp "$REPO/tools/stream_cache_run.sh" "$WORK/"
container=(podman run --rm --device="nvidia.com/gpu=$CUDA_VISIBLE_DEVICES"
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO"
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
    -e HF_HUB_OFFLINE=1 "$CACHE_IMAGE")
"${container[@]}" pip freeze > "$WORK/pip_freeze.txt"
"${container[@]}" python -m unittest discover -s tools -p 'test_kvcache_policy.py'
"${container[@]}" python -m unittest discover -s tools -p 'test_stream_cache.py'
extra=()
if [ -n "$MODEL_CONFIG" ]; then
    extra=(--model-config "$MODEL_CONFIG")
fi
"${container[@]}" python tools/stream_cache_sweep.py run \
    --host "$HOST" --checkpoint "$CKPT" --inputs "$INPUTS" \
    --sweep "$SWEEP" --out "$WORK/run" "${extra[@]}"
echo 'JOB OK'
