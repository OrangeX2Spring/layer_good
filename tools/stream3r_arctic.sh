#!/bin/bash
# Inside a CAMP GPU allocation with localhost/optpose loaded.
# Usage: bash tools/stream3r_arctic.sh ["scene ..."]
set -euo pipefail

ROOT=/mnt/projects/gr/3DRecon
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCENES=${1:-"box_grab_01 ketchup_grab_01 espressomachine_grab_01"}
: "${SLURM_JOB_ID:?Run inside a GPU allocation}"
: "${CUDA_VISIBLE_DEVICES:?No allocated GPU}"
test "$(hostname -s)" != head
read -r -a scene_list <<< "$SCENES"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$(mktemp -d "/tmp/stream3r_arctic_${STAMP}_${SLURM_JOB_ID}_XXXXXX")"
OUT="$WORK/run"
ARCHIVE="$ROOT/stream3r_out/$(basename "$WORK").tar"
KVT="$ROOT/kvt_arctic_out/arctic_r518_20260915T152911Z.tar"
CKPT="$ROOT/stream3r_ckpt"
mkdir -p "$ROOT/stream3r_out"

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

test -f "$CKPT/model.safetensors"
test -f "$KVT"
test -f "$ROOT/kvt_arctic_out/prepared.tar"
git -C "$REPO" rev-parse HEAD > "$WORK/superproject_commit.txt"
git -C "$REPO/stream3r" rev-parse HEAD > "$WORK/model_commit.txt"
git -C "$REPO/stream3r" diff HEAD > "$WORK/model_diff.patch"
git -C "$REPO/stream3r" archive HEAD > "$WORK/model_source.tar"
cp "$REPO/tools/stream3r_arctic.py" "$REPO/tools/stream3r_arctic.sh" \
    "$REPO/tools/fourrc_arctic.py" "$WORK/"
sha256sum "$CKPT/model.safetensors" > "$WORK/checkpoint.sha256"
if [ -f "$CKPT/config.json" ]; then
    cp "$CKPT/config.json" "$WORK/checkpoint_config.json"
fi
podman image inspect localhost/optpose > "$WORK/container.json"
nvidia-smi > "$WORK/nvidia-smi.txt"
ulimit -m > "$WORK/memory_grant_kb.txt"
printf 'scenes=%s\nframes=30\nsampling=even,consecutive\nmode=causal\nexecution=batched\n' \
    "$SCENES" > "$WORK/arguments.txt"

container=(podman run --rm --device="nvidia.com/gpu=$CUDA_VISIBLE_DEVICES"
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO"
    -e PYTHONPATH="$REPO/stream3r"
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
    -e HF_HUB_OFFLINE=1 localhost/optpose)
"${container[@]}" pip freeze > "$WORK/pip_freeze.txt"
for sampling in even consecutive; do
    "${container[@]}" python "$REPO/tools/stream3r_arctic.py" prepare \
        --prepared "$ROOT/kvt_arctic_out/prepared.tar" --kvt "$KVT" \
        --out "$OUT/$sampling" --scenes "${scene_list[@]}" --sampling "$sampling"
    for scene in "${scene_list[@]}"; do
        for condition in original masked; do
            echo "RUN $sampling $scene $condition"
            "${container[@]}" python "$REPO/tools/stream3r_arctic.py" run \
                --out "$OUT/$sampling" --scene "$scene" \
                --condition "$condition" --ckpt "$CKPT"
        done
    done
done
echo "STREAM3R ARCTIC OK"
