#!/bin/bash
# Run inside an existing CAMP GPU allocation with localhost/genrecon loaded.
# Usage: bash tools/fourrc_arctic.sh ["scene ..."] [frame_count] [even|consecutive]
# Match STream3R's first 30 post-offset frames:
# bash tools/fourrc_arctic.sh "box_grab_01 ketchup_grab_01 espressomachine_grab_01" 30 consecutive
set -euo pipefail

ROOT=/mnt/projects/gr/3DRecon
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCENES=${1:-"box_grab_01 ketchup_grab_01 espressomachine_grab_01"}
FRAMES=${2:-30}
SAMPLING=${3:-even}
case "$SAMPLING" in
    even|consecutive) ;;
    *) echo "Sampling must be even or consecutive" >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Run inside a GPU allocation}"
: "${CUDA_VISIBLE_DEVICES:?No allocated GPU}"
test "$(hostname -s)" != head
read -r -a scene_list <<< "$SCENES"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$(mktemp -d "/tmp/fourrc_${STAMP}_${SLURM_JOB_ID}_XXXXXX")"
OUT="$WORK/run"
ARCHIVE="$ROOT/4rc_out/$(basename "$WORK").tar"
KVT="$ROOT/kvt_arctic_out/arctic_r518_20260915T152911Z.tar"
CKPT="$ROOT/4rc_ckpt"
mkdir -p "$ROOT/4rc_out"

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
git -C "$REPO" rev-parse HEAD > "$WORK/superproject_commit.txt"
git -C "$REPO/4rc" rev-parse HEAD > "$WORK/model_commit.txt"
git -C "$REPO/4rc" diff HEAD > "$WORK/model_diff.patch"
git -C "$REPO/4rc" archive HEAD > "$WORK/model_source.tar"
cp "$REPO/tools/fourrc_arctic.py" "$REPO/tools/fourrc_arctic.sh" "$WORK/"
sha256sum "$CKPT/model.safetensors" > "$WORK/checkpoint.sha256"
podman image inspect localhost/genrecon > "$WORK/container.json"
nvidia-smi > "$WORK/nvidia-smi.txt"
ulimit -m > "$WORK/memory_grant_kb.txt"
printf 'scenes=%s\nframes=%s\nsampling=%s\n' "$SCENES" "$FRAMES" "$SAMPLING" > "$WORK/arguments.txt"

container=(podman run --rm --device="nvidia.com/gpu=$CUDA_VISIBLE_DEVICES"
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO/4rc"
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
    -e HF_HUB_OFFLINE=1 -e MPLCONFIGDIR=/tmp/fourrc_mpl
    localhost/genrecon)
"${container[@]}" pip freeze > "$WORK/pip_freeze.txt"
"${container[@]}" python "$REPO/tools/fourrc_arctic.py" prepare \
    --prepared "$ROOT/kvt_arctic_out/prepared.tar" --kvt "$KVT" \
    --out "$OUT" --scenes "${scene_list[@]}" --frames "$FRAMES" --sampling "$SAMPLING"

for scene in "${scene_list[@]}"; do
    for condition in original masked; do
        echo "RUN $SAMPLING $scene $condition"
        "${container[@]}" python inference.py \
            --input "$OUT/$scene/$condition" --checkpoint_dir "$CKPT" \
            --save "$OUT/$scene/$condition.npz"
        "${container[@]}" python "$REPO/tools/fourrc_arctic.py" export \
            --out "$OUT" --scene "$scene" --condition "$condition"
    done
done
echo "FOURRC ARCTIC OK"
