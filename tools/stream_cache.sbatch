#!/bin/bash
# One host/container per job. Default: STream3R's shipped example, smoke sweep.
# Optional environment: CACHE_HOST, CACHE_IMAGE, CACHE_IMAGE_TAR, CACHE_CHECKPOINT,
# CACHE_INPUT_TAR (archive containing source manifest and pixels), CACHE_MANIFEST,
# CACHE_SWEEP, CACHE_WIDTH. All extraction and prepared data stay in /tmp.
#SBATCH --job-name=stream-cache
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/stream_cache_slurm-%j.log
set -euo pipefail
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Submit through sbatch}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
HOST=${CACHE_HOST:-stream3r}
case "$HOST" in stream3r|streamvggt|longstream) ;; *) exit 2 ;; esac
IMAGE=${CACHE_IMAGE:-localhost/optpose}
IMAGE_TAR=${CACHE_IMAGE_TAR:-$ROOT/optpose.tar}
WORK="$(mktemp -d "/tmp/cache_${HOST}_${SLURM_JOB_ID}_XXXXXX")"
mkdir -p "$ROOT/stream_cache_out"
ARCHIVE="$ROOT/stream_cache_out/$(basename "$WORK").tar"
MONITOR=
finish() {
    status=$?
    trap - EXIT
    if [ -n "$MONITOR" ] && kill -0 "$MONITOR" 2>/dev/null; then
        kill "$MONITOR"
    fi
    printf '%s\n' "$status" > "$WORK/exit_status.txt"
    # run/inputs contains the exact model pixels. Do not duplicate extracted data.
    tar --exclude=./extracted --exclude=./prepared -cf "$ARCHIVE.part" -C "$WORK" .
    mv "$ARCHIVE.part" "$ARCHIVE"
    echo "ARCHIVED $ARCHIVE (exit status $status)"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee "$WORK/run.log") 2>&1
hostname
ulimit -m > "$WORK/memory_grant_kb.txt"
nvidia-smi > "$WORK/nvidia-smi.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
    --format=csv -l 30 > "$WORK/gpu_usage.csv" &
MONITOR=$!
scontrol show job "$SLURM_JOB_ID" > "$WORK/slurm_job.txt"
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
# The tiny attention contract test also reads StreamVGGT's source.
git -C "$REPO" -c url.https://github.com/.insteadOf=git@github.com: \
    submodule update --init "$HOST" streamvggt
cp "$REPO/tools/stream_cache.sbatch" "$REPO/tools/stream_cache_job.py" \
    "$REPO/tools/stream_cache_long.sbatch" "$WORK/"
git -C "$REPO" rev-parse HEAD > "$WORK/commit.txt"
git -C "$REPO" submodule status > "$WORK/submodules.txt"
mkdir "$WORK/git_provenance"
git -C "$REPO" rev-parse HEAD > "$WORK/git_provenance/superproject_commit.txt"
git -C "$REPO" diff --binary HEAD > "$WORK/git_provenance/superproject.patch"
git -C "$REPO/$HOST" rev-parse HEAD > "$WORK/git_provenance/model_commit.txt"
git -C "$REPO/$HOST" diff --binary HEAD > "$WORK/git_provenance/model.patch"
if ! podman image exists "$IMAGE"; then
    podman load -i "$IMAGE_TAR"
fi
podman image inspect "$IMAGE" > "$WORK/container.json"
CKPT=${CACHE_CHECKPOINT:-}
if [ "$HOST" = stream3r ]; then
    CKPT=${CKPT:-$ROOT/stream3r_ckpt}
fi
: "${CKPT:?Set CACHE_CHECKPOINT for this host; downloads are a separate job}"
test -e "$CKPT"
extra=()
if [ -n "${CACHE_INPUT_TAR:-}" ]; then
    mkdir "$WORK/extracted"
    tar -xf "$CACHE_INPUT_TAR" -C "$WORK/extracted"
    sha256sum "$CACHE_INPUT_TAR" > "$WORK/input_archive.sha256"
    extra=(--manifest "$WORK/extracted/${CACHE_MANIFEST:-manifest.json}")
fi
# TUM needs no staged input archive: the ZIPs are read-only on /mnt/datasets and
# cost no quota, so the clip is cut straight into job-local /tmp. Runs inside the
# image because a bare node is not guaranteed to have python3.
if [ -n "${CACHE_TUM_ZIP:-}" ]; then
    test -z "${CACHE_INPUT_TAR:-}"
    test -e "$CACHE_TUM_ZIP"
    podman run --rm --network=none -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO" \
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 "$IMAGE" \
        python tools/stream_cache_tum.py --zip "$CACHE_TUM_ZIP" \
        --out "$WORK/extracted" --start "${CACHE_TUM_START:-0}" \
        --count "${CACHE_TUM_COUNT:?Set CACHE_TUM_COUNT}" \
        --stride "${CACHE_TUM_STRIDE:-1}"
    sha256sum "$CACHE_TUM_ZIP" > "$WORK/input_archive.sha256"
    cp "$WORK/extracted/manifest.json" "$WORK/source_manifest.json"
    extra=(--manifest "$WORK/extracted/manifest.json")
fi
if [ -n "${CACHE_ARCTIC_SCENE:-}" ]; then
    test -z "${CACHE_TUM_ZIP:-}" && test -z "${CACHE_INPUT_TAR:-}"
    PREPARED="$ROOT/kvt_arctic_out/prepared.tar"
    KVT="$ROOT/kvt_arctic_out/arctic_r518_20260915T152911Z.tar"
    test -f "$PREPARED" && test -f "$KVT"
    podman run --rm --network=none -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO" \
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 "$IMAGE" \
        python tools/stream_cache_arctic.py --prepared "$PREPARED" --kvt "$KVT" \
        --scene "$CACHE_ARCTIC_SCENE" --count "${CACHE_ARCTIC_COUNT:-121}" \
        --out "$WORK/extracted"
    sha256sum "$PREPARED" "$KVT" > "$WORK/input_archive.sha256"
    cp "$WORK/extracted/manifest.json" "$WORK/source_manifest.json"
    extra=(--manifest "$WORK/extracted/manifest.json")
fi
podman run --rm --network=host --device="nvidia.com/gpu=$CUDA_VISIBLE_DEVICES" \
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO" \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$IMAGE" \
    python tools/stream_cache_job.py --host "$HOST" --checkpoint "$CKPT" \
    --work "$WORK" --width "${CACHE_WIDTH:-308}" \
    --max-pair-gap "${CACHE_MAX_PAIR_GAP:-0.1}" \
    --sweep "${CACHE_SWEEP:-$REPO/tools/stream_cache_sweep.json}" "${extra[@]}"
echo 'JOB OK'
