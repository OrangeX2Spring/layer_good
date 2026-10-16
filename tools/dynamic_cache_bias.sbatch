#!/bin/bash
# First submit prepare; review its evidence before submitting one pilot clip.
#SBATCH --job-name=dynamic-cache
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/dynamic_cache_slurm-%j.log
set -euo pipefail
MODE=${1:?Choose prepare or pilot}
case "$MODE" in prepare|pilot) ;; *) echo "Unknown mode: $MODE"; exit 2 ;; esac
MANIFEST=${2:-}
if [ "$MODE" = pilot ]; then
    test -n "$MANIFEST"
    test -f "$MANIFEST"
    MANIFEST=$(realpath "$MANIFEST")
fi
test "$(uname -s)" = Linux
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Submit inside an allocation}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
OUT="$ROOT/dynamic_cache_out"
WORK=$(mktemp -d "/tmp/dynamic_cache_${SLURM_JOB_ID}_XXXXXX")
mkdir -p "$OUT" "$WORK/context"
ARCHIVE="$OUT/dynamic_cache_${SLURM_JOB_ID}_${MODE}.tar"
finish() {
    status=$?
    trap - EXIT
    echo "$status" > "$WORK/exit_status.txt"
    tar -C "$WORK" -cf "$ARCHIVE.partial" .
    mv "$ARCHIVE.partial" "$ARCHIVE"
    echo "DYNAMIC CACHE ARCHIVED $ARCHIVE exit=$status"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee "$WORK/run.log") 2>&1
hostname
echo "mem_grant_kb=$(ulimit -m)"
nvidia-smi > "$WORK/context/nvidia-smi.txt"
scontrol show job "$SLURM_JOB_ID" > "$WORK/context/slurm_job.txt"
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
git -C "$REPO" -c url.https://github.com/.insteadOf=git@github.com: \
    submodule update --init infinitevggt
git -C "$REPO" rev-parse HEAD > "$WORK/context/parent_commit.txt"
git -C "$REPO/infinitevggt" rev-parse HEAD > "$WORK/context/model_commit.txt"
git -C "$REPO" diff --binary HEAD > "$WORK/context/parent.patch"
git -C "$REPO/infinitevggt" diff --binary HEAD > "$WORK/context/model.patch"
git -C "$REPO/infinitevggt" archive HEAD src tests LICENSE.txt requirements.txt \
    > "$WORK/context/model_source.tar"
cp "$REPO/tools/dynamic_cache_bias.py" "$REPO/tools/dynamic_cache_bias.sbatch" \
   "$REPO/tools/DYNAMIC_CACHE_BIAS.md" "$REPO/tools/DYNAMIC_CACHE_BIAS_PLAN.md" \
   "$REPO/tools/stream_cache_job.py" "$REPO/tools/stream3r_ycbv_metrics.py" \
   "$REPO/tools/test_dynamic_cache_bias.py" \
   "$WORK/context/"
# Bounded inventory only. An archive name is NOT a verified motion annotation.
for directory in /mnt/datasets "$ROOT" "$ROOT/stream3r_data" "$ROOT/dynamic_cache_data"; do
    if [ -d "$directory" ]; then
        echo "DATASET ROOT $directory"
        find "$directory" -maxdepth 2 \
            \( -iname '*sintel*' -o -iname '*bonn*' -o -iname '*scannet*' \
               -o -iname '*tum*' -o -iname '*walking*' -o -iname '*sitting*' \) \
            -printf '%y %s %p\n'
        if [ "$directory" = /mnt/datasets ]; then
            find "$directory" -mindepth 1 -maxdepth 1 -printf '%f\n'
        fi
    else
        echo "DATASET ROOT ABSENT $directory"
    fi
done > "$WORK/context/dataset_inventory.txt"
cat "$WORK/context/dataset_inventory.txt"
IMAGE=localhost/optpose
IMAGE_TAR="$ROOT/optpose.tar"
CKPT="$ROOT/streamvggt_ckpt/checkpoints.pth"
test -f "$IMAGE_TAR"
test -f "$CKPT"
sha256sum "$IMAGE_TAR" "$CKPT" > "$WORK/context/assets.sha256"
# Always load the recorded archive so an existing tag cannot silently differ.
podman load -i "$IMAGE_TAR"
podman image inspect "$IMAGE" > "$WORK/context/container.json"
podman run --rm --network=host \
    --device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}" \
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO" \
    -e "PYTHONPATH=$REPO/infinitevggt/src:$REPO/tools" \
    -e "HF_HOME=$ROOT/.hf_cache" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MPLBACKEND=Agg \
    "$IMAGE" bash -ec '
    set -o pipefail
    python tools/stream_cache_job.py --setup-only --host infinitevggt \
        --checkpoint "$3" --work "$1" --width 308 --sweep tools/stream_cache_sweep.json
    python -m unittest discover -s infinitevggt/tests -p test_dynamic_cache_probe.py \
        2>&1 | tee "$1/context/contracts.log"
    python -m unittest discover -s tools -p test_dynamic_cache_bias.py \
        2>&1 | tee "$1/context/input_metric_contracts.log"
    if [ "$2" = prepare ]; then
        python tools/dynamic_cache_bias.py smoke --work "$1" --checkpoint "$3"
        echo "PREPARE ENV OK; DATA INVENTORY ONLY; review annotations before pilot"
    else
        python tools/dynamic_cache_bias.py pilot --work "$1" \
            --checkpoint "$3" --manifest "$4"
    fi
    ' -- "$WORK" "$MODE" "$CKPT" "$MANIFEST"
echo "DYNAMIC CACHE JOB OK $MODE"
