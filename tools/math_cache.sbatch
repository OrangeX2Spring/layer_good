#!/bin/bash
# User submits; first-stage operator ablation only, never automatic tracking.
# Optional MATH_CACHE_HOST=streamvggt needs MATH_CACHE_CHECKPOINT (existing file).
#SBATCH --job-name=math-cache
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/math_cache_slurm-%j.log
set -euo pipefail
test "$(uname -s)" = Linux
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Submit through sbatch}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
HOST=${MATH_CACHE_HOST:-pi3}
OUT="$ROOT/math_cache_out"
WORK="$(mktemp -d "/tmp/math_cache_${SLURM_JOB_ID}_XXXXXX")"
mkdir -p "$OUT" "$WORK/context"
ARCHIVE="$OUT/math_${HOST}_${SLURM_JOB_ID}.tar"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$WORK/exit_status.txt"
    tar -C "$WORK" -cf "$ARCHIVE.partial" .
    mv "$ARCHIVE.partial" "$ARCHIVE"
    echo "MATH CACHE ARCHIVED $ARCHIVE exit=$status"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee "$WORK/run.log") 2>&1
hostname
ulimit -m > "$WORK/context/memory_grant_kb.txt"
nvidia-smi > "$WORK/context/nvidia-smi.txt"
scontrol show job "$SLURM_JOB_ID" > "$WORK/context/slurm_job.txt"
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
git -C "$REPO/kv_tracker" -c fetch.recurseSubmodules=false \
    -c url.https://github.com/.insteadOf=git@github.com: fetch origin
git -C "$REPO" submodule update --no-fetch kv_tracker
case "$HOST" in
    pi3)
        IMAGE=localhost/kvt
        IMAGE_TAR="$ROOT/kvt.tar"
        test -f "$REPO/kv_tracker/thirdparty/Pi3/pi3/models/pi3.py"
        MODEL_PATH="$REPO/kv_tracker/thirdparty/Pi3"
        extra=()
        git -C "$MODEL_PATH" archive HEAD > "$WORK/context/pi3_source.tar"
        ;;
    streamvggt)
        IMAGE=localhost/optpose
        IMAGE_TAR="$ROOT/optpose.tar"
        : "${MATH_CACHE_CHECKPOINT:?Set the existing StreamVGGT checkpoint file}"
        test -f "$MATH_CACHE_CHECKPOINT"
        git -C "$REPO" -c url.https://github.com/.insteadOf=git@github.com: \
            submodule update --init streamvggt
        MODEL_PATH="$REPO/streamvggt/src"
        extra=(--checkpoint "$MATH_CACHE_CHECKPOINT")
        sha256sum "$MATH_CACHE_CHECKPOINT" > "$WORK/context/checkpoint.sha256"
        git -C "$REPO/streamvggt" archive HEAD > "$WORK/context/streamvggt_source.tar"
        ;;
    *) echo "Unknown MATH_CACHE_HOST: $HOST" >&2; exit 2 ;;
esac
git -C "$REPO" rev-parse HEAD > "$WORK/context/parent_commit.txt"
git -C "$REPO/kv_tracker" rev-parse HEAD > "$WORK/context/tracker_commit.txt"
git -C "$REPO" diff --binary HEAD > "$WORK/context/parent.patch"
git -C "$REPO/kv_tracker" diff --binary HEAD > "$WORK/context/tracker.patch"
git -C "$REPO/kv_tracker" archive HEAD > "$WORK/context/tracker_source.tar"
cp "$REPO/tools/math_cache.sbatch" "$REPO/tools/math_cache_ablation.py" \
   "$REPO/tools/math_cache_ablation.json" "$REPO/tools/test_math_cache.py" \
   "$REPO/tools/MATH_CACHE_ABLATION.md" "$WORK/context/"
sha256sum "$IMAGE_TAR" > "$WORK/context/container_tar.sha256"
if ! podman image exists "$IMAGE"; then
    podman load -i "$IMAGE_TAR"
fi
podman image inspect "$IMAGE" > "$WORK/context/container.json"
CONTAINER=(podman run --rm --network=none
    --device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO"
    -e "PYTHONPATH=$MODEL_PATH:$REPO/kv_tracker:$REPO/tools"
    -e "HF_HOME=$ROOT/.hf_cache" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MPLBACKEND=Agg "$IMAGE")
"${CONTAINER[@]}" python -m unittest discover -s tools -p test_math_cache.py \
    2>&1 | tee "$WORK/context/contracts.log"
for scene in freiburg1_desk freiburg3_structure_notexture_far; do
    zip="/mnt/datasets/tum-rgbd/rgbd_dataset_${scene}.zip"
    test -f "$zip"
    "${CONTAINER[@]}" python tools/math_cache_ablation.py capture --host "$HOST" \
        --zip "$zip" --out "$WORK/$scene" "${extra[@]}"
    "${CONTAINER[@]}" python tools/math_cache_ablation.py replay \
        --manifest "$WORK/$scene/traces/manifest.json" --out "$WORK/${scene}_ablation"
done
touch "$WORK/OPERATOR_ABLATION_COMPLETE"
echo 'MATH CACHE OPERATOR JOB COMPLETE: inspect reports; no automatic tracking stage'
