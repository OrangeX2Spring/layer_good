#!/bin/bash
# Bounded live depth-memory pilot. User submits; no automatic full evaluation.
#SBATCH --job-name=layer-cache
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/layer_cache_slurm-%j.log
set -euo pipefail
MODE=${1:-pilot}
case "$MODE" in pilot|special) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac
test "$(uname -s)" = Linux
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Submit through sbatch}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
WORK="$(mktemp -d "/tmp/layer_cache_${SLURM_JOB_ID}_XXXXXX")"
OUT="$ROOT/layer_cache_out"
mkdir -p "$OUT" "$WORK/context"
TAG=layer_cache
if [ "$MODE" = special ]; then TAG=layer_cache_special; fi
ARCHIVE="$OUT/${TAG}_${SLURM_JOB_ID}.tar"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$WORK/exit_status.txt"
    tar -C "$WORK" -cf "$ARCHIVE.partial" .
    mv "$ARCHIVE.partial" "$ARCHIVE"
    echo "LAYER CACHE ARCHIVED $ARCHIVE exit=$status"
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
THIRDPARTY="$REPO/kv_tracker/thirdparty"
test -f "$THIRDPARTY/Pi3/pi3/models/pi3.py"
test -f "$THIRDPARTY/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
git -C "$REPO" rev-parse HEAD > "$WORK/context/parent_commit.txt"
git -C "$REPO/kv_tracker" rev-parse HEAD > "$WORK/context/tracker_commit.txt"
git -C "$REPO" diff --binary HEAD > "$WORK/context/parent.patch"
git -C "$REPO/kv_tracker" diff --binary HEAD > "$WORK/context/tracker.patch"
git -C "$REPO/kv_tracker" archive HEAD > "$WORK/context/tracker_source.tar"
git -C "$THIRDPARTY/Pi3" archive HEAD > "$WORK/context/pi3_source.tar"
cp "$REPO/tools/layer_cache.sbatch" "$REPO/tools/layer_cache_ablation.py" \
   "$REPO/tools/test_layer_cache.py" "$REPO/tools/LAYER_CACHE_ABLATION.md" \
   "$REPO/tools/kvt_tum_run.py" "$REPO/tools/kvt_tum_selector.py" "$WORK/context/"
sha256sum "$ROOT/kvt.tar" > "$WORK/context/container_tar.sha256"
sha256sum "$THIRDPARTY/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt" \
    > "$WORK/context/sam_checkpoint.sha256"
if ! podman image exists localhost/kvt; then
    podman load -i "$ROOT/kvt.tar"
fi
podman image inspect localhost/kvt > "$WORK/context/container.json"
CONTAINER=(podman run --rm --network=none
    --device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO/kv_tracker"
    -e "PYTHONPATH=$THIRDPARTY/Pi3:$THIRDPARTY/segment-anything-2-real-time:$REPO/kv_tracker:$REPO/tools"
    -e "HF_HOME=$ROOT/.hf_cache" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MPLBACKEND=Agg localhost/kvt)
extra=()
if [ "$MODE" = special ]; then
    REFERENCE="$OUT/layer_cache_26086.tar"
    test -f "$REFERENCE"
    extra=(--reference-archive "$REFERENCE")
    "${CONTAINER[@]}" python -m unittest \
        test_layer_cache.LayerCacheTests.test_special_control_preserves_every_frame_special_and_exact_bytes \
        2>&1 | tee "$WORK/context/contracts.log"
else
    "${CONTAINER[@]}" python -m unittest discover -s "$REPO/tools" -p test_layer_cache.py \
        2>&1 | tee "$WORK/context/contracts.log"
fi
"${CONTAINER[@]}" python "$REPO/tools/layer_cache_ablation.py" --work "$WORK" "${extra[@]}"
touch "$WORK/LAYER_CACHE_PILOT_COMPLETE"
echo 'LAYER CACHE JOB COMPLETE: inspect summary; no automatic next job'
