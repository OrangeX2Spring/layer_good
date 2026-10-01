#!/bin/bash
# Bounded second-backbone pilot; no downstream job is submitted.
#SBATCH --job-name=stream-depth
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/stream_depth_slurm-%j.log
set -euo pipefail
MODE=${1:-pilot}
case "$MODE" in pilot|anchor|overnight|full) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac
test "$(uname -s)" = Linux
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Submit through sbatch}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
WORK="$(mktemp -d "/tmp/stream_depth_${SLURM_JOB_ID}_XXXXXX")"
mkdir -p "$ROOT/stream_depth_out" "$WORK/context"
TAG=stream_depth
if [ "$MODE" = anchor ]; then TAG=stream_depth_anchor; fi
if [ "$MODE" = overnight ]; then TAG=stream_depth_overnight; fi
if [ "$MODE" = full ]; then TAG=stream_depth_full; fi
ARCHIVE="$ROOT/stream_depth_out/${TAG}_${SLURM_JOB_ID}.tar"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$WORK/exit_status.txt"
    tar -C "$WORK" -cf "$ARCHIVE.partial" .
    mv "$ARCHIVE.partial" "$ARCHIVE"
    echo "STREAM DEPTH ARCHIVED $ARCHIVE exit=$status"
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
git -C "$REPO/streamvggt" -c fetch.recurseSubmodules=false \
    -c url.https://github.com/.insteadOf=git@github.com: fetch origin
git -C "$REPO" submodule update --no-fetch streamvggt
git -C "$REPO" rev-parse HEAD > "$WORK/context/parent_commit.txt"
git -C "$REPO/streamvggt" rev-parse HEAD > "$WORK/context/model_commit.txt"
git -C "$REPO" diff --binary HEAD > "$WORK/context/parent.patch"
git -C "$REPO/streamvggt" diff --binary HEAD > "$WORK/context/model.patch"
git -C "$REPO/streamvggt" archive HEAD > "$WORK/context/model_source.tar"
cp "$REPO/tools/stream_depth.sbatch" "$REPO/tools/stream_depth_ablation.py" \
   "$REPO/tools/test_stream_depth.py" "$REPO/tools/STREAM_DEPTH_ABLATION.md" \
   "$REPO/tools/stream_cache_sweep.py" "$REPO/tools/stream_cache_tum.py" \
   "$REPO/tools/stream3r_ycbv_metrics.py" "$REPO/tools/stream_cache_job.py" \
   "$REPO/tools/stream_depth_overnight.py" "$WORK/context/"
sha256sum "$ROOT/optpose.tar" > "$WORK/context/container_tar.sha256"
if ! podman image exists localhost/optpose; then
    podman load -i "$ROOT/optpose.tar"
fi
podman image inspect localhost/optpose > "$WORK/context/container.json"
CKPT="$ROOT/streamvggt_ckpt/checkpoints.pth"
test -f "$CKPT"
if [ "$MODE" != pilot ]; then
    test -f "$ROOT/stream_depth_out/stream_depth_26089.tar"
fi
if [ "$MODE" = full ]; then
    : "${REVIEWED_ANCHOR_ARCHIVE:?Full stage requires manual anchor review first}"
    : "${REVIEWED_ANCHOR_SHA256:?Supply the manually reviewed archive SHA}"
    test -f "$REVIEWED_ANCHOR_ARCHIVE"
fi
CONTAINER=(podman run --rm --network=host
    --device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw -w "$REPO"
    -e "PYTHONPATH=$REPO/streamvggt/src:$REPO/tools"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 localhost/optpose)
# The setup and all subprocesses share this disposable container. Separate
# --rm invocations would discard the missing packages installed during setup.
"${CONTAINER[@]}" bash -ec '
    python tools/stream_cache_job.py --setup-only --host streamvggt \
        --checkpoint "$2" --work "$1" --width 308 \
        --sweep tools/stream_cache_sweep.json
    set -o pipefail
    python -m unittest discover -s tools -p test_stream_depth.py \
        2>&1 | tee "$1/context/contracts.log"
    if [ "$3" = overnight ]; then
        python tools/stream_depth_overnight.py diagnostics --work "$1" \
            --checkpoint "$2" --reference-archive "$4"
        exit
    fi
    if [ "$3" = full ]; then
        python tools/stream_depth_overnight.py full --work "$1" \
            --checkpoint "$2" --reference-archive "$4" \
            --reviewed-anchor-archive "$5" --reviewed-anchor-sha256 "$6"
        exit
    fi
    extra=()
    if [ "$3" = anchor ]; then
        extra=(--history-policy anchor --reference-archive "$4")
    fi
    python tools/stream_depth_ablation.py pilot --work "$1" --checkpoint "$2" "${extra[@]}"
' -- "$WORK" "$CKPT" "$MODE" "$ROOT/stream_depth_out/stream_depth_26089.tar" \
    "${REVIEWED_ANCHOR_ARCHIVE:-}" "${REVIEWED_ANCHOR_SHA256:-}"
touch "$WORK/STREAM_DEPTH_PILOT_COMPLETE"
echo 'STREAM DEPTH JOB COMPLETE: inspect summary; no automatic next job'
