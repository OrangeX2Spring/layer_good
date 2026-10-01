#!/bin/bash
# Separate user-submitted pilots; never chain a full evaluation behind them.
#SBATCH --job-name=online-cache
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/online_cache_slurm-%j.log
set -euo pipefail
HOST=${1:?Choose pi3 or streamvggt}
case "$HOST" in pi3|streamvggt) ;; *) exit 2 ;; esac
test "$(uname -s)" = Linux
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Submit with sbatch}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
WORK="$(mktemp -d "/tmp/online_cache_${SLURM_JOB_ID}_XXXXXX")"
OUT="$ROOT/online_cache_out"
mkdir -p "$OUT" "$WORK/context" "$WORK/datasets"
ARCHIVE="$OUT/online_cache_${HOST}_${SLURM_JOB_ID}.tar"
finish() {
    status=$?
    trap - EXIT
    echo "$status" > "$WORK/exit_status.txt"
    tar -C "$WORK" -cf "$ARCHIVE.partial" .
    mv "$ARCHIVE.partial" "$ARCHIVE"
    echo "ONLINE CACHE ARCHIVED $ARCHIVE exit=$status"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee "$WORK/run.log") 2>&1
hostname
nvidia-smi > "$WORK/context/nvidia-smi.txt"
ulimit -m > "$WORK/context/memory_grant_kb.txt"
scontrol show job "$SLURM_JOB_ID" > "$WORK/context/slurm_job.txt"
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
for module in kv_tracker streamvggt; do
    git -C "$REPO/$module" -c fetch.recurseSubmodules=false \
        -c url.https://github.com/.insteadOf=git@github.com: fetch origin
    git -C "$REPO" submodule update --no-fetch "$module"
    git -C "$REPO/$module" rev-parse HEAD > "$WORK/context/${module}_commit.txt"
    git -C "$REPO/$module" diff --binary HEAD > "$WORK/context/${module}.patch"
    git -C "$REPO/$module" archive HEAD > "$WORK/context/${module}_source.tar"
done
git -C "$REPO" rev-parse HEAD > "$WORK/context/parent_commit.txt"
git -C "$REPO" diff --binary HEAD > "$WORK/context/parent.patch"
THIRD="$REPO/kv_tracker/thirdparty"
test -f "$THIRD/Pi3/pi3/models/pi3.py"
git -C "$THIRD/Pi3" archive HEAD > "$WORK/context/pi3_source.tar"
git -C "$THIRD/Pi3" rev-parse HEAD > "$WORK/context/pi3_commit.txt"
cp "$REPO/tools/online_cache_experiment.py" "$REPO/tools/online_cache.sbatch" \
   "$REPO/tools/test_online_cache.py" "$REPO/tools/ONLINE_CACHE_EXPERIMENT.md" \
   "$REPO/tools/test_stream_depth.py" "$REPO/tools/stream_depth_ablation.py" \
   "$REPO/tools/kvt_patch_select.py" "$REPO/tools/kvt_arctic_run.py" \
   "$REPO/tools/kvt_tum_run.py" "$REPO/tools/stream_cache_sweep.py" \
   "$REPO/tools/stream_cache_tum.py" "$REPO/tools/stream_cache_arctic.py" \
   "$REPO/tools/stream_cache_job.py" "$WORK/context/"
CKPT="$ROOT/streamvggt_ckpt/checkpoints.pth"
if [ "$HOST" = pi3 ]; then
    IMAGE=localhost/kvt
    IMAGE_TAR="$ROOT/kvt.tar"
    test -f "$THIRD/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
    sha256sum "$ROOT/kvt_arctic_out/prepared.tar" \
        "$ROOT/kvt_arctic_out/arctic_r518_20260915T152911Z.tar" > "$WORK/context/object_inputs.sha256"
else
    IMAGE=localhost/optpose
    IMAGE_TAR="$ROOT/optpose.tar"
    test -f "$CKPT"
    sha256sum "$CKPT" > "$WORK/context/checkpoint.sha256"
fi
sha256sum "$IMAGE_TAR" > "$WORK/context/image.sha256"
if ! podman image exists "$IMAGE"; then podman load -i "$IMAGE_TAR"; fi
podman image inspect "$IMAGE" > "$WORK/context/container.json"
CONTAINER=(podman run --rm --network=host
    --device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw
    -v "$WORK/datasets:$REPO/kv_tracker/datasets:rw" -w "$REPO"
    -e "PYTHONPATH=$REPO/streamvggt/src:$THIRD/Pi3:$THIRD/segment-anything-2-real-time:$REPO/kv_tracker:$REPO/tools"
    -e "HF_HOME=$ROOT/.hf_cache" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MPLBACKEND=Agg "$IMAGE")
"${CONTAINER[@]}" bash -ec '
    if [ "$2" = streamvggt ]; then
        python tools/stream_cache_job.py --setup-only --host streamvggt \
            --checkpoint "$3" --work "$1" --width 308 --sweep tools/stream_cache_sweep.json
    fi
    set -o pipefail
    contracts=(
        test_online_cache.OnlineContracts.test_sensitivity_matches_explicit_removal
        test_online_cache.OnlineContracts.test_pi3_fixed_cap_and_protected_values
        test_online_cache.OnlineContracts.test_pi3_probe_does_not_change_query
    )
    if [ "$2" = streamvggt ]; then
        contracts+=(test_online_cache.OnlineContracts.test_stream_caps_and_original_positions)
    fi
    python -m unittest "${contracts[@]}" \
        2>&1 | tee "$1/context/contracts.log"
    if [ "$2" = streamvggt ]; then
        python -m unittest discover -s tools -p test_stream_depth.py \
            2>&1 | tee "$1/context/stream_regression.log"
    fi
    python tools/online_cache_experiment.py "$2" --work "$1" --checkpoint "$3"
' -- "$WORK" "$HOST" "$CKPT"
echo "ONLINE CACHE JOB COMPLETE $HOST"
