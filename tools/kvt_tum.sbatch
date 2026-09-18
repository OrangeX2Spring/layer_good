#!/bin/bash
# Submit only after publishing the tools. All work runs inside this allocation.
#SBATCH --job-name=kvt-tum
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
set -euo pipefail
test "$(uname -s)" = Linux
test "$(hostname)" != head
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
OUT="$ROOT/kvt_tum_out"
TAG="tum_${SLURM_JOB_ID:?Submit through sbatch}"
WORK="/tmp/$TAG"
mkdir -p "$OUT"
mkdir "$WORK"
mkdir "$WORK/context"
MONITOR_PID=
finish() {
    status=$?
    trap - EXIT
    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID"
        wait "$MONITOR_PID" || test "$?" -eq 143
    fi
    echo "$status" > "$WORK/context/exit_status.txt"
    # Completed runs and inputs already have their own archives. Preserve any
    # failed/partial run as well, without duplicating all source images here.
    if [ -d "$WORK/runs" ]; then
        tar -C "$WORK" -cf "$OUT/${TAG}_all_runs.tar.partial" runs
        mv "$OUT/${TAG}_all_runs.tar.partial" "$OUT/${TAG}_all_runs.tar"
    fi
    tar --exclude=./inputs --exclude=./runs -C "$WORK" \
        -cf "$OUT/${TAG}_context.tar.partial" .
    mv "$OUT/${TAG}_context.tar.partial" "$OUT/${TAG}_context.tar"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

hostname
ulimit -m
nvidia-smi
sinfo -N -o '%.10N %.6P %.9T %.15G' > "$WORK/context/sinfo.txt"
scontrol show job "$SLURM_JOB_ID" > "$WORK/context/slurm_job.txt"
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
git -C "$REPO/kv_tracker" -c fetch.recurseSubmodules=false \
    -c url.https://github.com/.insteadOf=git@github.com: fetch origin
git -C "$REPO" submodule update --no-fetch kv_tracker
# No dependency installs or model downloads. Pi3/SAM must already be initialized.
test -f "$REPO/kv_tracker/thirdparty/Pi3/pi3/models/pi3.py"
test -f "$REPO/kv_tracker/thirdparty/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
git -C "$REPO" rev-parse HEAD > "$WORK/context/parent_commit.txt"
git -C "$REPO/kv_tracker" rev-parse HEAD > "$WORK/context/tracker_commit.txt"
git -C "$REPO/kv_tracker" submodule status > "$WORK/context/nested_submodules.txt"
git -C "$REPO" diff > "$WORK/context/parent.diff"
git -C "$REPO/kv_tracker" diff > "$WORK/context/tracker.diff"
cp "$REPO/tools/kvt_tum.sbatch" "$REPO/tools/kvt_tum_run.py" \
   "$REPO/tools/kvt_tum_selector.py" "$REPO/tools/kvt_tum_sweep.py" \
   "$REPO/tools/kvt_tum_viz.py" "$REPO/tools/kvt_artifacts.py" \
   "$REPO/tools/kvt_confidence_videos.py" \
   "$REPO/tools/test_kvt_tum.py" "$REPO/tools/KVT_TUM.md" "$WORK/context/"
git -C "$REPO/kv_tracker" archive HEAD > "$WORK/context/tracker_source.tar"
git -C "$REPO/kv_tracker/thirdparty/Pi3" archive HEAD > "$WORK/context/pi3_source.tar"
cp "$REPO/kv_tracker/thirdparty/Pi3/pi3/models/pi3.py" "$WORK/context/pi3.py"
sha256sum "$ROOT/kvt.tar" > "$WORK/context/container_tar.sha256"

if ! podman image exists localhost/kvt; then
    podman load -i "$ROOT/kvt.tar"
fi
podman image inspect localhost/kvt > "$WORK/context/container.json"
THIRDPARTY="$REPO/kv_tracker/thirdparty"
CONTAINER=(podman run --rm --network=host
    -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO/kv_tracker"
    --device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
    -e "PYTHONPATH=$THIRDPARTY/Pi3:$THIRDPARTY/segment-anything-2-real-time:$REPO/kv_tracker"
    -e PYTHONUNBUFFERED=1 -e MPLBACKEND=Agg -e "HF_HOME=$ROOT/.hf_cache"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 localhost/kvt)

nvidia-smi --query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 > "$WORK/context/gpu_samples.csv" &
MONITOR_PID=$!
"${CONTAINER[@]}" python -m pip list --format=json > "$WORK/context/packages.json"
"${CONTAINER[@]}" python "$REPO/tools/test_kvt_tum.py" 2>&1 | tee "$WORK/context/tests.log"
"${CONTAINER[@]}" python "$REPO/tools/kvt_tum_sweep.py" \
    --work "$WORK" --out "$OUT" --tag "$TAG" 2>&1 | tee "$WORK/context/sweep.log"
test -f "$WORK/JOB_OK"
echo 'JOB OK'
