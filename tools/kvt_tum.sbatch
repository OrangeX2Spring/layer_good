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
MODE=${1:-sweep}
case "$MODE" in sweep|combined|correspondence|correspondence-gates|llm-packet|replay-fidelity|llm-compare) ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac
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
    if [ "$MODE" = correspondence-gates ] && [ -d "$WORK/arctic_review" ]; then
        tar -C "$WORK/arctic_review" \
            -cf "$ROOT/kvt_arctic_out/${TAG}_review.tar" .
    fi
    # Completed runs and inputs already have their own archives. Preserve any
    # failed/partial run as well, without duplicating all source images here.
    if [ -d "$WORK/runs" ]; then
        # Safety net for partial/failed runs only. Completed runs already have
        # their own archive, so excluding the two heavy artefacts keeps this
        # from doubling the job's storage cost against a 233 GB quota.
        tar -C "$WORK" --exclude='*/final_scene.npz' --exclude='*/viz' \
            -cf "$OUT/${TAG}_all_runs.tar.partial" runs
        mv "$OUT/${TAG}_all_runs.tar.partial" "$OUT/${TAG}_all_runs.tar"
    fi
    if [ "$MODE" = correspondence-gates ] && [ "$status" -ne 0 ]; then
        arctic_data=/tmp/data/arctic_data/data/cropped_images_grab_only_subset/s01/box_grab_01
        if [ -d "$arctic_data" ]; then
            find "$arctic_data" -mindepth 1 -maxdepth 1 -type d -name "${TAG}_*" -print0 \
                | tar --null -T - -cf "$ROOT/kvt_arctic_out/${TAG}_partial.tar"
        fi
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
if [ "$MODE" = llm-packet ]; then
    cp "$REPO/tools/kvt_tum_llm_packet.py" "$REPO/tools/KVT_TUM_LLM_PROMPT.md" \
       "$REPO/tools/KVT_LLM_REFERENCE.md" "$WORK/context/"
fi
if [ "$MODE" = replay-fidelity ]; then
    cp "$REPO/tools/kvt_tum_replay.py" "$REPO/tools/KVT_LLM_REFERENCE.md" \
       "$WORK/context/"
fi
if [ "$MODE" = llm-compare ]; then
    cp "$REPO/tools/kvt_tum_compare.py" "$REPO/tools/KVT_LLM_REFERENCE.md" \
       "$WORK/context/"
fi
if [ "$MODE" = combined ]; then
    cp "$REPO/tools/kvt_tum_combined.py" "$REPO/tools/kvt_combined_run.py" \
       "$REPO/tools/test_kvt_combined.py" "$REPO/tools/STREAM_CACHE_COMBINED.md" \
       "$WORK/context/"
fi
if [ "$MODE" = correspondence ] || [ "$MODE" = correspondence-gates ]; then
    cp "$REPO/tools/kvt_tum_combined.py" "$REPO/tools/kvt_tum_correspondence.py" \
       "$REPO/tools/kvt_correspondence_tum.py" "$REPO/tools/test_kvt_correspondence.py" \
       "$REPO/tools/KVT_CORRESPONDENCE.md" "$WORK/context/"
fi
if [ "$MODE" = correspondence-gates ]; then
    cp "$REPO/tools/kvt_correspondence_both_gates.py" \
       "$REPO/tools/kvt_correspondence_pilot.py" \
       "$REPO/tools/kvt_correspondence_selector.py" \
       "$REPO/tools/kvt_arctic_run.py" "$REPO/tools/kvt_online_selector.py" \
       "$REPO/tools/kvt_correspondence.sbatch" "$WORK/context/"
    sha256sum "$ROOT/kvt_arctic_out/prepared.tar" > "$WORK/context/arctic_inputs.sha256"
    cp "$ROOT/kvt_arctic_out/prepared.tar" \
       "$ROOT/kvt_arctic_out/${TAG}_inputs.tar"
    test -f "$HOME/.hf_env"
    source "$HOME/.hf_env"
fi
sha256sum "$ROOT/kvt.tar" > "$WORK/context/container_tar.sha256"

if ! podman image exists localhost/kvt; then
    podman load -i "$ROOT/kvt.tar"
fi
podman image inspect localhost/kvt > "$WORK/context/container.json"
THIRDPARTY="$REPO/kv_tracker/thirdparty"
EXTRA_ENV=()
OFFLINE_ENV=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
if [ "$MODE" = correspondence-gates ]; then
    EXTRA_ENV=(-e HF_TOKEN -e HF_XET_HIGH_PERFORMANCE=1)
    OFFLINE_ENV=()
fi
CONTAINER=(podman run --rm --network=host
    -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO/kv_tracker"
    --device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
    -e "PYTHONPATH=$THIRDPARTY/Pi3:$THIRDPARTY/segment-anything-2-real-time:$REPO/kv_tracker"
    -e PYTHONUNBUFFERED=1 -e MPLBACKEND=Agg -e "HF_HOME=$ROOT/.hf_cache"
    "${OFFLINE_ENV[@]}"
    "${EXTRA_ENV[@]}" localhost/kvt)

nvidia-smi --query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 > "$WORK/context/gpu_samples.csv" &
MONITOR_PID=$!
if [ "$MODE" = llm-packet ]; then
    "${CONTAINER[@]}" python "$REPO/tools/kvt_tum_llm_packet.py" \
        --work "$WORK" --out "$OUT" --tag "$TAG" \
        2>&1 | tee "$WORK/context/llm_packet.log"
    test -f "$WORK/JOB_OK"
    echo 'LLM PACKET JOB OK'
    exit 0
fi
if [ "$MODE" = correspondence-gates ]; then
    case "${2:-both}" in
        both) GATE_ARGS=() ;;
        arctic-only) GATE_ARGS=(--arctic-only) ;;
        *) echo "Unknown correspondence gate selection: $2" >&2; exit 2 ;;
    esac
    "${CONTAINER[@]}" python "$REPO/tools/kvt_correspondence_both_gates.py" \
        --work "$WORK" --tum-out "$OUT" --arctic-out "$ROOT/kvt_arctic_out" \
        --tag "$TAG" "${GATE_ARGS[@]}" 2>&1 | tee "$WORK/context/both_gates.log"
    if [ "${2:-both}" = both ]; then
        test -f "$WORK/both_gates.json"
    else
        test -f "$WORK/arctic_gate.json"
    fi
    test -f "$WORK/JOB_OK"
    echo "CORRESPONDENCE ${2:-both} GATES JOB OK"
    exit 0
fi
"${CONTAINER[@]}" python -m pip list --format=json > "$WORK/context/packages.json"
"${CONTAINER[@]}" python "$REPO/tools/test_kvt_tum.py" 2>&1 | tee "$WORK/context/tests.log"
if [ "$MODE" = replay-fidelity ]; then
    "${CONTAINER[@]}" python "$REPO/tools/kvt_tum_replay.py" \
        --work "$WORK" --out "$OUT" --tag "$TAG" \
        2>&1 | tee "$WORK/context/replay_fidelity.log"
    test -f "$WORK/fidelity.json"
elif [ "$MODE" = llm-compare ]; then
    "${CONTAINER[@]}" python "$REPO/tools/kvt_tum_compare.py" \
        --work "$WORK" --out "$OUT" --tag "$TAG" \
        2>&1 | tee "$WORK/context/compare.log"
    test -f "$WORK/comparison.json"
elif [ "$MODE" = combined ]; then
    "${CONTAINER[@]}" python "$REPO/tools/test_kvt_combined.py" \
        2>&1 | tee "$WORK/context/combined_tests.log"
    "${CONTAINER[@]}" python "$REPO/tools/kvt_combined_run.py" \
        --work "$WORK" --out "$OUT" --tag "$TAG" \
        --scene "${2:?Choose a TUM scene}" --stage "${3:-pilot}" \
        2>&1 | tee "$WORK/context/sweep.log"
elif [ "$MODE" = correspondence ]; then
    "${CONTAINER[@]}" python "$REPO/tools/test_kvt_correspondence.py" \
        2>&1 | tee "$WORK/context/correspondence_tests.log"
    "${CONTAINER[@]}" python "$REPO/tools/kvt_correspondence_tum.py" \
        --work "$WORK" --out "$OUT" --tag "$TAG" \
        --scene "${2:?Choose a TUM scene}" --stage "${3:-gate}" \
        2>&1 | tee "$WORK/context/correspondence.log"
else
    "${CONTAINER[@]}" python "$REPO/tools/kvt_tum_sweep.py" \
        --work "$WORK" --out "$OUT" --tag "$TAG" 2>&1 | tee "$WORK/context/sweep.log"
fi
test -f "$WORK/JOB_OK"
echo 'JOB OK'
