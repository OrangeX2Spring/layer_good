#!/bin/bash
# Opt-in experimental models. User operates every CAMP job and review gate.
#SBATCH --job-name=kvt-loops
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/loop_transformer_slurm-%j.log
set -euo pipefail
STAGE=${1:-contracts}
VARIANT=${2:-all}
case "$STAGE" in contracts|train|pilot|profile|full|overnight) ;;
    *) echo "Unknown loop stage: $STAGE" >&2; exit 2 ;; esac
case "$VARIANT" in all|elastic|relaxed|shared_kv|adaptive|token|refiner|nested|combined) ;;
    *) echo "Unknown loop variant: $VARIANT" >&2; exit 2 ;; esac
test "$(uname -s)" = Linux
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?GPU allocation required}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
WORK="$(mktemp -d "/tmp/loop_transformer_${SLURM_JOB_ID}_XXXXXX")"
OUT="$ROOT/loop_transformer_out"
mkdir -p "$OUT" "$WORK/context"
ARCHIVE="$OUT/loop_${SLURM_JOB_ID}_${STAGE}.tar"
MONITOR_PID=
finish() {
    status=$?
    trap - EXIT
    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID"
        wait "$MONITOR_PID" || test "$?" -eq 143
    fi
    printf '%s\n' "$status" > "$WORK/exit_status.txt"
    # Preserve exact resized pixels, masks/manifests, partial models and logs.
    # Original source PNGs are reproducible from the separately hashed TUM ZIP.
    tar --exclude='*/inputs/*/rgb' --exclude='reviewed_training' \
        -C "$WORK" -cf "$ARCHIVE.partial" .
    mv "$ARCHIVE.partial" "$ARCHIVE"
    echo "LOOP ARCHIVED $ARCHIVE exit=$status"
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
git -C "$REPO" rev-parse HEAD > "$WORK/context/parent_commit.txt"
git -C "$REPO/kv_tracker" rev-parse HEAD > "$WORK/context/tracker_commit.txt"
git -C "$REPO" diff --binary HEAD > "$WORK/context/parent.patch"
git -C "$REPO/kv_tracker" diff --binary HEAD > "$WORK/context/tracker.patch"
git -C "$REPO/kv_tracker" archive HEAD > "$WORK/context/tracker_source.tar"
git -C "$THIRDPARTY/Pi3" archive HEAD > "$WORK/context/pi3_source.tar"
cp "$REPO/tools/loop_transformer.py" "$REPO/tools/test_loop_transformer.py" \
   "$REPO/tools/loop_transformer.sbatch" "$REPO/tools/LOOP_TRANSFORMER_IMPLEMENTATION.md" \
   "$REPO/tools/LOOP_TRANSFORMER_RESEARCH.md" "$REPO/tools/kvt_tum_run.py" \
   "$REPO/tools/kvt_tum_sweep.py" "$WORK/context/"
sha256sum "$ROOT/kvt.tar" > "$WORK/context/container_tar.sha256"
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
nvidia-smi --query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 > "$WORK/context/gpu_samples.csv" &
MONITOR_PID=$!
if [ "$STAGE" = contracts ]; then
    "${CONTAINER[@]}" python -m unittest discover -s "$REPO/tools" \
        -p test_loop_transformer.py -v
    echo "LOOP ALL VARIANT GPU CONTRACTS OK"
else
    if [[ "$STAGE" = overnight || "$STAGE" = train ]]; then
        "${CONTAINER[@]}" python -m unittest discover -s "$REPO/tools" \
            -p test_loop_transformer.py -v
        echo "LOOP ${STAGE^^} CONTRACTS OK"
    fi
    extra=()
    if [ "$STAGE" = train ]; then
        # Same policy the overnight run used for its trained variants, for every variant.
        extra+=(--steps 200 --learning-rate 1e-5 --aggregate-gate)
    fi
    if [[ "$STAGE" = pilot || "$STAGE" = profile || "$STAGE" = full ]]; then
        SOURCE=${3:?Supply the reviewed training archive}
        test -f "$SOURCE"
        sha256sum "$SOURCE" > "$WORK/context/training_archive.sha256"
        extra+=(--checkpoint "$SOURCE")
    fi
    if [ "$STAGE" = full ]; then
        GATE=${4:?Supply the reviewed matching pilot_gate.json}
        test -f "$GATE"
        cp "$GATE" "$WORK/context/pilot_gate.json"
        extra+=(--reviewed-pilot "$GATE")
    fi
    "${CONTAINER[@]}" python "$REPO/tools/loop_transformer.py" \
        --work "$WORK" --stage "$STAGE" --variant "$VARIANT" "${extra[@]}"
fi
touch "$WORK/JOB_OK"
echo "LOOP ${STAGE^^} JOB OK: review evidence before the next stage"
