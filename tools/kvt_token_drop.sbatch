#!/bin/bash
# Object-mode token dropping against the original tracker, three ARCTIC S01 objects.
# Protocol: tools/KVT_TOKEN_DROP.md. Submit after publishing both repository commits.
# Mode: drop (default, job 25837), a2 (kept background, mass bias, attention probe),
# or a2k K (original plus one kept-background arm and its probe, K patches per frame).
#SBATCH --job-name=kvt-token-drop
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/kvt_token_drop-%j.log
set -euo pipefail
MODE=${1:-drop}
case "$MODE" in drop) PREFIX=token_drop ;; a2) PREFIX=token_mass ;;
    a2k) KBG=${2:?a2k needs K}; PREFIX=token_mass_k${KBG} ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Submit through sbatch}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
OUT="$ROOT/kvt_arctic_out"
TAG="${PREFIX}_${SLURM_JOB_ID}"
DATA=/tmp/data/arctic_data/data/cropped_images_grab_only_subset/s01

hostname
nvidia-smi
ulimit -m
# Fetch and pin inside the allocation, using HTTPS because outbound SSH is blocked.
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
git -C "$REPO/kv_tracker" -c fetch.recurseSubmodules=false \
    -c url.https://github.com/.insteadOf=git@github.com: fetch origin
git -C "$REPO" submodule update --no-fetch kv_tracker
echo "COMMIT $(git -C "$REPO" rev-parse HEAD) tracker $(git -C "$REPO/kv_tracker" rev-parse HEAD)"

CONTEXT=/tmp/"$TAG"_context
mkdir -p "$CONTEXT"
cp "$REPO/tools/kvt_token_drop.sbatch" "$REPO/tools/kvt_arctic_run.py" \
   "$REPO/tools/test_kvt_token_drop.py" "$REPO/tools/KVT_TOKEN_DROP.md" "$CONTEXT/"
cp "$REPO/kv_tracker/main.py" "$CONTEXT/tracker_main.py"
cp "$REPO/kv_tracker/kv_tracker/token_drop.py" "$REPO/kv_tracker/kv_tracker/pi3_utilts.py" "$CONTEXT/"
cp "$REPO/kv_tracker/thirdparty/Pi3/pi3/models/pi3.py" "$CONTEXT/pi3.py"
sha256sum "$OUT/prepared.tar" "$OUT/arctic_r518_20260915T152911Z.tar" > "$CONTEXT/inputs.sha256"
git -C "$REPO" rev-parse HEAD > "$CONTEXT/superproject_commit.txt"
git -C "$REPO/kv_tracker" rev-parse HEAD > "$CONTEXT/tracker_commit.txt"
finish() {
    status=$?
    trap - EXIT
    echo "$status" > "$CONTEXT/exit_status.txt"
    tar -C "$CONTEXT" -cf "$OUT/${TAG}_context.tar" .
    if [ "$status" -ne 0 ] && [ -d "$DATA" ]; then
        find "$DATA" -mindepth 2 -maxdepth 2 -type d -name "${TAG}_*" -print0 \
            | tar --null -T - -cf "$OUT/${TAG}_partial.tar"
    fi
    exit "$status"
}
trap finish EXIT

# Patch mapping, all-kept equivalence to upstream Pi3, partial-mask cache contract.
bash "$REPO/tools/kvt_run.sh" token-drop-check -v 2>&1 | tee "$CONTEXT/token_drop_tests.log"

# Same-allocation original: must reproduce the recorded r518 ATE, and gives the timing baseline.
bash "$REPO/tools/kvt_run.sh" arctic-run --results "${TAG}_original" --no-viz \
    --online-policy original --max-keyframes 0 --expect-baseline
if [ "$MODE" = a2k ]; then
    bash "$REPO/tools/kvt_run.sh" arctic-run --results "${TAG}_bg${KBG}" --no-viz \
        --online-policy original --max-keyframes 0 --token-drop --keep-background "$KBG" \
        --check-masks-from "${TAG}_original"
    bash "$REPO/tools/kvt_run.sh" arctic-run --results "${TAG}_probe_bg${KBG}" --no-viz \
        --online-policy original --max-keyframes 0 --token-drop --keep-background "$KBG" \
        --attention-probe --check-masks-from "${TAG}_original"
    echo "TOKEN MASS K JOB OK"
    exit 0
fi
# Token drop: same frames, same SAM masks (checked byte for byte), fewer patches computed.
bash "$REPO/tools/kvt_run.sh" arctic-run --results "${TAG}_dropped" --no-viz \
    --online-policy original --max-keyframes 0 --token-drop \
    --check-masks-from "${TAG}_original"
if [ "$MODE" = drop ]; then
    echo "TOKEN DROP JOB OK"
    exit 0
fi
# A2 timed arms, same allocation: k background patches, without and with mass bias.
for K in 4 16; do
    bash "$REPO/tools/kvt_run.sh" arctic-run --results "${TAG}_bg${K}" --no-viz \
        --online-policy original --max-keyframes 0 --token-drop --keep-background "$K" \
        --check-masks-from "${TAG}_original"
    bash "$REPO/tools/kvt_run.sh" arctic-run --results "${TAG}_bg${K}mass" --no-viz \
        --online-policy original --max-keyframes 0 --token-drop --keep-background "$K" \
        --background-mass --check-masks-from "${TAG}_original"
done
# Probe runs read attention only; their times are not comparable. -1 keeps every
# background patch: the dense computation, through the token-drop path.
for ARM in dense:-1: dropped:0: bg4:4: bg4mass:4:--background-mass \
           bg16:16: bg16mass:16:--background-mass; do
    IFS=: read -r NAME K EXTRA <<< "$ARM"
    bash "$REPO/tools/kvt_run.sh" arctic-run --results "${TAG}_probe_${NAME}" --no-viz \
        --online-policy original --max-keyframes 0 --token-drop --keep-background "$K" \
        $EXTRA --attention-probe --check-masks-from "${TAG}_original"
done
echo "TOKEN MASS JOB OK"
