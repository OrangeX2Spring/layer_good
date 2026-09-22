#!/bin/bash
# Each gate or full stage is a separate submission after reviewing the prior gate.
#SBATCH --job-name=kvt-correspond
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/kvt_correspondence_slurm-%j.log
set -euo pipefail
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
OUT="$ROOT/kvt_arctic_out"
STAGE="${1:-gate}"
case "$STAGE" in gate|full) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
TAG="correspondence_${SLURM_JOB_ID}_${STAGE}"
SCENE="${2:-box_grab_01}"
case "$SCENE" in
    box_grab_01|ketchup_grab_01|espressomachine_grab_01) ;;
    *) echo "Unknown ARCTIC scene: $SCENE" >&2; exit 2 ;;
esac
hostname
test "$(hostname)" != head
nvidia-smi
ulimit -m
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
git -C "$REPO/kv_tracker" -c fetch.recurseSubmodules=false \
    -c url.https://github.com/.insteadOf=git@github.com: fetch origin
git -C "$REPO" submodule update --no-fetch kv_tracker
CONTEXT="/tmp/${TAG}_context"
mkdir "$CONTEXT"
finish() {
    status=$?
    trap - EXIT
    echo "$status" > "$CONTEXT/exit_status.txt"
    tar -C "$CONTEXT" -cf "$OUT/${TAG}_context.tar" .
    if [ -d "/tmp/${TAG}_review" ]; then
        tar -C "/tmp/${TAG}_review" -cf "$OUT/${TAG}_review.tar" .
    fi
    if [ "$status" -ne 0 ]; then
        data=/tmp/data/arctic_data/data/cropped_images_grab_only_subset/s01
        if [ -d "$data/$SCENE" ]; then
            find "$data/$SCENE" -mindepth 1 -maxdepth 1 -type d -name "${TAG}_*" -print0 \
                | tar --null -T - -cf "$OUT/${TAG}_partial.tar"
        fi
    fi
    exit "$status"
}
trap finish EXIT
git -C "$REPO" rev-parse HEAD > "$CONTEXT/parent_revision.txt"
git -C "$REPO/kv_tracker" rev-parse HEAD > "$CONTEXT/tracker_revision.txt"
git -C "$REPO" diff HEAD > "$CONTEXT/parent.diff"
git -C "$REPO/kv_tracker" diff HEAD > "$CONTEXT/tracker.diff"
cp "$REPO/tools/kvt_correspondence.sbatch" "$REPO/tools/kvt_correspondence_pilot.py" \
   "$REPO/tools/kvt_correspondence_selector.py" "$REPO/tools/test_kvt_correspondence.py" \
   "$REPO/tools/kvt_arctic_run.py" "$REPO/tools/kvt_online_selector.py" \
   "$REPO/tools/kvt_run.sh" "$REPO/tools/KVT_CORRESPONDENCE.md" "$CONTEXT/"
cp "$REPO/kv_tracker/main.py" "$CONTEXT/tracker_main.py"
cp "$REPO/kv_tracker/kv_tracker/correspondence_cache.py" \
   "$REPO/kv_tracker/kv_tracker/combined_cache.py" "$CONTEXT/"
cp "$REPO/kv_tracker/thirdparty/Pi3/pi3/models/pi3.py" "$CONTEXT/"
cp "$REPO/kv_tracker/thirdparty/Pi3/pi3/models/layers/attention.py" "$CONTEXT/"
sha256sum "$OUT/prepared.tar" > "$CONTEXT/inputs.sha256"
cp "$OUT/prepared.tar" "$OUT/${TAG}_inputs.tar"
bash "$REPO/tools/kvt_run.sh" correspondence-check
bash "$REPO/tools/kvt_run.sh" correspondence-pilot --tag "$TAG" \
    --scene "$SCENE" --out "/tmp/${TAG}_review" --stage "$STAGE"
echo "JOB OK: $STAGE $SCENE"
