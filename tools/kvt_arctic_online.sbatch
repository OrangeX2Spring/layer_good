#!/bin/bash
# Online selectors only. Submit after publishing both repository commits.
#SBATCH --job-name=kvt-online
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/kvt_arctic_online_slurm-%j.log
set -euo pipefail
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
OUT="$ROOT/kvt_arctic_out"
TAG="online_${SLURM_JOB_ID}"
DATA=/tmp/data/arctic_data/data/cropped_images_grab_only_subset/s01

hostname
nvidia-smi
ulimit -m
# Fetch and pin inside the allocation, using HTTPS because outbound SSH is blocked.
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
git -C "$REPO/kv_tracker" -c fetch.recurseSubmodules=false \
    -c url.https://github.com/.insteadOf=git@github.com: fetch origin
git -C "$REPO" submodule update --no-fetch kv_tracker
git -C "$REPO" rev-parse HEAD
git -C "$REPO/kv_tracker" rev-parse HEAD

# Persist source and exact input archive identity even if the first model run fails.
CONTEXT=/tmp/"$TAG"_context
mkdir -p "$CONTEXT"
cp "$REPO/tools/kvt_arctic_online.sbatch" "$REPO/tools/kvt_arctic_run.py" \
   "$REPO/tools/kvt_online_selector.py" "$REPO/tools/test_kvt_online_selector.py" "$CONTEXT/"
cp "$REPO/kv_tracker/main.py" "$CONTEXT/tracker_main.py"
cp "$REPO/kv_tracker/thirdparty/Pi3/pi3/models/pi3.py" "$CONTEXT/pi3.py"
sha256sum "$OUT/prepared.tar" > "$CONTEXT/inputs.sha256"
git -C "$REPO" rev-parse HEAD > "$CONTEXT/superproject_commit.txt"
git -C "$REPO/kv_tracker" rev-parse HEAD > "$CONTEXT/tracker_commit.txt"
# One shared inputs tar per overnight job, rather than duplicating RGB in each run.
cp "$OUT/prepared.tar" "$OUT/${TAG}_inputs.tar"
finish() {
    status=$?
    trap - EXIT
    echo "$status" > "$CONTEXT/exit_status.txt"
    tar -C "$CONTEXT" -cf "$OUT/${TAG}_context.tar" .
    if [ "$status" -ne 0 ] && [ -d "$DATA" ]; then
        # Preserve partial results as well as previously archived completed runs.
        find "$DATA" -mindepth 2 -maxdepth 2 -type d -name "${TAG}_*" -print0 \
            | tar --null -T - -cf "$OUT/${TAG}_partial.tar"
    fi
    exit "$status"
}
trap finish EXIT

bash "$REPO/tools/kvt_run.sh" online-check

# Every configuration gets a 128-frame prefix run and a full run. Full runs must
# reproduce prefix poses (1e-4 atol/rtol), exact decisions, and baseline masks.
run_policy() {
    name="$1"
    shift
    prefix="${TAG}_${name}_prefix"
    result="${TAG}_${name}"
    bash "$REPO/tools/kvt_run.sh" arctic-run --results "$prefix" --no-viz \
        --prefix-frames 128 "$@"
    checks=(--compare-prefix "$prefix")
    if [ "$name" = original ]; then
        checks+=(--expect-baseline)
    else
        checks+=(--check-masks-from "${TAG}_original")
    fi
    bash "$REPO/tools/kvt_run.sh" arctic-run --results "$result" --no-viz \
        "${checks[@]}" "$@"
    # These exact job-owned directories have already been archived and verified.
    for scene in box_grab_01 ketchup_grab_01 espressomachine_grab_01; do
        rm -rf "$DATA/$scene/$prefix"
        if [ "$name" != original ]; then
            rm -rf "$DATA/$scene/$result"
        fi
    done
}

run_policy original --online-policy original --max-keyframes 0
run_policy semantic010 --online-policy semantic --novelty-threshold 0.10
run_policy original_cap32 --online-policy original
for angle in 20 25 40; do
    run_policy "angular${angle}" --online-policy angular --angle-degrees "$angle"
done
for interval in 30 60; do
    run_policy "interval${interval}" --online-policy interval --interval "$interval"
done
for seed in 0 1 2; do
    run_policy "random30_s${seed}" --online-policy random --interval 30 --seed "$seed"
done
for threshold in 0.05 0.20; do
    run_policy "semantic${threshold}" --online-policy semantic --novelty-threshold "$threshold"
done

echo "JOB OK"
