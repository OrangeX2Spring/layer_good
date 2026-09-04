#!/bin/bash
# Extract the HouseCat6D pieces the OPT-Pose loaders read, into a PERSISTENT root.
#
#   bash tools/opt_pose_extract_housecat.sh [dest] [scene ...]
#
# Default dest is /mnt, not /tmp, which departs from the GenRecon pattern on
# purpose. HouseCat6DPoseDataset writes its frame-annotation cache under
# <data_root>/cache, and the causal experiment re-runs many times; on job-local
# /tmp both the extraction and the cache rebuild are re-paid every allocation.
# The cost is file count: one test scene is roughly 8.5k files against the
# 100,000-file project quota (that count, not the 250 GB, is the binding limit).
# Pass /tmp/data/housecat6d as dest if the quota is tight.
#
# Only what the loader reads is extracted. `test_scene.zip` is 31 GB and a full
# extraction is slow: network read, single-threaded unzip, hundreds of thousands
# of small files. Both depth folders are needed -- `_load_housecat_depth` reads
# `depth_gt` and `_load_housecat_depth_sensor` reads `depth`
# (training/data/datasets/housecat.py:1746-1747).
#
# The `-d "$DEST"` targets are hardcoded here on purpose: a pasted unzip line
# that loses its -d extracts into the current directory on /mnt and burns the
# quota (docs/opt-pose-cluster.md, gotcha 2).
#
# Re-running is a no-op once a PREVIOUS RUN FINISHED, which is recorded by the
# .extract_ok stamp written at the very end. Checking for the directories instead
# was wrong and cost a silent failure on 2026-09-04: unzip aborted partway, having
# already created test/test_scene1/rgb with a single file in it, and the next run
# reported "already extracted: EXTRACT OK" over a scene with 1 frame of 513. A
# directory existing proves an extraction STARTED. Delete the stamp to force a
# fresh extraction.
set -euo pipefail

# Newer unzip refuses this archive: "invalid zip file with overlapped components
# (possible zip bomb)". Measured on muenchen 2026-09-04; passau's older unzip
# extracts it without complaint, so this is node-dependent and silent until it
# is not. The archive is a TUM dataset release, not a bomb.
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

DEST="${1:-/mnt/projects/gr/3DRecon/housecat6d}"
shift || true
SCENES=("${@:-test_scene1}")
SRC=/mnt/datasets/housecat6d

MODELS="$DEST/obj_models_small_size_final/objects.pkl"
STAMP="$DEST/.extract_ok"

if [ -f "$STAMP" ] && [ -f "$MODELS" ]; then
  echo "already extracted: $DEST"
  cat "$STAMP"
  echo "EXTRACT OK"
  exit 0
fi

mkdir -p "$DEST/test"

if [ ! -f "$MODELS" ]; then
  echo "[1/2] object models from obj_models.zip"
  unzip -qo "$SRC/obj_models.zip" "obj_models_small_size_final/*" -d "$DEST"
fi

echo "[2/2] scenes from test_scene.zip: ${SCENES[*]}"
for SCENE in "${SCENES[@]}"; do
  unzip -qo "$SRC/test_scene.zip" "$SCENE/*" \
    -x "*/sim_obj/*" "*/pol/*" "*/normal/*" "*/occlusion/*" "*/_debug/*" "*/grasps/*" \
    -d "$DEST/test"
done

# Go/no-go. Every miss inside the loader is a bare `continue`, so a wrong layout
# produces "0 sequences" and a run over nothing rather than an error -- the same
# failure shape as the KV-Tracker ATE=nan trap.
test -f "$MODELS"
for SCENE in "${SCENES[@]}"; do
  D="$DEST/test/$SCENE"
  test -f "$D/intrinsics.txt"
  test -d "$D/rgb"
  test -d "$D/depth"
  test -d "$D/depth_gt"
  test -d "$D/instance"
  test -d "$D/nocs"
  test -d "$D/labels"
  printf '%s: %s rgb frames\n' "$SCENE" "$(ls "$D/rgb" | wc -l)"
done

du -sh "$DEST"

# Written last, and only if every check above passed. This is what a later run
# trusts, so it records what was extracted and how much of it.
: > "$STAMP"
for SCENE in "${SCENES[@]}"; do
  printf '%s: %s rgb frames\n' "$SCENE" "$(ls "$DEST/test/$SCENE/rgb" | wc -l)" >> "$STAMP"
done

echo "EXTRACT OK"
echo "pass to test_causal_housecat6d.py:  --data_root $DEST"
