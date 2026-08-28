#!/bin/bash
# Extract one ScanNet++ scene for GenRecon `--mode Scannet_colmap` into job-local /tmp.
#
#   bash tools/genrecon_extract_scannetpp.sh <scene_id> [dest]
#
# Run this INSIDE the GPU allocation. /tmp is node-local and wiped when the job
# ends, so extracting in a separate `data` job would be invisible to the run.
#
# Three paths are needed per scene, and they live in TWO archives:
#   dslr/nerfstudio/transforms_undistorted.json  -> scannetpp_dslr_undistorted.zip
#   dslr/resized_undistorted_images/             -> scannetpp_dslr_undistorted.zip
#   dslr/colmap/points3D.txt                     -> scannetpp_dslr.zip  (NOT the
#                                                   undistorted one - verified 2026-08-29)
#
# The `-d "$DEST"` targets are hardcoded here on purpose: a pasted unzip line that
# loses its -d extracts into the current directory on /mnt and burns the 100k-file
# quota. That has already happened once (docs/opt-pose-cluster.md, gotcha 2).
set -euo pipefail

SCENE="${1:?usage: genrecon_extract_scannetpp.sh <scene_id> [dest]}"
DEST="${2:-/tmp/data/scannetpp}"
SRC=/mnt/datasets/scannetpp

mkdir -p "$DEST"

echo "[1/2] poses + images from scannetpp_dslr_undistorted.zip"
unzip -qo "$SRC/scannetpp_dslr_undistorted.zip" \
  "data/$SCENE/dslr/nerfstudio/*" \
  "data/$SCENE/dslr/resized_undistorted_images/*" \
  -d "$DEST"

echo "[2/2] COLMAP points from scannetpp_dslr.zip"
unzip -qo "$SRC/scannetpp_dslr.zip" \
  "data/$SCENE/dslr/colmap/points3D.txt" \
  -d "$DEST"

SCENE_DIR="$DEST/data/$SCENE"

# Go/no-go: every one of these is read by get_chunks.py / get_images.py. A missing
# file does not raise - it produces an empty or degenerate run, the same failure
# shape as the KV-Tracker ATE=nan trap.
test -f "$SCENE_DIR/dslr/nerfstudio/transforms_undistorted.json"
test -f "$SCENE_DIR/dslr/colmap/points3D.txt"
test -d "$SCENE_DIR/dslr/resized_undistorted_images"

echo "--- $SCENE ---"
ls "$SCENE_DIR/dslr"
printf 'images: %s\n' "$(ls "$SCENE_DIR/dslr/resized_undistorted_images" | wc -l)"
du -sh "$SCENE_DIR"
echo "EXTRACT OK"
echo "pass to reconstruct_scene.py:  --path $SCENE_DIR"
