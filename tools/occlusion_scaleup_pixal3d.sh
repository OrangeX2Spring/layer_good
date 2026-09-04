#!/bin/bash
# Prepare the prior experiment's visible/completed inputs and run Pixal3D.
# Runs inside localhost/pixal3d, not on the host.
set -euo pipefail

YCBV_ROOT="${1:?usage: occlusion_scaleup_pixal3d.sh <ycbv_root> <completion_root> <output_root> <cases_tsv> [completion_index]}"
COMPLETION_ROOT="${2:?usage: occlusion_scaleup_pixal3d.sh <ycbv_root> <completion_root> <output_root> <cases_tsv> [completion_index]}"
OUTPUT_ROOT="${3:?usage: occlusion_scaleup_pixal3d.sh <ycbv_root> <completion_root> <output_root> <cases_tsv> [completion_index]}"
CASES="${4:?usage: occlusion_scaleup_pixal3d.sh <ycbv_root> <completion_root> <output_root> <cases_tsv> [completion_index]}"
COMPLETION_INDEX="${5:-3}"
# Which conditions to run. "visible" alone needs no completions at all, which is
# how the pipeline gets exercised before pix2gestalt has produced any.
CONDITIONS="${6:-completed visible}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_ROOT="$OUTPUT_ROOT/inputs"
PIXAL_ROOT="$OUTPUT_ROOT/pixal3d"

mkdir -p "$INPUT_ROOT" "$PIXAL_ROOT"

while IFS=$'\t' read -r SCENE IMAGE GT_ID OBJ_ID SLUG; do
  if [ "$SCENE" = scene_id ]; then
    continue
  fi

  COMPLETION_DIR=$(printf '%s/scene_%06d_image_%06d_gt_%06d' \
    "$COMPLETION_ROOT" "$SCENE" "$IMAGE" "$GT_ID")

  for CONDITION in $CONDITIONS; do
    if [ "$CONDITION" = completed ]; then
      test -s "$COMPLETION_DIR/completion_$(printf '%02d' "$COMPLETION_INDEX").png"
    fi
    INPUT_DIR="$INPUT_ROOT/$SLUG/$CONDITION"
    mkdir -p "$INPUT_DIR" "$PIXAL_ROOT/$SLUG"

    PREPARE_ARGS=(
      --ycbv-root "$YCBV_ROOT"
      --scene-id "$SCENE"
      --image-id "$IMAGE"
      --gt-id "$GT_ID"
      --mode "$CONDITION"
      --padding 0.2
      --size 1024
      --output-dir "$INPUT_DIR"
    )
    if [ "$CONDITION" = completed ]; then
      PREPARE_ARGS+=(
        --completion-dir "$COMPLETION_DIR"
        --completion-index "$COMPLETION_INDEX"
      )
    fi

    python "$REPO/tools/prepare_pixal3d_input.py" "${PREPARE_ARGS[@]}"
    FOV=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["fov_x_rad"])' \
      "$INPUT_DIR/meta.json")

    pushd "$REPO/pixal3d" >/dev/null
    python inference.py \
      --image "$INPUT_DIR/input.png" \
      --output "$PIXAL_ROOT/$SLUG/$CONDITION.glb" \
      --seed 42 \
      --fov "$FOV" \
      --low_vram \
      --resolution 1024
    popd >/dev/null

    test -s "$PIXAL_ROOT/$SLUG/$CONDITION.glb"
  done
done < "$CASES"

echo "PIXAL3D RUN OK"
