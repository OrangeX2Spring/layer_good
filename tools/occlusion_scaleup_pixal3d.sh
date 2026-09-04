#!/bin/bash
# Prepare the prior experiment's visible/completed inputs and run Pixal3D.
# Runs inside localhost/pixal3d, not on the host.
#
# Failures are isolated per case and per stage, and the script runs every case
# before reporting. Slurm job 21203 is why: `set -euo pipefail` over the loop
# meant one failure on the first of six cases discarded the other five, and
# because input preparation and Pixal3D inference sat in the same failure
# domain, SAM 3D was left with nothing to read either. Preparation is cheap,
# CPU-only and independent of the model, so it is now separated: a Pixal3D
# failure costs a GLB, not the whole run.
set -uo pipefail

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

PREPARED=0
RECONSTRUCTED=0
FAILED=()

while IFS=$'\t' read -r SCENE IMAGE GT_ID OBJ_ID SLUG; do
  if [ "$SCENE" = scene_id ]; then
    continue
  fi

  COMPLETION_DIR=$(printf '%s/scene_%06d_image_%06d_gt_%06d' \
    "$COMPLETION_ROOT" "$SCENE" "$IMAGE" "$GT_ID")

  for CONDITION in $CONDITIONS; do
    CASE="$SLUG/$CONDITION"

    if [ "$CONDITION" = completed ] &&
       [ ! -s "$COMPLETION_DIR/completion_$(printf '%02d' "$COMPLETION_INDEX").png" ]; then
      echo "PIXAL3D SKIP $CASE: no completion under $COMPLETION_DIR"
      FAILED+=("$CASE:no-completion")
      continue
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

    # Stage 1: the input. SAM 3D consumes exactly this, so it is worth having
    # even when the reconstruction below fails.
    if ! python "$REPO/tools/prepare_pixal3d_input.py" "${PREPARE_ARGS[@]}"; then
      echo "PIXAL3D FAIL $CASE: prepare"
      FAILED+=("$CASE:prepare")
      continue
    fi
    PREPARED=$((PREPARED + 1))

    FOV=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["fov_x_rad"])' \
      "$INPUT_DIR/meta.json")
    if [ -z "$FOV" ]; then
      echo "PIXAL3D FAIL $CASE: no fov_x_rad in $INPUT_DIR/meta.json"
      FAILED+=("$CASE:fov")
      continue
    fi

    # Stage 2: the reconstruction. A subshell rather than pushd/popd so a
    # failure cannot leave the loop in the wrong directory.
    if ( cd "$REPO/pixal3d" && python inference.py \
           --image "$INPUT_DIR/input.png" \
           --output "$PIXAL_ROOT/$SLUG/$CONDITION.glb" \
           --seed 42 \
           --fov "$FOV" \
           --low_vram \
           --resolution 1024 ) &&
       [ -s "$PIXAL_ROOT/$SLUG/$CONDITION.glb" ]; then
      RECONSTRUCTED=$((RECONSTRUCTED + 1))
      echo "PIXAL3D OK $CASE"
    else
      echo "PIXAL3D FAIL $CASE: inference"
      FAILED+=("$CASE:inference")
    fi
  done
done < "$CASES"

echo "PIXAL3D SUMMARY inputs=$PREPARED glbs=$RECONSTRUCTED failed=${#FAILED[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'PIXAL3D FAILED %s\n' "${FAILED[@]}"
  exit 1
fi

echo "PIXAL3D RUN OK"
