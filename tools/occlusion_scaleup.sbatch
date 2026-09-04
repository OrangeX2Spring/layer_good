#!/bin/bash
# Unattended YCB-V occlusion scale-up: Pix2Gestalt -> Pixal3D -> SAM 3D.
#
# Submit from head after this commit is present in the cluster checkout:
#
#   sbatch tools/occlusion_scaleup.sbatch [conditions] [pix2gestalt_dir] [pix2gestalt_python]
#
# conditions defaults to "visible", which needs no completions and therefore no
# pix2gestalt -- and pix2gestalt is not installed on this cluster at all (checked
# 2026-09-04: no directory, no checkpoint, no environment). Running the visible
# condition is also how Pixal3D and SAM 3D get exercised here for the first time.
#
# For "completed visible" you must supply a pix2gestalt directory and its
# interpreter, OR have completions already sitting in
# $OUTPUT_ROOT/completions/scene_%06d_image_%06d_gt_%06d/completion_03.png,
# produced off-cluster and rsynced in. With completions present the pix2gestalt
# stage is skipped automatically.
#
# YCB-V is extracted by the job itself into job-local /tmp: on this cluster it
# exists only as a 14 GB zip, and anything put in /tmp dies with the allocation,
# so it cannot be prepared in advance from an interactive session.
#
# Log:     /mnt/projects/gr/3DRecon/occlusion_scaleup_slurm-<jobid>.log
# Results: /mnt/projects/gr/3DRecon/occlusion_scaleup/
#
# muenchen is an sm_86 node compatible with both saved images. The nodelist is a
# hard constraint; override with `sbatch -w <compatible-node> ...` if appropriate.
#SBATCH --job-name=occlusion-scaleup
#SBATCH --partition=24g
#SBATCH --gres=gpu:1
#SBATCH --nodelist=muenchen
#SBATCH --propagate=NONE
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/gr/3DRecon/occlusion_scaleup_slurm-%j.log
set -uo pipefail

CONDITIONS="${1:-visible}"
PIX2GESTALT_DIR="${2:-}"
PIX2GESTALT_PYTHON="${3:-}"

ROOT=/mnt/projects/gr/3DRecon
YCBV_ROOT=/tmp/data/ycbv
REPO="$ROOT/layer_good"
OUTPUT_ROOT="${4:-$ROOT/occlusion_scaleup}"
CASES="$REPO/tools/occlusion_scaleup_cases.tsv"
COMPLETION_ROOT="$OUTPUT_ROOT/completions"
COMPLETION_INDEX=3

PIXAL_IMAGE_TAR="$ROOT/pixal3d.tar"
SAM3D_IMAGE_TAR="$ROOT/sam3d.tar"
SAM3D_CONFIG="$ROOT/sam3d_ckpt/hf/pipeline.yaml"

hostname
nvidia-smi
echo "conditions: $CONDITIONS"

set -e
test -s "$PIXAL_IMAGE_TAR"
test -s "$SAM3D_IMAGE_TAR"
test -s "$SAM3D_CONFIG"
test -s "$CASES"

bash "$REPO/tools/occlusion_scaleup_extract_ycbv.sh" "$YCBV_ROOT"
set +e

mkdir -p "$COMPLETION_ROOT" "$OUTPUT_ROOT"
export HF_HOME="$ROOT/.hf_cache"
export HF_XET_HIGH_PERFORMANCE=1
if [ -f ~/.hf_env ]; then
  source ~/.hf_env
  export HF_TOKEN
fi
PODMAN_HF_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  PODMAN_HF_ARGS=(-e HF_TOKEN)
fi

# Stage 1, and it is skipped whenever the completed condition is not requested.
# It is also skipped when completions already exist, which is the supported route:
# produce them where pix2gestalt works and rsync them in.
NEED_COMPLETIONS=0
case " $CONDITIONS " in *" completed "*) NEED_COMPLETIONS=1 ;; esac
HAVE=$(ls "$COMPLETION_ROOT"/scene_*/completion_$(printf '%02d' "$COMPLETION_INDEX").png 2>/dev/null | wc -l)

if [ "$NEED_COMPLETIONS" -eq 1 ] && [ "$HAVE" -lt 6 ]; then
  test -s "$PIX2GESTALT_DIR/inference.py"
  test -s "$PIX2GESTALT_DIR/ckpt/epoch=000005.ckpt"
  test -x "$PIX2GESTALT_PYTHON"
while IFS=$'\t' read -r SCENE IMAGE GT_ID OBJ_ID SLUG; do
  if [ "$SCENE" = scene_id ]; then
    continue
  fi
  "$PIX2GESTALT_PYTHON" "$REPO/tools/run_pix2gestalt_ycbv.py" \
    --ycbv-root "$YCBV_ROOT" \
    --pix2gestalt-dir "$PIX2GESTALT_DIR" \
    --scene-id "$SCENE" \
    --image-id "$IMAGE" \
    --gt-id "$GT_ID" \
    --output-dir "$COMPLETION_ROOT" \
    --device cuda:0 \
    --n-samples 4 \
    --ddim-steps 100 \
    --guidance-scale 2.0 \
    --precision autocast \
    --seed 0 \
    --crop-padding 0.2
done < "$CASES"
else
  echo "pix2gestalt stage skipped (conditions=$CONDITIONS, completions found=$HAVE)"
fi

podman load -i "$PIXAL_IMAGE_TAR"
podman run --rm \
  -v /mnt:/mnt:rw \
  -v /tmp:/tmp:rw \
  -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" \
  --network=host \
  -e HF_HOME \
  -e HF_XET_HIGH_PERFORMANCE \
  "${PODMAN_HF_ARGS[@]}" \
  -e ATTN_BACKEND=sdpa \
  -e SPARSE_ATTN_BACKEND=sdpa \
  localhost/pixal3d \
  bash tools/occlusion_scaleup_pixal3d.sh \
    "$YCBV_ROOT" "$COMPLETION_ROOT" "$OUTPUT_ROOT" "$CASES" "$COMPLETION_INDEX" "$CONDITIONS"
PIXAL_RC=$?
echo "pixal3d stage exit $PIXAL_RC"

podman load -i "$SAM3D_IMAGE_TAR"
podman run --rm \
  -v /mnt:/mnt:rw \
  -v /tmp:/tmp:rw \
  -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" \
  --network=host \
  -e CONDA_PREFIX=/usr/local/cuda \
  -e HF_HOME \
  -e HF_XET_HIGH_PERFORMANCE \
  "${PODMAN_HF_ARGS[@]}" \
  localhost/sam3d \
  python tools/occlusion_scaleup_sam3d.py \
    --config "$SAM3D_CONFIG" \
    --cases "$CASES" \
    --input-root "$OUTPUT_ROOT/inputs" \
    --output-root "$OUTPUT_ROOT/sam3d" \
    --seed 42 \
    --conditions $CONDITIONS
SAM3D_RC=$?
echo "sam3d stage exit $SAM3D_RC"

# Both stages are allowed to fail without discarding the other: an unattended job
# that loses six finished reconstructions because the seventh OOMed is worse than
# one that reports which half worked.
echo "OCCLUSION SCALE-UP JOB DONE  pixal3d=$PIXAL_RC sam3d=$SAM3D_RC"
[ "$PIXAL_RC" -eq 0 ] && [ "$SAM3D_RC" -eq 0 ] && echo "OCCLUSION SCALE-UP JOB OK"
