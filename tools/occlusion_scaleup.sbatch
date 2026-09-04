#!/bin/bash
# Unattended YCB-V occlusion scale-up: Pix2Gestalt -> Pixal3D -> SAM 3D.
#
# Submit from head after this commit is present in the cluster checkout:
#
#   sbatch tools/occlusion_scaleup.sbatch \
#     /path/to/ycbv \
#     /path/to/pix2gestalt \
#     /path/to/pix2gestalt/python
#
# The Pix2Gestalt Python argument is the interpreter from its working environment,
# not the system Python. The job reproduces the previous experiment's completion
# index 3, seed and crop settings, then runs visible/completed through both 3D models.
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
#SBATCH --output=/mnt/projects/gr/3DRecon/occlusion_scaleup_slurm-%j.log
set -euo pipefail

YCBV_ROOT="${1:?usage: sbatch tools/occlusion_scaleup.sbatch <ycbv_root> <pix2gestalt_dir> <pix2gestalt_python> [output_root]}"
PIX2GESTALT_DIR="${2:?usage: sbatch tools/occlusion_scaleup.sbatch <ycbv_root> <pix2gestalt_dir> <pix2gestalt_python> [output_root]}"
PIX2GESTALT_PYTHON="${3:?usage: sbatch tools/occlusion_scaleup.sbatch <ycbv_root> <pix2gestalt_dir> <pix2gestalt_python> [output_root]}"

ROOT=/mnt/projects/gr/3DRecon
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

test -d "$YCBV_ROOT/test/000051"
test -s "$PIX2GESTALT_DIR/inference.py"
test -s "$PIX2GESTALT_DIR/ckpt/epoch=000005.ckpt"
test -x "$PIX2GESTALT_PYTHON"
test -s "$PIXAL_IMAGE_TAR"
test -s "$SAM3D_IMAGE_TAR"
test -s "$SAM3D_CONFIG"
test -s "$CASES"

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
    "$YCBV_ROOT" "$COMPLETION_ROOT" "$OUTPUT_ROOT" "$CASES" "$COMPLETION_INDEX"

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
    --seed 42

echo "OCCLUSION SCALE-UP JOB OK"
