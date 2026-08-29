#!/bin/bash
# Inference half of the GenRecon batch job. Runs INSIDE localhost/genrecon.
# Invoked by tools/genrecon_run.sbatch; not meant to be run by hand.
#
#   bash tools/genrecon_run.sh <scene_id>
#
# The scene is already extracted to /tmp by the sbatch wrapper, on the node and
# not in here: `unzip` is absent from this image (the KV-Tracker image had the
# same gap, docs/kv-tracker-cluster.md trap 7) and adding it would make the
# saved tar stale.
#
# Outputs go straight to /mnt, never /tmp, so they survive the allocation.
set -euxo pipefail

SCENE="${1:?usage: genrecon_run.sh <scene_id>}"
ROOT=/mnt/projects/gr/3DRecon
OUT="$ROOT/genrecon_out/$SCENE"
mkdir -p "$OUT"

cd "$(dirname "$0")/../genrecon"

# Watch the log for "[VRAM] <stage>: peak X.XX GB" - that answers whether the 512
# pipeline fits in 24 GB. On OOM, lower --proj_batch_voxels (default 8192) first.
python reconstruct_scene.py \
  --mode Scannet_colmap \
  --path "/tmp/data/scannetpp/data/$SCENE" \
  --output_path "$OUT" \
  --ss_ckpt    "$ROOT/genrecon_ckpt/sparse_structure.pt" \
  --shape_ckpt "$ROOT/genrecon_ckpt/shape_slat.pt" \
  --tex_ckpt   "$ROOT/genrecon_ckpt/texture_slat.pt" \
  --num_imgs_per_scene 32

python chunked_to_glb.py \
  --inputs       "$OUT/to_glb_inputs.pt" \
  --chunk_inputs "$OUT/chunk_inputs.pt" \
  --output_dir   "$OUT"

ls -la "$OUT"
echo "RUN OK"
