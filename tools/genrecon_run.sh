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
#
# Checkpoint paths are NOT where the download puts them. `load_train_config`
# (genrecon/pipelines/setup_utils.py:322) derives the training config as
# `dirname(dirname(ckpt))/config.json`, i.e. it expects a training *run* directory:
#   genrecon_ckpt/<stage>/config.json  +  genrecon_ckpt/<stage>/ckpts/<name>.pt
# Flat .pt files make it look for /mnt/projects/gr/3DRecon/config.json and die.
# Neither README documents this. The config.json per stage is the repo's own
# training config, which train.py:150 copies into the run dir:
#   ss    <- configs/gen/ss_flow_img/genrecon.json
#   shape <- configs/gen/slat_flow_img2shape/genrecon_512.json
#   tex   <- configs/gen/slat_flow_imgshape2tex/genrecon_512.json
# Each declares exactly one model, so find_model_cfg's name-based fallback resolves
# regardless of the checkpoint filename.
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
  --ss_ckpt    "$ROOT/genrecon_ckpt/ss/ckpts/sparse_structure.pt" \
  --shape_ckpt "$ROOT/genrecon_ckpt/shape/ckpts/shape_slat.pt" \
  --tex_ckpt   "$ROOT/genrecon_ckpt/tex/ckpts/texture_slat.pt" \
  --num_imgs_per_scene 32

python chunked_to_glb.py \
  --inputs       "$OUT/to_glb_inputs.pt" \
  --chunk_inputs "$OUT/chunk_inputs.pt" \
  --output_dir   "$OUT"

ls -la "$OUT"
echo "RUN OK"
