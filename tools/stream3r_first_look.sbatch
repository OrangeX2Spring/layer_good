#!/bin/bash
# STream3R, first look: one reconstruction we can open in MeshLab.
#
#   sbatch tools/stream3r_first_look.sbatch
#
# Prerequisite, run once:
#   sbatch tools/hf_download.sbatch yslan/STream3R model \
#       /mnt/projects/gr/3DRecon/stream3r_ckpt
#
# This is NOT the paper's evaluation. eval/mv_recon/launch.py:75 builds a dict of
# 7scenes AND NRGBD and loops over both with no flag to select one, so running it
# would mean a second 25 GB download or patching the fork. Neither is wanted. The
# question here is only "does the reconstruction look like the room".
#
# Two subjects, both already on the cluster:
#   examples/static_room   10 PNGs shipped in the repo. Zero data cost, and it is
#                          the authors' own example - if this looks wrong the
#                          fault is ours, not the data handling.
#   one NRGBD scene        out of neural_rgbd.tar.gz (job 25555). NRGBD is the set
#                          behind the paper's headline table.
#
# NO CONTAINER IS BUILT. optpose.tar already carries torch/torchvision/numpy/
# Pillow/huggingface_hub/einops - the whole of STream3R's inference closure - and
# is borrowed read-only via PYTHONPATH. See tools/stream3r_shell.sh. The one
# assumption being tested is torch 2.5.0 against the README's 2.8.0; nothing read
# needs 2.6+, but the import and forward pass below are what actually settle it.
#
# Partition 12g/passau: nothing compiles, attention is plain
# scaled_dot_product_attention so sm_75 is fine, RTX 5000 has 16 GB against
# STream3R's 5.5-9 GB at these frame counts, and it sidesteps the 24g under-12 GB
# auto-stop. Do not add bf16 here - Turing has none.
#
# Everything staged into ONE tar on /mnt: outputs and the exact input frames.
# A picture whose inputs were left in job-local /tmp is not reconstructible.
#
# Log: /mnt/projects/gr/3DRecon/stream3r_first_look_slurm-<jobid>.log
#SBATCH --job-name=stream3r-look
#SBATCH --partition=12g
#SBATCH --nodelist=passau
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/gr/3DRecon/stream3r_first_look_slurm-%j.log
set -euo pipefail

ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
CKPT="$ROOT/stream3r_ckpt"
TARBALL="$ROOT/stream3r_data/neural_rgbd.tar.gz"
WORK=/tmp/stream3r
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$WORK/out"
ARCHIVE="$ROOT/stream3r_first_look_${STAMP}_${SLURM_JOB_ID}.tar"

hostname
echo "mem_grant_kb=$(ulimit -m) nproc=$(nproc)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

test -d "$CKPT" || { echo "checkpoint missing - run tools/hf_download.sbatch first" >&2; exit 1; }
test -s "$TARBALL"
ls -l "$CKPT"

mkdir -p "$WORK/nrgbd" "$OUT"

# Record the archive layout in the log. `head` closes the pipe, so tar dies of
# SIGPIPE - expected, hence the guards. Without them `pipefail` fails the job.
echo "--- neural_rgbd.tar.gz, first entries ---"
tar -tzf "$TARBALL" 2>/dev/null | head -20 || true

# NRGBD reads <ROOT>/<scene>/images/* (eval/mv_recon/data.py:470), so find the
# first member on that path and take the scene it belongs to. Discovered rather
# than hardcoded: the archive's internal rooting is not documented anywhere.
SCENE_PREFIX="$(tar -tzf "$TARBALL" 2>/dev/null | grep -m1 '/images/' | sed 's#\(.*\)/images/.*#\1#' || true)"
test -n "$SCENE_PREFIX" || { echo "no */images/* member found in $TARBALL" >&2; exit 1; }
echo "scene prefix: $SCENE_PREFIX"

tar -xzf "$TARBALL" -C "$WORK/nrgbd" "$SCENE_PREFIX"
SCENE_DIR="$WORK/nrgbd/$SCENE_PREFIX/images"
test -d "$SCENE_DIR"
echo "$(ls "$SCENE_DIR" | wc -l) frames in $SCENE_DIR"

podman load -i "$ROOT/optpose.tar"   # ~9 min, silent throughout. Not hung.

# Read-only borrow: no commit, no save, the OPT-Pose image is left exactly as it
# was. PYTHONPATH makes the checkout importable without touching the image.
podman run \
  -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" \
  --name=stream3r-look --replace --network=host \
  -e PYTHONPATH="$REPO/stream3r" \
  -e HF_HOME="$ROOT/.hf_cache" \
  localhost/optpose \
  bash -c "
    set -euo pipefail
    python -c 'import torch, torchvision, numpy, PIL, huggingface_hub; print(\"deps ok\", torch.__version__)'
    python -c 'from stream3r.models.stream3r import STream3R; print(\"stream3r import ok\")'

    python tools/stream3r_visualize.py \
      --images stream3r/examples/static_room \
      --ckpt '$CKPT' --out '$OUT/static_room' \
      --modes causal full --max-frames 10

    python tools/stream3r_visualize.py \
      --images '$SCENE_DIR' \
      --ckpt '$CKPT' --out '$OUT/nrgbd' \
      --modes causal full --max-frames 16
  "

# The frames that produced the pictures travel with the pictures. Job-local /tmp
# is erased when the allocation ends; a single tar keeps the 100,000-file quota
# intact while preserving the record.
cp -r stream3r/examples/static_room "$OUT/static_room/input_frames"
mkdir -p "$OUT/nrgbd/input_frames"
python3 - "$SCENE_DIR" "$OUT/nrgbd" <<'PY'
import json, shutil, sys, os
scene_dir, out = sys.argv[1], sys.argv[2]
used = json.load(open(os.path.join(out, "manifest.json")))["images"]
for p in used:
    shutil.copy2(p, os.path.join(out, "input_frames", os.path.basename(p)))
print(f"staged {len(used)} input frames")
PY

tar -cf "$ARCHIVE" -C "$WORK" out
ls -la "$ARCHIVE"
getusage "$ROOT"

echo "FIRST LOOK OK"
