#!/bin/bash
# STream3R multi-view reconstruction eval on NRGBD, for numbers comparable to the
# paper's table.
#
#   sbatch tools/stream3r_eval_nrgbd.sbatch
#
# Paper, NRGBD row (Ours): Acc 0.057 / med 0.014, Comp 0.028 / med 0.013,
# NC 0.910 / med 0.993. Those are averages over the NRGBD scenes; a single scene
# is not comparable to them, which is why the whole archive is extracted.
#
# Why this is possible without 7-Scenes. NRGBD carries its own ground truth -
# NRGBD._get_views builds it from <scene>/depth/depth{i}.png plus poses.txt, with
# intrinsics hardcoded (fx=fy=554.2562584220408, cx=320, cy=240). There is no
# separate GT mesh. Upstream's launch.py nevertheless built a literal dict of
# 7scenes AND NRGBD and looped over both, and each constructor scans its ROOT at
# construction time, so the NRGBD number required a 25 GB download of a dataset it
# does not use. Our fork adds --datasets (commit 6e18980); the protocol values
# themselves - full_video=True, kf_every=500, num_seq=1 - are upstream's, untouched.
#
# kf_every=500 with _get_views' `img_idxs[::min(kf_every, len//2)]` means roughly
# three frames per scene. That is the authors' protocol, not a shortcut: it is
# far sparser than the 16 frames tools/stream3r_first_look.sbatch used.
#
# --conf_thresh is passed explicitly at its default. Upstream's eval/mv_recon/run.sh
# ends on a dangling backslash after --model_name, so whatever arguments it meant
# to pass are absent from the shipped script and the published command does not
# fully specify the published protocol. Recording the value we used is the only
# way this stays reproducible.
#
# No container is built. optpose.tar is borrowed read-only; the five packages it
# lacks are installed into the container's ephemeral layer and vanish with it, so
# the OPT-Pose image is never modified. They are needed only because
# `from stream3r.utils.utils import ...` executes stream3r/utils/__init__.py,
# which imports the lightning training scaffold.
#
# Log: /mnt/projects/gr/3DRecon/stream3r_eval_slurm-<jobid>.log
#SBATCH --job-name=stream3r-eval
#SBATCH --partition=12g
#SBATCH --nodelist=passau
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/projects/gr/3DRecon/stream3r_eval_slurm-%j.log
set -euo pipefail

ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
CKPT="$ROOT/stream3r_ckpt"
TARBALL="$ROOT/stream3r_data/neural_rgbd.tar.gz"
WORK=/tmp/stream3r_eval
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$WORK/eval_results"
ARCHIVE="$ROOT/stream3r_eval_nrgbd_${STAMP}_${SLURM_JOB_ID}.tar"

cd "$REPO"

hostname
echo "mem_grant_kb=$(ulimit -m) nproc=$(nproc)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
test -d "$CKPT"
test -s "$TARBALL"

mkdir -p "$WORK" "$OUT"

# The whole archive: the paper's number is an average over NRGBD's scenes.
echo "--- extracting $(du -h "$TARBALL" | cut -f1) ---"
time tar -xzf "$TARBALL" -C "$WORK"
DATA_ROOT="$(dirname "$(find "$WORK" -maxdepth 3 -type d -name images | head -1)" | xargs dirname)"
echo "data root: $DATA_ROOT"
ls "$DATA_ROOT"

# Preflight. NRGBD._get_views reads images/img{i}.png and depth/depth{i}.png, but
# the archive was observed to ship depth_filtered/. Fail loudly with the listing
# rather than substituting: filtered depth is not raw depth, and silently swapping
# them would change the protocol while still producing plausible numbers. Every
# miss in these loaders is a bare `continue`, so a layout mismatch otherwise
# surfaces as a run over zero frames, not as an error.
SCENE="$(ls "$DATA_ROOT" | head -1)"
echo "--- layout of $SCENE ---"
ls "$DATA_ROOT/$SCENE"
for need in images depth poses.txt; do
  if [ ! -e "$DATA_ROOT/$SCENE/$need" ]; then
    echo "MISSING: $DATA_ROOT/$SCENE/$need" >&2
    echo "present instead:" >&2
    ls "$DATA_ROOT/$SCENE" >&2
    exit 1
  fi
done
ls "$DATA_ROOT/$SCENE/images" | head -3
ls "$DATA_ROOT/$SCENE/depth" | head -3

# launch.py hardcodes ROOT="./data/neural_rgbd" relative to the repo root.
mkdir -p stream3r/data
ln -sfn "$DATA_ROOT" stream3r/data/neural_rgbd

podman load -i "$ROOT/optpose.tar"   # ~9 min, silent throughout. Not hung.

podman run \
  -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO/stream3r" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" \
  --name=stream3r-eval --replace --network=host \
  -e PYTHONPATH="$REPO/stream3r" \
  -e HF_HOME="$ROOT/.hf_cache" \
  localhost/optpose \
  bash -c "
    set -euo pipefail
    # Ephemeral: this layer dies with the container, optpose.tar is unchanged.
    pip install --no-cache-dir -q lightning lightning-utilities rich wandb pillow_heif
    # lightning declares torch>=2.1, which 2.5.0 satisfies - but assert it rather
    # than trust it. A silently upgraded torch would invalidate every number below.
    python -c \"import torch; assert torch.__version__.startswith('2.5.0'), torch.__version__; print('torch intact', torch.__version__)\"
    python -c 'from eval.mv_recon.launch import accuracy, completion; print(\"eval imports ok\")'

    python eval/mv_recon/launch.py \
      --output_dir='$OUT' \
      --size=518 \
      --model_name=stream3r \
      --datasets NRGBD \
      --conf_thresh=0.0 \
      --revisit=1
  "

cp "$ROOT/stream3r_eval_slurm-${SLURM_JOB_ID}.log" "$OUT/job.log" || true
tar -cf "$ARCHIVE" -C "$WORK" eval_results
ls -la "$ARCHIVE"
echo "--- per-scene results ---"
cat "$OUT"/*/logs_all.txt 2>/dev/null || find "$OUT" -type f | head -20
getusage "$ROOT"

echo "EVAL OK"
