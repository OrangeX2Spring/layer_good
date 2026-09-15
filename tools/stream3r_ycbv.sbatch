#!/bin/bash
# STream3R over YCB-V object-centric video: the baseline for a semantic KV cache.
#
#   sbatch tools/stream3r_ycbv.sbatch                       # six verified scenes
#   sbatch tools/stream3r_ycbv.sbatch "48 49 50 51 52 53 54 55 56 57 58 59"
#
# Why this exists. The proposal is a training-free / small module doing object
# tracking with semantic-level KV caching. STream3R already caches K/V per patch
# token and already truncates that cache - but POSITIONALLY: stream_session.py's
# `window` mode keeps frame 0's tokens plus the last window_size frames'. A
# semantic cache would keep the tokens belonging to the tracked object instead.
# That is an indexing change inside _update_cache, no training - which is what
# makes the proposal a small module rather than a model.
#
# Two things this job establishes before any module is written:
#
#   1. How STream3R behaves on object-centric video at all. It is trained on
#      room-scale scenes; every number so far here is from rooms (NRGBD,
#      static_room). Whether it transfers to a tabletop with a moving camera and
#      a dominant background is unmeasured.
#   2. What fraction of the KV cache an object actually occupies
#      (tools/stream3r_ycbv_tokens.py). If it is a few percent, a semantic cache
#      is one to two orders of magnitude smaller than the frame cache that drives
#      causal mode to 45.4 GB at 200 frames - and unlike window mode, it would not
#      throw the object's own history away. That is the win condition.
#
# Why YCB-V rather than the HouseCat6D clips: mask_visib gives a per-object
# per-frame VISIBLE mask, and scene_gt_info.json gives visib_fract, so the
# semantic labels and the occlusion signal are both already annotated. The
# extraction script is the occlusion scale-up's, unchanged.
#
# Sampling is a contiguous stride, NOT the even spread used for NRGBD. The target
# here is tracking, so temporal continuity is the thing the causal cache exploits;
# spanning the whole clip would destroy exactly what is being measured.
#
# 32 frames at stride 5 covers 160 frames of video and needs a 24 GB card. Job 25580
# died at exactly this on passau's 16 GB, because the first version of this script
# sized the run from the paper's VRAM table - and that table (20 frames 9.02 GB,
# 40 frames 12.92 GB) describes StreamSession, which processes ONE FRAME AT A TIME
# with a KV cache. A single model(images, mode=...) call materialises every frame's
# activations at once and is far larger. Verified interactively on muenchen: 32
# frames fit on 24 GB, 32 frames do not fit on 16 GB, 16 frames do.
#
# Defaults to the six scenes tools/occlusion_scaleup_extract_ycbv.sh is verified
# on. Pass the full list as an argument once this is known to work - the
# extraction is the slow and untested part for the other six, not the inference.
#
# No container is built: optpose.tar is borrowed read-only. See tools/stream3r_shell.sh.
#
# Log: /mnt/projects/gr/3DRecon/stream3r_ycbv_slurm-<jobid>.log
#SBATCH --job-name=stream3r-ycbv
#SBATCH --partition=24g
#SBATCH --nodelist=muenchen
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/projects/gr/3DRecon/stream3r_ycbv_slurm-%j.log
set -euo pipefail

ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
CKPT="$ROOT/stream3r_ckpt"
DATA=/tmp/data/ycbv
WORK=/tmp/stream3r_ycbv
OUT="$WORK/out"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$ROOT/stream3r_ycbv_${STAMP}_${SLURM_JOB_ID}.tar"

SCENES="${1:-51 52 53 55 56 57}"
NUM_FRAMES=32
STRIDE=5

cd "$REPO"

hostname
echo "mem_grant_kb=$(ulimit -m) nproc=$(nproc)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
test -d "$CKPT"
echo "scenes: $SCENES"

mkdir -p "$OUT"
bash tools/occlusion_scaleup_extract_ycbv.sh "$DATA" "$SCENES"

podman load -i "$ROOT/optpose.tar"   # ~9 min, silent throughout. Not hung.

podman run \
  -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" \
  --name=stream3r-ycbv --replace --network=host \
  -e PYTHONPATH="$REPO/stream3r" \
  -e HF_HOME="$ROOT/.hf_cache" \
  localhost/optpose \
  bash -c "
    set -euo pipefail
    for S in $SCENES; do
      D=\$(printf '$DATA/test/%06d' \"\$S\")
      O=\$(printf '$OUT/scene%06d' \"\$S\")
      echo \"===== scene \$S =====\"
      ls \"\$D/rgb\" | wc -l

      python tools/stream3r_visualize.py \
        --images \"\$D/rgb\" \
        --ckpt '$CKPT' --out \"\$O\" \
        --modes causal window full \
        --sampling stride --stride $STRIDE --max-frames $NUM_FRAMES

      # No model, no GPU: pure mask accounting against the frames just used.
      python tools/stream3r_ycbv_tokens.py \
        --scene-dir \"\$D\" \
        --manifest \"\$O/manifest.json\" \
        --out \"\$O/tokens.json\"

      # Ground-truth metrics, read back from the .npz the run above saved:
      # no model, no GPU. Depth error is split object vs background, which is
      # the split the semantic-cache proposal turns on.
      python tools/stream3r_ycbv_metrics.py \
        --scene-dir \"\$D\" \
        --pred-dir \"\$O\" \
        --out \"\$O/metrics.json\"
    done
  "

tar -cf "$ARCHIVE" -C "$WORK" out
ls -la "$ARCHIVE"

echo "===== token-fraction summary across scenes ====="
python3 - "$OUT" <<'PY'
import json, sys, glob, os
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "tokens.json"))):
    s = json.load(open(f))["summary"]
    t = s["token_fraction_per_object"]
    print(f"{os.path.basename(os.path.dirname(f))}: "
          f"objects/frame={s['objects_per_frame']} "
          f"tokens/frame={s['tokens_total_per_frame']} "
          f"per-object median={t['median']*100:.2f}% max={t['max']*100:.2f}% "
          f"all-objects mean={s['all_objects_token_fraction_mean']*100:.1f}% "
          f"visib_fract median={s['visib_fract']['median']:.2f}")
PY

getusage "$ROOT"
echo "YCBV SWEEP OK"
