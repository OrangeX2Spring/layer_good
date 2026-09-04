#!/bin/bash
# Preflight for tools/occlusion_scaleup.sbatch. Checks every precondition the job
# asserts, plus the ones it assumes without asserting, and reports them all
# instead of dying on the first miss.
#
#   bash tools/occlusion_scaleup_preflight.sh [ycbv_root] [pix2gestalt_dir]
#
# Run it on an allocated node, not on head: it touches /mnt and /tmp only, but
# head's cgroup makes even that unpleasant. Deliberately NOT `set -e` -- the point
# is a complete list of what is missing.
set -uo pipefail

YCBV_ROOT="${1:-/tmp/data/ycbv}"
PIX2GESTALT_DIR="${2:-}"
ROOT=/mnt/projects/gr/3DRecon
BOP=/mnt/datasets/bop/ycbv
FAIL=0

ok()   { printf '  OK    %s\n' "$1"; }
bad()  { printf '  MISS  %s\n' "$1"; FAIL=$((FAIL+1)); }
note() { printf '  note  %s\n' "$1"; }

check_file() { if [ -s "$1" ]; then ok "$2 ($(du -h "$1" | cut -f1))"; else bad "$2 -- $1"; fi; }
check_dir()  { if [ -d "$1" ]; then ok "$2"; else bad "$2 -- $1"; fi; }

echo "== container images and checkpoints =="
check_file "$ROOT/pixal3d.tar" "Pixal3D image"
check_file "$ROOT/sam3d.tar" "SAM 3D image"
check_file "$ROOT/sam3d_ckpt/hf/pipeline.yaml" "SAM 3D pipeline config"

echo "== Pixal3D weights =="
# inference.py defaults to MODEL_PATH = "TencentARC/Pixal3D", fetched from HF at
# runtime. Nothing has ever run Pixal3D on this cluster, so this is expected to be
# absent on a first pass -- it is a download, not a blocker, but it is several GB
# and it happens inside the job unless it is warmed first.
HFDIR="$ROOT/.hf_cache/hub/models--TencentARC--Pixal3D"
if [ -d "$HFDIR" ]; then ok "Pixal3D weights cached ($(du -sh "$HFDIR" | cut -f1))"
else note "Pixal3D weights NOT cached -- the first run downloads them into HF_HOME"; fi

echo "== YCB-V =="
if [ -d "$YCBV_ROOT/test/000051" ]; then
  ok "extracted at $YCBV_ROOT"
  for S in 51 52 53 55 56 57; do
    D=$(printf '%s/test/%06d' "$YCBV_ROOT" "$S")
    N=$(ls "$D/rgb" 2>/dev/null | wc -l)
    if [ -d "$D/rgb" ] && [ -d "$D/depth" ] && [ -d "$D/mask" ] && [ -d "$D/mask_visib" ] \
       && [ -s "$D/scene_gt.json" ] && [ -s "$D/scene_gt_info.json" ] && [ -s "$D/scene_camera.json" ]; then
      ok "scene $S complete ($N rgb frames)"
    else
      bad "scene $S incomplete ($N rgb frames; needs rgb depth mask mask_visib + three json)"
    fi
  done
else
  bad "NOT extracted at $YCBV_ROOT/test/000051"
  check_file "$BOP/ycbv_test_all.zip" "source zip to extract from"
fi

echo "== pix2gestalt =="
if [ -n "$PIX2GESTALT_DIR" ]; then
  check_file "$PIX2GESTALT_DIR/inference.py" "pix2gestalt inference.py"
  check_file "$PIX2GESTALT_DIR/ckpt/epoch=000005.ckpt" "pix2gestalt checkpoint"
else
  HIT=$(find /mnt/projects/gr -maxdepth 4 -iname "*gestalt*" 2>/dev/null | head -1)
  if [ -n "$HIT" ]; then note "found $HIT -- pass its directory as argument 2"
  else bad "pix2gestalt is not on this cluster (no directory, no checkpoint, no environment)"; fi
fi

echo "== completions (the alternative to running pix2gestalt here) =="
COMP="$ROOT/occlusion_scaleup/completions"
if [ -d "$COMP" ]; then
  N=$(ls -d "$COMP"/scene_* 2>/dev/null | wc -l)
  M=$(ls "$COMP"/scene_*/completion_03.png 2>/dev/null | wc -l)
  if [ "$M" -eq 6 ]; then ok "6 completions present ($N case dirs)"
  else bad "$M of 6 completion_03.png present in $COMP"; fi
else
  bad "no completions at $COMP -- produce them off-cluster and rsync them here"
fi

echo "== repo pieces =="
REPO="$(cd "$(dirname "$0")/.." && pwd)"
for F in tools/occlusion_scaleup_cases.tsv tools/occlusion_scaleup_pixal3d.sh \
         tools/occlusion_scaleup_sam3d.py tools/run_pix2gestalt_ycbv.py \
         tools/prepare_pixal3d_input.py pixal3d/inference.py sam3d/notebook/inference.py; do
  check_file "$REPO/$F" "$F"
done

echo "== node =="
printf '  note  %s, /tmp free %s\n' "$(hostname)" "$(df -h /tmp | tail -1 | awk '{print $4}')"
printf '  note  GPU %s\n' "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

echo
if [ "$FAIL" -eq 0 ]; then echo "PREFLIGHT OK"; else echo "PREFLIGHT: $FAIL missing"; exit 1; fi
