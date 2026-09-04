#!/bin/bash
# Extract only what the occlusion scale-up reads from ycbv_test_all.zip (14 GB).
#
#   bash tools/occlusion_scaleup_extract_ycbv.sh [dest] [scene ...]
#
# Six scenes, and within them only rgb, depth, mask, mask_visib and the three
# json. A full extraction is 14 GB of mostly frames nothing here opens.
#
# Newer unzip refuses these BOP archives -- "invalid zip file with overlapped
# components (possible zip bomb)" on muenchen, silence on passau. Node-dependent,
# so it is set here rather than remembered.
#
# Re-running is a no-op once a previous run FINISHED, recorded by .extract_ok.
# Checking for the directories instead would wave through a partial unzip, which
# is exactly what happened to the HouseCat6D script on 2026-09-04.
set -euo pipefail
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

DEST="${1:-/tmp/data/ycbv}"
shift || true
SCENES=("${@:-51 52 53 55 56 57}")
if [ "${#SCENES[@]}" -eq 1 ]; then read -r -a SCENES <<< "${SCENES[0]}"; fi
SRC=/mnt/datasets/bop/ycbv/ycbv_test_all.zip
STAMP="$DEST/.extract_ok"

if [ -f "$STAMP" ]; then
  echo "already extracted: $DEST"; cat "$STAMP"; echo "EXTRACT OK"; exit 0
fi

test -s "$SRC"
mkdir -p "$DEST"

for S in "${SCENES[@]}"; do
  D=$(printf 'test/%06d' "$S")
  echo "[extract] $D"
  unzip -qo "$SRC" "$D/rgb/*" "$D/depth/*" "$D/mask/*" "$D/mask_visib/*" \
    "$D/scene_gt.json" "$D/scene_gt_info.json" "$D/scene_camera.json" -d "$DEST"
done

# Go/no-go before the stamp is written: every path the loaders open.
: > "$STAMP"
for S in "${SCENES[@]}"; do
  D=$(printf '%s/test/%06d' "$DEST" "$S")
  test -d "$D/rgb"; test -d "$D/depth"; test -d "$D/mask"; test -d "$D/mask_visib"
  test -s "$D/scene_gt.json"; test -s "$D/scene_gt_info.json"; test -s "$D/scene_camera.json"
  printf 'scene %s: %s rgb frames\n' "$S" "$(ls "$D/rgb" | wc -l)" >> "$STAMP"
done

cat "$STAMP"
du -sh "$DEST"
echo "EXTRACT OK"
echo "pass as ycbv_root:  $DEST"
