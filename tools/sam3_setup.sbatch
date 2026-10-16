#!/bin/bash
# Download/build SAM3, persist the image, then run the real TUM person pilot.
# Submit via allocation-wrapped git pull; see DYNAMIC_CACHE_BIAS.md.
#SBATCH --job-name=sam3-tum-pilot
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/sam3_setup-%j.log
set -euo pipefail
test "$(uname -s)" = Linux
test "$(hostname -s)" != head
: "${SLURM_JOB_ID:?Run inside an allocation}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
WORK=$(mktemp -d "/tmp/sam3_setup_${SLURM_JOB_ID}_XXXXXX")
mkdir -p "$ROOT/dynamic_cache_out"
finish() {
    status=$?
    trap - EXIT
    echo "$status" > "$WORK/exit_status.txt"
    tar -C "$WORK" -cf "$ROOT/dynamic_cache_out/sam3_setup_${SLURM_JOB_ID}.tar.partial" .
    mv "$ROOT/dynamic_cache_out/sam3_setup_${SLURM_JOB_ID}.tar.partial" \
        "$ROOT/dynamic_cache_out/sam3_setup_${SLURM_JOB_ID}.tar"
    echo "SAM3 SETUP ARCHIVED exit=$status"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee "$WORK/run.log") 2>&1
hostname
echo "mem_grant_kb=$(ulimit -m)"
nvidia-smi
getusage "$ROOT"
df -h "$ROOT" /tmp
git -C "$REPO" -c fetch.recurseSubmodules=false pull --ff-only
git -C "$REPO" submodule update --init third_party/sam3
git -C "$REPO" rev-parse HEAD > "$WORK/parent_commit.txt"
git -C "$REPO/third_party/sam3" rev-parse HEAD > "$WORK/sam3_commit.txt"
cp "$REPO/tools/sam3_build.sh" "$REPO/tools/sam3_setup.sbatch" \
    "$REPO/tools/hf_download.sbatch" "$WORK/"
mkdir "$WORK/source"
git -C "$REPO/third_party/sam3" archive HEAD | tar -x -C "$WORK/source"

# Use the established authenticated hf/Xet downloader with this GPU memory grant.
# Its #SBATCH directives do not apply when invoked through bash here.
bash "$REPO/tools/hf_download.sbatch" facebook/sam3 model \
    "$ROOT/sam3_ckpt" sam3.pt LICENSE README.md
test -s "$ROOT/sam3_ckpt/sam3.pt"
sha256sum "$ROOT/sam3_ckpt/sam3.pt" > "$WORK/checkpoint.sha256"

BASE=docker.io/library/python:3.12-slim-bookworm
CONTAINER="sam3_build_${SLURM_JOB_ID}"
export SAM3_IMAGE="localhost/sam3:job-${SLURM_JOB_ID}"
export SAM3_IMAGE_TAR="$ROOT/sam3_${SLURM_JOB_ID}.tar"
export SAM3_CHECKPOINT="$ROOT/sam3_ckpt/sam3.pt"
test ! -e "$SAM3_IMAGE_TAR"
podman pull "$BASE"
podman image inspect "$BASE" > "$WORK/base_image.json"
podman run --name "$CONTAINER" --network=host \
    --device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}" \
    -v /mnt:/mnt:ro -v /tmp:/tmp:rw \
    "$BASE" bash "$WORK/sam3_build.sh" "$WORK/source"
podman commit "$CONTAINER" "$SAM3_IMAGE"
podman image inspect "$SAM3_IMAGE" > "$WORK/sam3_image.json"
# Save immediately to a new path. Never replace either shared existing image.
podman save -o "$SAM3_IMAGE_TAR.partial" "$SAM3_IMAGE"
test -s "$SAM3_IMAGE_TAR.partial"
tar -tf "$SAM3_IMAGE_TAR.partial" > "$WORK/image_members.txt"
mv "$SAM3_IMAGE_TAR.partial" "$SAM3_IMAGE_TAR"
sha256sum "$SAM3_IMAGE_TAR" > "$WORK/image.sha256"
printf 'export SAM3_IMAGE=%q\nexport SAM3_IMAGE_TAR=%q\nexport SAM3_CHECKPOINT=%q\n' \
    "$SAM3_IMAGE" "$SAM3_IMAGE_TAR" "$SAM3_CHECKPOINT" > "$WORK/pilot_env.sh"
cp "$WORK/pilot_env.sh" "$ROOT/dynamic_cache_out/sam3_${SLURM_JOB_ID}_env.sh"
echo "SAM3 IMAGE SAVED $SAM3_IMAGE_TAR"
# Imports and save have succeeded. First inference is the actual selected clip.
bash "$REPO/tools/dynamic_cache_bias.sbatch" pilot-tum
echo "SAM3 SETUP AND REAL PILOT OK"
