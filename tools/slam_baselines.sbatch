#!/bin/bash
# VGGT-SLAM 1.0/2.0 and MASt3R-SLAM on TUM in the borrowed kvt image.
# Protocol, published targets and gates: tools/SLAM_BASELINES.md.
#
#   bash tools/slam_baselines.sbatch prepare   # once: wheels, sources, checkpoints (12g jena)
#   bash tools/slam_baselines.sbatch pilot     # fr1_xyz, five arms (24g stuttgart)
#   bash tools/slam_baselines.sbatch full      # five fr1 + fr3 office (24g stuttgart)
#
# Submitted with `sbatch --wrap` after a pull inside the allocation, and with
# -o ../slam_baselines_slurm-%j.log; exact commands in tools/SLAM_BASELINES.md. VGGT-1B
# comes first from tools/hf_download.sbatch.
#
# No image is built, committed or saved: kvt.tar is borrowed and the added packages
# live in the container's ephemeral layer (tools/slam_baselines_env.sh). prepare
# compiles, so it needs a GPU memory grant; jena's card is never used. pilot/full run
# on stuttgart (RTX 4090), the GPU of the KV-Tracker comparisons.
#
# Log: /mnt/projects/gr/3DRecon/slam_baselines_slurm-<jobid>.log
#SBATCH --job-name=slam-baselines
#SBATCH --account=students
#SBATCH --qos=students_normal
#SBATCH --partition=24g
#SBATCH --nodelist=stuttgart
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --propagate=NONE
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/projects/gr/3DRecon/slam_baselines_slurm-%j.log
set -euo pipefail

MODE=${1:?prepare, pilot or full}
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
SB="$ROOT/slam_baselines"
OUT="$ROOT/slam_baselines_out"
WORK="/tmp/slam_baselines_${SLURM_JOB_ID}"
TORCH_HOME="$SB/torch_home"
MASTR=https://download.europe.naverlabs.com/ComputerVision/MASt3R
SALAD=https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt

hostname
echo "mem_grant_kb=$(ulimit -m) nproc=$(nproc)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo "parent $(git -C "$REPO" rev-parse HEAD)"
mkdir -p "$WORK/context" "$OUT"

fetch() {  # url destination; .partial then rename, so a cut download never looks complete
  if [ ! -s "$2" ]; then
    wget -q -O "$2.partial" "$1"
    mv "$2.partial" "$2"
  fi
}

if ! podman image exists localhost/kvt; then
  podman load -i "$ROOT/kvt.tar"
fi
podman image inspect localhost/kvt > "$WORK/context/container.json"
CONTAINER=(podman run --rm --network=host --shm-size=16g
  -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO"
  --device="nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
  -e PYTHONUNBUFFERED=1 -e MPLBACKEND=Agg -e "TORCH_HOME=$TORCH_HOME")

if [ "$MODE" = prepare ]; then
  test -s "$SB/vggt_1b/model.pt"
  mkdir -p "$SB/ckpt" "$TORCH_HOME/hub/checkpoints"
  for f in MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
           MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
           MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl; do
    fetch "$MASTR/$f" "$SB/ckpt/$f"
  done
  fetch "$SALAD" "$TORCH_HOME/hub/checkpoints/dino_salad.ckpt"
  # VGGT-SLAM loads torch.hub.load_state_dict_from_url(.../VGGT-1B/.../model.pt).
  ln -sfn "$SB/vggt_1b/model.pt" "$TORCH_HOME/hub/checkpoints/model.pt"
  ls -lL "$SB/ckpt" "$TORCH_HOME/hub/checkpoints"
  # Wheels for stuttgart's RTX 4090 (8.9) and the A5000 nodes (8.6), not jena's card.
  "${CONTAINER[@]}" -e "TORCH_CUDA_ARCH_LIST=8.6;8.9" -e MAX_JOBS=4 localhost/kvt bash -c "bash tools/slam_baselines_env.sh build && python -c \"import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14'); print('dinov2 hub cached')\""
  sha256sum "$SB"/ckpt/* "$TORCH_HOME/hub/checkpoints/dino_salad.ckpt" "$SB/vggt_1b/model.pt" > "$SB/checkpoints.sha256"
  getusage "$ROOT"
  echo "SLAM BASELINES prepare JOB OK"
  exit 0
fi

case "$MODE" in
  pilot) SEQS=(freiburg1_xyz) ;;
  full) SEQS=(freiburg1_xyz freiburg1_rpy freiburg1_desk freiburg1_desk2 freiburg1_room
              freiburg3_long_office_household) ;;
  *) echo "mode must be prepare, pilot or full" >&2; exit 2 ;;
esac
mkdir -p "$WORK/data/tum"
for s in "${SEQS[@]}"; do
  unzip -q "/mnt/datasets/tum-rgbd/rgbd_dataset_$s.zip" -d "$WORK/data/tum"
  sha256sum "/mnt/datasets/tum-rgbd/rgbd_dataset_$s.zip" >> "$WORK/context/datasets.sha256"
done
cp "$SB"/SHAS.txt "$SB"/wheels.lock "$SB"/base_versions.txt "$SB"/checkpoints.sha256 \
   "$REPO"/tools/slam_baselines* "$REPO"/tools/SLAM_BASELINES.md "$WORK/context/"
git -C "$REPO" rev-parse HEAD > "$WORK/context/parent_revision.txt"

"${CONTAINER[@]}" localhost/kvt bash -c "bash tools/slam_baselines_env.sh install && \
  python tools/slam_baselines_run.py --mode $MODE --data $WORK/data --work $WORK"

cp "$ROOT/slam_baselines_slurm-${SLURM_JOB_ID}.log" "$WORK/context/job.log"
ARCHIVE="$OUT/slam_${MODE}_${SLURM_JOB_ID}.tar"
tar -cf "$ARCHIVE" --exclude=data -C /tmp "slam_baselines_${SLURM_JOB_ID}"
ls -l "$ARCHIVE"
getusage "$ROOT"
echo "SLAM BASELINES $MODE JOB OK"
