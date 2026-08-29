#!/bin/bash
# Batch job: one GenRecon scene reconstruction, end to end.
#
#   sbatch tools/genrecon_run.sbatch [scene_id]        # default 7b6477cb95
#
# Submit from head. Frees the node when done, unlike an interactive allocation.
# Log: /mnt/projects/gr/3DRecon/genrecon_slurm-<jobid>.log
# Results: /mnt/projects/gr/3DRecon/genrecon_out/<scene_id>/ (on /mnt, so they
# survive the job; /tmp is wiped when the allocation ends).
#
# --nodelist is a hard constraint, so the job waits for muenchen rather than
# landing elsewhere. That is deliberate: this cluster's submit plugin refuses
# --exclude ("not allowed, it was removed automatically"), and koblenz is a
# TITAN RTX (sm_75) which flash-attn 2.7.3 has no kernels for. Override with
# `sbatch -w bonn ...` if muenchen is tied up; bonn/heidelberg/stuttgart are the
# other sm_86+ nodes in 24g.
#SBATCH --job-name=genrecon
#SBATCH --partition=24g
#SBATCH --gres=gpu:1
#SBATCH --nodelist=muenchen
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/genrecon_slurm-%j.log
set -euxo pipefail

SCENE="${1:-7b6477cb95}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"

hostname
nvidia-smi

# DINOv3 is gated and the pipeline fetches it at construction time, along with the
# TRELLIS.2 VAE decoders and RMBG-2.0. Token lives in ~/.hf_env (chmod 600), never
# inline. $HOME inside the container is /root, so it is sourced out here and passed
# in with -e.
source ~/.hf_env
export HF_TOKEN
: "${HF_TOKEN:?HF_TOKEN was not set by ~/.hf_env}"
export HF_HOME="$ROOT/.hf_cache"
export HF_XET_HIGH_PERFORMANCE=1

# Extraction runs on the node, not in the container: /tmp is node-local (so a data
# -partition job would extract somewhere this job cannot see), and `unzip` is not
# installed in the image.
bash "$REPO/tools/genrecon_extract_scannetpp.sh" "$SCENE"

podman load -i "$ROOT/genrecon.tar"
podman run --rm -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" --network=host \
  -e HF_HOME -e HF_TOKEN -e HF_XET_HIGH_PERFORMANCE \
  localhost/genrecon bash tools/genrecon_run.sh "$SCENE"

echo "JOB OK"
