#!/bin/bash
# Batch job: Step 0 of the causal-readout experiment on OPT-Pose.
#
#   sbatch tools/opt_pose_causal.sbatch [num_ref] [num_seqs]     # defaults 3 10
#
# Submit from head. Log:     /mnt/projects/gr/3DRecon/optpose_causal_slurm-<jobid>.log
#      Results:              /mnt/projects/gr/3DRecon/optpose_causal_out/
#
# Partition choice. The first HouseCat6D run was single-view and measured ~7 GB
# steady with a 10.8 GB transient at checkpoint load, on passau (RTX 5000, 16 GB).
# This job is S = num_ref + 1 frames, so global attention grows from 1374 to
# S*1374 tokens and the DPT heads decode S dense maps; 10-14 GB is the estimate,
# and 12g has no idle-GPU auto-stop rule to trip over. If it OOMs, move to 24g
# (`sbatch -p 24g -w muenchen ...`) -- above 12 GB it is a legitimate 24g job
# under auto-stop rule (c). Judge with `jobstats <jobid>`, not with torch's own
# max_memory_allocated: the two disagreed by 6 GB on GenRecon job 21027.
#
# --nodelist is a hard constraint (this cluster's submit plugin refuses
# --exclude), so the job waits for passau rather than landing on a 2080Ti with
# 12 GB. Check `sinfo -p 12g -N -o "%.10N %.9T %.15G"` before submitting.
#
# QoS students_normal allows one job with one GPU, so an interactive srun of
# your own will keep this in PD. Release it with `exit` first.
#SBATCH --job-name=optpose-causal
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --nodelist=passau
#SBATCH --propagate=NONE
#SBATCH --output=/mnt/projects/gr/3DRecon/optpose_causal_slurm-%j.log
set -euxo pipefail

NUM_REF="${1:-3}"
NUM_SEQS="${2:-10}"
ROOT=/mnt/projects/gr/3DRecon
REPO="$ROOT/layer_good"
DATA_ROOT="$ROOT/housecat6d"

hostname
nvidia-smi

# sonata is ungated and the checkpoint is already local, so HF_TOKEN is not
# required here -- it only avoids the unauthenticated-download throttle if the
# 434 MB sonata weights are not in the cache yet. Sourced if present, not demanded.
export HF_HOME="$ROOT/.hf_cache"
export HF_XET_HIGH_PERFORMANCE=1
if [ -f ~/.hf_env ]; then source ~/.hf_env; export HF_TOKEN; fi

# Extraction runs on the node, not in the container: `unzip` is absent from the
# image and adding it would make the saved tar stale.
bash "$REPO/tools/opt_pose_extract_housecat.sh" "$DATA_ROOT"

mkdir -p "$ROOT/optpose_causal_out"

podman load -i "$ROOT/optpose.tar"
podman run --rm -v /mnt:/mnt:rw -v /tmp:/tmp:rw -w "$REPO" \
  --device=nvidia.com/gpu="$CUDA_VISIBLE_DEVICES" --network=host \
  -e HF_HOME -e HF_XET_HIGH_PERFORMANCE \
  localhost/optpose bash tools/opt_pose_causal_run.sh "$NUM_REF" "$NUM_SEQS"

echo "JOB OK"
