#!/bin/bash
# Download STream3R's evaluation sets from HuggingFace.
#
#   sbatch tools/stream3r_download.sbatch
#
# What and why. STream3R's eval scripts read MonST3R/Spann3R preprocessed trees,
# not the original dataset releases. /mnt/datasets has none of neural_rgbd,
# 7scenes, sintel, bonn or nyu-v2, so the preprocessed sets come from the
# authors' own upload, yslan/pointmap_regression_evalsets (ungated, 228.6 GB in
# total). Only two are pulled here: neural_rgbd (15.6 GB) is the set behind the
# headline NRGBD table, and nyu-v2 (0.93 GB) is the cheapest end-to-end check
# that model, checkpoint and data layout agree.
#
# The job shape - GPU attached purely for the memory grant, 12g rather than
# data, hf_xet rather than wget - is explained once in tools/hf_download.sbatch,
# which is the generic version of this job. Prefer it for new downloads:
#
#   sbatch tools/hf_download.sbatch yslan/pointmap_regression_evalsets dataset \
#       /mnt/projects/gr/3DRecon/stream3r_data neural_rgbd.tar.gz nyu-v2.tar.gz
#
# This file is kept because it records which eval sets matter and why. Ran as
# job 25555: 16.5 GB in 2m57s, DOWNLOAD OK.
#
# Log: /mnt/projects/gr/3DRecon/stream3r_dl_slurm-<jobid>.log
#SBATCH --job-name=stream3r-dl
#SBATCH --partition=12g
#SBATCH --nodelist=jena
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/gr/3DRecon/stream3r_dl_slurm-%j.log
set -euo pipefail

ROOT=/mnt/projects/gr/3DRecon
DEST="$ROOT/stream3r_data"
REPO=yslan/pointmap_regression_evalsets

hostname
# The grant is the number that matters. `free -g` describes the node and lies.
# Expect ~30 GB here, not the 4096000 kB a GPU-free allocation gets.
echo "mem_grant_kb=$(ulimit -m) nproc=$(nproc)"

source /mnt/home/chunquancheng/.hf_env
export HF_HOME="$ROOT/.hf_cache"
# The hf_xet client's parallel path. HF_HUB_ENABLE_HF_TRANSFER is deprecated.
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p "$DEST"

python3 -m venv /tmp/hfdl
/tmp/hfdl/bin/pip install -q --upgrade pip
/tmp/hfdl/bin/pip install -q "huggingface_hub[hf_xet]"
/tmp/hfdl/bin/python -c "import huggingface_hub as h, hf_xet; print('hub', h.__version__, '| hf_xet ok')"

# Settles the recurring "am I authenticated?" question in the log, in writing.
# The repo is ungated, so the token only ever affected rate limits.
/tmp/hfdl/bin/hf auth whoami

# Resumes whatever .cache/huggingface/download/ already holds.
/tmp/hfdl/bin/hf download "$REPO" \
  neural_rgbd.tar.gz nyu-v2.tar.gz \
  --repo-type dataset \
  --local-dir "$DEST"

# 15575329963 and 931121601 bytes (both verified, job 25555). Trust the byte count, not a progress bar.
ls -l "$DEST"
getusage "$ROOT"

echo "DOWNLOAD OK"
