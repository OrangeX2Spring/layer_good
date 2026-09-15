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
# Why this is a GPU job for a download. Two attempts died first. wget crawled at
# 0.6 MB/s because it pulls one TCP stream from a Xet-backed repo; hf_xet does
# chunked parallel transfer instead, but its workers were OOM-killed in 21
# seconds on a GPU-free `data` allocation:
#
#   error: Detected 1 oom_kill event in StepId=25547.0.
#
# That is the 4 GB grant, not the node - memory on this cluster is granted per
# GPU and --mem* is refused. So a card is attached purely for the grant. It is
# never touched: jena's sm_52 TITAN X is unusable with current CUDA wheels,
# which is exactly why borrowing it costs nobody anything. Note the partition is
# 12g, not data: `data` carries no GRES, and every working GPU recipe in tools/
# goes through 12g or 24g with an explicit --nodelist.
#
# The tarballs stay packed on /mnt permanently. They unpack to tens of thousands
# of small frames, and the 100,000-file quota is the binding limit, not the
# 250 GB - extraction belongs in job-local /tmp inside whichever allocation runs
# the eval.
#
# huggingface_hub is not installed on the bare node, and does not belong on
# quota'd storage for a one-off. The venv goes in job-local /tmp and dies with
# the job; rebuilding it next time costs 30 seconds.
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

# 15575329963 and 933473431 bytes. Trust the byte count, not a progress bar.
ls -l "$DEST"
getusage "$ROOT"

echo "DOWNLOAD OK"
