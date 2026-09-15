#!/bin/bash
# Download anything from HuggingFace onto CAMP, fast and without being OOM-killed.
#
#   sbatch tools/hf_download.sbatch <repo_id> <model|dataset> <dest_dir> [file ...]
#
# Examples:
#   sbatch tools/hf_download.sbatch yslan/STream3R model \
#       /mnt/projects/gr/3DRecon/stream3r_ckpt
#   sbatch tools/hf_download.sbatch yslan/pointmap_regression_evalsets dataset \
#       /mnt/projects/gr/3DRecon/stream3r_data neural_rgbd.tar.gz nyu-v2.tar.gz
#
# With no file list the whole repo is fetched - check the repo size first, some
# are hundreds of GB against a 250 GB quota.
#
# ---------------------------------------------------------------------------
# Why this script exists. Measured 2026-09-15 pulling 16.5 GB of STream3R eval
# sets; three attempts failed before this shape worked.
#
#   wget, one TCP stream          0.588 MB/s and falling, ~2.9 h remaining
#   hf_xet on a GPU-free `data`   OOM-killed after 67 MB in 21 seconds
#   this script                   16.5 GB in 2m57s, ~93 MB/s   (job 25555)
#
# Three things each cost an attempt:
#
# 1. USE hf download, NOT wget/curl. HF repos are Xet-backed; wget gets a single
#    stream from the CDN while hf_xet does chunked parallel transfer. This is a
#    160x difference, not a tuning detail. HF_XET_HIGH_PERFORMANCE=1 is what
#    turns the parallel path on (HF_HUB_ENABLE_HF_TRANSFER is deprecated).
#
# 2. IT NEEDS A GPU, for the memory grant and nothing else. Memory on this
#    cluster is granted per GPU and --mem* is refused, so a GPU-free allocation
#    gets 2 CPUs and 4 GB however much the node reports - and hf_xet's parallel
#    workers blow that in seconds:
#
#      error: Detected 1 oom_kill event in StepId=25547.0.
#
#    With --gres=gpu:1 the same node grants 24 GB and 4 CPUs. Read `ulimit -m`,
#    never `free -g`.
#
# 3. THE PARTITION IS 12g, NOT data. `data` carries no GRES, so --gres there is
#    unsatisfiable. jena sits in both partitions; 12g is the one that can hand
#    out its card. jena is the polite place to borrow one: its sm_52 TITAN X is
#    unusable with current CUDA wheels, so nobody loses compute. The card is
#    never touched here.
#
# Not an authentication problem, ever. The repo may well be ungated - a token
# only affects rate limits. `hf auth whoami` runs below so the log answers the
# question in writing instead of it being re-litigated.
#
# huggingface_hub is not installed on bare nodes and does not belong on quota'd
# storage for a one-off: the venv goes in job-local /tmp and dies with the job.
# Rebuilding it costs ~30 s, against ~40K files if it lived on /mnt.
#
# Archives stay PACKED on /mnt. The binding quota is 100,000 files, not 250 GB;
# unpack into job-local /tmp inside whichever allocation consumes them.
#
# Log: /mnt/projects/gr/3DRecon/hf_download_slurm-<jobid>.log
#SBATCH --job-name=hf-download
#SBATCH --partition=12g
#SBATCH --nodelist=jena
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/gr/3DRecon/hf_download_slurm-%j.log
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: sbatch tools/hf_download.sbatch <repo_id> <model|dataset> <dest_dir> [file ...]" >&2
  exit 2
fi

REPO_ID=$1
REPO_TYPE=$2
DEST=$3
shift 3
FILES=("$@")

case "$REPO_TYPE" in
  model|dataset) ;;
  *) echo "repo_type must be 'model' or 'dataset', got '$REPO_TYPE'" >&2; exit 2 ;;
esac

hostname
# The go/no-go line. ~24 GB means the card brought its memory grant with it;
# 4096000 means it did not and hf_xet will be killed shortly.
echo "mem_grant_kb=$(ulimit -m) nproc=$(nproc)"

test -s /mnt/home/chunquancheng/.hf_env
source /mnt/home/chunquancheng/.hf_env
export HF_HOME=/mnt/projects/gr/3DRecon/.hf_cache
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p "$DEST"

python3 -m venv /tmp/hfdl
/tmp/hfdl/bin/pip install -q --upgrade pip
/tmp/hfdl/bin/pip install -q "huggingface_hub[hf_xet]"
# Without hf_xet the CLI silently falls back to plain HTTP and you are back at
# wget speed, so assert it rather than trusting the install.
/tmp/hfdl/bin/python -c "import huggingface_hub as h, hf_xet; print('hub', h.__version__, '| hf_xet ok')"

/tmp/hfdl/bin/hf auth whoami

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "no file list given - fetching the whole repo"
  /tmp/hfdl/bin/hf download "$REPO_ID" --repo-type "$REPO_TYPE" --local-dir "$DEST"
else
  # Resumes whatever <dest>/.cache/huggingface/download/ already holds.
  /tmp/hfdl/bin/hf download "$REPO_ID" "${FILES[@]}" \
    --repo-type "$REPO_TYPE" --local-dir "$DEST"
fi

# Byte counts, not a progress bar: -O with --continue can silently append.
ls -l "$DEST"
getusage /mnt/projects/gr/3DRecon

echo "DOWNLOAD OK"
