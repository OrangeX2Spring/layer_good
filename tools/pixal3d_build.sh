#!/bin/bash
# Build the Pixal3D inference environment.
#
# Run this INSIDE a pytorch:2.6.0-cuda12.4-cudnn9-devel container with the
# repository mounted. A GPU is not required because every CUDA architecture is
# named explicitly. On success the last line printed is "BUILD OK".
#
# Host RAM matters more than the GPU here: this compiles five CUDA extensions.
# Use a `data`-partition allocation on a node with >= 60 GB (see the table in
# docs/camp-cluster-field-notes.md) and never `head`, whose memory cgroup kills
# compilers. `data` also sidesteps the 24g idle-GPU auto-stop rule.
#
# Pixal3D's README defines the environment as "TRELLIS.2's install, then
# requirements.txt, then natten, then utils3d 0.0.2". The TRELLIS.2 half is
# already solved: tools/genrecon_build.sh compiled the same five extensions on
# `muenchen` on 2026-08-29, and o-voxel there is byte-identical in source list
# and pyproject to TRELLIS.2's. This script follows that proven order.
#
# Deviations from the README, each deliberate:
#   1. natten IS required, and a prebuilt wheel replaces the README's source
#      build. This reverses the earlier decision to drop it, which was wrong and
#      cost Slurm job 21203: `grep -rn natten pixal3d/` hits only README.md and
#      requirements-hfdemo.txt, but natten does not arrive through this
#      checkout at all. image_conditioned_proj.py:_load_naf does
#      torch.hub.load("valeoai/NAF"), and NAF imports natten. A grep over the
#      source cannot see a torch.hub dependency fetched at runtime.
#   2. No flash-attn. Both pixal3d/modules/attention/config.py and
#      pixal3d/modules/sparse/config.py accept 'sdpa', and the README documents
#      the sdpa fallback. This avoids the flash-attn ABI trap that cost
#      GenRecon a build (docs/genrecon-cluster.md).
#   3. TORCH_CUDA_ARCH_LIST set explicitly. Without it every extension targets
#      only the GPU present at build time, pinning the saved image to one node.
#      8.6 = RTX 3090 / A5000, 8.9 = RTX 4090, 9.0 = H200. sm_75 is absent: the
#      released flow checkpoints are bfloat16, so Turing is not a target.
#
# Two failures from the 2026-09-01 attempt, both fixed here:
#   A. Install order. o-voxel went first, and its pyproject declares
#      `cumesh @ git+...` and `flex_gemm @ git+...` as *runtime* dependencies,
#      so pip cloned and compiled both again from inside the o-voxel install -
#      and then kept going after o-voxel had already failed. Installing
#      CuMesh and FlexGEMM first makes those requirements already-satisfied.
#   B. `c++: fatal error: Killed signal terminated program cc1plus`, plus a
#      bare `Killed` from nvcc. That is the OOM killer, not a compile error:
#      MAX_JOBS was hardcoded to 4 with too little host RAM to back it.
set -euxo pipefail

REPO="$(cd "$(dirname "$0")/../pixal3d" && pwd)"
cd "$REPO"

hostname
free -g || true
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

apt-get update
# libjpeg-dev for pillow; ninja-build because requirements.txt does not carry
# ninja and torch.utils.cpp_extension needs it; the GL/EGL set is nvdiffrast's
# runtime requirement.
apt-get install -y --no-install-recommends \
  git build-essential ninja-build \
  libjpeg-dev libgl1 libglx0 libegl1 libgles2 libglvnd0 \
  libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1
rm -rf /var/lib/apt/lists/*

# ── Python deps (README steps 2 and 4) ────────────────────────────────────────
# The order is the README's and the downgrade is intended: requirements.txt
# pulls MoGe, which drags in the current utils3d (1.3), and Pixal3D wants 0.0.2.
# The verification block below imports MoGe so that incompatibility, if any,
# surfaces here rather than after the checkpoints have loaded.
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir \
  https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# natten (README step 3), as a prebuilt wheel rather than the README's source
# build. It is needed by valeoai/NAF, which _load_naf pulls through torch.hub -
# see deviation 1 above. 0.17.5 is the newest natten built against torch 2.6.0;
# 0.21.x, which the README names, exists only for newer torch. NAF tries
# `from natten.functional import na2d_av, na2d_qk` before falling back to
# `from natten import na2d`, and 0.17.5 provides that first path. Pinned by
# direct URL for the same reason utils3d is: the wheel index moves.
pip install --no-cache-dir \
  https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.5/natten-0.17.5%2Btorch260cu124-cp311-cp311-linux_x86_64.whl

# ── CUDA extensions ───────────────────────────────────────────────────────────
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"

# What bounds this build is the Slurm memory grant, NOT the node's free RAM and
# NOT the core count. On this cluster `--mem*` is refused outright -
#   slurm_job_submit: Memory requests (--mem*) are not allowed. Memory is
#   allocated automatically based on the number of GPUs.
# - so a GPU-free allocation gets a 4 GB grant regardless of the node. Slurm
# sets RLIMIT_RSS from that grant, which makes `ulimit -m` the number that
# matters; /proc/meminfo describes the node and is misleading here (61.8 GB
# MemAvailable on `jena` while the job could use 4).
#
# Measured 2026-09-01: a `--partition=data` allocation with no GPU died as
# "Detected 1 oom_kill event in StepId=21113.0" on ONE nvcc pass over CuMesh's
# src/clean_up.cu at MAX_JOBS=2. MAX_JOBS=1 would have died too. Request a GPU
# to get a workable grant, even where the GPU itself goes unused.
rss_kb=$(ulimit -m)
if [ "$rss_kb" = "unlimited" ]; then
  mem_gb=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
else
  mem_gb=$(( rss_kb / 1024 / 1024 ))
fi

# Fail here rather than 20 minutes into the compile. One nvcc pass over a single
# .cu for three architectures needs well over 4 GB on its own.
if [ "$mem_gb" -lt 8 ]; then
  echo "FATAL: this allocation can use ${mem_gb} GB (ulimit -m = ${rss_kb})." >&2
  echo "       Too little to compile. --mem is refused on this cluster; memory" >&2
  echo "       is granted per GPU, so re-allocate with --gres=gpu:1." >&2
  exit 1
fi

# Budget 10 GB per parallel job, capped at the core count.
if [ -z "${MAX_JOBS:-}" ]; then
  MAX_JOBS=$(( mem_gb / 10 ))
  if [ "$MAX_JOBS" -lt 1 ]; then MAX_JOBS=1; fi
  if [ "$MAX_JOBS" -gt "$(nproc)" ]; then MAX_JOBS=$(nproc); fi
fi
export MAX_JOBS
echo "mem_gb=${mem_gb} nproc=$(nproc) MAX_JOBS=${MAX_JOBS}"

mkdir -p /tmp/extensions

git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install --no-cache-dir /tmp/extensions/nvdiffrast --no-build-isolation

git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
pip install --no-cache-dir /tmp/extensions/nvdiffrec --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh
pip install --no-cache-dir /tmp/extensions/CuMesh --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM
pip install --no-cache-dir /tmp/extensions/FlexGEMM --no-build-isolation

# o-voxel is not vendored in Pixal3D; it comes from TRELLIS.2. Only o-voxel and
# its eigen submodule are wanted, hence the shallow clone - the full history of
# TRELLIS.2 plus eigen is several hundred MB for two directories.
#
# --no-deps is load-bearing, not tidiness. o-voxel's pyproject declares cumesh and
# flex_gemm by git URL, and pip re-clones a direct-URL requirement to read its
# metadata even when the package is already installed - which is how the
# 2026-09-01 run ended up compiling CuMesh a second time. Every one of o-voxel's
# declared dependencies (torch, numpy, plyfile, trimesh, tqdm, zstandard,
# easydict, cumesh, flex_gemm) is already installed by the time this line runs.
git clone --depth 1 --recursive --shallow-submodules \
  https://github.com/microsoft/TRELLIS.2.git /tmp/extensions/TRELLIS.2
pip install --no-cache-dir /tmp/extensions/TRELLIS.2/o-voxel \
  --no-build-isolation --no-deps

# ── Verify ────────────────────────────────────────────────────────────────────
pip list | grep -i -E 'nvdiffr|cumesh|flex|voxel|utils3d|moge|natten|torch|transformers|diffusers'

cd "$REPO"
ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa python - <<'PY'
import torch
import cumesh
import flex_gemm
import nvdiffrast.torch
import nvdiffrec_render
import o_voxel
import utils3d
from natten.functional import na2d_av, na2d_qk
from moge.model.v2 import MoGeModel
from pixal3d.modules.attention import config as attention_config
from pixal3d.modules.sparse import config as sparse_config
from pixal3d.pipelines import Pixal3DImageTo3DPipeline

assert attention_config.BACKEND == "sdpa"
assert sparse_config.ATTN == "sdpa"
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("all imports ok")
PY

echo "BUILD OK"
