#!/bin/bash
# Build the Pixal3D inference environment.
#
# Run this INSIDE a pytorch:2.6.0-cuda12.4-cudnn9-devel container with the
# repository mounted. A GPU is not required because every CUDA architecture is
# specified explicitly. On success the last line printed is "BUILD OK".
set -euxo pipefail

REPO="$(cd "$(dirname "$0")/../pixal3d" && pwd)"
cd "$REPO"

python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

apt-get update
apt-get install -y --no-install-recommends \
  git build-essential cmake ninja-build \
  libjpeg-dev libgl1 libglx0 libegl1 libgles2 libglvnd0 \
  libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1
rm -rf /var/lib/apt/lists/*

pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir \
  https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# Build for the eligible CAMP targets: Ampere/Ada 24g nodes and H200 fallback.
# The released flow checkpoints use bfloat16, so the Turing and older nodes are
# not runtime targets.
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
export NATTEN_CUDA_ARCH="8.6;8.9;9.0"
export NATTEN_N_WORKERS=4
export MAX_JOBS=4

pip install --no-cache-dir natten==0.21.0 --no-build-isolation

mkdir -p /tmp/extensions

git clone --recursive https://github.com/microsoft/TRELLIS.2.git /tmp/extensions/TRELLIS.2
pip install --no-cache-dir /tmp/extensions/TRELLIS.2/o-voxel --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh
pip install --no-cache-dir /tmp/extensions/CuMesh --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM
pip install --no-cache-dir /tmp/extensions/FlexGEMM --no-build-isolation

git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install --no-cache-dir /tmp/extensions/nvdiffrast --no-build-isolation

git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
pip install --no-cache-dir /tmp/extensions/nvdiffrec --no-build-isolation

cd "$REPO"
ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa python - <<'PY'
import torch
import natten
import cumesh
import flex_gemm
import nvdiffrast.torch
import nvdiffrec_render
import o_voxel
from pixal3d.pipelines import Pixal3DImageTo3DPipeline

assert natten.HAS_LIBNATTEN
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("natten", natten.__version__, "libnatten", natten.HAS_LIBNATTEN)
print("all imports ok")
PY

echo "BUILD OK"
