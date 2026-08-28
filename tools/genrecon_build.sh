#!/bin/bash
# Build the GenRecon inference environment.
#
# Run this INSIDE a pytorch:2.6.0-cuda12.4-cudnn9-devel container that has the repo
# mounted and a GPU attached. See docs/genrecon-cluster.md.
# On success the last line printed is "BUILD OK".
#
# Deviations from the repo's own setup.sh, each deliberate:
#   1. No conda env. AGENTS.md rules out Conda on network storage; we install into
#      the image's base environment and `podman save` the image instead.
#   2. flash-attn comes from the GitHub release wheel, not `pip install flash-attn`.
#      PyPI ships only an sdist, so the setup.sh form compiles from source for hours.
#      The base image is Python 3.11, and a cu12/torch2.6/cp311/abiTRUE wheel exists.
#   3. pillow-simd dropped. Its last release predates Python 3.11 and it conflicts
#      with the Pillow that torchvision and transformers already require. It is a
#      decode-speed optimisation, not a dependency.
#   4. TORCH_CUDA_ARCH_LIST set explicitly. Without it every CUDA extension targets
#      only the GPU present at build time, pinning the saved image to one node.
#      8.6 = RTX 3090 / A5000, 8.9 = RTX 4090, 9.0 = H200 (the fallback if 24 GB
#      turns out to be too little). sm_75 is deliberately absent: flash-attn 2.7.3
#      has no Turing kernels, so 12g nodes cannot run this regardless.
set -euxo pipefail

REPO="$(cd "$(dirname "$0")/../genrecon" && pwd)"
cd "$REPO"

python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

apt-get update
# libjpeg-dev per setup.sh; the GL/EGL set is what nvdiffrast needs at runtime.
apt-get install -y --no-install-recommends \
  git build-essential libjpeg-dev \
  libgl1 libglx0 libegl1 libgles2 libglvnd0 \
  libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1
rm -rf /var/lib/apt/lists/*

# ── Basic deps (setup.sh --basic, minus pillow-simd) ──────────────────────────
pip install --no-cache-dir \
  imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
  transformers==4.57.3 gradio==6.0.1 tensorboard pandas lpips zstandard \
  kornia timm plyfile
pip install --no-cache-dir \
  git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# ── flash-attn 2.7.3, prebuilt (see deviation 2) ──────────────────────────────
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# ── CUDA extensions ───────────────────────────────────────────────────────────
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
mkdir -p /tmp/extensions

git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install --no-cache-dir /tmp/extensions/nvdiffrast --no-build-isolation

git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
pip install --no-cache-dir /tmp/extensions/nvdiffrec --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh
pip install --no-cache-dir /tmp/extensions/CuMesh --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM
pip install --no-cache-dir /tmp/extensions/FlexGEMM --no-build-isolation

# o-voxel is bundled in the repo. Copy it out first: building in place would write
# build/ and *.so into the repo on /mnt, against the 100,000-file quota.
cp -r "$REPO/o-voxel" /tmp/extensions/o-voxel
pip install --no-cache-dir /tmp/extensions/o-voxel --no-build-isolation

# ── Verify ────────────────────────────────────────────────────────────────────
pip list | grep -i -E 'nvdiffr|cumesh|flex|voxel|flash|torch|transformers'

cd "$REPO"
python - <<'PY'
import torch, flash_attn, nvdiffrast.torch, cumesh, flex_gemm, o_voxel
from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline
print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__)
print("all imports ok")
PY

echo "BUILD OK"
