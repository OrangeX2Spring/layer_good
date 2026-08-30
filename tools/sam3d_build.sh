#!/bin/bash
# Build the SAM 3D Objects inference environment.
#
# Run this INSIDE a pytorch:2.5.1-cuda12.1-cudnn9-devel container that has the repo
# mounted and a GPU attached. See docs/sam3d-cluster.md.
# On success the last line printed is "BUILD OK".
#
# A GPU must be attached even though the build is not GPU-heavy: doc/setup.md warns
# that pytorch3d otherwise compiles without GPU support and fails much later with
# "RuntimeError: Not compiled with GPU support". Build on 24g (muenchen), not on the
# h200 that inference needs - h200 is only reachable via students_opportunistic,
# which is preemptible, and the job-local podman store dies with the job.
#
# Deviations from doc/setup.md, each deliberate:
#   1. No mamba/conda env. AGENTS.md rules out Conda on network storage; we install
#      into the image's base environment and `podman save` the image instead. The
#      base image is Python 3.11 and CUDA 12.1, matching environments/default.yml
#      (python=3.11.0, cuda-version=12.1).
#   2. `pip install -e . --no-deps` plus an explicit dependency list, instead of
#      `pip install -e '.[dev]'`. requirements.txt is a kitchen sink - it carries
#      sagemaker, wandb, mosaicml-streaming, auto_gptq, bitsandbytes, bpy, librosa,
#      decord and smplx, none of which an AST scan of sam3d_objects/, demo.py and
#      notebook/ finds imported anywhere. auto_gptq compiles CUDA and bpy is a ~1 GB
#      Blender build, so installing them is expensive as well as pointless.
#      ESCAPE HATCH: if something turns out to be missing at runtime, fall back to
#      the upstream sequence (`.[dev]`, then `.[p3d]`, then `.[inference]`) rather
#      than guessing - it is slow but faithful.
#   3. flash-attn 2.8.3 from the GitHub release wheel, not PyPI. Take the abiFALSE
#      variant: the official pytorch images are built with _GLIBCXX_USE_CXX11_ABI=0,
#      and the abiTRUE wheel installs cleanly then fails at `import flash_attn` with
#      an undefined c10::Error symbol. That cost one full build on GenRecon
#      (docs/genrecon-cluster.md, "Build gotchas"). The check is asserted below.
#   4. TORCH_CUDA_ARCH_LIST set explicitly. 8.6 = A5000/3090 (where we build),
#      8.9 = RTX 4090, 9.0 = H200 (where inference must run, since doc/setup.md
#      requires 32 GB VRAM). Without this every extension targets only the build
#      node's GPU and the saved image is pinned to that node.
#   5. nvdiffrast and vox2seq are installed even though NO requirements file
#      declares them. Both are on the default inference path:
#        - nvdiffrast is the default renderer
#          (pipeline/inference_pipeline.py:91, used at
#          tdfy_dit/utils/postprocessing_utils.py:555). pytorch3d is the documented
#          alternative if this ever becomes a problem.
#        - vox2seq is imported by tdfy_dit/modules/sparse/attention/serialized_attn.py
#          and the structured-latent VAE really does use serialized blocks
#          (models/structured_latent_vae/base.py:17-27, SerializeMode.Z_ORDER).
#      utils3d is NOT installed explicitly: it arrives transitively via MoGe, which
#      pins it at 3913c65d. That is fragile - if MoGe is ever dropped, utils3d
#      silently disappears and pipeline/utils/pointmap.py fails at import.
#   6. diffoctreerast, diff_gaussian_rasterization and torchsparse are deliberately
#      NOT built. All three are guarded: the first two warn and disable their
#      renderer when absent (octree_renderer.py:195-198, gaussian_render.py:24-30),
#      and torchsparse is the alternative to the declared spconv backend
#      (modules/sparse/__init__.py:5 - BACKEND defaults to "spconv").
#   7. opencv-python-headless replaces opencv-python. Same version, no GUI libs,
#      which is what a container wants.
set -euxo pipefail

REPO="$(cd "$(dirname "$0")/../sam3d" && pwd)"
cd "$REPO"

python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# Deviation 3: assert the ABI before installing the wheel, rather than discovering
# it at import time after a long build.
python - <<'PY'
import torch, sys
abi = torch._C._GLIBCXX_USE_CXX11_ABI
print("_GLIBCXX_USE_CXX11_ABI =", abi)
if abi:
    sys.exit("this image is cxx11abiTRUE; switch the flash-attn wheel below to abiTRUE")
PY

apt-get update
# The GL/EGL set is what nvdiffrast needs at runtime; libgomp/libX11 are open3d's.
apt-get install -y --no-install-recommends \
  git build-essential curl \
  libgl1 libglx0 libegl1 libgles2 libglvnd0 \
  libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1
rm -rf /var/lib/apt/lists/*

# ── The package itself, without its kitchen-sink requirements (deviation 2) ────
pip install --no-cache-dir -e . --no-deps

# ── Dependencies actually imported on the inference path ──────────────────────
# Pins are the repo's own where it pins them; unpinned otherwise.
pip install --no-cache-dir \
  hydra-core==1.3.2 lightning==2.3.3 timm==0.9.16 optree==0.14.1 \
  easydict==1.13 loguru==0.7.2 astor==0.8.1 igraph==0.11.8 \
  open3d==0.18.0 pymeshfix==0.17.0 xatlas==0.0.9 seaborn==0.13.2 \
  opencv-python-headless==4.9.0.80 gradio==5.49.0 \
  safetensors scipy tqdm trimesh imageio matplotlib plyfile huggingface_hub

# MoGe also supplies utils3d (pinned there at 3913c65d) - see deviation 5.
pip install --no-cache-dir \
  "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# ── Prebuilt CUDA wheels ──────────────────────────────────────────────────────
pip install --no-cache-dir spconv-cu121==2.3.8 xformers==0.0.28.post3

pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# kaolin ships only on NVIDIA's own index, keyed to the exact torch build.
pip install --no-cache-dir kaolin==0.17.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html

# ── CUDA extensions built from source ─────────────────────────────────────────
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
mkdir -p /tmp/extensions

pip install --no-cache-dir --no-build-isolation \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

pip install --no-cache-dir --no-build-isolation \
  "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install --no-cache-dir /tmp/extensions/nvdiffrast --no-build-isolation

# vox2seq is a TRELLIS extension with no PyPI release and no pin anywhere in this
# repo (deviation 5), and it is worse than that: microsoft/TRELLIS DELETED the whole
# extensions/ directory in e74d605 ("remove unused extensions"), while its own
# setup.sh:238 still says `cp -r extensions/vox2seq`. A shallow clone of main
# therefore yields nothing at all, and the source has to be recovered from history.
# Measured 2026-08-30; cost one build to find. The clone is deliberately NOT
# --depth 1 (~1.07 GiB) because we need to reach back past that deletion.
git clone https://github.com/microsoft/TRELLIS.git /tmp/extensions/TRELLIS
git -C /tmp/extensions/TRELLIS checkout e74d605^ -- extensions/vox2seq
pip install --no-cache-dir /tmp/extensions/TRELLIS/extensions/vox2seq --no-build-isolation

# ── The hydra patch, vendored rather than run ─────────────────────────────────
# ./patching/hydra downloads hydra/core/utils.py from a personal fork
# (gleize/hydra @ 78f00766, = upstream PR facebookresearch/hydra#2863) and overwrites
# the installed file. Done explicitly here so the source and SHA are recorded in the
# build rather than hidden in a script. If that fork ever disappears, the patch has
# to be sourced from the PR itself.
HYDRA_DIR="$(python -c 'import os, hydra; print(os.path.dirname(hydra.__file__))')"
python -c 'import hydra, sys; sys.exit(0 if hydra.__version__ == "1.3.2" else "hydra is not 1.3.2; the patch does not apply")'
curl -fsSL -o "$HYDRA_DIR/core/utils.py" \
  https://raw.githubusercontent.com/gleize/hydra/78f00766b5f37672aa7232ebbf01bdd74246bd60/hydra/core/utils.py

# ── Verify ────────────────────────────────────────────────────────────────────
pip list | grep -i -E 'torch|flash|kaolin|gsplat|nvdiffr|vox2seq|spconv|xformers|hydra|moge|utils3d'

# LIDRA_SKIP_INIT is required, not optional: sam3d_objects/__init__.py imports
# sam3d_objects.init unless it is set, and that module is internal Meta code that
# was never shipped publicly. notebook/inference.py:6 sets it for the same reason,
# so anything importing the package outside that entry point must set it too.
cd "$REPO"
LIDRA_SKIP_INIT=1 python - <<'PY'
import torch, flash_attn, nvdiffrast.torch, vox2seq, kaolin, gsplat
import pytorch3d, spconv, xformers, utils3d, moge, hydra
# inference_pipeline_pointmap is the _target_ the released pipeline.yaml instantiates.
import sam3d_objects.pipeline.inference_pipeline_pointmap  # noqa: F401
print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__, "hydra", hydra.__version__)
print("all imports ok")
PY

echo "BUILD OK"
