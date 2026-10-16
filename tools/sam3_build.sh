#!/bin/bash
# Runs inside a fresh Python 3.12 container on an allocated CAMP GPU.
set -euo pipefail
SOURCE=${1:?SAM3 source directory in job-local /tmp}
test -f "$SOURCE/pyproject.toml"
export PIP_NO_CACHE_DIR=1
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST=8.9
apt-get update
apt-get install -y --no-install-recommends gcc libc6-dev
# The pinned SAM3 imports pkg_resources, removed by newer setuptools releases.
python -m pip install setuptools==80.9.0 wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
# Declared dependencies plus the image-builder's transitive video-module imports.
# No optional flash-attn/cc_torch build: this image path uses PyTorch SDPA.
python -m pip install 'numpy==1.26.4' 'timm==1.0.17' 'ftfy==6.1.1' \
    tqdm regex iopath typing_extensions huggingface_hub \
    scipy einops pycocotools psutil pillow
python -m pip install --no-deps --no-build-isolation "$SOURCE"
python -m pip check
python -c 'import torch, torchvision, triton; from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); assert callable(torch.nn.functional.scaled_dot_product_attention); print("SAM3 IMPORTS OK", torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
python -m pip freeze
