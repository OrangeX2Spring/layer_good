#!/bin/bash
# Build the OPT-Pose inference environment.
#
# Run this INSIDE a pytorch:2.5.0-cuda12.4-cudnn9-devel container that has the
# repo mounted and a GPU attached. See docs/opt-pose-cluster.md step 7.
# On success the last line printed is "BUILD OK".
set -euxo pipefail

REPO="$(cd "$(dirname "$0")/../opt_pose" && pwd)"
cd "$REPO"

python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

apt-get update
apt-get install -y --no-install-recommends \
  libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1
rm -rf /var/lib/apt/lists/*

pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir spconv-cu124
pip install --no-cache-dir torch-scatter \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install --no-cache-dir -e .

# PointNet++ CUDA extension. The arch list is set explicitly so the saved image
# runs on passau (RTX 5000, sm_75) and on the 24g nodes (RTX 3090 / A5000,
# sm_86); without it the build targets only the GPU present at build time.
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
cd geo_models/pointnet2
python setup.py install
cd "$REPO"

python - <<'PY'
import torch, cv2, numpy, spconv.pytorch, torch_scatter, open3d
import pointnet2._ext
import sonata
from opt.models.opt import OPT
print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available())
print("numpy", numpy.__version__, "opencv", cv2.__version__)
print("open3d", open3d.__version__)
print("all imports ok")
PY

echo "BUILD OK"
