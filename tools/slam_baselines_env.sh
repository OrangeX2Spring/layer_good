#!/bin/bash
# Runs INSIDE the borrowed kvt container (tools/SLAM_BASELINES.md).
#
#   bash tools/slam_baselines_env.sh build     # once, from slam_baselines.sbatch prepare
#   bash tools/slam_baselines_env.sh install   # every run, from slam_baselines.sbatch pilot|full
#
# kvt.tar is never committed or saved. Packages go into the container's ephemeral
# layer and die with it. `build` compiles the CUDA/C extensions into wheels and stores
# every added package as a wheel in $SB/wheels, so `install` is offline and exact.
# Both assert that Python, torch, CUDA, torchvision and numpy are still the image's own.
set -euo pipefail

SB=/mnt/projects/gr/3DRecon/slam_baselines
SRC=/tmp/slam_src
WHEELS="$SB/wheels"

# Pinned upstreams, recorded 2026-10-12 (git ls-remote / clone HEAD).
MAST3R_SLAM_SHA=e6f4e3d474fad0e11f561482012be864ba8c3f17
VGGT_SLAM_V1_SHA=1c548f7dd72e9ae418bbbd74e7f9c16a886405bc   # branch version1.0
VGGT_SLAM_V2_SHA=35327ac28b7d193df9ccc39ba6346052bb6f1207   # main, VGGT-SLAM 2.0
VGGT_SHA=a288dd0f14786c93483e45524328726ab7b1b4ce           # facebookresearch/vggt (v1 setup.sh)
VGGT_SPARK_SHA=6e6e16107b88e8e76c751826af10d4295d87ecd2     # MIT-SPARK/VGGT_SPARK (v2 setup.sh)
SALAD_SHA=33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d          # Dominic101/salad
LIETORCH_SHA=e7df86554156b36846008d8ddbcc4d8521a16554

base_versions() {
  python -c "import sys, torch, torchvision, numpy; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torchvision.__version__, numpy.__version__)"
}

assert_base() {  # the image's versions, recorded by `build` before any install
  test "$(base_versions)" = "$(cat "$SB/base_versions.txt")"
  echo "base intact: python torch cuda torchvision numpy = $(cat "$SB/base_versions.txt")"
}

checkout() {  # url dir sha
  git clone -q "$1" "$SRC/$2"
  git -C "$SRC/$2" checkout -q "$3"
  git -C "$SRC/$2" submodule update -q --init --recursive
  echo "$2 $(git -C "$SRC/$2" rev-parse HEAD)" >> "$SRC/SHAS.txt"
}

import_checks() {  # imports only; kernels run on the target GPU in the run job's pilot
  local ms="$SRC/MASt3R-SLAM"
  (cd "$ms" && PYTHONPATH="$ms/thirdparty/mast3r:$ms/thirdparty/in3d" python -c "import lietorch, mast3r_slam_backends, curope, asmk.hamming, imgui, moderngl, pyrealsense2, evo; import mast3r_slam.visualization, mast3r_slam.tracker, mast3r_slam.global_opt; from mast3r_slam.mast3r_utils import load_mast3r, load_retriever; print('MASt3R-SLAM imports ok')")
  for v in v1 v2; do
    local vg=$([ $v = v1 ] && echo vggt || echo vggt_spark)
    (cd "$SRC/VGGT-SLAM-$v" && PYTHONPATH="$SRC/VGGT-SLAM-$v:$SRC/$vg:$SRC/salad" python -c "import gtsam, open3d, viser; from vggt.models.vggt import VGGT; from vggt_slam.solver import Solver; print('VGGT-SLAM $v imports ok, gtsam', gtsam.__version__)")
  done
}

case "${1:?build or install}" in
build)
  test ! -e "$WHEELS" || { echo "$WHEELS exists; move it aside first" >&2; exit 1; }
  rm -rf "$SRC"; mkdir -p "$SRC" "$WHEELS"
  checkout https://github.com/rmurai0610/MASt3R-SLAM.git MASt3R-SLAM "$MAST3R_SLAM_SHA"
  checkout https://github.com/MIT-SPARK/VGGT-SLAM.git VGGT-SLAM-v1 "$VGGT_SLAM_V1_SHA"
  checkout https://github.com/MIT-SPARK/VGGT-SLAM.git VGGT-SLAM-v2 "$VGGT_SLAM_V2_SHA"
  checkout https://github.com/facebookresearch/vggt.git vggt "$VGGT_SHA"
  checkout https://github.com/MIT-SPARK/VGGT_SPARK.git vggt_spark "$VGGT_SPARK_SHA"
  checkout https://github.com/Dominic101/salad.git salad "$SALAD_SHA"
  checkout https://github.com/princeton-vl/lietorch.git lietorch "$LIETORCH_SHA"
  base_versions > "$SB/base_versions.txt"
  pip freeze | sort > "$SB/base_freeze.txt"
  # Never move anything the image already has; a conflict fails here, loudly.
  # Only name==version lines: editable (-e) and direct-URL entries are not constraints.
  grep -E '^[A-Za-z0-9_.-]+==' "$SB/base_freeze.txt" > /tmp/constraints.txt
  assert_base
  echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"
  # MASt3R-SLAM's setup.py defines its extension only when this is True.
  python -c "import torch; assert torch.cuda.is_available(); print('cuda available for the build')"
  pip install --no-cache-dir -q -c /tmp/constraints.txt cython
  ms="$SRC/MASt3R-SLAM"
  # Compiled packages, against the image's torch (no build isolation). mast3r and in3d
  # are not wheeled: mast3r finds dust3r relative to its own source file
  # (mast3r/utils/path_to_dust3r.py) and in3d loads resources beside its source, so
  # both run from src.tar via PYTHONPATH, the equivalent of upstream's `pip install -e`.
  for p in "$SRC/lietorch" "$ms" "$ms/thirdparty/mast3r/dust3r/croco/models/curope" \
           "$ms/thirdparty/mast3r/asmk" "$ms/thirdparty/in3d/thirdparty/pyimgui"; do
    echo "--- wheel $p ---"
    pip wheel --no-deps --no-build-isolation --no-cache-dir -w "$WHEELS" "$p"
  done
  pip install --no-cache-dir --no-deps "$WHEELS"/*.whl
  # Declared runtime dependencies (MASt3R-SLAM pyproject, mast3r setup.py, in3d setup.py,
  # VGGT-SLAM requirements.txt). Dropped: torch/torchvision pins (the image keeps its own),
  # gradio (only mast3r/demo.py and VGGT-SLAM app.py import it), cmake/virtualenv
  # (build tooling), torchcodec (optional, upstream comments it out).
  pip install --no-cache-dir -c /tmp/constraints.txt \
    einops pyrealsense2 evo natsort plyfile pyyaml \
    scikit-learn roma matplotlib tqdm opencv-python scipy trimesh tensorboard pyglet \
    "huggingface-hub[torch]>=0.22" \
    moderngl==5.12.0 moderngl-window==2.4.6 glfw pyglm msgpack "trimesh[easy]" \
    open3d safetensors pytorch_metric_learning pytorch-lightning termcolor viser==0.2.23 \
    omegaconf requests lz4 gtsam-develop ftfy regex
  assert_base
  pip freeze | sort > "$SB/full_freeze.txt"
  comm -13 "$SB/base_freeze.txt" "$SB/full_freeze.txt" | grep -E '^[A-Za-z0-9_.-]+==' > "$SB/wheels.lock"
  # pip wheel, not download: some transitive packages publish only an sdist.
  pip wheel --no-cache-dir --no-deps -w "$WHEELS" -r "$SB/wheels.lock"
  import_checks
  cp "$SRC/SHAS.txt" "$SB/SHAS.txt"
  tar -cf "$SB/src.tar" -C /tmp slam_src
  ls -l "$WHEELS" | tail -n +2 | wc -l
  echo "ENV BUILD OK"
  ;;
install)
  test -s "$SB/wheels.lock"
  pip install --no-cache-dir -q --no-index --no-deps "$WHEELS"/*.whl
  assert_base
  rm -rf "$SRC"; tar -xf "$SB/src.tar" -C /tmp
  import_checks
  echo "ENV INSTALL OK"
  ;;
esac
