#!/bin/bash
# Add natten to the saved pixal3d image, in place, without rebuilding it.
#
#   sbatch tools/pixal3d_add_natten.sbatch
#
# Why this exists. Slurm job 21203 died at pipeline construction with
# `ModuleNotFoundError: No module named 'natten'`, on Pixal3D's first execution
# on this cluster. natten is not imported anywhere under pixal3d/, which is why
# the build dropped it - but image_conditioned_proj.py:_load_naf calls
# torch.hub.load("valeoai/NAF"), and NAF imports natten. The dependency arrives
# at runtime and a grep over the checkout cannot see it.
#
# It is not optional. IMAGE_COND_CONFIGS sets use_naf_upsample=True for
# shape_512, shape_1024 and tex_1024, and proj_channels is embed_dim*2 when it
# is on, so NAF is part of the checkpoint's architecture rather than a toggle.
#
# A prebuilt wheel means no compiler runs here, so this is a five-minute change
# to a 14.6 GB image rather than the three-hour rebuild tools/pixal3d_build.sh
# does. The GPU is attached only for the memory grant (docs/camp-cluster-field-
# notes.md: `--mem*` is refused and a GPU-free allocation gets 4 GB); jena's
# sm_52 TITAN X is unusable with current CUDA wheels, so nothing is taken from
# anyone. `data` also has no idle-GPU auto-stop rule, which 24g does.
#
# Log: /mnt/projects/gr/3DRecon/pixal3d_natten_slurm-<jobid>.log
#SBATCH --job-name=pixal3d-natten
#SBATCH --partition=data
#SBATCH --nodelist=jena
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/gr/3DRecon/pixal3d_natten_slurm-%j.log
set -euo pipefail

ROOT=/mnt/projects/gr/3DRecon
TAR="$ROOT/pixal3d.tar"
NEW_TAR="$ROOT/pixal3d_natten.tar"
WHEEL=https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.5/natten-0.17.5%2Btorch260cu124-cp311-cp311-linux_x86_64.whl

hostname
echo "mem_grant_kb=$(ulimit -m) nproc=$(nproc)"
test -s "$TAR"
test ! -e "$NEW_TAR"   # podman save refuses to overwrite; fail before the load

# ~9 minutes, silent throughout. Not hung.
podman load -i "$TAR"

# --replace rather than --rm: the container has to survive the run so that
# podman commit has something to commit.
podman run \
  -v /mnt:/mnt:rw -v /tmp:/tmp:rw \
  --name=pixal3d-natten --replace --network=host \
  -e TORCH_HOME=/mnt/projects/gr/3DRecon/.torch_cache \
  localhost/pixal3d \
  bash -c "
    set -euo pipefail
    pip install --no-cache-dir '$WHEEL'
    pip list | grep -i natten
    # The real entry point, not just the module: NAF is what actually needs
    # natten, and the zipball is already in TORCH_HOME from job 21203. Loading
    # it on CPU here is the check that job could not make.
    python - <<'PY'
import torch
from natten.functional import na2d_av, na2d_qk
naf = torch.hub.load('valeoai/NAF', 'naf', pretrained=True, device='cpu', trust_repo=True)
print('natten ok, NAF loaded:', type(naf).__name__)
PY
  "

# Save to a new path and swap. Deleting the only copy first opens a window in
# which losing the allocation costs the whole image; the podman store is
# job-local and dies with the job.
podman commit pixal3d-natten pixal3d
podman images | grep -i pixal3d
podman save -o "$NEW_TAR" localhost/pixal3d
ls -la "$ROOT"/pixal3d*.tar

# Only now is the old one expendable.
rm "$TAR"
mv "$NEW_TAR" "$TAR"
ls -la "$TAR"

echo "PIXAL3D NATTEN OK"
