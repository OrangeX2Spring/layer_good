"""Verify the Point2Pose inference closure inside a container.

Run from the repository root, inside a GPU allocation, after staging the wheels
that `kvt.tar` does not already carry:

    python tools/point2pose_env_check.py

Prints one line per module and exits non-zero if any import fails, so it can gate
a job. Importing `point2pose.pipeline.modular_pipeline` is the real test: it pulls
in sam2 via the segmenter package even though the YCBInEOAT config sets
`use_segmenter: false`.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "point_to_pose"))

# (import path, attribute holding a version string or None)
MODULES = [
    ("torch", "__version__"),
    ("torchvision", "__version__"),
    ("cv2", "__version__"),
    ("numpy", "__version__"),
    ("scipy", "__version__"),
    ("open3d", "__version__"),
    ("omegaconf", "__version__"),
    ("trimesh", "__version__"),
    ("numba", "__version__"),
    ("skimage", "__version__"),
    ("gtsam", "__version__"),
    ("lightglue", None),
    ("tapnet.torch.tapir_model", None),
    ("sam2.build_sam", None),
    ("point2pose.pipeline.modular_pipeline", None),
]

failures = []
for name, version_attr in MODULES:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)
        continue
    version = getattr(module, version_attr, "?") if version_attr else ""
    print(f"ok   {name} {version}".rstrip())

if "torch" not in failures:
    torch = importlib.import_module("torch")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")

if failures:
    print(f"ENV FAIL: {', '.join(failures)}")
    sys.exit(1)
print("ENV OK")
