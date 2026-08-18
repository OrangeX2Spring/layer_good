#!/usr/bin/env python3
"""Report a generated mesh's real-world dimensions, without needing a GT mesh.

Pixal3D exports into the unit cube, and prepare_pixal3d_input.py plus the depth
observation give the world-units-per-metre scale, so the reconstruction's
extents convert directly to millimetres. That makes thickness checkable against
a tape measure rather than against an alignment.

003_cracker_box is 213 x 164 x 60 mm, so the number to watch is the smallest
extent: 60 mm is correct, 120 mm is the failure this experiment is about.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from ycbv_bop import MM_TO_M  # noqa: F401  (documents the unit convention)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, nargs="+", required=True)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Output of prepare_pixal3d_input.py, for the metric scale")
    parser.add_argument("--expected-mm", type=float, nargs=3,
                        help="Known object dimensions, longest first, for comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pixal3d.utils.depth_constraint import metric_scale

    meta = json.loads((args.input_dir / "meta.json").read_text())
    points = np.load(args.input_dir / "visible_points_rect.npy")
    focal_real = 0.5 * (meta["K"][0][0] + meta["K"][1][1])
    scale = metric_scale(focal_real, meta["crop_side_px"], float(np.median(points[:, 2])))

    for path in args.mesh:
        mesh = trimesh.load(path, force="mesh", process=False)
        sampled = np.asarray(trimesh.sample.sample_surface(mesh, 20000)[0])
        centered = sampled - sampled.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pca = centered @ vt.T
        low, high = np.percentile(pca, [1.0, 99.0], axis=0)
        extents_mm = np.sort(high - low)[::-1] / scale * 1000.0
        line = f"{path.name:<24} {extents_mm[0]:7.1f} {extents_mm[1]:7.1f} {extents_mm[2]:7.1f} mm"
        if args.expected_mm:
            ratio = extents_mm[2] / args.expected_mm[2]
            line += f"   thickness x{ratio:.2f} of expected"
        print(line)


if __name__ == "__main__":
    main()
