#!/usr/bin/env python3
"""Score a Pixal3D reconstruction against the YCB-V ground-truth mesh.

Three numbers, in decreasing robustness:

  flatness_inflation  Ratio of (thinnest / longest) PCA extent, generated over
                      ground truth. Needs no alignment at all, and measures the
                      exact failure under investigation: 1.0 means the aspect
                      ratio is right, 3.0 means a thin object came out three
                      times too thick.
  chamfer_norm        Symmetric Chamfer after similarity alignment, as a
                      fraction of the GT mesh's longest extent. Overall shape
                      error once scale and pose are factored out.
  sensor_residual_mm  Median distance from the visible depth points to the
                      aligned reconstruction. Whether the result actually
                      explains what the camera saw, as opposed to being a
                      plausible object of the right general shape.

Pixal3D emits meshes in a normalised canonical frame, so alignment is searched
over the 24 axis-aligned rotations of the PCA frame rather than assumed.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

MM_TO_M = 1e-3
NUM_SAMPLES = 20000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Generated GLB")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Output of prepare_pixal3d_input.py (meta.json, point clouds)")
    parser.add_argument("--models-dir", type=Path, required=True,
                        help="YCB-V models/ or models_eval/ directory")
    parser.add_argument("--icp-iterations", type=int, default=40)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sample_surface(mesh_path, scale=1.0):
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    assert isinstance(loaded, trimesh.Trimesh), type(loaded)
    points, _ = trimesh.sample.sample_surface(loaded, NUM_SAMPLES)
    return np.asarray(points, dtype=np.float64) * scale


def pca_frame(points):
    """Center and rotate points onto their principal axes (descending spread).

    Returns (rotated, mean, basis); the inverse map is `rotated @ basis.T + mean`.
    """
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mean, full_matrices=False)
    basis = vt.T
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.0
    return (points - mean) @ basis, mean, basis


def robust_extents(points):
    """Per-axis 1st-to-99th percentile span, insensitive to stray facets."""
    low, high = np.percentile(points, [1.0, 99.0], axis=0)
    return np.sort(high - low)[::-1]


def signed_permutations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for row, col in enumerate(perm):
                R[row, col] = signs[row]
            if np.linalg.det(R) > 0:
                yield R


def umeyama(source, target):
    """Similarity transform (scale, R, t) mapping source onto target."""
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    src_c, dst_c = source - src_mean, target - dst_mean
    u, s, vt = np.linalg.svd(dst_c.T @ src_c / source.shape[0])
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    R = u @ correction @ vt
    scale = float(s @ np.diag(correction).ravel()) / float((src_c ** 2).sum() / source.shape[0])
    return scale, R, dst_mean - scale * R @ src_mean


def chamfer(source, target_tree, target, source_tree):
    forward = target_tree.query(source)[0]
    backward = source_tree.query(target)[0]
    return float(forward.mean() + backward.mean()) / 2.0


def align(source, target, iterations):
    """Similarity ICP from 24 axis-aligned inits; returns (chamfer, transform)."""
    target_tree = cKDTree(target)
    best = (np.inf, None)
    for R_init in signed_permutations():
        scale, R, t = 1.0, R_init, np.zeros(3)
        for _ in range(iterations):
            moved = scale * (source @ R.T) + t
            _, indices = target_tree.query(moved)
            scale, R, t = umeyama(source, target[indices])
        moved = scale * (source @ R.T) + t
        error = chamfer(moved, target_tree, target, cKDTree(moved))
        if error < best[0]:
            best = (error, (scale, R, t))
    return best


def main():
    args = parse_args()
    meta = json.loads((args.input_dir / "meta.json").read_text())

    gt_path = args.models_dir / f"obj_{meta['obj_id']:06d}.ply"
    gt_points = sample_surface(gt_path, scale=MM_TO_M)
    gen_points = sample_surface(args.mesh)

    gt_pca, gt_mean, gt_basis = pca_frame(gt_points)
    gen_pca, _, _ = pca_frame(gen_points)
    gt_extents, gen_extents = robust_extents(gt_pca), robust_extents(gen_pca)
    gt_flatness = float(gt_extents[2] / gt_extents[0])
    gen_flatness = float(gen_extents[2] / gen_extents[0])

    gt_scale = float(gt_extents[0])
    error, (scale, R, t) = align(gen_pca / gt_scale, gt_pca / gt_scale, args.icp_iterations)

    # Bring the reconstruction into the GT object frame, where the sensor points
    # already live, so the residual is measured against real observations.
    gen_in_gt_pca = scale * (gen_pca @ R.T) + t * gt_scale
    gen_in_obj = gen_in_gt_pca @ gt_basis.T + gt_mean

    sensor_points = np.load(args.input_dir / "visible_points_obj.npy")
    residual = cKDTree(gen_in_obj).query(sensor_points)[0]

    result = {
        "mesh": str(args.mesh),
        "obj_id": meta["obj_id"],
        "mode": meta["mode"],
        "visib_fract": meta["visib_fract"],
        "fov_x_deg": meta["fov_x_deg"],
        "gt_extents_mm": (gt_extents / MM_TO_M).tolist(),
        "gen_extents_normalised": (gen_extents / gen_extents[0]).tolist(),
        "gt_flatness": gt_flatness,
        "gen_flatness": gen_flatness,
        "flatness_inflation": gen_flatness / gt_flatness,
        "chamfer_norm": error,
        "chamfer_mm": error * gt_scale / MM_TO_M,
        "sensor_residual_mm": float(np.median(residual)) / MM_TO_M,
        "sensor_points": int(sensor_points.shape[0]),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
