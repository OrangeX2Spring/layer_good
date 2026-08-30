#!/usr/bin/env python3
"""Verify the sensor-depth to Pixal3D-voxel-grid mapping before relying on it.

Depth can only constrain the generated occupancy if we know exactly where a
metric point lands in Pixal3D's normalised cube. The mapping lives in
pixal3d/utils/depth_constraint.py; this script checks it by reprojection:
transform the visible depth points into Pixal3D world units, project them
through Pixal3D's virtual camera, and see whether they land inside the input
image's alpha. A correct mapping puts essentially all of them inside; a sign
error puts them somewhere obvious instead.

Also reports the recovered world extents, which should match the object's real
dimensions once divided by scale_world_per_metre -- the check that caught depth
outliers inflating the depth axis by 7x.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pixal3d"))
# Pixal3D is a submodule since 2026-08-30; the package is one level deeper.
from pixal3d.utils.depth_constraint import (  # noqa: E402
    metric_scale,
    pixal3d_camera,
    project,
    sensor_to_world,
    voxel_constraint,
    world_to_voxel,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Output of prepare_pixal3d_input.py")
    parser.add_argument("--grid-resolution", type=int, default=32,
                        help="Sparse-structure resolution the constraint targets")
    parser.add_argument("--dump-voxels", action="store_true",
                        help="Write the free/occupied voxel sets as a coloured PLY")
    return parser.parse_args()


def write_voxel_ply(path, free, occupied):
    """Red = carved free space, green = forced occupied. Viewable in Blender/MeshLab."""
    points = np.concatenate([free, occupied]).astype(np.float32)
    colors = np.concatenate([
        np.tile([255, 0, 0], (free.shape[0], 1)),
        np.tile([0, 255, 0], (occupied.shape[0], 1)),
    ]).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def main():
    args = parse_args()
    meta = json.loads((args.input_dir / "meta.json").read_text())
    resolution = meta["size"]

    alpha = np.array(Image.open(args.input_dir / "input.png").convert("RGBA"))[..., 3] > 0
    points_cam = np.load(args.input_dir / "visible_points_rect.npy")

    fov_x = meta["fov_x_rad"]
    distance, f_pixels = pixal3d_camera(fov_x, resolution)
    center_depth = float(np.median(points_cam[:, 2]))
    focal_real = 0.5 * (meta["K"][0][0] + meta["K"][1][1])
    scale = metric_scale(focal_real, meta["crop_side_px"], center_depth)

    points_world = sensor_to_world(points_cam, scale, center_depth, distance)
    pixels, _ = project(points_world, f_pixels, distance, resolution)

    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
    hit = np.zeros(u.shape, dtype=bool)
    hit[inside] = alpha[v[inside], u[inside]]

    voxels = world_to_voxel(points_world, args.grid_resolution)
    in_cube = np.all((voxels >= 0) & (voxels < args.grid_resolution), axis=1)

    scene_path = args.input_dir / "scene_points_rect.npy"
    constraint = voxel_constraint(
        points_cam, focal_real, meta["crop_side_px"], fov_x, args.grid_resolution,
        scene_points_cam=np.load(scene_path) if scene_path.exists() else None,
    )
    extent = (points_world.max(axis=0) - points_world.min(axis=0))

    print(json.dumps({
        "mode": meta["mode"],
        "fov_x_deg": meta["fov_x_deg"],
        "distance": distance,
        "scale_world_per_metre": scale,
        "center_depth_m": center_depth,
        "points": int(points_cam.shape[0]),
        "fraction_in_image": float(inside.mean()),
        "fraction_on_alpha": float(hit.mean()),
        "fraction_in_cube": float(in_cube.mean()),
        "world_extent": extent.tolist(),
        "extent_cm": (extent / scale * 100.0).tolist(),
        "grid_resolution": args.grid_resolution,
        "free_voxels": int(constraint["free"].shape[0]),
        "occupied_voxels": int(constraint["occupied"].shape[0]),
        "total_voxels": args.grid_resolution ** 3,
    }, indent=2))

    if args.dump_voxels:
        ply_path = args.input_dir / "constraint_voxels.ply"
        write_voxel_ply(ply_path, constraint["free"], constraint["occupied"])
        print(ply_path)

    overlay = np.array(Image.open(args.input_dir / "input.png").convert("RGB"))
    overlay[v[inside & hit], u[inside & hit]] = (0, 255, 0)
    overlay[v[inside & ~hit], u[inside & ~hit]] = (255, 0, 0)
    out_path = args.input_dir / "camera_mapping_overlay.png"
    Image.fromarray(overlay).save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
