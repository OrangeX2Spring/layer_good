#!/usr/bin/env python3
"""Verify the sensor-depth to Pixal3D-voxel-grid mapping before relying on it.

Depth can only constrain the generated occupancy if we know exactly where a
metric point lands in Pixal3D's normalised cube. That mapping is derived here
and checked by reprojection: transform the visible depth points into Pixal3D
world units, project them through Pixal3D's virtual camera, and see whether
they land inside the input image's alpha. A correct mapping puts essentially
all of them inside; a sign error puts them somewhere obvious instead.

Conventions, read out of trainers/flow_matching/mixins/image_conditioned_proj.py:

  voxel (i,j,k) -> g = linspace(-1,1,R) per axis  (meshgrid indexing='ij')
  world         -> (gx, -gz, gy) / mesh_scale / 2      so mesh_scale=1 spans [-0.5, 0.5]
  camera-to-world = [[1,0,0,0],[0,0,-1,-d],[0,1,0,0],[0,0,0,1]]
                    i.e. camera at (0,-d,0) looking along +y, Blender axes
  f_px          = (16 / tan(fov/2)) * resolution / 32
  distance      = 0.5 / tan(fov/2)   (closed form of inference.distance_from_fov)

The last line means the unit cube exactly fills the image width, which is what
fixes the metric scale: the crop's width in metres at the object's depth
corresponds to 1.0 world unit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Blender-alignment rotation applied to the raw grid, from ProjGrid.__init__.
GRID_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
CAMERA_TO_WORLD_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Output of prepare_pixal3d_input.py")
    parser.add_argument("--grid-resolution", type=int, default=64,
                        help="Sparse-structure resolution the constraint will target")
    return parser.parse_args()


def pixal3d_camera(fov_x, resolution):
    """Virtual camera Pixal3D reconstructs in: (distance, focal in pixels)."""
    distance = 0.5 / np.tan(fov_x / 2.0)
    f_pixels = (16.0 / np.tan(fov_x / 2.0)) * resolution / 32.0
    return float(distance), float(f_pixels)


def sensor_to_world(points_cam, scale, center_depth, distance):
    """Rectified OpenCV camera points (metres) -> Pixal3D world units.

    OpenCV (x right, y down, z forward) becomes Blender (x right, y up, z back),
    the object centre is placed at the world origin, and the whole thing is
    scaled so the crop width maps to one world unit.
    """
    x = scale * points_cam[:, 0]
    y = -scale * points_cam[:, 1]
    z = -(scale * (points_cam[:, 2] - center_depth) + distance)
    camera_frame = np.stack([x, y, z], axis=1)
    return camera_frame @ CAMERA_TO_WORLD_R.T + np.array([0.0, -distance, 0.0])


def project(points_world, f_pixels, distance, resolution):
    """Pixal3D's projection, mirrored from project_points_to_image_batch."""
    camera_frame = (points_world - np.array([0.0, -distance, 0.0])) @ CAMERA_TO_WORLD_R
    x_cam, y_cam, z_cam = camera_frame[:, 0], camera_frame[:, 1], camera_frame[:, 2]
    depth = -z_cam
    u = f_pixels * x_cam / (depth + 1e-8) + resolution / 2.0
    v = -f_pixels * y_cam / (depth + 1e-8) + resolution / 2.0
    return np.stack([u, v], axis=1), depth


def world_to_voxel(points_world, grid_resolution, mesh_scale=1.0):
    """World units -> integer voxel indices of the sparse-structure grid."""
    grid = points_world * mesh_scale * 2.0 @ np.linalg.inv(GRID_ROTATION).T
    index = (grid + 1.0) / 2.0 * (grid_resolution - 1)
    return np.rint(index).astype(np.int64)


def main():
    args = parse_args()
    meta = json.loads((args.input_dir / "meta.json").read_text())
    resolution = meta["size"]

    alpha = np.array(Image.open(args.input_dir / "input.png").convert("RGBA"))[..., 3] > 0
    points_cam = np.load(args.input_dir / "visible_points_rect.npy")
    assert points_cam.ndim == 2 and points_cam.shape[1] == 3, points_cam.shape

    fov_x = meta["fov_x_rad"]
    distance, f_pixels = pixal3d_camera(fov_x, resolution)

    center_depth = float(np.median(points_cam[:, 2]))
    focal_real = 0.5 * (meta["K"][0][0] + meta["K"][1][1])
    scale = focal_real / (meta["crop_side_px"] * center_depth)

    points_world = sensor_to_world(points_cam, scale, center_depth, distance)
    pixels, depth = project(points_world, f_pixels, distance, resolution)

    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
    hit = np.zeros(u.shape, dtype=bool)
    hit[inside] = alpha[v[inside], u[inside]]

    voxels = world_to_voxel(points_world, args.grid_resolution)
    in_cube = np.all((voxels >= 0) & (voxels < args.grid_resolution), axis=1)

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
        "world_extent": (points_world.max(axis=0) - points_world.min(axis=0)).tolist(),
        "occupied_voxels": int(np.unique(voxels[in_cube], axis=0).shape[0]),
    }, indent=2))

    overlay = np.array(Image.open(args.input_dir / "input.png").convert("RGB"))
    overlay[v[inside & hit], u[inside & hit]] = (0, 255, 0)
    overlay[v[inside & ~hit], u[inside & ~hit]] = (255, 0, 0)
    out_path = args.input_dir / "camera_mapping_overlay.png"
    Image.fromarray(overlay).save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
