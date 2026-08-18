"""Turn observed sensor depth into hard occupancy constraints on the 32^3 grid.

Pixal3D generates the sparse structure from image conditioning alone, so
nothing stops it from placing geometry where the depth camera has already seen
empty space, or from making an object thicker than the measured surface allows.
For a box resting on a table and viewed from above, the visible top face pins
the thickness directly -- information the generative prior currently ignores.

The mapping below is the one Pixal3D itself uses, read out of
trainers/flow_matching/mixins/image_conditioned_proj.py:

  voxel (i,j,k) -> g = linspace(-1,1,R) per axis  (meshgrid indexing='ij')
  world         -> (gx, -gz, gy) / mesh_scale / 2      mesh_scale=1 spans [-0.5, 0.5]
  camera-to-world = [[1,0,0,0],[0,0,-1,-d],[0,1,0,0],[0,0,0,1]]
                    camera at (0,-d,0) looking along +y, Blender axes
  f_px          = (16 / tan(fov/2)) * resolution / 32
  distance      = 0.5 / tan(fov/2)   (closed form of inference.distance_from_fov)

That last identity means the unit cube exactly fills the image width, which is
what fixes the metric scale: the crop's width in metres at the object's depth
is one world unit.

Verified by reprojection on YCB-V scene 54 (cracker box): the back-projected
depth lands 100% inside the input alpha and 100% inside the cube, and recovers
extents of 16.5 x 20.6 cm against the object's true 16.4 x 21.3 cm.
"""

import numpy as np

GRID_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
CAMERA_TO_WORLD_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def pixal3d_camera(fov_x, resolution):
    """Virtual camera Pixal3D reconstructs in: (distance, focal in pixels)."""
    distance = 0.5 / np.tan(fov_x / 2.0)
    f_pixels = (16.0 / np.tan(fov_x / 2.0)) * resolution / 32.0
    return float(distance), float(f_pixels)


def metric_scale(focal_real, crop_side_px, center_depth):
    """World units per metre: the crop's width at the object's depth is 1.0."""
    return float(focal_real / (crop_side_px * center_depth))


def sensor_to_world(points_cam, scale, center_depth, distance):
    """Rectified OpenCV camera points (metres) -> Pixal3D world units.

    OpenCV (x right, y down, z forward) becomes Blender (x right, y up, z back),
    the object centre is placed at the world origin, and the result is scaled so
    the crop width maps to one world unit.
    """
    assert points_cam.ndim == 2 and points_cam.shape[1] == 3, points_cam.shape
    x = scale * points_cam[:, 0]
    y = -scale * points_cam[:, 1]
    z = -(scale * (points_cam[:, 2] - center_depth) + distance)
    return np.stack([x, y, z], axis=1) @ CAMERA_TO_WORLD_R.T + np.array([0.0, -distance, 0.0])


def project(points_world, f_pixels, distance, resolution):
    """Pixal3D's projection, mirrored from project_points_to_image_batch."""
    camera_frame = (points_world - np.array([0.0, -distance, 0.0])) @ CAMERA_TO_WORLD_R
    depth = -camera_frame[:, 2]
    u = f_pixels * camera_frame[:, 0] / (depth + 1e-8) + resolution / 2.0
    v = -f_pixels * camera_frame[:, 1] / (depth + 1e-8) + resolution / 2.0
    return np.stack([u, v], axis=1), depth


def voxel_centers(grid_resolution, mesh_scale=1.0):
    """World-space centre of every voxel, in the grid's own (i,j,k) order."""
    one_dim = np.linspace(-1.0, 1.0, grid_resolution)
    grid = np.stack(np.meshgrid(one_dim, one_dim, one_dim, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3) @ GRID_ROTATION.T / mesh_scale / 2.0


def world_to_voxel(points_world, grid_resolution, mesh_scale=1.0):
    """World units -> integer voxel indices of the sparse-structure grid."""
    grid = points_world * mesh_scale * 2.0 @ np.linalg.inv(GRID_ROTATION).T
    return np.rint((grid + 1.0) / 2.0 * (grid_resolution - 1)).astype(np.int64)


def voxel_constraint(
    points_cam,
    focal_real,
    crop_side_px,
    fov_x,
    grid_resolution,
    mesh_scale=1.0,
    buffer_resolution=128,
    margin_voxels=1.5,
):
    """Split the voxel grid into observed-free and observed-surface sets.

    Builds a z-buffer of the observed surface in Pixal3D's virtual camera, then
    classifies every voxel by comparing its own depth against it:

      free      strictly in front of the observed surface -- the sensor saw
                through this voxel, so nothing may be generated there
      occupied  within a margin of the observed surface

    Voxels whose pixel has no observed depth (the occluded region) are left
    unconstrained, so the generative prior still owns everything the camera
    could not see. Returns int arrays of (i, j, k) indices.
    """
    assert points_cam.ndim == 2 and points_cam.shape[1] == 3, points_cam.shape
    distance, f_pixels = pixal3d_camera(fov_x, buffer_resolution)
    center_depth = float(np.median(points_cam[:, 2]))
    scale = metric_scale(focal_real, crop_side_px, center_depth)

    surface_world = sensor_to_world(points_cam, scale, center_depth, distance)
    pixels, surface_depth = project(surface_world, f_pixels, distance, buffer_resolution)
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < buffer_resolution) & (v >= 0) & (v < buffer_resolution)

    z_buffer = np.full((buffer_resolution, buffer_resolution), np.inf)
    np.minimum.at(z_buffer, (v[inside], u[inside]), surface_depth[inside])

    centers = voxel_centers(grid_resolution, mesh_scale)
    voxel_pixels, voxel_depth = project(centers, f_pixels, distance, buffer_resolution)
    vu = np.rint(voxel_pixels[:, 0]).astype(np.int64)
    vv = np.rint(voxel_pixels[:, 1]).astype(np.int64)
    in_view = (vu >= 0) & (vu < buffer_resolution) & (vv >= 0) & (vv < buffer_resolution)

    observed = np.full(centers.shape[0], np.inf)
    observed[in_view] = z_buffer[vv[in_view], vu[in_view]]
    known = np.isfinite(observed)

    margin = margin_voxels / (mesh_scale * (grid_resolution - 1))
    free = known & (voxel_depth < observed - margin)
    occupied = known & (np.abs(voxel_depth - observed) <= margin)

    index = np.argwhere(np.ones((grid_resolution,) * 3, dtype=bool))
    return {
        "free": index[free],
        "occupied": index[occupied],
        "scale_world_per_metre": scale,
        "center_depth_m": center_depth,
    }
