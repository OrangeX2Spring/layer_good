"""Shared BOP/YCB-V IO and crop-camera geometry for the occlusion experiment.

The central piece is `rectify_homography`. Pixal3D assumes a centered pinhole
camera whose FOV it either estimates with MoGe-2 or receives via `--fov`. A
plain array slice around an off-center object violates both assumptions: the
object's principal point is not at the crop center, and MoGe-2 sees a tight
object crop that is far outside its training distribution. Rectifying with a
virtual camera that looks straight down the object's center ray fixes the first,
and the exact crop FOV (returned here) fixes the second.
"""

import json

import numpy as np

# BOP stores depth maps as uint16; `depth_scale` converts them to millimetres.
MM_TO_M = 1e-3


def load_json(path):
    with open(path) as f:
        return json.load(f)


def scene_camera(scene_dir, image_id):
    """Return (K, depth_scale) for one image of a BOP scene."""
    entry = load_json(scene_dir / "scene_camera.json")[str(image_id)]
    K = np.array(entry["cam_K"], dtype=np.float64).reshape(3, 3)
    return K, float(entry["depth_scale"])


def object_pose(scene_dir, image_id, gt_id):
    """Return (R_m2c, t_m2c in metres, obj_id) for one annotated instance."""
    entry = load_json(scene_dir / "scene_gt.json")[str(image_id)][gt_id]
    R = np.array(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t = np.array(entry["cam_t_m2c"], dtype=np.float64) * MM_TO_M
    return R, t, int(entry["obj_id"])


def square_crop(bbox, padding):
    """Padded square crop as (center_u, center_v, side) in pixels.

    Deliberately unclamped: the rectifying warp samples outside the image
    border on its own, so clamping to the image rectangle would only shift the
    crop centre off the object and reintroduce the skew we are removing.
    """
    x, y, width, height = bbox
    side = max(width, height) * (1.0 + 2.0 * padding)
    return x + width / 2.0, y + height / 2.0, side


def look_at_rotation(K, center_u, center_v):
    """Rotation taking the ray through (center_u, center_v) onto the +Z axis."""
    d = np.linalg.inv(K) @ np.array([center_u, center_v, 1.0])
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(d, z)
    sin_angle = np.linalg.norm(axis)
    if sin_angle < 1e-9:
        return np.eye(3)
    axis = axis / sin_angle
    angle = np.arctan2(sin_angle, float(d @ z))
    kx, ky, kz = axis
    skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rectify_homography(K, center_u, center_v, side):
    """Homography mapping the source image into a centered virtual camera.

    Returns (H, K_new, R, fov_x). Applying `H` with an output size of
    (side, side) yields a crop in which the object centre lies on the optical
    axis and the horizontal FOV is exactly `fov_x` radians, independent of any
    later resize.
    """
    assert K.shape == (3, 3), K.shape
    f = 0.5 * (K[0, 0] + K[1, 1])
    R = look_at_rotation(K, center_u, center_v)
    K_new = np.array([[f, 0.0, side / 2.0], [0.0, f, side / 2.0], [0.0, 0.0, 1.0]])
    H = K_new @ R @ np.linalg.inv(K)
    fov_x = 2.0 * np.arctan(side / (2.0 * f))
    return H, K_new, R, float(fov_x)


def crop_to_image_transform(crop_box, crop_resolution):
    """Homography from a resized axis-aligned crop back to source-image pixels.

    Lets a pix2gestalt completion, produced in the old 256x256 crop frame, be
    warped straight into the rectified frame without a resampling round-trip.
    """
    left, top, right, bottom = crop_box
    return np.array([
        [(right - left) / crop_resolution, 0.0, left],
        [0.0, (bottom - top) / crop_resolution, top],
        [0.0, 0.0, 1.0],
    ])


def backproject(depth_png, depth_scale, K, mask):
    """Back-project masked depth pixels to a point cloud in the camera frame.

    Returns an (N, 3) array in metres. Depth sensors are least reliable at
    occlusion boundaries, so callers should erode `mask` before trusting the
    result as a geometric constraint.
    """
    assert depth_png.shape == mask.shape, (depth_png.shape, mask.shape)
    v, u = np.nonzero(mask & (depth_png > 0))
    z = depth_png[v, u].astype(np.float64) * depth_scale * MM_TO_M
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)
