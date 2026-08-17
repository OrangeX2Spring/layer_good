#!/usr/bin/env python3
"""Build a rectified RGBA Pixal3D input from a YCB-V instance, with exact FOV.

Supersedes compose_pix2gestalt_ycbv.py: same compositing of pix2gestalt
completions under the real visible pixels, but the crop is a rectifying warp
rather than an array slice, and the exact crop FOV is written to meta.json so
inference can be run with `--fov` instead of MoGe-2's estimate.

Three modes give the input axis of the experiment:
  completed  RGB where visible, pix2gestalt completion elsewhere, amodal alpha
  visible    RGB, visible-mask alpha (no completion at all -- the lower bound)
  control    RGB, amodal alpha, intended for an unoccluded view of the same
             object, where amodal == visible (the upper bound: a real photo
             with nothing to complete)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ycbv_bop import (
    backproject,
    crop_to_image_transform,
    load_json,
    object_pose,
    rectify_homography,
    scene_camera,
    square_crop,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ycbv-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--gt-id", type=int, required=True)
    parser.add_argument("--mode", choices=("completed", "visible", "control"), required=True)
    parser.add_argument("--completion-dir", type=Path,
                        help="Output dir of run_pix2gestalt_ycbv.py (mode=completed)")
    parser.add_argument("--completion-index", type=int, default=0)
    parser.add_argument("--padding", type=float, default=0.2,
                        help="Must match the padding used for the pix2gestalt crop")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--depth-erosion", type=int, default=3,
                        help="Mask erosion in pixels before back-projecting depth")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.mode == "completed") != (args.completion_dir is not None):
        raise ValueError("--completion-dir is required for and only for mode=completed")

    scene_dir = args.ycbv_root / args.split / f"{args.scene_id:06d}"
    stem = f"{args.image_id:06d}"
    inst = f"{args.image_id:06d}_{args.gt_id:06d}"

    K, depth_scale = scene_camera(scene_dir, args.image_id)
    R_m2c, t_m2c, obj_id = object_pose(scene_dir, args.image_id, args.gt_id)
    info = load_json(scene_dir / "scene_gt_info.json")[str(args.image_id)][args.gt_id]

    rgb = np.array(Image.open(scene_dir / "rgb" / f"{stem}.png").convert("RGB"))
    visible = np.array(Image.open(scene_dir / "mask_visib" / f"{inst}.png").convert("L")) > 127
    amodal = np.array(Image.open(scene_dir / "mask" / f"{inst}.png").convert("L")) > 127
    depth = np.array(Image.open(scene_dir / "depth" / f"{stem}.png"))

    center_u, center_v, side = square_crop(info["bbox_obj"], args.padding)
    side = int(round(side))
    H, K_new, R_rect, fov_x = rectify_homography(K, center_u, center_v, side)

    def warp(image, interpolation):
        return cv2.warpPerspective(image, H, (side, side), flags=interpolation)

    rgb_rect = warp(rgb, cv2.INTER_LANCZOS4)
    visible_rect = warp(visible.astype(np.uint8), cv2.INTER_NEAREST) > 0
    amodal_rect = warp(amodal.astype(np.uint8), cv2.INTER_NEAREST) > 0

    if args.mode == "completed":
        metadata = json.loads((args.completion_dir / "metadata.json").read_text())
        assert metadata["image_id"] == args.image_id, metadata
        assert metadata["gt_id"] == args.gt_id, metadata
        completion_path = args.completion_dir / f"completion_{args.completion_index:02d}.png"
        completion = np.array(Image.open(completion_path).convert("RGB"))
        # completion lives in the old 256x256 slice frame; carry it through that
        # frame's mapping back to source pixels before applying the rectification.
        H_completion = H @ crop_to_image_transform(metadata["crop_box"], completion.shape[0])
        color = cv2.warpPerspective(completion, H_completion, (side, side), flags=cv2.INTER_LANCZOS4)
        color[visible_rect] = rgb_rect[visible_rect]
        alpha = amodal_rect
    elif args.mode == "visible":
        color, alpha = rgb_rect, visible_rect
    else:
        color, alpha = rgb_rect, amodal_rect

    color = cv2.resize(color, (args.size, args.size), interpolation=cv2.INTER_LANCZOS4)
    alpha_resized = cv2.resize(
        alpha.astype(np.uint8) * 255, (args.size, args.size), interpolation=cv2.INTER_NEAREST
    )
    color[alpha_resized == 0] = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.dstack((color, alpha_resized))).save(args.output_dir / "input.png")

    kernel = np.ones((2 * args.depth_erosion + 1,) * 2, np.uint8)
    trusted = cv2.erode(visible.astype(np.uint8), kernel) > 0
    points_cam = backproject(depth, depth_scale, K, trusted)
    # Two frames the evaluator needs: the rectified camera Pixal3D implicitly
    # renders from, and the object's own frame, where the GT mesh lives.
    np.save(args.output_dir / "visible_points_rect.npy", points_cam @ R_rect.T)
    np.save(args.output_dir / "visible_points_obj.npy", (points_cam - t_m2c) @ R_m2c)

    meta = {
        "scene_id": args.scene_id,
        "image_id": args.image_id,
        "gt_id": args.gt_id,
        "obj_id": obj_id,
        "mode": args.mode,
        "visib_fract": info["visib_fract"],
        "fov_x_rad": fov_x,
        "fov_x_deg": float(np.degrees(fov_x)),
        "crop_center_uv": [center_u, center_v],
        "crop_side_px": side,
        "size": args.size,
        "K": K.tolist(),
        "K_rect": K_new.tolist(),
        "R_rect": R_rect.tolist(),
        "cam_R_m2c": R_m2c.tolist(),
        "cam_t_m2c_m": t_m2c.tolist(),
        "num_visible_points": int(points_cam.shape[0]),
    }
    if args.mode == "completed":
        meta["completion"] = str(completion_path)
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"{args.output_dir}  fov={meta['fov_x_deg']:.2f}deg  points={meta['num_visible_points']}")


if __name__ == "__main__":
    main()
