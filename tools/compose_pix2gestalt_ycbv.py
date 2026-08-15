#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build high-resolution RGBA Pixal3D inputs from existing Pix2Gestalt "
            "completions and YCB-V ground-truth masks."
        )
    )
    parser.add_argument("--ycbv-root", type=Path, required=True)
    parser.add_argument("--completion-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = json.loads((args.completion_dir / "metadata.json").read_text())

    scene_dir = args.ycbv_root / "test" / f"{metadata['scene_id']:06d}"
    image_id = metadata["image_id"]
    gt_id = metadata["gt_id"]
    left, top, right, bottom = metadata["crop_box"]

    rgb = np.array(
        Image.open(scene_dir / "rgb" / f"{image_id:06d}.png").convert("RGB")
    )[top:bottom, left:right]
    visible_mask = np.array(
        Image.open(
            scene_dir / "mask_visib" / f"{image_id:06d}_{gt_id:06d}.png"
        ).convert("L")
    )[top:bottom, left:right]
    amodal_mask = np.array(
        Image.open(scene_dir / "mask" / f"{image_id:06d}_{gt_id:06d}.png").convert("L")
    )[top:bottom, left:right]

    crop_height, crop_width = rgb.shape[:2]
    visible = visible_mask > 127
    amodal = amodal_mask > 127
    completion_paths = sorted(args.completion_dir.glob("completion_[0-9][0-9].png"))
    if not completion_paths:
        raise FileNotFoundError(f"No completion_XX.png files in {args.completion_dir}")

    for completion_path in completion_paths:
        completion = np.array(Image.open(completion_path).convert("RGB"))
        completion = cv2.resize(
            completion, (crop_width, crop_height), interpolation=cv2.INTER_LANCZOS4
        )

        composite = completion.copy()
        composite[visible] = rgb[visible]

        composite = cv2.resize(
            composite, (args.size, args.size), interpolation=cv2.INTER_LANCZOS4
        )
        alpha = cv2.resize(
            amodal.astype(np.uint8) * 255,
            (args.size, args.size),
            interpolation=cv2.INTER_NEAREST,
        )
        composite[alpha == 0] = 0

        rgba = np.dstack((composite, alpha))
        index = completion_path.stem.rsplit("_", 1)[1]
        output_path = args.completion_dir / f"pixal3d_input_{index}_{args.size}.png"
        Image.fromarray(rgba).save(output_path)
        print(output_path)


if __name__ == "__main__":
    main()
