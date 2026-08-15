#!/usr/bin/env python3
import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pix2gestalt on one object instance from a BOP-format YCB-V scene."
    )
    parser.add_argument("--ycbv-root", type=Path, required=True)
    parser.add_argument("--pix2gestalt-dir", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=54)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--gt-id", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pix2gestalt_ycbv"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--precision", choices=("autocast", "fp32"), default="autocast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-visible-fraction", type=float, default=0.25)
    parser.add_argument("--max-visible-fraction", type=float, default=0.75)
    parser.add_argument("--min-visible-pixels", type=int, default=1500)
    parser.add_argument("--crop-padding", type=float, default=0.2)
    return parser.parse_args()


def load_json(path):
    with path.open() as f:
        return json.load(f)


def choose_instance(scene_dir, args):
    gt = load_json(scene_dir / "scene_gt.json")
    gt_info = load_json(scene_dir / "scene_gt_info.json")

    if (args.image_id is None) != (args.gt_id is None):
        raise ValueError("--image-id and --gt-id must be supplied together")

    if args.image_id is not None:
        image_key = str(args.image_id)
        return args.image_id, args.gt_id, gt[image_key][args.gt_id], gt_info[image_key][args.gt_id]

    candidates = []
    for image_key, infos in gt_info.items():
        for gt_id, info in enumerate(infos):
            visible_fraction = info["visib_fract"]
            if not args.min_visible_fraction <= visible_fraction <= args.max_visible_fraction:
                continue
            if info["px_count_visib"] < args.min_visible_pixels:
                continue
            candidates.append(
                (
                    abs(visible_fraction - 0.5),
                    -info["px_count_visib"],
                    int(image_key),
                    gt_id,
                    gt[image_key][gt_id],
                    info,
                )
            )

    if not candidates:
        raise ValueError("No instance matches the visibility and size filters")

    _, _, image_id, gt_id, annotation, info = min(candidates)
    return image_id, gt_id, annotation, info


def square_crop_box(bbox, image_width, image_height, padding):
    x, y, width, height = bbox
    side = max(width, height) * (1 + 2 * padding)
    center_x = x + width / 2
    center_y = y + height / 2
    left = max(0, int(round(center_x - side / 2)))
    top = max(0, int(round(center_y - side / 2)))
    right = min(image_width, int(round(center_x + side / 2)))
    bottom = min(image_height, int(round(center_y + side / 2)))
    return left, top, right, bottom


def load_model_cpu_first(config_path, checkpoint, device):
    from ldm.util import instantiate_from_config

    print(f"Loading model from {checkpoint} on CPU")
    checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
    if "global_step" in checkpoint_data:
        print(f"Global Step: {checkpoint_data['global_step']}")
    state_dict = checkpoint_data["state_dict"]
    model = instantiate_from_config(OmegaConf.load(config_path).model)
    model.load_state_dict(state_dict, strict=False)
    del state_dict, checkpoint_data
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scene_dir = args.ycbv_root / "test" / f"{args.scene_id:06d}"
    image_id, gt_id, annotation, info = choose_instance(scene_dir, args)

    rgb_path = scene_dir / "rgb" / f"{image_id:06d}.png"
    visible_mask_path = scene_dir / "mask_visib" / f"{image_id:06d}_{gt_id:06d}.png"
    amodal_mask_path = scene_dir / "mask" / f"{image_id:06d}_{gt_id:06d}.png"

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    visible_mask = np.array(Image.open(visible_mask_path).convert("L"))
    amodal_mask = np.array(Image.open(amodal_mask_path).convert("L"))

    left, top, right, bottom = square_crop_box(
        info["bbox_obj"], rgb.shape[1], rgb.shape[0], args.crop_padding
    )
    rgb_crop = rgb[top:bottom, left:right]
    visible_crop = visible_mask[top:bottom, left:right]
    amodal_crop = amodal_mask[top:bottom, left:right]

    rgb_256 = cv2.resize(rgb_crop, (256, 256), interpolation=cv2.INTER_AREA)
    visible_256 = cv2.resize(visible_crop, (256, 256), interpolation=cv2.INTER_NEAREST)
    amodal_256 = cv2.resize(amodal_crop, (256, 256), interpolation=cv2.INTER_NEAREST)
    visible_rgb = np.repeat(visible_256[:, :, None], 3, axis=2)

    output_dir = args.output_dir / f"scene_{args.scene_id:06d}_image_{image_id:06d}_gt_{gt_id:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_256).save(output_dir / "input_crop.png")
    Image.fromarray(visible_256).save(output_dir / "visible_mask.png")
    Image.fromarray(amodal_256).save(output_dir / "gt_amodal_mask.png")
    Image.fromarray(np.dstack((rgb_256, visible_256))).save(output_dir / "visible_object.png")

    sys.path.insert(0, str(args.pix2gestalt_dir.resolve()))
    from inference import run_pix2gestalt

    config_path = args.pix2gestalt_dir / "configs" / "sd-finetune-pix2gestalt-c_concat-256.yaml"
    checkpoint = args.checkpoint or args.pix2gestalt_dir / "ckpt" / "epoch=000005.ckpt"
    model = load_model_cpu_first(config_path, checkpoint, args.device)
    predictions = run_pix2gestalt(
        model,
        args.device,
        rgb_256,
        visible_rgb,
        scale=args.guidance_scale,
        n_samples=args.n_samples,
        ddim_steps=args.ddim_steps,
        precision=args.precision,
    )

    for index, prediction in enumerate(predictions):
        Image.fromarray(prediction).save(output_dir / f"completion_{index:02d}.png")

    metadata = {
        "scene_id": args.scene_id,
        "image_id": image_id,
        "gt_id": gt_id,
        "obj_id": annotation["obj_id"],
        "visib_fract": info["visib_fract"],
        "bbox_obj": info["bbox_obj"],
        "crop_box": [left, top, right, bottom],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output_dir)


if __name__ == "__main__":
    main()
