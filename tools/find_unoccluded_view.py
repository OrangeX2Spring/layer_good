#!/usr/bin/env python3
"""Find unoccluded views of an object, to serve as the experiment's control.

The same physical object appears in many YCB-V frames, some of them barely
occluded. Running Pixal3D on such a frame gives a control input that is a real
photograph rather than a render, so the control and the occluded condition
differ only in occlusion -- no domain gap is introduced by the comparison.
"""

import argparse
from pathlib import Path

from ycbv_bop import load_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ycbv-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--obj-id", type=int, required=True)
    parser.add_argument("--min-visib-fract", type=float, default=0.98)
    parser.add_argument("--min-visible-pixels", type=int, default=1500)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    split_dir = args.ycbv_root / args.split

    candidates = []
    for scene_dir in sorted(split_dir.iterdir()):
        if not (scene_dir / "scene_gt.json").exists():
            continue
        gt = load_json(scene_dir / "scene_gt.json")
        gt_info = load_json(scene_dir / "scene_gt_info.json")
        for image_key, annotations in gt.items():
            for gt_id, annotation in enumerate(annotations):
                if annotation["obj_id"] != args.obj_id:
                    continue
                info = gt_info[image_key][gt_id]
                if info["visib_fract"] < args.min_visib_fract:
                    continue
                if info["px_count_visib"] < args.min_visible_pixels:
                    continue
                candidates.append((
                    -info["px_count_visib"], int(scene_dir.name), int(image_key), gt_id,
                    info["visib_fract"],
                ))

    if not candidates:
        raise SystemExit(f"No view of obj {args.obj_id} met the visibility filters")

    print(f"{'scene':>6} {'image':>6} {'gt':>3} {'visib':>6} {'px':>8}")
    for neg_px, scene_id, image_id, gt_id, visib in sorted(candidates)[: args.top_k]:
        print(f"{scene_id:6d} {image_id:6d} {gt_id:3d} {visib:6.3f} {-neg_px:8d}")


if __name__ == "__main__":
    main()
