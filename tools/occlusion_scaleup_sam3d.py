#!/usr/bin/env python3
"""Run SAM 3D on the prepared visible/completed YCB-V inputs."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def serialise(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "sam3d" / "notebook"))
    from inference import Inference

    pipeline = Inference(str(args.config), compile=False)
    with args.cases.open(newline="") as handle:
        cases = list(csv.DictReader(handle, delimiter="\t"))

    for case in cases:
        for condition in ("completed", "visible"):
            input_path = args.input_root / case["slug"] / condition / "input.png"
            rgba = np.asarray(Image.open(input_path).convert("RGBA"), dtype=np.uint8)
            mask = rgba[..., 3] > 127
            assert mask.any(), input_path

            output_dir = args.output_root / case["slug"] / condition
            output_dir.mkdir(parents=True, exist_ok=True)

            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = pipeline(rgba[..., :3], mask, seed=args.seed)
            elapsed = time.perf_counter() - start

            output["gs"].save_ply(str(output_dir / "splat.ply"))
            assert output["glb"] is not None
            output["glb"].export(str(output_dir / "model.glb"))

            metadata = {
                "scene_id": int(case["scene_id"]),
                "image_id": int(case["image_id"]),
                "gt_id": int(case["gt_id"]),
                "obj_id": int(case["obj_id"]),
                "condition": condition,
                "seed": args.seed,
                "input": str(input_path),
                "elapsed_seconds": elapsed,
                "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
            for key in ("rotation", "translation", "scale", "iou", "iou_before_optim", "optim_accepted"):
                if key in output:
                    metadata[key] = serialise(output[key])
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

            assert (output_dir / "splat.ply").stat().st_size > 0
            assert (output_dir / "model.glb").stat().st_size > 0
            print(f"{case['slug']} {condition}: {elapsed:.1f}s, {metadata['peak_cuda_gb']:.2f} GiB")

    print("SAM3D RUN OK")


if __name__ == "__main__":
    main()
