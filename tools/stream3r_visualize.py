#!/usr/bin/env python3
"""Run STream3R over a folder of frames and write a coloured point cloud per mode.

    python tools/stream3r_visualize.py \
        --images /tmp/stream3r/nrgbd/<scene>/images \
        --ckpt   /mnt/projects/gr/3DRecon/stream3r_ckpt \
        --out    /tmp/stream3r/out/<name> \
        --modes causal full --max-frames 16

This is a first look, not an evaluation: it answers "does the reconstruction
look like the room" and nothing else. No metric is computed and none should be
quoted from it.

Two deliberate choices.

The confidence cut is a PERCENTILE, not a constant. STream3R's DPT head uses the
`expp1` activation, so `world_points_conf` is exp(x)+1 - it lives in [1, inf),
not [0, 1], and a threshold carried over from a sigmoid-based model would be
meaningless here. The KV-Tracker cavity investigation spent a week on phantom
geometry that turned out to be an uncalibrated retention threshold
(tools/FINDINGS.md 12), so the distribution is printed every run and the cut is
stated in the manifest rather than buried in a default.

Both modes are run on identical frames. `causal` is what STream3R is for;
`full` is the bidirectional VGGT-style pass over the same input. The difference
between the two .ply files is the only thing here that bears on the OPT-Pose
frozen-cache line, where the same question was asked of a model that was never
trained to answer it.
"""
import argparse
import json
import os
import shutil
import numpy as np
import torch

from stream3r.models.stream3r import STream3R
from stream3r.models.components.utils.load_fn import load_and_preprocess_images

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".PNG", ".JPG")


def write_ply(path, xyz, rgb):
    """Binary little-endian PLY. Hand-rolled so this does not depend on plyfile
    or open3d being present in whichever image we are borrowing."""
    assert xyz.ndim == 2 and xyz.shape[1] == 3, xyz.shape
    assert rgb.shape == xyz.shape, (rgb.shape, xyz.shape)
    assert xyz.dtype == np.float32, xyz.dtype
    assert rgb.dtype == np.uint8, rgb.dtype

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    rec = np.empty(
        len(xyz),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory of input frames")
    ap.add_argument("--ckpt", required=True,
                    help="local checkpoint dir, or a hub id like yslan/STream3R")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", nargs="+", default=["causal", "full"])
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--conf-percentile", type=float, default=50.0,
                    help="drop points below this percentile of world_points_conf")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} cuda {torch.version.cuda} device {device}", flush=True)

    names = sorted(
        os.path.join(args.images, f)
        for f in os.listdir(args.images)
        if f.endswith(IMAGE_EXT)
    )
    assert names, f"no images under {args.images}"
    # Evenly spaced rather than the first N: a contiguous head of a video clip is
    # a few centimetres of baseline. The bottle-v8_small rerun (CLAUDE.md) was
    # voided for exactly this - three "reference" frames 0.36 deg apart.
    if len(names) > args.max_frames:
        idx = np.linspace(0, len(names) - 1, args.max_frames).round().astype(int)
        names = [names[i] for i in idx]
    print(f"{len(names)} frames from {args.images}", flush=True)

    # Stage the exact frames next to the outputs, here rather than in the caller:
    # this is the code that selected them, and it keeps the caller's only
    # post-inference action a single `tar`. Job 25559 lost its point clouds to a
    # relative path in an external staging step that ran with the job's cwd
    # (the submit directory) instead of the repo's.
    frames_dir = os.path.join(args.out, "input_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for p in names:
        shutil.copy2(p, os.path.join(frames_dir, os.path.basename(p)))
    print(f"staged {len(names)} input frames to {frames_dir}", flush=True)

    images = load_and_preprocess_images(names).to(device)
    assert images.ndim == 4 and images.shape[1] == 3, images.shape
    print(f"input tensor {tuple(images.shape)} dtype {images.dtype}", flush=True)

    model = STream3R.from_pretrained(args.ckpt).to(device)
    model.eval()

    manifest = {
        "images": names,
        "num_frames": len(names),
        "input_shape": list(images.shape),
        "ckpt": args.ckpt,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "conf_percentile": args.conf_percentile,
        "modes": {},
    }

    for mode in args.modes:
        print(f"--- mode={mode} ---", flush=True)
        with torch.no_grad():
            pred = model(images, mode=mode)

        xyz = pred["world_points"][0].reshape(-1, 3).float().cpu().numpy()
        conf = pred["world_points_conf"][0].reshape(-1).float().cpu().numpy()
        rgb = pred["images"][0].permute(0, 2, 3, 1).reshape(-1, 3).float().cpu().numpy()
        assert xyz.shape == rgb.shape, (xyz.shape, rgb.shape)
        assert conf.shape[0] == xyz.shape[0], (conf.shape, xyz.shape)

        # expp1 output: >= 1 by construction. Printed so the cut can be judged.
        pct = [float(np.percentile(conf, p)) for p in (1, 25, 50, 75, 99)]
        print(f"conf min {conf.min():.4f} max {conf.max():.4f} "
              f"pct1/25/50/75/99 {['%.4f' % v for v in pct]}", flush=True)

        thresh = float(np.percentile(conf, args.conf_percentile))
        keep = conf >= thresh
        print(f"{int(keep.sum())} of {len(keep)} points kept at conf >= {thresh:.4f}",
              flush=True)

        ply = os.path.join(args.out, f"{mode}.ply")
        write_ply(ply,
                  np.ascontiguousarray(xyz[keep], dtype=np.float32),
                  np.ascontiguousarray(
                      (rgb[keep].clip(0, 1) * 255).round(), dtype=np.uint8))
        print(f"wrote {ply}", flush=True)

        manifest["modes"][mode] = {
            "points_total": int(len(keep)),
            "points_kept": int(keep.sum()),
            "conf_threshold": thresh,
            "conf_min": float(conf.min()),
            "conf_max": float(conf.max()),
            "conf_percentiles": {str(p): v for p, v in zip((1, 25, 50, 75, 99), pct)},
            "ply": os.path.basename(ply),
        }

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("VISUALIZE OK", flush=True)


if __name__ == "__main__":
    main()
