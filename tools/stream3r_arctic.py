"""Run matched ARCTIC frames through STream3R on CAMP; export each instant separately."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from stream3r.models.stream3r import STream3R
from stream3r.models.components.utils.load_fn import load_and_preprocess_images
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from fourrc_arctic import SCENES, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--prepared", type=Path, required=True)
    prep.add_argument("--kvt", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    prep.add_argument("--sampling", choices=("even", "consecutive"), required=True)
    prep.set_defaults(frames=30)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--scene", choices=SCENES, required=True)
    run.add_argument("--condition", choices=("original", "masked"), required=True)
    run.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        # Reuse the exact 4RC frame selection, source/mask checks, and GT staging.
        prepare(args, sampling=args.sampling)
        for scene in args.scenes:
            path = args.out / scene / "inputs.json"
            record = json.loads(path.read_text())
            record["masking"] = "saved KV-Tracker SAM mask, before STream3R resize; no bbox crop"
            record["inference"] = {"model": "STream3R", "mode": "causal",
                                   "execution": "batched", "preprocess": "crop",
                                   "width": 518, "patch_size": 14, "dtype": "float32"}
            path.write_text(json.dumps(record, indent=2) + "\n")
        print("STREAM3R PREPARE OK", flush=True)
        return
    assert torch.cuda.is_available(), "Run on an allocated GPU"
    root = args.out / args.scene
    record = json.loads((root / "inputs.json").read_text())
    count = len(record["frames"])
    paths = [root / args.condition / f"{i:05d}.png" for i in range(count)]
    sizes = {Image.open(path).size for path in paths}
    assert len(sizes) == 1, "This pilot expects equal-sized source frames"
    images = load_and_preprocess_images(paths, mode="crop").cuda()
    assert images.ndim == 4 and images.shape[:2] == (count, 3)
    model = STream3R.from_pretrained(str(args.ckpt)).cuda().eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    # Match the previously verified STream3R batched causal inference path.
    with torch.no_grad():
        pred = model(images, mode="causal")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    h, w = images.shape[-2:]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pred["pose_enc"], (h, w))
    arrays = {key: pred[key][0].float().cpu().numpy() for key in
              ("world_points", "world_points_conf", "depth", "depth_conf", "pose_enc")}
    arrays["extrinsics"] = extrinsics[0].float().cpu().numpy()
    arrays["intrinsics"] = intrinsics[0].float().cpu().numpy()
    for key, value in arrays.items():
        assert value.shape[0] == count and np.isfinite(value).all(), key
    assert arrays["world_points"].shape == (count, h, w, 3)
    assert arrays["world_points_conf"].shape == (count, h, w)
    np.savez_compressed(root / f"{args.condition}.npz", **arrays)
    target = root / f"{args.condition}_clouds"
    target.mkdir(exist_ok=False)
    rgb = images.permute(0, 2, 3, 1).cpu().numpy()
    rows = []
    for i, frame in enumerate(record["frames"]):
        colors = np.rint(rgb[i].clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(colors).save(target / f"{i:05d}_model_input.png")
        mask_image = Image.open(root / "masks" / f"{i:05d}.png").convert("L")
        assert mask_image.size == next(iter(sizes))
        width, height = mask_image.size
        resized = (518, round(height * (518 / width) / 14) * 14)
        top = max(0, (resized[1] - 518) // 2)
        crop = (0, top, 518, top + min(resized[1], 518))
        mask_image = mask_image.resize(resized, Image.Resampling.NEAREST).crop(crop)
        mask = np.asarray(mask_image) > 127
        assert mask.shape == (h, w) and mask.any()
        mask_image.save(target / f"{i:05d}_model_mask.png")
        points = arrays["world_points"][i][mask]
        confidence = arrays["world_points_conf"][i][mask]
        vertices = np.empty(len(points), dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("confidence", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        for channel, name in enumerate(("x", "y", "z")):
            vertices[name] = points[:, channel]
        vertices["confidence"] = confidence
        for channel, name in enumerate(("red", "green", "blue")):
            vertices[name] = colors[..., channel][mask]
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(vertices)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property float confidence\nproperty uchar red\n"
                  "property uchar green\nproperty uchar blue\nend_header\n")
        with (target / f"{i:05d}_object.ply").open("wb") as stream:
            stream.write(header.encode("ascii"))
            stream.write(vertices.tobytes())
        rows.append({"sample_index": i, "image_id": frame["image_id"],
                     "object_points": len(points), "resize_wh": list(resized),
                     "crop_xyxy": list(crop), "model_hw": [h, w],
                     "confidence_quantiles": np.quantile(confidence, [0, .1, .5, .9, 1]).tolist()})
    summary = {
        "scene": args.scene, "condition": args.condition, "frames": rows,
        "mode": "causal", "execution": "batched, no StreamSession cache",
        "checkpoint": str(args.ckpt), "input_shape": list(images.shape),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "parameter_dtype": str(next(model.parameters()).dtype),
        "inference_seconds": seconds, "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "timing_scope": "one synchronized forward, no warmup; excludes load and export",
        "geometry": "per-frame observed object surface in STream3R world coordinates",
        "units": "model units, no metric alignment",
        "filter": "SAM object mask only; no confidence filtering",
        "caveat": "Not fused geometry, ground-truth accuracy, object pose, or streaming FPS"}
    (root / f"{args.condition}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"EXPORT OK {args.scene} {args.condition}: {count} finite frames; "
          f"{seconds:.3f} s; allocated {allocated / 2**30:.3f} GiB; "
          f"reserved {reserved / 2**30:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
