"""Prepare matched ARCTIC inputs and export 4RC per-frame object geometry.

Run on CAMP only. Inference itself uses the unchanged 4RC inference.py CLI.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

SCENES = ("box_grab_01", "ketchup_grab_01", "espressomachine_grab_01")


def prepare(args, sampling="even"):
    assert sampling in ("even", "consecutive")
    args.out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.prepared) as source, tarfile.open(args.kvt) as baseline:
        manifest = json.load(source.extractfile("manifest.json"))
        kvt_manifest = json.load(baseline.extractfile("manifest.json"))
        assert manifest == kvt_manifest["prepare_manifest"], "Different source inputs"
        assert manifest["loader_offset"] == kvt_manifest["offset"] == 2
        assert kvt_manifest["resize_dim"] == 518
        for name in ("manifest.json", "metrics.json"):
            (args.out / f"kvt_{name}").write_bytes(baseline.extractfile(name).read())
        for scene in args.scenes:
            root = args.out / scene
            for folder in ("original", "masked", "masks", "source", "gt"):
                (root / folder).mkdir(parents=True)
            frames = manifest["scenes"][scene]["frames"][2:]
            assert 2 <= args.frames <= min(30, len(frames))
            mask_names = {m.name for m in baseline.getmembers()
                          if m.isfile() and m.name.startswith(f"{scene}/results/sam_masks/")}
            assert mask_names == {f"{scene}/results/sam_masks/{i:05d}.png"
                                  for i in range(len(frames))}
            indices = (np.linspace(0, len(frames) - 1, args.frames, dtype=int)
                       if sampling == "even" else np.arange(args.frames))
            selected = []
            for output_index, index in enumerate(indices):
                frame = frames[index]
                raw = source.extractfile(frame["path"]).read()
                mask_path = f"{scene}/results/sam_masks/{index:05d}.png"
                mask_bytes = baseline.extractfile(mask_path).read()
                rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                mask = np.asarray(Image.open(io.BytesIO(mask_bytes)).convert("L")) > 127
                assert rgb.shape[:2] == mask.shape and mask.any()
                name = f"{output_index:05d}"
                (root / "source" / f"{name}.jpg").write_bytes(raw)
                (root / "masks" / f"{name}.png").write_bytes(mask_bytes)
                Image.fromarray(rgb).save(root / "original" / f"{name}.png")
                Image.fromarray(rgb * mask[..., None]).save(root / "masked" / f"{name}.png")
                selected.append({**frame, "sample_index": output_index,
                                 "kvt_index": int(index), "mask_member": mask_path,
                                 "rgb_sha256": hashlib.sha256(raw).hexdigest(),
                                 "mask_sha256": hashlib.sha256(mask_bytes).hexdigest()})
            for suffix in ("object.npy", "egocam.dist.npy"):
                name = f"{scene}.{suffix}"
                (root / "gt" / name).write_bytes(
                    source.extractfile(f"data/raw_seqs/s01/{name}").read())
            record = {"scene": scene, "frames": selected,
                      "conditions": ["original", "masked"],
                      "sampling": ("evenly spaced over full post-offset sequence"
                                   if sampling == "even" else "first consecutive post-offset frames"),
                      "masking": "saved KV-Tracker SAM mask, before 4RC resize; no bbox crop",
                      "prepared_archive": str(args.prepared), "kvt_archive": str(args.kvt),
                      "inference": {"size": 512, "patch_size": 14,
                                    "track_query_idx": args.frames // 2,
                                    "dtype": "bf16-mixed", "use_center_as_anchor": False}}
            (root / "inputs.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"PREPARED {scene}: {len(selected)} matched frames", flush=True)
    print("PREPARE OK", flush=True)


def export(args):
    root = args.out / args.scene
    record = json.loads((root / "inputs.json").read_text())
    target = root / f"{args.condition}_clouds"
    target.mkdir(exist_ok=False)
    rows = []
    with np.load(root / f"{args.condition}.npz", allow_pickle=False) as predictions:
        assert int(predictions["n_frames"]) == len(record["frames"])
        for i, frame in enumerate(record["frames"]):
            points = predictions[f"pred_{i}_pts"][0]
            confidence = predictions[f"pred_{i}_conf"][0]
            model_rgb = predictions[f"view_{i}_img"][0]
            assert points.ndim == 3 and points.shape[-1] == 3
            assert confidence.shape == points.shape[:2]
            assert model_rgb.shape == (3, *points.shape[:2])
            for key in ("pts", "conf", "track", "conf_track", "extrinsic", "intrinsic"):
                assert np.isfinite(predictions[f"pred_{i}_{key}"]).all(), (i, key)
            colors = np.rint((model_rgb.transpose(1, 2, 0) + 1) * 127.5)
            colors = colors.clip(0, 255).astype(np.uint8)
            Image.fromarray(colors).save(target / f"{i:05d}_model_input.png")

            # Match load_images(size=512, patch_size=14): long-edge resize,
            # then central crop. Nearest-neighbour is used only for labels.
            mask_image = Image.open(root / "masks" / f"{i:05d}.png").convert("L")
            width, height = mask_image.size
            resized = tuple(int(round(x * 512 / max(width, height)))
                            for x in (width, height))
            cx, cy = resized[0] // 2, resized[1] // 2
            halfw = (2 * cx) // 14 * 14 // 2
            halfh = (2 * cy) // 14 * 14 // 2
            if resized[0] == resized[1]:
                halfh = 3 * halfw / 4
            crop = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
            mask_image = mask_image.resize(resized, Image.Resampling.NEAREST).crop(crop)
            mask = np.asarray(mask_image) > 127
            assert mask.shape == points.shape[:2] and mask.any()
            mask_image.save(target / f"{i:05d}_model_mask.png")

            # One time instant per PLY: never fuse moving-object world points.
            vertices = np.empty(int(mask.sum()), dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("confidence", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1")])
            for channel, name in enumerate(("x", "y", "z")):
                vertices[name] = points[..., channel][mask]
            vertices["confidence"] = confidence[mask]
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
                         "object_points": len(vertices),
                         "confidence_quantiles": np.quantile(confidence[mask],
                                                             [0, .1, .5, .9, 1]).tolist(),
                         "model_hw": list(mask.shape), "resize_wh": list(resized),
                         "crop_xyxy": list(crop)})
    summary = {"scene": args.scene, "condition": args.condition, "frames": rows,
               "geometry": "per-frame observed object surface in 4RC world coordinates",
               "units": "model units, no metric alignment",
               "filter": "SAM object mask only; no confidence filtering",
               "caveat": "Not fused geometry, ground-truth accuracy, or full-sequence ATE"}
    (root / f"{args.condition}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"EXPORT OK {args.scene} {args.condition}: {len(rows)} finite frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--prepared", type=Path, required=True)
    prep.add_argument("--kvt", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    prep.add_argument("--frames", type=int, default=30)
    prep.add_argument("--sampling", choices=("even", "consecutive"), default="even")
    exp = sub.add_parser("export")
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument("--scene", choices=SCENES, required=True)
    exp.add_argument("--condition", choices=("original", "masked"), required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args, sampling=args.sampling)
    else:
        export(args)
