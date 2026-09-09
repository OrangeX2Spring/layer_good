"""Prepare an RGB-only KV-Tracker manifest and separate depth reference on Linux."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kv_tracker"))
from kv_tracker.image import pi3_resize_image
from kvt_artifacts import write_ply


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, required=True, help="Exclusive sorted RGB index")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resize-dim", type=int, default=308)
    parser.add_argument("--depth-units-per-metre", type=float, required=True)
    parser.add_argument("--crop-margin", type=float,
                        help="Crop a square window this many times the largest "
                             "object extent, centred on the object each frame")
    args = parser.parse_args()
    assert args.stride > 0 and 0 <= args.start < args.stop
    assert args.depth_units_per_metre > 0 and args.resize_dim >= 14
    files = sorted((args.scene / "rgb").glob("*.png"))
    if args.stop > len(files):
        raise ValueError(f"stop={args.stop}, but only {len(files)} RGB frames")
    files = files[args.start:args.stop:args.stride]
    if len(files) < 2:
        raise ValueError("Select at least two frames")
    args.out.mkdir(parents=True, exist_ok=False)
    side, windows = None, None
    if args.crop_margin is not None:
        assert args.crop_margin >= 1.0
        boxes = []
        for path in files:
            instance = cv2.imread(str(args.scene / "instance" / path.name), cv2.IMREAD_UNCHANGED)
            if instance is None:
                raise FileNotFoundError(path)
            m = (instance[..., 2] if instance.ndim == 3 else instance) == args.instance_id
            if not m.any():
                raise ValueError(f"Empty instance mask at {path}")
            ys, xs = np.nonzero(m)
            boxes.append((ys.min(), ys.max(), xs.min(), xs.max()))
        boxes = np.array(boxes)
        extent = int(max((boxes[:, 1] - boxes[:, 0]).max(), (boxes[:, 3] - boxes[:, 2]).max()) + 1)
        height, width = instance.shape[:2]
        # One window size for the whole clip: the focal length stays fixed and only
        # the principal point moves, so per-frame zoom cannot confound the geometry.
        side = int(min(round(extent * args.crop_margin), height, width))
        y0 = np.clip((boxes[:, 0] + boxes[:, 1]) // 2 - side // 2, 0, height - side)
        x0 = np.clip((boxes[:, 2] + boxes[:, 3]) // 2 - side // 2, 0, width - side)
        assert ((boxes[:, 0] >= y0) & (boxes[:, 1] < y0 + side)
                & (boxes[:, 2] >= x0) & (boxes[:, 3] < x0 + side)).all(), \
            f"crop margin {args.crop_margin} cannot contain a {extent} px object in every frame"
        windows = list(zip(y0.tolist(), x0.tolist()))
        (args.out / "frames").mkdir()
        print(f"CROP {side}x{side} px window; largest object extent {extent} px")
    frames, thumbnails = [], []
    for index, path in enumerate(files):
        with (args.scene / "labels" / f"{path.stem}_label.pkl").open("rb") as handle:
            label = pickle.load(handle)  # Trusted local HouseCat6D release only.
        slot = list(label["instance_ids"]).index(args.instance_id)
        model = str(label["model_list"][slot])
        if index == 0:
            model_name = model
        if model != model_name:
            raise ValueError(f"Instance identity changed at {path}")
        mask_path = args.scene / "instance" / path.name
        if windows is not None:
            y0, x0 = windows[index]
            full_rgb = cv2.imread(str(path))
            full_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if full_rgb is None or full_mask is None:
                raise FileNotFoundError(path)
            cropped_rgb = args.out / "frames" / f"{path.stem}_rgb.png"
            cropped_mask = args.out / "frames" / f"{path.stem}_instance.png"
            assert cv2.imwrite(str(cropped_rgb), full_rgb[y0:y0 + side, x0:x0 + side])
            assert cv2.imwrite(str(cropped_mask), full_mask[y0:y0 + side, x0:x0 + side])
            path, mask_path = cropped_rgb, cropped_mask
        frames.append({"id": path.stem, "rgb": str(path.resolve()),
                       "instance": str(mask_path.resolve())})
        if index in np.linspace(0, len(files) - 1, min(12, len(files)), dtype=int):
            rgb = cv2.imread(str(path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or mask is None:
                raise FileNotFoundError(path)
            # Match the released HouseCat6D loader: OpenCV channel 2.
            mask = (mask[..., 2] if mask.ndim == 3 else mask) == args.instance_id
            overlay = rgb.copy()
            overlay[mask] = (overlay[mask] * 0.6 + np.array([0, 100, 0])).astype(np.uint8)
            tile = cv2.resize(overlay, (320, 240))
            cv2.putText(tile, path.stem, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
            thumbnails.append(tile)
    sheet = np.zeros((240 * ((len(thumbnails) + 3) // 4), 1280, 3), np.uint8)
    for i, tile in enumerate(thumbnails):
        sheet[i // 4 * 240:(i // 4 + 1) * 240, i % 4 * 320:(i % 4 + 1) * 320] = tile
    assert cv2.imwrite(str(args.out / "contact_sheet.jpg"), sheet)
    rgb = cv2.imread(frames[0]["rgb"])
    instance = cv2.imread(frames[0]["instance"], cv2.IMREAD_UNCHANGED)
    if rgb is None or instance is None:
        raise FileNotFoundError(files[0])
    mask = (instance[..., 2] if instance.ndim == 3 else instance) == args.instance_id
    if not mask.any():
        raise ValueError("Initial instance mask is empty")
    assert cv2.imwrite(str(args.out / "initial_mask.png"), mask.astype(np.uint8) * 255)
    resized = pi3_resize_image(rgb, (args.resize_dim, args.resize_dim))
    h, w = resized.shape[:2]
    depth_path = args.scene / "depth_gt" / files[0].name
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if depth.ndim == 3:
        depth = depth[..., 1].astype(np.uint16) * 256 + depth[..., 2].astype(np.uint16)
        depth[depth == 32001] = 0
    assert depth.ndim == 2 and depth.dtype == np.uint16
    intrinsics = np.loadtxt(args.scene / "intrinsics.txt")
    if windows is not None:
        # Crop at source resolution, before the resize, so the window coordinates
        # and the principal-point shift are in the same pixel units.
        y0, x0 = windows[0]
        depth = depth[y0:y0 + side, x0:x0 + side]
        intrinsics = intrinsics.copy()
        intrinsics[0, 2] -= x0
        intrinsics[1, 2] -= y0
    assert depth.shape == rgb.shape[:2]
    depth = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    depth /= args.depth_units_per_metre
    # OpenCV resize pixel-centre convention; same grid as the point maps.
    yy, xx = np.indices((h, w))
    u = (xx + 0.5) * rgb.shape[1] / w - 0.5
    v = (yy + 0.5) * rgb.shape[0] / h - 0.5
    xyz = np.stack(((u - intrinsics[0, 2]) * depth / intrinsics[0, 0],
                    (v - intrinsics[1, 2]) * depth / intrinsics[1, 1], depth), axis=-1)
    reference_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    np.savez_compressed(args.out / "reference.npz", xyz=xyz, valid=depth > 0,
                        object_mask=reference_mask, rgb=resized[..., ::-1])
    reference_keep = reference_mask & (depth > 0)
    write_ply(args.out / "reference_object.ply", xyz[reference_keep], resized[..., ::-1][reference_keep])
    assert cv2.imwrite(str(args.out / "reference_rgb.png"), resized)
    manifest = {"scene": str(args.scene.resolve()), "model": model_name,
                "instance_id": args.instance_id, "resize_dim": args.resize_dim,
                "initial_mask": str((args.out / "initial_mask.png").resolve()),
                "depth_units_per_metre": args.depth_units_per_metre,
                "crop_side_px": side,
                "frames": frames}
    (args.out / "input.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PREPARED {len(frames)} frames of {model_name}; inspect contact_sheet.jpg")


if __name__ == "__main__":
    main()
