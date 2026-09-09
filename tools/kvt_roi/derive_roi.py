"""Derive the handle ROI, alignment anchors and evaluation configs from a reference."""
import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np

# Fractions of the measured opening, not fixed sizes: the box has to scale with
# the hole or it measures a patch of a large one. 0.65/0.55 reproduce the
# 1.4 cm and 1.2 cm boxes hand-chosen for the cup's 2.1 cm opening.
BOXES = (("evaluation.json", 0.65), ("evaluation_tight.json", 0.55))


def flood_holes(mask):
    """Complement pixels unreachable from the border: the enclosed handle opening."""
    comp = ~mask
    out = np.zeros_like(comp)
    out[0] |= comp[0]; out[-1] |= comp[-1]
    out[:, 0] |= comp[:, 0]; out[:, -1] |= comp[:, -1]
    while True:
        grown = out.copy()
        grown[1:] |= out[:-1]; grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]; grown[:, :-1] |= out[:, 1:]
        grown &= comp
        if np.array_equal(grown, out):
            return comp & ~out
        out = grown


def dilate(mask, steps=1):
    for _ in range(steps):
        grown = mask.copy()
        grown[1:] |= mask[:-1]; grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]; grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask


def write_png(path, image):
    assert image.dtype == np.uint8 and image.ndim == 2
    raw = b"".join(b"\x00" + image[i].tobytes() for i in range(image.shape[0]))

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", image.shape[1], image.shape[0], 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.002)
    parser.add_argument("--max-alignment-rmse-m", type=float, default=0.005)
    parser.add_argument("--anchor-clear-px", type=int, default=12)
    args = parser.parse_args()
    reference = np.load(args.reference)
    mask, xyz, valid = reference["object_mask"], reference["xyz"], reference["valid"]
    depth = xyz[..., 2]

    hole = flood_holes(mask)
    if not hole.any():
        raise ValueError("The object mask encloses no opening in this reference frame")
    ring = dilate(hole, 2) & mask & valid
    plane_depth = float(np.median(depth[ring]))

    # Per-pixel rays are affine in (u, v); fit them where depth exists so the
    # empty hole pixels, which carry no depth return, can still be unprojected.
    known = depth > 0
    rays = np.zeros_like(xyz)
    rays[known] = xyz[known] / depth[known][:, None]
    vy, vx = np.nonzero(known)
    coefficients, *_ = np.linalg.lstsq(np.c_[vx, vy, np.ones(vx.size)], rays[known][:, :2], rcond=None)
    hy, hx = np.nonzero(hole)
    hole_xyz = np.c_[np.c_[hx, hy, np.ones(hx.size)] @ coefficients, np.ones(hx.size)] * plane_depth
    centre = hole_xyz.mean(0)
    footprint = float(min(np.ptp(hole_xyz[:, 0]), np.ptp(hole_xyz[:, 1])))

    # Reject silhouette edges and depth steps: anchors must be reliable surface.
    padded = np.where(known, depth, np.nan)
    neighbourhood = np.stack([np.roll(np.roll(padded, i, 0), j, 1)
                              for i in (-1, 0, 1) for j in (-1, 0, 1)])
    spread = np.nanmax(neighbourhood, 0) - np.nanmin(neighbourhood, 0)
    anchor = (~dilate(~(mask & valid), 3)) & (np.nan_to_num(spread, nan=1.0) < 0.003) \
        & ~dilate(hole, args.anchor_clear_px)
    if anchor.sum() < 20:
        raise ValueError(f"Only {anchor.sum()} anchor pixels survived")

    args.out.mkdir(parents=True, exist_ok=True)
    image = np.zeros(mask.shape, np.uint8)
    image[anchor] = 255
    write_png(args.out / "anchor_mask.png", image)

    roi_to_reference = np.eye(4)
    roi_to_reference[:3, 3] = centre
    cloud = xyz[known]
    surface = xyz[mask & valid]
    span = max(np.ptp(surface[:, 0]), np.ptp(surface[:, 1])) * 1.6 / 2
    view_centre = surface[:, :2].mean(0)
    print(f"hole {int(hole.sum())} px, ring depth {plane_depth:.4f} m, "
          f"anchors {int(anchor.sum())} px, object depth valid "
          f"{int((mask & valid).sum())}/{int(mask.sum())}")
    print(f"roi centre {centre.round(4)}")
    print(f"hole footprint {footprint * 100:.1f} cm across at the handle plane")
    for name, fraction in BOXES:
        side = round(footprint * fraction / args.voxel_size_m) * args.voxel_size_m
        if side < 2 * args.voxel_size_m:
            raise ValueError(f"{name}: opening spans only {side / args.voxel_size_m:.1f} voxels")
        extent = np.full(3, side)
        low, high = centre - extent / 2, centre + extent / 2
        inside = ((cloud >= low) & (cloud <= high)).all(1)
        gap = np.maximum(np.maximum(low - cloud[~inside], cloud[~inside] - high), 0)
        clearance = float(np.linalg.norm(gap, axis=1).min())
        if inside.any():
            raise ValueError(f"{name}: {int(inside.sum())} GT points inside the supposedly empty box")
        label = "x".join(f"{v * 100:g}" for v in extent.round(4))
        evidence = (
            f"Reference {args.reference.parent.name}. The annotation mask encloses a "
            f"{int(hole.sum())} px hole at the handle and the table is visible through the "
            f"loop in the reference RGB. GT depth is dense on the object "
            f"({int((mask & valid).sum())}/{int(mask.sum())} pixels valid) and returns nothing "
            f"inside the hole. The {label} cm box contains no GT point; the nearest valid GT "
            f"point is {clearance * 1000:.1f} mm away. The scanned mesh was not consulted.")
        cfg = {"reference": args.reference.name, "anchor_mask": "anchor_mask.png",
               "roi_to_reference": roi_to_reference.tolist(),
               "roi_extent_m": extent.tolist(), "voxel_size_m": args.voxel_size_m,
               "max_alignment_rmse_m": args.max_alignment_rmse_m,
               "empty_region_evidence": evidence,
               "render_reference_to_view": np.eye(4).tolist(),
               "render_bounds_m": [float(view_centre[0] - span), float(view_centre[0] + span),
                                   float(view_centre[1] - span), float(view_centre[1] + span)]}
        (args.out / name).write_text(json.dumps(cfg, indent=2) + "\n")
        voxels = int(np.prod(np.ceil(extent / args.voxel_size_m)))
        print(f"  {name}: {label} cm, {voxels} voxels, clearance {clearance * 1000:.1f} mm")


if __name__ == "__main__":
    main()
