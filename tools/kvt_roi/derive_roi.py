"""Derive the handle ROI, alignment anchors and evaluation configs from a reference."""
import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np

# The box scales with the opening, and which fraction is usable depends on what
# sits behind it: on the teapot 0.65 lands 0.3 mm from a real surface while 0.55
# clears comfortably. Search downward and keep the two largest that clear the
# stated margin, rather than fixing fractions that happened to suit the cup.
NAMES = ("evaluation.json", "evaluation_tight.json")
FRACTIONS = (0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40)


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
    parser.add_argument("--min-clearance-m", type=float, default=0.002)
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
    # Orient the box along the line of sight through the opening. An axis-aligned
    # prism does not follow that ray for an off-axis hole: at greater depth the
    # same metric x,y maps to pixels drifting inward, onto the object itself.
    forward = centre / np.linalg.norm(centre)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(forward, right), forward))
    assert np.allclose(rotation.T @ rotation, np.eye(3)) and np.linalg.det(rotation) > 0
    local_hole = (hole_xyz - centre) @ rotation
    footprint = float(min(np.ptp(local_hole[:, 0]), np.ptp(local_hole[:, 1])))

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
    roi_to_reference[:3, :3] = rotation
    roi_to_reference[:3, 3] = centre
    cloud = (xyz[known] - centre) @ rotation
    surface = xyz[mask & valid]
    span = max(np.ptp(surface[:, 0]), np.ptp(surface[:, 1])) * 1.6 / 2
    view_centre = surface[:, :2].mean(0)
    print(f"hole {int(hole.sum())} px, ring depth {plane_depth:.4f} m, "
          f"anchors {int(anchor.sum())} px, object depth valid "
          f"{int((mask & valid).sum())}/{int(mask.sum())}")
    print(f"roi centre {centre.round(4)}")
    print(f"hole footprint {footprint * 100:.1f} cm across at the handle plane")
    viable = []
    for fraction in FRACTIONS:
        side = round(footprint * fraction / args.voxel_size_m) * args.voxel_size_m
        if side < 2 * args.voxel_size_m:
            continue
        # The free space through a handle is a tube, not a cube: how deep it runs
        # is set by whatever the sensor sees in front of and behind the opening.
        column = cloud[(np.abs(cloud[:, :2]) < side / 2).all(1), 2]
        front, behind = column[column < 0], column[column > 0]
        limits = [side / 2]
        if front.size:
            limits.append(-front.max() - args.voxel_size_m)
        if behind.size:
            limits.append(behind.min() - args.voxel_size_m)
        depth = np.floor(2 * min(limits) / args.voxel_size_m) * args.voxel_size_m
        if depth < 2 * args.voxel_size_m:
            continue
        extent = np.array([side, side, depth])
        inside = (np.abs(cloud) < extent / 2).all(1)
        if inside.any():
            continue
        gap = np.maximum(np.abs(cloud[~inside]) - extent / 2, 0)
        clearance = float(np.linalg.norm(gap, axis=1).min())
        if clearance < args.min_clearance_m:
            continue
        if viable and np.array_equal(viable[-1][0], extent):
            continue  # neighbouring fractions round onto the same voxel grid
        viable.append((extent, clearance))
    if len(viable) < 2:
        raise ValueError(f"Only {len(viable)} box sizes clear "
                         f"{args.min_clearance_m * 1000:.1f} mm of real geometry; "
                         f"this opening cannot be measured from this frame")
    for name, (extent, clearance) in zip(NAMES, viable):
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
