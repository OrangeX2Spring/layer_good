"""Attribute ROI points back to the view and pixel that produced them."""
import argparse
import json
from pathlib import Path

import numpy as np


def erode(mask):
    out = mask.copy()
    out[1:] &= mask[:-1]; out[:-1] &= mask[1:]
    out[:, 1:] &= mask[:, :-1]; out[:, :-1] &= mask[:, 1:]
    return out


def dilate(mask):
    out = mask.copy()
    out[1:] |= mask[:-1]; out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]; out[:, :-1] |= mask[:, 1:]
    return out


def depth_from_edge(mask, limit=40):
    """Erosions a pixel survives: its distance to the nearest non-mask pixel."""
    distance = np.zeros(mask.shape, int)
    current = mask.copy()
    for step in range(1, limit + 1):
        current = erode(current)
        if not current.any():
            break
        distance[current] = step
    return distance


def distance_to(seed, limit=60):
    distance = np.full(seed.shape, limit + 1, int)
    distance[seed] = 0
    current = seed.copy()
    for step in range(1, limit + 1):
        grown = dilate(current)
        distance[grown & ~current] = step
        current = grown
    return distance


def flood_holes(mask):
    comp = ~mask
    out = np.zeros_like(comp)
    out[0] |= comp[0]; out[-1] |= comp[-1]
    out[:, 0] |= comp[:, 0]; out[:, -1] |= comp[:, -1]
    while True:
        grown = dilate(out) & comp
        if np.array_equal(grown, out):
            return comp & ~out
        out = grown


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()

    kf = np.load(args.keyframes)
    al = np.load(args.alignment)
    cfg = json.loads(args.evaluation.read_text())
    roi = np.asarray(cfg["roi_to_reference"], float)
    extent = np.asarray(cfg["roi_extent_m"], float)
    inverse = np.linalg.inv(roi)

    xyz, masks, conf = kf["xyz"], kf["masks"], kf["confidence"]
    threshold = float(kf["threshold"])
    ids, first = np.unique(kf["frame_ids"], return_index=True)
    order = np.argsort(first)
    keep_views = first[order]
    xyz, masks, conf = xyz[keep_views], masks[keep_views], conf[keep_views]

    aligned = float(al["scale"]) * xyz @ np.asarray(al["rotation"]).T + np.asarray(al["translation"])
    local = aligned @ inverse[:3, :3].T + inverse[:3, 3]
    retained = masks & (conf > threshold)
    inside = (np.abs(local) < extent / 2).all(-1) & retained
    print(f"{int(retained.sum())} retained points, {int(inside.sum())} in the ROI, "
          f"across {len(xyz)} views")

    per_view = inside.reshape(len(xyz), -1).sum(1)
    print("per view:", per_view.tolist())
    print(f"views contributing: {(per_view > 0).sum()} of {len(xyz)}")

    edge, hole_d, conf_in, conf_all = [], [], [], []
    for v in range(len(xyz)):
        if not per_view[v]:
            continue
        depth = depth_from_edge(masks[v])
        holes = flood_holes(masks[v])
        to_hole = distance_to(holes) if holes.any() else np.full(masks[v].shape, 999)
        sel = inside[v]
        edge.append(depth[sel])
        hole_d.append(to_hole[sel])
        conf_in.append(conf[v][sel])
        conf_all.append(conf[v][retained[v]])
    edge = np.concatenate(edge); hole_d = np.concatenate(hole_d)
    conf_in = np.concatenate(conf_in); conf_all = np.concatenate(conf_all)

    print("\nsource pixels, distance INTO the mask from its silhouette edge (px):")
    for p in (10, 25, 50, 75, 90):
        print(f"  p{p:<3}: {np.percentile(edge, p):6.1f}")
    print(f"  within 3 px of the edge: {(edge <= 3).mean():.1%}")
    print("\ndistance from the source pixel to the enclosed opening (px):")
    for p in (10, 25, 50, 75, 90):
        print(f"  p{p:<3}: {np.percentile(hole_d, p):6.1f}")
    print(f"  within 3 px of the opening: {(hole_d <= 3).mean():.1%}")
    print(f"\nconfidence: ROI points median {np.median(conf_in):.3f}, "
          f"all retained median {np.median(conf_all):.3f}, threshold {threshold:.3f}")


if __name__ == "__main__":
    main()
