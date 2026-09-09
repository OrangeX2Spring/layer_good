"""Report the enclosed mask opening per frame, to choose a reference frame."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from derive_roi import flood_holes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    assert args.stride > 0 and args.top > 0
    files = sorted((args.scene / "rgb").glob("*.png"))[::args.stride]
    rows = []
    for path in files:
        instance = cv2.imread(str(args.scene / "instance" / path.name), cv2.IMREAD_UNCHANGED)
        if instance is None:
            raise FileNotFoundError(path)
        mask = (instance[..., 2] if instance.ndim == 3 else instance) == args.instance_id
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        extent = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        rows.append((int(flood_holes(mask).sum()), path.stem, int(mask.sum()), int(extent)))
    if not rows:
        raise ValueError(f"Instance {args.instance_id} appears in no scanned frame")
    enclosing = [r for r in rows if r[0] > 0]
    print(f"scanned {len(rows)} frames at stride {args.stride}; "
          f"{len(enclosing)} enclose an opening")
    print("  hole_px  frame   object_px  extent_px  hole/object")
    for hole, stem, area, extent in sorted(enclosing, reverse=True)[:args.top]:
        print(f"  {hole:7d}  {stem}  {area:9d}  {extent:9d}  {hole / area:.4f}")


if __name__ == "__main__":
    main()
