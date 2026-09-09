"""Report the enclosed mask opening and the viewpoint arc, to choose a clip."""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from derive_roi import flood_holes


def viewing_direction(label, instance_id):
    """Camera position in the object's own frame: -R.T @ t, from the annotation."""
    slot = list(label["instance_ids"]).index(instance_id)
    rotation = np.asarray(label["rotations"][slot], float)
    translation = np.asarray(label["translations"][slot], float)
    centre = -rotation.T @ translation
    return centre / np.linalg.norm(centre)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--arc-only", action="store_true",
                        help="Report only the viewpoint arc; needs labels/, not images")
    parser.add_argument("--all-instances", action="store_true",
                        help="With --arc-only, report every instance in the scene")
    args = parser.parse_args()
    if args.all_instances and not args.arc_only:
        raise ValueError("--all-instances only makes sense with --arc-only")
    if (args.instance_id is None) != bool(args.all_instances):
        raise ValueError("Pass exactly one of --instance-id or --all-instances")
    assert args.stride > 0 and args.top > 0
    if args.arc_only:
        # labels/ alone is enough, so a whole test split can be checked from a
        # small unzip rather than a full scene extraction.
        files = [p.with_name(p.name.replace("_label.pkl", ".png"))
                 for p in sorted((args.scene / "labels").glob("*_label.pkl"))]
    else:
        files = sorted((args.scene / "rgb").glob("*.png"))
    files = files[::args.stride]
    if not files:
        raise FileNotFoundError(f"No frames under {args.scene}")
    if args.all_instances:
        with (args.scene / "labels" / files[0].name.replace(".png", "_label.pkl")).open("rb") as handle:
            ids = list(pickle.load(handle)["instance_ids"])
        for instance_id in ids:
            print(f"instance {instance_id}: ", end="")
            report(args, files, instance_id)
        return
    report(args, files, args.instance_id)


def report(args, files, instance_id):
    rows, directions = [], []
    for path in files:
        with (args.scene / "labels" / f"{path.stem}_label.pkl").open("rb") as handle:
            label = pickle.load(handle)  # Trusted local HouseCat6D release only.
        if instance_id not in list(label["instance_ids"]):
            continue
        directions.append(viewing_direction(label, instance_id))
        if args.arc_only:
            continue
        instance = cv2.imread(str(args.scene / "instance" / path.name), cv2.IMREAD_UNCHANGED)
        if instance is None:
            raise FileNotFoundError(path)
        mask = (instance[..., 2] if instance.ndim == 3 else instance) == instance_id
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        extent = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        rows.append((int(flood_holes(mask).sum()), path.stem, int(mask.sum()), int(extent)))
    if not directions:
        print(f"instance {instance_id} appears in no scanned frame")
        return
    v = np.array(directions)
    elevation = np.degrees(np.arcsin(np.clip(v[:, 1], -1, 1)))
    azimuth = np.degrees(np.arctan2(v[:, 0], v[:, 2]))
    arc = np.degrees(np.arccos(np.clip(v @ v.T, -1, 1))).max()
    # The tracker adds a keyframe past 10 degrees of view change (main.py:115), so
    # the count of occupied 10-degree cells is what the schedule can actually reach.
    cells = len({(int(a // 10), int(e // 10)) for a, e in zip(azimuth, elevation)})
    model = "?"
    with (args.scene / "labels" / files[0].name.replace(".png", "_label.pkl")).open("rb") as handle:
        label = pickle.load(handle)
        if instance_id in list(label["instance_ids"]):
            model = str(label["model_list"][list(label["instance_ids"]).index(instance_id)])
    print(f"arc {arc:5.0f} deg (az {np.ptp(azimuth):3.0f}, el {np.ptp(elevation):3.0f}), "
          f"{cells:3d} ten-degree cells -> ~{cells} keyframes   {model}")
    if args.arc_only:
        return
    enclosing = [r for r in rows if r[0] > 0]
    print(f"scanned {len(rows)} frames at stride {args.stride}; "
          f"{len(enclosing)} enclose an opening")
    print("  hole_px  frame   object_px  extent_px  hole/object")
    for hole, stem, area, extent in sorted(enclosing, reverse=True)[:args.top]:
        print(f"  {hole:7d}  {stem}  {area:9d}  {extent:9d}  {hole / area:.4f}")


if __name__ == "__main__":
    main()
