"""Evaluate captured keyframe maps; export PLYs, playback arrays, and an MP4."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def align(source, target):
    """Least-squares Sim(3), target = scale * source @ rotation.T + translation."""
    assert source.shape == target.shape and source.ndim == 2 and source.shape[1] == 3
    x, y = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError("Alignment anchors must not be collinear")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = (singular * sign).sum() / np.mean(np.sum(x * x, axis=1))
    translation = target.mean(0) - scale * source.mean(0) @ rotation.T
    residual = np.sqrt(np.mean(np.sum((scale * source @ rotation.T + translation - target) ** 2, axis=1)))
    return scale, rotation, translation, residual


def write_ply(path, xyz, rgb):
    assert xyz.shape == rgb.shape and xyz.shape[1] == 3
    vertices = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, i]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())
    if path.stat().st_size != len(header.encode("ascii")) + vertices.nbytes:
        raise IOError(f"Incomplete PLY: {path}")


def project_cloud(xyz, rgb, view, bounds, size=640):
    points = xyz @ view[:3, :3].T + view[:3, 3]
    xmin, xmax, ymin, ymax = bounds
    uv = np.column_stack(((points[:, 0] - xmin) / (xmax - xmin),
                          (points[:, 1] - ymin) / (ymax - ymin)))
    valid = np.isfinite(points).all(1) & (uv >= 0).all(1) & (uv < 1).all(1)
    uv = (uv[valid] * size).astype(int)
    # Orthographic z-buffer: one pixel per point; no splat size change over time.
    order = np.argsort(points[valid, 2], kind="stable")
    pixels = uv[order, 1] * size + uv[order, 0]
    _, first = np.unique(pixels, return_index=True)
    image = np.full((size * size, 3), 24, np.uint8)
    image[pixels[first]] = rgb[valid][order[first]][:, ::-1]
    return image.reshape(size, size, 3)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.capture / "capture_complete.json").is_file():
        raise ValueError("Capture did not finish successfully")
    cfg = json.loads(args.evaluation.read_text())
    base = args.evaluation.resolve().parent
    with np.load(base / cfg["reference"]) as reference:
        target = reference["xyz"]
        valid = reference["valid"] & reference["object_mask"]
        reference_rgb = reference["rgb"]
    anchor_image = cv2.imread(str(base / cfg["anchor_mask"]), cv2.IMREAD_GRAYSCALE)
    if anchor_image is None or anchor_image.shape != valid.shape:
        raise ValueError("anchor_mask must match the reference point-map grid")
    anchor = (anchor_image > 0) & valid
    if anchor.sum() < 20:
        raise ValueError("Select at least 20 reliable surface pixels for alignment")
    roi_to_reference = np.asarray(cfg["roi_to_reference"], dtype=float)
    extent = np.asarray(cfg["roi_extent_m"], dtype=float)
    voxel = float(cfg["voxel_size_m"])
    assert roi_to_reference.shape == (4, 4) and extent.shape == (3,)
    assert np.all(extent > 0) and voxel > 0 and cfg["max_alignment_rmse_m"] > 0
    assert np.allclose(roi_to_reference[3], [0, 0, 0, 1])
    assert np.allclose(roi_to_reference[:3, :3].T @ roi_to_reference[:3, :3], np.eye(3))
    reference_to_roi = np.linalg.inv(roi_to_reference)
    anchor_roi = target[anchor] @ reference_to_roi[:3, :3].T + reference_to_roi[:3, 3]
    if np.any((np.abs(anchor_roi) < extent / 2).all(1)):
        raise ValueError("Alignment anchors overlap the supposedly empty region")
    if not cfg["empty_region_evidence"].strip():
        raise ValueError("Record how reference geometry and RGB verify this region is empty")
    grid_shape = np.ceil(extent / voxel).astype(int)
    total_voxels = int(np.prod(grid_shape))
    view = np.asarray(cfg["render_reference_to_view"], dtype=float)
    bounds = cfg["render_bounds_m"]
    assert view.shape == (4, 4) and len(bounds) == 4
    assert bounds[1] > bounds[0] and bounds[3] > bounds[2]
    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5)
                        for z in (-0.5, 0.5)]) * extent
    corners = corners @ roi_to_reference[:3, :3].T + roi_to_reference[:3, 3]
    projected = corners @ view[:3, :3].T + view[:3, 3]
    corner_pixels = np.column_stack(((projected[:, 0] - bounds[0]) / (bounds[1] - bounds[0]),
                                     (projected[:, 1] - bounds[2]) / (bounds[3] - bounds[2]))) * 640
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("snapshots", "final_views", "panels"):
        (args.out / name).mkdir()
    (args.out / "evaluation.json").write_text(json.dumps(cfg, indent=2) + "\n")
    write_ply(args.out / "reference_object.ply", target[valid], reference_rgb[valid])
    files = sorted((args.capture / "keyframes").glob("*.npz"))
    if not files:
        raise ValueError("No keyframe snapshots")
    metrics, playback = [], []
    previous_xyz, previous_ids, previous_masks = None, None, None
    threshold = None
    for step, path in enumerate(files):
        with np.load(path) as data:
            ids, first = np.unique(data["frame_ids"], return_index=True)
            # Remove the duplicated bootstrap view, retaining acquisition order.
            order = np.argsort(first)
            first, ids = first[order], ids[order]
            xyz, rgb = data["xyz"][first], data["rgb"][first]
            masks, confidence = data["masks"][first], data["confidence"][first]
            if threshold is None:
                threshold = float(data["threshold"])
            if threshold != float(data["threshold"]):
                raise ValueError("Confidence threshold changed across snapshots")
            if not np.isfinite(xyz).all() or not np.isfinite(confidence).all():
                raise ValueError(f"Non-finite predictions in {path}")
            scale, rotation, translation, residual = align(xyz[0][anchor], target[anchor])
            xyz = scale * xyz @ rotation.T + translation
            keep = masks & (confidence > threshold)
            roi = xyz @ reference_to_roi[:3, :3].T + reference_to_roi[:3, 3]
            inside = (np.abs(roi) < extent / 2).all(-1) & keep
            cells = np.floor((roi[inside] + extent / 2) / voxel).astype(int)
            occupied = len(np.unique(cells, axis=0))
            old = np.isin(ids, previous_ids) if previous_ids is not None else np.zeros(len(ids), bool)
            row = {"snapshot": step, "source_frame": int(ids[-1]), "keyframes": len(ids),
                   "retained_points": int(keep.sum()), "roi_points": int(inside.sum()),
                   "roi_fraction": float(inside.sum() / keep.sum()) if keep.any() else None,
                   "occupied_voxels": occupied, "total_roi_voxels": total_voxels,
                   "roi_occupancy": occupied / total_voxels,
                   "roi_confidence_mean": float(confidence[inside].mean()) if inside.any() else None,
                   "roi_points_old_views": int(inside[old].sum()),
                   "roi_points_new_views": int(inside[~old].sum()),
                   "alignment_rmse_m": float(residual), "alignment_valid": bool(residual <= cfg["max_alignment_rmse_m"]),
                   "old_view_displacement_rmse_m": None, "confidence_threshold": threshold}
            if previous_ids is not None:
                assert np.array_equal(ids[old], previous_ids)
                common = masks[old] & previous_masks
                if common.any():
                    row["old_view_displacement_rmse_m"] = float(np.sqrt(np.mean(np.sum(
                        (xyz[old][common] - previous_xyz[common]) ** 2, axis=-1))))
            previous_xyz, previous_ids, previous_masks = xyz, ids, masks
            metrics.append(row)
            cloud_path = args.out / "snapshots" / f"{step:06d}"
            write_ply(cloud_path.with_suffix(".ply"), xyz[keep], rgb[keep])
            np.savez_compressed(cloud_path.with_suffix(".npz"), xyz=xyz[keep], rgb=rgb[keep])
            np.savez_compressed(cloud_path.with_name(cloud_path.name + "_alignment.npz"),
                                scale=scale, rotation=rotation, translation=translation,
                                anchor_mask=anchor, residual_m=residual)
            playback.append({"file": str(cloud_path.with_suffix(".npz").relative_to(args.out)),
                             "source_frame": int(ids[-1]), "keyframes": len(ids)})
            image = project_cloud(xyz[keep], rgb[keep], view, bounds)
            for i in range(8):
                for bit in (1, 2, 4):
                    j = i ^ bit
                    if i < j:
                        cv2.line(image, tuple(corner_pixels[i].astype(int)),
                                 tuple(corner_pixels[j].astype(int)), (0, 255, 255), 1)
            latest = cv2.resize(data["latest_rgb"][..., ::-1], (640, 640))
            panel = np.zeros((800, 1280, 3), np.uint8)
            panel[:640, :640], panel[:640, 640:] = latest, image
            title = f"Actual history | frame {ids[-1]} | {len(ids)} distinct keyframes | ROI {occupied / total_voxels:.4f}"
            cv2.putText(panel, title, (12, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            status = f"Alignment RMSE {residual:.5f} m | valid={row['alignment_valid']} | fixed confidence > {threshold:.4f}"
            cv2.putText(panel, status, (12, 707), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # Fixed axes, occupancy [0, 1], snapshot [0, last]. Invalid samples red.
            for j, measurement in enumerate(metrics):
                px = 20 + int(1200 * j / max(1, len(files) - 1))
                py = 785 - int(60 * measurement["roi_occupancy"])
                color = (0, 220, 0) if measurement["alignment_valid"] else (0, 0, 255)
                cv2.circle(panel, (px, py), 3, color, -1)
            assert cv2.imwrite(str(args.out / "panels" / f"{step:06d}.png"), panel)
            if step == len(files) - 1:
                final_views = []
                for i, frame_id in enumerate(ids):
                    stem = args.out / "final_views" / f"{i:06d}"
                    write_ply(stem.with_suffix(".ply"), xyz[i][keep[i]], rgb[i][keep[i]])
                    np.savez_compressed(stem.with_suffix(".npz"), xyz=xyz[i][keep[i]], rgb=rgb[i][keep[i]])
                    final_views.append({"file": str(stem.with_suffix(".npz").relative_to(args.out)),
                                        "source_frame": int(frame_id)})
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (args.out / "playback.json").write_text(json.dumps({"actual": playback, "reveal": final_views,
        "reveal_note": "Final reconstruction revealed per view; NOT actual reconstruction history"}, indent=2) + "\n")
    writer = cv2.VideoWriter(str(args.out / "actual_history.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (1280, 800))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 encoder")
    for path in sorted((args.out / "panels").glob("*.png")):
        panel = cv2.imread(str(path))
        for _ in range(12):
            writer.write(panel)
    # Finish with a frozen-final-cloud orbit, explicitly separate from time evolution.
    center = target[valid].mean(0) @ view[:3, :3].T + view[:3, 3]
    for angle in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        c, s = np.cos(angle), np.sin(angle)
        orbit = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        orbit_view = view.copy()
        orbit_view[:3, :3] = orbit @ view[:3, :3]
        orbit_view[:3, 3] = orbit @ (view[:3, 3] - center) + center
        panel[:640, 640:] = project_cloud(xyz[keep], rgb[keep], orbit_view, bounds)
        panel[640:] = 0
        cv2.putText(panel, "Final reconstruction orbit (frozen geometry, not additional views)",
                    (12, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        writer.write(panel)
    writer.release()
    video = cv2.VideoCapture(str(args.out / "actual_history.mp4"))
    decoded = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        assert frame.shape == (800, 1280, 3)
        decoded += 1
    video.release()
    if decoded != len(files) * 12 + 72:
        raise RuntimeError(f"Video verification failed: decoded {decoded} frames")
    print(f"ARTIFACTS OK: {len(metrics)} snapshots, {decoded} video frames; "
          f"{sum(row['alignment_valid'] for row in metrics)} valid alignments")


if __name__ == "__main__":
    main()
