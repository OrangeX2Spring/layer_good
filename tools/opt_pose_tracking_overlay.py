#!/usr/bin/env python3
"""Render tracked pose as a 3D box locked to the object; no model or GPU needed.

The diagnostic report shows depth maps and latency curves. This renders the
figure a pose result is normally judged by: the object's 3D bounding box drawn
into every frame using each method's tracked camera pose. If the tracking is
right the boxes stay on the object; if it drifts they slide off, which is
visible without reading a number.

Geometry uses numpy only so it can be checked off-cluster; matplotlib and cv2
are used for drawing alone.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [("original", "#1f77b4", "Original bidirectional"),
           ("cached", "#2ca02c", "Frozen cache")]
EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def project_to_so3(r):
    u, _, vt = np.linalg.svd(r)
    sign = np.ones(r.shape[:-2] + (3,))
    sign[..., -1] = np.linalg.det(u @ vt)
    return (u * sign[..., None, :]) @ vt


def quat_xyzw_to_matrix(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def similarity(source, target):
    """Least-squares Sim(3) mapping source points onto target points."""
    assert source.shape == target.shape and source.ndim == 2 and source.shape[1] == 3
    x, y = source - source.mean(0), target - target.mean(0)
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    assert singular[1] > singular[0] * 1e-6, f"Degenerate fit: {singular}"
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = float(singular @ sign / np.mean(np.sum(x * x, axis=1)))
    assert scale > 0
    return scale, rotation, target.mean(0) - scale * rotation @ source.mean(0)


def orientation_gauge(predicted, annotated):
    """Sim(3) whose rotation is fitted to the ORIENTATIONS, then scale and shift to
    the centres with that rotation held fixed.

    Fitting the rotation on camera centres instead is what the accuracy report was
    doing wrong: on a thin trajectory that rotation is nearly free, and here it comes
    out ~180 degrees off, which puts the projected box behind the camera. See
    tools/FINDINGS.md, "revised Step 1b" section 5.
    """
    assert predicted["rotation"].shape == annotated["rotation"].shape
    rotation = project_to_so3(np.einsum("nij,nkj->ik", annotated["rotation"], predicted["rotation"]))
    rotated = predicted["center"] @ rotation.T
    x, y = rotated - rotated.mean(0), annotated["center"] - annotated["center"].mean(0)
    scale = float((x * y).sum() / (x * x).sum())
    assert scale > 0, f"Degenerate scale: {scale}"
    return scale, rotation, annotated["center"].mean(0) - scale * rotated.mean(0)


def frame_geometry(path):
    """Per-frame intrinsics, annotated extrinsic, and the canonical->camera fit.

    NOCS is dense inside the mask but the depth sensor is not: on this clip it
    returns 1-29% of masked pixels. The fit uses the overlap, then the FULL mask's
    NOCS extent gives the whole object rather than only the part depth saw.
    """
    data = np.load(path)
    intrinsics = data["intrinsics"][0, 0].astype(np.float64)
    mask = data["source_inst_masks"][0] > 0
    depth = data["source_depths"][0].astype(np.float64)
    nocs = data["nocs_gt"][0, 0].astype(np.float64)
    selected = mask & (depth > 0)
    assert selected.sum() >= 100, f"Too few depth samples on the object: {selected.sum()}"
    rows, columns = np.nonzero(selected)
    z = depth[selected]
    camera_points = np.stack([(columns - intrinsics[0, 2]) / intrinsics[0, 0] * z,
                              (rows - intrinsics[1, 2]) / intrinsics[1, 1] * z, z], 1)
    scale, rotation, translation = similarity(nocs[selected], camera_points)
    residual = np.linalg.norm(nocs[selected] @ (scale * rotation).T + translation - camera_points, axis=1)
    extrinsic = data["source_extrinsics"][0].astype(np.float64)
    # Match the accuracy report's convention: the stored extrinsic is world->camera,
    # so the camera's orientation in the world is its transpose and the centre is
    # -orientation @ translation. fp16 rotations are projected back to SO(3).
    orientation = np.swapaxes(project_to_so3(extrinsic[:3, :3]), -1, -2)
    return {
        "image": data["source_images"][0],
        "intrinsics": intrinsics,
        "mask_bbox": (columns.min(), columns.max(), rows.min(), rows.max()),
        "canonical_to_camera": (scale, rotation, translation),
        "fit_residual_m": float(np.median(residual)),
        "annotated_rotation": orientation,
        "annotated_center": -(orientation @ extrinsic[:3, 3]),
        "nocs_in_mask": nocs[mask],
    }


def project(points_camera, intrinsics):
    assert points_camera.ndim == 2 and points_camera.shape[1] == 3
    z = np.maximum(points_camera[:, 2], 1e-6)
    return np.stack([points_camera[:, 0] / z * intrinsics[0, 0] + intrinsics[0, 2],
                     points_camera[:, 1] / z * intrinsics[1, 1] + intrinsics[1, 2]], 1)


def build(run):
    manifest = json.loads((run / "manifest.json").read_text())
    sequence = manifest["sequences"][0]
    entries = sequence["frames"][1:]
    assert len(entries) >= 3
    frames = [frame_geometry(run / e["input"]) for e in entries]

    # One canonical box for the whole clip, from the union of the masked NOCS.
    stacked = np.concatenate([f["nocs_in_mask"] for f in frames])
    low, high = np.percentile(stacked, 1, axis=0), np.percentile(stacked, 99, axis=0)
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])

    # The object is static, so the box has one pose in the annotated world frame.
    # Averaging across frames only makes sense if they already agree; assert that.
    world = []
    for f in frames:
        scale, rotation, translation = f["canonical_to_camera"]
        camera = corners @ (scale * rotation).T + translation
        world.append(camera @ f["annotated_rotation"].T + f["annotated_center"])
    world = np.stack(world)
    spread = float(np.abs(world - world.mean(0)).max())
    assert spread < 0.05, f"Object box is not static in the annotated frame: {spread:.4f} m"
    box_world = world.mean(0)

    centers = np.stack([f["annotated_center"] for f in frames])
    orientations = np.stack([f["annotated_rotation"] for f in frames])
    predictions = {}
    for method, _, _ in METHODS:
        encodings = np.stack([
            np.load(run / method / sequence["directory"] / f"query{i:05d}.npz")["pose_enc"][0, 0].astype(np.float64)
            for i in range(len(entries))])
        assert np.isfinite(encodings).all()
        rotation = np.swapaxes(quat_xyzw_to_matrix(encodings[:, 3:7]), -1, -2)
        center = -(rotation @ encodings[:, :3, None])[..., 0]
        scale, align, shift = orientation_gauge(
            {"rotation": rotation, "center": center},
            {"rotation": orientations, "center": centers})
        predictions[method] = {"rotation": rotation, "center": center,
                               "aligned_center": scale * center @ align.T + shift,
                               "to_prediction_frame": (scale, align, shift)}
    return manifest, sequence, entries, frames, box_world, centers, predictions


def box_pixels(frame, box_world, method_state, index):
    """Project the annotated-frame box through one method's tracked camera pose."""
    scale, align, shift = method_state["to_prediction_frame"]
    # Overall scale cancels in the projection, so no metric claim is made here.
    local = (box_world - shift) @ align / scale
    camera = (local - method_state["center"][index]) @ method_state["rotation"][index]
    assert camera[:, 2].min() > 0, f"Box is behind the camera: z {camera[:, 2].min():.4f}"
    return project(camera, frame["intrinsics"]), camera[:, 2].min()


def annotated_box_pixels(frame, box_world):
    camera = (box_world - frame["annotated_center"]) @ frame["annotated_rotation"]
    return project(camera, frame["intrinsics"]), camera[:, 2].min()


def draw_box(axis, pixels, color, label=None, width=1.8):
    for a, b in EDGES:
        axis.plot(*zip(pixels[a], pixels[b]), color=color, linewidth=width,
                  solid_capstyle="round", label=label)
        label = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--filmstrip", type=int, default=6)
    args = parser.parse_args()
    out = args.run / "overlay"
    out.mkdir(exist_ok=False)

    manifest, sequence, entries, frames, box_world, centers, predictions = build(args.run)
    residuals = [f["fit_residual_m"] for f in frames]
    print(f"canonical->camera fit residual: median {np.median(residuals)*1000:.2f} mm, "
          f"max {max(residuals)*1000:.2f} mm")
    extent = box_world.max(0) - box_world.min(0)
    print(f"object box extent: {np.round(extent*100, 2)} cm")

    size = (1280, 720)
    movie = out / f"{sequence['directory']}_pose_overlay.mp4"
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    assert writer.isOpened(), f"Cannot open video writer: {movie}"
    picks = np.unique(np.linspace(0, len(entries) - 1, args.filmstrip).astype(int))
    strip, rows = [], []

    for index, (entry, frame) in enumerate(zip(entries, frames)):
        figure = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        axis = figure.add_axes([0.02, 0.02, 0.62, 0.90])
        axis.imshow(frame["image"])
        axis.set_xlim(0, frame["image"].shape[1])
        axis.set_ylim(frame["image"].shape[0], 0)
        axis.axis("off")
        gt_pixels, _ = annotated_box_pixels(frame, box_world)
        draw_box(axis, gt_pixels, "#000000", "Annotation", width=2.6)
        record = {"frame": entry["ids"][0]}
        for method, color, label in METHODS:
            pixels, _ = box_pixels(frame, box_world, predictions[method], index)
            draw_box(axis, pixels, color, label)
            record[f"{method}_box_corner_px"] = float(np.abs(pixels - gt_pixels).max())
        rows.append(record)
        axis.legend(loc="upper right", framealpha=0.85, fontsize=11)

        side = figure.add_axes([0.68, 0.55, 0.30, 0.36], projection="3d")
        side.plot(*centers.T, color="black", linewidth=1.4, label="Annotation")
        for method, color, _ in METHODS:
            side.plot(*predictions[method]["aligned_center"].T, color=color, linewidth=1.2)
        side.scatter(*centers[index], color="red", s=28)
        span = np.abs(centers - centers.mean(0)).max()
        middle = centers.mean(0)
        for setter, value in zip((side.set_xlim, side.set_ylim, side.set_zlim), middle):
            setter(value - span, value + span)
        side.set_title("Camera path (equal aspect)", fontsize=10)
        side.tick_params(labelsize=6)

        text = figure.add_axes([0.66, 0.04, 0.33, 0.45])
        text.axis("off")
        text.text(0, 1.0, f"{sequence['name']}", fontsize=14, weight="bold", va="top")
        text.text(0, 0.88, f"frame {entry['ids'][0]}   ({index + 1} of {len(entries)})",
                  fontsize=11, va="top")
        text.text(0, 0.72, "3 frozen reference views;\nevery later frame reads the cache.",
                  fontsize=10, va="top")
        text.text(0, 0.52, "Frozen cache   123.8 ms / frame\nOriginal         349.9 ms / frame",
                  fontsize=11, va="top", family="monospace")
        text.text(0, 0.30, "Rotation vs annotation\n  original  2.300 deg\n  cached    2.262 deg",
                  fontsize=11, va="top", family="monospace")
        text.text(0, 0.06, "Box drawn with each method's own tracked pose,\n"
                           "after one global alignment fitted over the clip.",
                  fontsize=8.5, va="top", color="#555555")

        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        assert pixels.shape[:2] == (size[1], size[0]), pixels.shape
        writer.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        if index in picks:
            strip.append((entry["ids"][0], pixels))
        plt.close(figure)
    writer.release()

    video = cv2.VideoCapture(str(movie))
    decoded = 0
    while video.read()[0]:
        decoded += 1
    video.release()
    assert decoded == len(entries), f"Decoded {decoded} of {len(entries)} frames"

    columns = len(strip)
    figure, axes = plt.subplots(1, columns, figsize=(3.1 * columns, 2.4), dpi=150)
    for ax, (frame_id, pixels) in zip(np.atleast_1d(axes), strip):
        ax.imshow(pixels[60:660, 40:760])
        ax.set_title(f"frame {frame_id}", fontsize=9)
        ax.axis("off")
    figure.suptitle(f"{sequence['name']} | annotation (black) vs original (blue) vs frozen cache (green)",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(out / f"{sequence['directory']}_filmstrip.png")
    plt.close(figure)

    with (out / f"{sequence['directory']}_box_agreement.csv").open("w") as handle:
        table = csv.DictWriter(handle, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)
    (out / "provenance.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((args.run / "manifest.json").read_bytes()).hexdigest(),
        "object_box_extent_m": (box_world.max(0) - box_world.min(0)).tolist(),
        "canonical_to_camera_residual_m": {"median": float(np.median(residuals)), "max": float(max(residuals))},
        "protocol": "Canonical object box from the union of masked NOCS over all queries, "
                    "placed by a per-frame NOCS<->depth Umeyama fit and averaged in the annotated "
                    "world frame. Projected with each method's tracked camera pose after ONE Sim(3) "
                    "fitted on all evaluated query centres, the same alignment the accuracy report "
                    "uses. Overall scale cancels in the projection; no metric claim. Box agreement "
                    "is a pixel distance between projected corners, not a pose error.",
    }, indent=2))
    print(f"decoded {decoded} frames")
    print("OVERLAY OK")


if __name__ == "__main__":
    main()
