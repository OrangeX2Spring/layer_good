"""Render the ARCTIC pilot the way the HouseCat6D artifacts were rendered.

One video per sequence, in the `actual_history.mp4` layout of `kvt_artifacts.py`:
the frame the tracker saw with its own mask on the left, the reconstructed object
with the tracked and ground-truth camera trails on the right, a text strip under
both, and a frozen-cloud orbit at the end. Plus the object as a PLY. No plots:
the numbers are the printed table and `metrics.json`.

Called at the end of `kvt_arctic_run.py`, and runnable on its own with
`bash tools/kvt_run.sh arctic-viz` while the results are still in job-local /tmp,
so a view can be reworked without re-tracking.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kv_tracker"))
from kv_tracker.eval_tools.evo_utils import align_pair
from kv_tracker.geometry import umeyama_alignment
import eval as kvt_eval
from kvt_artifacts import write_ply
from kvt_confidence_videos import render

DATASET_DIR = Path("datasets/arctic_data/data/cropped_images_grab_only_subset/s01")
SIZE, STRIP, FPS, ORBIT = 640, 160, 30, 72


def project(points, center, rotation, span, size=SIZE):
    """The projection `render` uses, so overlays land on the points they mark."""
    local = (points - center) @ rotation
    return np.floor((local[:, :2] / span + 0.5) * size).astype(int)


def align_to_ground_truth(scene, results_dir):
    """Both trails and the cloud in the GT metric frame, under one Sim(3)."""
    ground_truth = kvt_eval.load_gt_arctic(scene)
    estimate = np.load(results_dir / "traj.npy")
    assert ground_truth.shape == estimate.shape, (scene, ground_truth.shape, estimate.shape)
    aligned, _ = align_pair({"traj_gt": ground_truth, "traj_est": estimate, "name": scene},
                            "traj", ret_np=True)
    aligned = np.asarray(aligned)
    rotation, translation, scale = umeyama_alignment(
        estimate[:, :3, 3].T, ground_truth[:, :3, 3].T, with_scale=True)
    # The transform that carries the cloud must be the one evo used on the poses.
    assert np.allclose(scale * (estimate[:, :3, 3] @ rotation.T) + translation,
                       aligned[:, :3, 3], atol=1e-6), scene
    return ground_truth, aligned, (rotation, translation, scale)


def object_points(results_dir, sim3):
    with np.load(results_dir / "keyframes.npz") as data:
        keep = data["masks"] & (data["confidence"] > float(data["threshold"]))
        points, colors = data["xyz"][keep], data["rgb"][keep]
        keyframes, threshold = len(data["frame_ids"]), float(data["threshold"])
    assert len(points), "No object points survived the confidence threshold"
    write_ply(results_dir / "viz" / "object.ply", points, colors)
    rotation, translation, scale = sim3
    return scale * (points @ rotation.T) + translation, colors, keyframes, threshold


def view_frame(points, trails):
    """One fixed view holding the object and both trails, tilted off-axis."""
    low, high = np.percentile(points, [1, 99], axis=0)
    extent = np.vstack([low, high] + trails)
    center = (extent.min(0) + extent.max(0)) / 2
    span = float(np.linalg.norm(extent.max(0) - extent.min(0)) * 1.1)
    tilt = np.deg2rad(20)
    return center, span, np.array([[1, 0, 0],
                                   [0, np.cos(tilt), -np.sin(tilt)],
                                   [0, np.sin(tilt), np.cos(tilt)]])


def panel(left, right, lines):
    canvas = np.full((SIZE + STRIP, 2 * SIZE, 3), 24, np.uint8)
    canvas[:SIZE, :SIZE], canvas[:SIZE, SIZE:] = left, right
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (14, SIZE + 34 + 40 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def tracking_video(scene, results_dir, row):
    frames = sorted((DATASET_DIR / scene / "0").glob("*.jpg"))[2:]
    masks = sorted((results_dir / "sam_masks").glob("*.png"))
    assert len(masks) == len(frames), (scene, len(masks), len(frames))

    ground_truth, aligned, sim3 = align_to_ground_truth(scene, results_dir)
    points, colors, keyframes, threshold = object_points(results_dir, sim3)
    gt_track, est_track = ground_truth[:, :3, 3], aligned[:, :3, 3]
    error = np.linalg.norm(est_track - gt_track, axis=1)
    ate = float(np.sqrt((error ** 2).mean()))
    # The curve drawn here and the reported ATE are the same quantity.
    assert abs(ate - row["ate_m"]) < 1e-6, (scene, ate, row["ate_m"])

    center, span, rotation = view_frame(points, [gt_track, est_track])
    cloud = render(points, colors, np.ones(len(points)), center, rotation, span, SIZE)
    cloud = np.ascontiguousarray(cloud[..., ::-1])
    gt_pixels = project(gt_track, center, rotation, span)
    est_pixels = project(est_track, center, rotation, span)

    path = results_dir / "viz" / "tracking.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (2 * SIZE, SIZE + STRIP))
    assert writer.isOpened(), f"Cannot open video encoder: {path}"
    fractions = []
    for index, (frame_path, mask_path) in enumerate(zip(frames, masks)):
        bgr = cv2.imread(str(frame_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        fractions.append(float(mask.mean()))
        bgr[mask] = (0.6 * bgr[mask] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 0, 255), 2)

        right = cloud.copy()
        cv2.polylines(right, [gt_pixels[:index + 1]], False, (80, 220, 80), 1, cv2.LINE_AA)
        cv2.polylines(right, [est_pixels[:index + 1]], False, (70, 70, 240), 1, cv2.LINE_AA)
        cv2.circle(right, tuple(est_pixels[index]), 5, (70, 70, 240), -1)
        cv2.circle(right, tuple(gt_pixels[index]), 5, (80, 220, 80), 1)
        writer.write(panel(
            cv2.resize(bgr, (SIZE, SIZE)), right,
            [f"{scene} | frame {index + 1}/{len(frames)} | "
             f"object {100 * fractions[-1]:.1f}% of frame | {keyframes} keyframes",
             f"error {error[index]:.3f} m | ATE RMSE {ate:.3f} m | "
             f"paper Table 4 {row['paper_ate_m']:.3f} m",
             f"green ground truth, red KV-Tracker (Sim(3) aligned) | "
             f"{len(points):,} object points, confidence > {threshold:.3f}"]))

    # Frozen final cloud, explicitly separate from the tracking above it.
    for step in range(ORBIT):
        angle = 2 * np.pi * step / ORBIT
        cosine, sine = np.cos(angle), np.sin(angle)
        turn = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
        orbit = render(points, colors, np.ones(len(points)), center, rotation @ turn, span, SIZE)
        writer.write(panel(
            np.full((SIZE, SIZE, 3), 24, np.uint8), np.ascontiguousarray(orbit[..., ::-1]),
            [f"{scene} | frozen final reconstruction, 360 degree orbit",
             f"ATE RMSE {ate:.3f} m | paper Table 4 {row['paper_ate_m']:.3f} m",
             f"{len(points):,} object points from {keyframes} keyframes, "
             f"confidence > {threshold:.3f}"]))
    writer.release()
    assert path.stat().st_size > 0, path
    return path, fractions, error, keyframes, len(points)


def visualize(scenes, results_name, metrics):
    rows = {row["scene"]: row for row in metrics["rows"]}
    for scene in scenes:
        results_dir = DATASET_DIR / scene / results_name
        (results_dir / "viz").mkdir(exist_ok=True)
        path, fractions, error, keyframes, points = tracking_video(
            scene, results_dir, rows[scene])
        rows[scene].update({"object_fraction_median": float(np.median(fractions)),
                            "object_fraction_min": float(np.min(fractions)),
                            "error_max_m": float(error.max()),
                            "keyframes": int(keyframes),
                            "object_points": int(points)})
        print(f"VIZ {scene}: {path.name} + object.ply | {points:,} points, "
              f"{keyframes} keyframes, object {100 * np.median(fractions):.1f}% of frame, "
              f"worst error {error.max():.3f} m", flush=True)
    print("VIZ OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="pilot")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--metrics", type=Path, required=True,
                        help="metrics.json from the run being visualised")
    args = parser.parse_args()
    visualize(args.scenes, args.results, json.loads(args.metrics.read_text()))


if __name__ == "__main__":
    main()
