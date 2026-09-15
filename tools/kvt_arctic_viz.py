"""Render the ARCTIC pilot's segmentation videos, trajectory plots and object clouds.

Called at the end of `kvt_arctic_run.py`, and runnable on its own via
`bash tools/kvt_run.sh arctic-viz --results pilot` while the results are still in
job-local /tmp, so a figure can be reworked without re-tracking. Reads only what the
run wrote: the recorded per-frame masks, `traj.npy`, `kf_idx.npy` and the last
keyframe reconstruction.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")  # No display in the container.
import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kv_tracker"))
from kv_tracker.eval_tools.evo_utils import align_pair
import eval as kvt_eval
from kvt_artifacts import write_ply
from kvt_confidence_videos import render

DATASET_DIR = Path("datasets/arctic_data/data/cropped_images_grab_only_subset/s01")


def segmentation_video(scene, results_dir, fps=30):
    """Every frame the tracker saw, with the mask that was actually used on it."""
    frames = sorted((DATASET_DIR / scene / "0").glob("*.jpg"))[2:]
    masks = sorted((results_dir / "sam_masks").glob("*.png"))
    assert len(masks) == len(frames), (scene, len(masks), len(frames))
    height, width = cv2.imread(str(frames[0])).shape[:2]
    path = results_dir / "viz" / "segmentation.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), f"Cannot open video encoder: {path}"
    fractions = []
    for index, (frame_path, mask_path) in enumerate(zip(frames, masks)):
        bgr = cv2.imread(str(frame_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        fractions.append(float(mask.mean()))
        bgr[mask] = (0.55 * bgr[mask] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 0, 255), 2)
        cv2.putText(bgr, f"{scene}  frame {index + 1}/{len(frames)}  "
                         f"object {100 * fractions[-1]:.1f}% of frame",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(bgr)
    writer.release()
    assert path.stat().st_size > 0, path
    return path, fractions


def trajectory_plot(scene, results_dir, row):
    """Aligned estimate against GT, plus the per-frame error the ATE summarises."""
    traj_gt = kvt_eval.load_gt_arctic(scene)
    traj_est = np.load(results_dir / "traj.npy")
    aligned, gt = align_pair({"traj_gt": traj_gt, "traj_est": traj_est, "name": scene},
                             "traj", ret_np=True)
    aligned = np.asarray(aligned)
    error = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    ate = float(np.sqrt((error ** 2).mean()))
    # Same quantity evo reported, recomputed: the curve below must summarise to it.
    assert abs(ate - row["ate_m"]) < 1e-6, (scene, ate, row["ate_m"])

    keyframes = np.load(results_dir / "kf_idx.npy") if (results_dir / "kf_idx.npy").is_file() \
        else np.zeros(0, int)
    figure = plt.figure(figsize=(14, 9))
    figure.suptitle(f"KV-Tracker on ARCTIC S01 {scene} — object-relative camera trajectory",
                    fontsize=13)

    space = figure.add_subplot(2, 2, 1, projection="3d")
    space.plot(*gt[:, :3, 3].T, color="black", linewidth=1.2, label="ground truth")
    space.plot(*aligned[:, :3, 3].T, color="tab:red", linewidth=1.0, label="ours (Sim(3) aligned)")
    space.set_box_aspect(np.ptp(gt[:, :3, 3], axis=0))
    space.set_xlabel("x (m)"), space.set_ylabel("y (m)"), space.set_zlabel("z (m)")
    space.legend(loc="upper left", fontsize=8)

    curve = figure.add_subplot(2, 2, 2)
    curve.plot(error, color="tab:red", linewidth=0.9, label="per-frame translation error")
    curve.axhline(ate, color="tab:red", linestyle="--", linewidth=1.0,
                  label=f"ATE RMSE {ate:.3f} m")
    curve.axhline(row["paper_ate_m"], color="tab:blue", linestyle=":", linewidth=1.2,
                  label=f"paper Table 4 {row['paper_ate_m']:.3f} m")
    for index, keyframe in enumerate(keyframes):
        curve.axvline(keyframe, color="tab:green", alpha=0.35, linewidth=0.8,
                      label="keyframe" if index == 0 else None)
    curve.set_xlabel("tracked frame"), curve.set_ylabel("error (m)")
    curve.legend(fontsize=8), curve.grid(alpha=0.3)

    axes = figure.add_subplot(2, 1, 2)
    for index, (name, color) in enumerate(zip("xyz", ("tab:blue", "tab:orange", "tab:green"))):
        axes.plot(gt[:, index, 3], color=color, linewidth=1.1, label=f"{name} ground truth")
        axes.plot(aligned[:, index, 3], color=color, linewidth=1.0, linestyle="--",
                  label=f"{name} ours")
    axes.set_xlabel("tracked frame"), axes.set_ylabel("position (m)")
    axes.legend(ncol=3, fontsize=8), axes.grid(alpha=0.3)

    path = results_dir / "viz" / "trajectory.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path, error, len(keyframes)


def object_cloud(scene, results_dir, frames=120, fps=12, size=720):
    """The retained object points from the last keyframe reconstruction."""
    with np.load(results_dir / "keyframes.npz") as data:
        keep = data["masks"] & (data["confidence"] > float(data["threshold"]))
        points, colors = data["xyz"][keep], data["rgb"][keep]
        keyframe_count, threshold = len(data["frame_ids"]), float(data["threshold"])
    assert len(points), f"{scene}: no object points survived the confidence threshold"
    ply = results_dir / "viz" / "object.ply"
    write_ply(ply, points, colors)

    # Percentile bounds, so a handful of stray points cannot set the field of view.
    low, high = np.percentile(points, [1, 99], axis=0)
    center, span = (low + high) / 2, float(np.linalg.norm(high - low) * 1.15)
    alpha = np.ones(len(points))
    tilt = np.deg2rad(20)
    lean = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    caption = (f"{scene} | {len(points):,} points | {keyframe_count} keyframes | "
               f"confidence > {threshold:.3f}")

    video = results_dir / "viz" / "object_turntable.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size + 34))
    assert writer.isOpened(), f"Cannot open video encoder: {video}"
    stills = []
    for index in range(frames):
        angle = 2 * np.pi * index / frames
        cosine, sine = np.cos(angle), np.sin(angle)
        turn = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
        image = render(points, colors, alpha, center, turn @ lean, span, size)
        panel = np.full((size + 34, size, 3), 24, np.uint8)
        panel[34:] = image[..., ::-1]
        cv2.putText(panel, caption, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
        writer.write(panel)
        if index % (frames // 4) == 0:
            stills.append(panel)
    writer.release()
    montage = np.vstack([np.hstack(stills[:2]), np.hstack(stills[2:4])])
    cv2.imwrite(str(results_dir / "viz" / "object_views.png"), montage)
    return ply, len(points)


def summary_plot(metrics, path):
    """The headline figure: our ATE against paper Table 4, per sequence."""
    rows = metrics["rows"]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(figsize=(9, 4.5))
    axes.bar(positions - 0.19, [row["paper_ate_m"] for row in rows], 0.38,
             color="tab:blue", label="paper Table 4")
    axes.bar(positions + 0.19, [row["ate_m"] for row in rows], 0.38,
             color="tab:red", label="ours")
    for position, row in zip(positions, rows):
        axes.text(position + 0.19, row["ate_m"], f"{row['ate_m']:.3f}",
                  ha="center", va="bottom", fontsize=9)
        axes.text(position - 0.19, row["paper_ate_m"], f"{row['paper_ate_m']:.3f}",
                  ha="center", va="bottom", fontsize=9)
    axes.set_xticks(positions)
    axes.set_xticklabels([row["scene"].replace("_grab_01", "") for row in rows])
    axes.set_ylabel("ATE RMSE (m)")
    axes.set_title("KV-Tracker on ARCTIC S01: three sequences, per-sequence comparison")
    axes.legend(), axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def visualize(scenes, results_name, metrics, summary_path):
    rows = {row["scene"]: row for row in metrics["rows"]}
    for scene in scenes:
        results_dir = DATASET_DIR / scene / results_name
        (results_dir / "viz").mkdir(exist_ok=True)
        video, fractions = segmentation_video(scene, results_dir)
        plot, error, keyframes = trajectory_plot(scene, results_dir, rows[scene])
        ply, points = object_cloud(scene, results_dir)
        rows[scene]["object_fraction_median"] = float(np.median(fractions))
        rows[scene]["object_fraction_min"] = float(np.min(fractions))
        rows[scene]["error_max_m"] = float(error.max())
        rows[scene]["keyframes"] = int(keyframes)
        rows[scene]["object_points"] = int(points)
        print(f"VIZ {scene}: {video.name}, {plot.name}, {ply.name} "
              f"({points:,} points, {keyframes} keyframes, object "
              f"{100 * np.median(fractions):.1f}% of frame, worst error "
              f"{error.max():.3f} m)", flush=True)
    summary_plot(metrics, summary_path)
    print("VIZ OK:", summary_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="pilot")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--metrics", type=Path, required=True,
                        help="metrics.json from the run being visualised")
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    visualize(args.scenes, args.results, metrics,
              Path("/tmp") / f"arctic_{args.results}_summary.png")


if __name__ == "__main__":
    main()
