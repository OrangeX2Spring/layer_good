#!/usr/bin/env python3
"""Render saved OPT tracking comparisons on remote Linux; no model or GPU needed."""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path)
p.add_argument("--fps", type=int, default=10, help="Playback FPS, independent of inference speed")
args = p.parse_args()
assert args.fps > 0
manifest = json.loads((args.run / "manifest.json").read_text())
cached = json.loads((args.run / "cached/results.json").read_text())
original = json.loads((args.run / "original/results.json").read_text())
readout = json.loads((args.run / "readout/results.json").read_text())
verification = json.loads((args.run / "verify/results.json").read_text())
assert len(manifest["sequences"]) == len(cached["records"]) == len(original["records"]) == len(readout["records"]) == len(verification["records"])
assert cached["gpu"] == original["gpu"] and cached["mode"] == original["mode"]
assert cached["dtype"] == original["dtype"]
assert (readout["gpu"], readout["mode"], readout["dtype"]) == (cached["gpu"], cached["mode"], cached["dtype"])
out = args.run / "report"
out.mkdir(exist_ok=False)
summary = []

for seq, cache_record, orig_record, read_record, checks in zip(
        manifest["sequences"], cached["records"], original["records"], readout["records"], verification["records"]):
    assert seq["name"] == cache_record["sequence"] == orig_record["sequence"] == read_record["sequence"] == checks["sequence"]
    tc = np.array([f["wall_ms"] for f in cache_record["frames"]])
    to = np.array([f["wall_ms"] for f in orig_record["frames"]])
    assert len(tc) == len(to) == len(seq["frames"]) - 1
    tr = np.array([f["wall_ms"] for f in read_record["frames"]])
    assert len(tr) == len(tc)
    build = cache_record["build"]["wall_ms"]
    saved = np.cumsum(to - tc) - build
    crossing = np.flatnonzero(saved > 0)
    delta_median = float(np.median(to) - np.median(tc))
    record = {
        "sequence": seq["name"], "queries": len(tc), "cache_build_ms": build,
        "cached_median_ms": float(np.median(tc)), "cached_p95_ms": float(np.percentile(tc, 95)),
        "original_median_ms": float(np.median(to)), "original_p95_ms": float(np.percentile(to, 95)),
        "cached_throughput_fps": 1000 * len(tc) / float(tc.sum()),
        "original_throughput_fps": 1000 * len(to) / float(to.sum()),
        "steady_speedup": float(to.sum() / tc.sum()),
        "readout_median_ms": float(np.median(tr)),
        "cache_speedup_vs_readout": float(tr.sum() / tc.sum()),
        "seconds_saved_vs_readout_including_build": float((tr.sum() - tc.sum() - build) / 1000),
        "speedup_including_build": float(to.sum() / (tc.sum() + build)),
        "seconds_saved_including_build": float(saved[-1] / 1000),
        "percent_time_saved_including_build": float(100 * saved[-1] / to.sum()),
        "observed_first_break_even_queries": int(crossing[0] + 1) if len(crossing) else None,
        "estimated_break_even_queries_at_median": math.floor(build / delta_median) + 1 if delta_median > 0 else None,
        "cache_bytes": cache_record["cache_bytes"],
        "cached_peak_allocated_bytes": max([cache_record["build"]["peak_allocated_bytes"]] + [f["peak_allocated_bytes"] for f in cache_record["frames"]]),
        "original_peak_allocated_bytes": max(f["peak_allocated_bytes"] for f in orig_record["frames"]),
        "cached_peak_reserved_bytes": max([cache_record["build"]["peak_reserved_bytes"]] + [f["peak_reserved_bytes"] for f in cache_record["frames"]]),
        "original_peak_reserved_bytes": max(f["peak_reserved_bytes"] for f in orig_record["frames"]),
    }
    rows = []
    movie = out / f"{seq['directory']}_comparison.mp4"
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1400, 800))
    assert writer.isOpened(), f"Cannot open video writer: {movie}"
    # Fixed clip-wide color scales. Read one frame at a time to bound host memory.
    depth_max = 0.0
    error_max = 0.0
    if cached["mode"] == "geometry":
        for index in range(len(tc)):
            with np.load(args.run / "original" / seq["directory"] / f"query{index:05d}.npz") as a, np.load(args.run / "cached" / seq["directory"] / f"query{index:05d}.npz") as b:
                depth_max = max(depth_max, float(a["depth"].max()), float(b["depth"].max()))
                error_max = max(error_max, float(np.abs(a["depth"] - b["depth"]).max()))
    for index, frame in enumerate(seq["frames"][1:]):
        with np.load(args.run / frame["input"]) as inp:
            rgb = inp["source_images"][0]
            mask = inp["source_inst_masks"][0].astype(bool)
        with np.load(args.run / "original" / seq["directory"] / f"query{index:05d}.npz") as data:
            a = dict(data)
        with np.load(args.run / "cached" / seq["directory"] / f"query{index:05d}.npz") as data:
            b = dict(data)
        with np.load(args.run / "readout" / seq["directory"] / f"query{index:05d}.npz") as data:
            r = dict(data)
        fidelity = {}
        for key in b:
            assert np.isfinite(b[key]).all() and np.isfinite(a[key]).all() and np.isfinite(r[key]).all()
            fidelity[key] = float(np.abs(b[key] - r[key]).max()) / max(float(np.abs(r[key]).max()), 1e-12)
            assert fidelity[key] <= checks["floors"][key], f"Stream fidelity failed: {seq['name']} {index} {key} {fidelity[key]}"
        # Camera encoding uses xyzw quaternions; sign-invariant angular difference.
        qa, qb = a["pose_enc"][0, 0, 3:7], b["pose_enc"][0, 0, 3:7]
        dot = abs(float(np.dot(qa, qb))) / max(float(np.linalg.norm(qa) * np.linalg.norm(qb)), 1e-12)
        row = {"frame": frame["ids"][0], "original_ms": float(to[index]), "cached_ms": float(tc[index]),
               "readout_ms": float(tr[index]), "cached_vs_readout_max_rel": max(fidelity.values()),
               "net_seconds_saved": float(saved[index] / 1000),
               "rotation_difference_deg": float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))),
               "translation_difference_model_units": float(np.linalg.norm(a["pose_enc"][0, 0, :3] - b["pose_enc"][0, 0, :3]))}
        if cached["mode"] == "geometry":
            da, db = a["depth"].squeeze(), b["depth"].squeeze()
            point_error = np.linalg.norm(a["world_points"][0, 0] - b["world_points"][0, 0], axis=-1)
            row["depth_mask_mae"] = float(np.abs(da - db)[mask].mean())
            row["pointmap_mask_mean_distance"] = float(point_error[mask].mean())
        rows.append(row)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=100, constrained_layout=True)
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title(f"Input crop | frame {frame['ids'][0]}")
        if cached["mode"] == "geometry":
            axes[0, 1].imshow(da, vmin=0, vmax=depth_max, cmap="viridis")
            axes[0, 1].set_title("Original bidirectional depth")
            axes[0, 2].imshow(db, vmin=0, vmax=depth_max, cmap="viridis")
            axes[0, 2].set_title("Frozen-cache depth | same scale")
            im = axes[1, 0].imshow(np.abs(da - db), vmin=0, vmax=max(error_max, 1e-12), cmap="magma")
            axes[1, 0].set_title("Absolute depth difference | model units")
            fig.colorbar(im, ax=axes[1, 0], fraction=0.035)
        else:
            for j, label in enumerate(("Original", "Frozen cache"), start=1):
                pose = (a if j == 1 else b)["pose_enc"][0, 0]
                axes[0, j].bar(["tx", "ty", "tz"], pose[:3])
                axes[0, j].set_title(f"{label} camera translation | model units")
                lim = max(float(np.abs(a["pose_enc"][..., :3]).max()), float(np.abs(b["pose_enc"][..., :3]).max()), 1e-6)
                axes[0, j].set_ylim(-lim * 1.1, lim * 1.1)
            axes[1, 0].plot([r["frame"] for r in rows], [r["rotation_difference_deg"] for r in rows])
            axes[1, 0].set_title("Camera rotation difference (degrees)")
        axes[1, 1].plot(to[:index+1], label="Original")
        axes[1, 1].plot(tc[:index+1], label="Frozen cache")
        axes[1, 1].plot(tr[:index+1], label="Recomputed causal readout", alpha=0.7)
        axes[1, 1].set_title("Synchronized model latency (ms)")
        axes[1, 1].legend()
        axes[1, 2].plot(saved[:index+1] / 1000)
        axes[1, 2].axhline(0, color="grey", linewidth=1)
        axes[1, 2].set_title("Cumulative seconds saved, including build")
        fig.suptitle(f"{seq['name']} | {cached['mode']} / {cached['dtype']} | differences, not GT errors\n"
                     f"Fixed {manifest['arguments']['num_ref']} references; playback {args.fps} FPS is not inference FPS")
        for ax in axes[0]:
            if cached["mode"] == "geometry" or ax is axes[0, 0]:
                ax.axis("off")
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        if index in {0, len(tc) - 1}:
            fig.savefig(out / f"{seq['directory']}_query{index:05d}.png")
        plt.close(fig)
    writer.release()
    # Decode the complete artifact, catching truncated or unsupported encodes.
    video = cv2.VideoCapture(str(movie))
    decoded = 0
    while True:
        ok, image = video.read()
        if not ok:
            break
        assert image.shape[:2] == (800, 1400)
        decoded += 1
    video.release()
    assert decoded == len(tc), f"Decoded {decoded}/{len(tc)} frames from {movie}"
    with (out / f"{seq['directory']}_frames.csv").open("w") as f:
        table = csv.DictWriter(f, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)
    record["rotation_difference_median_deg"] = float(np.median([r["rotation_difference_deg"] for r in rows]))
    summary.append(record)
assert summary
(out / "summary.json").write_text(json.dumps(summary, indent=2))
with (out / "summary.csv").open("w") as f:
    table = csv.DictWriter(f, fieldnames=list(summary[0]))
    table.writeheader()
    table.writerows(summary)
print(json.dumps(summary, indent=2))
print("REPORT OK: all videos decoded at the expected dimensions and frame count")
