"""CPU-only evaluation of archived virtual-camera poses; run on remote Linux."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def align_centers(pred, gt):
    """One least-squares Sim(3), fitted to all query camera centers."""
    assert pred.shape == gt.shape and pred.ndim == 2 and pred.shape[1] == 3
    x, y = pred - pred.mean(0), gt - gt.mean(0)
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    # Collinear/stationary trajectories cannot determine this alignment rotation.
    assert singular[1] > singular[0] * 1e-6, f"Degenerate trajectory: {singular}"
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = float(singular @ sign / np.mean(np.sum(x * x, axis=1)))
    assert scale > 0
    translation = gt.mean(0) - scale * rotation @ pred.mean(0)
    return scale, rotation, translation, singular


def evaluate(pred_r, pred_c, gt_r, gt_c):
    scale, rotation, translation, singular = align_centers(pred_c, gt_c)
    centers = scale * pred_c @ rotation.T + translation
    orientations = rotation @ pred_r
    position_error = np.linalg.norm(centers - gt_c, axis=1)
    rotation_error = np.degrees(Rotation.from_matrix(
        np.swapaxes(gt_r, -1, -2) @ orientations).magnitude())
    # Consecutive relative rotation is independent of constant world alignment.
    pred_delta = np.swapaxes(pred_r[:-1], -1, -2) @ pred_r[1:]
    gt_delta = np.swapaxes(gt_r[:-1], -1, -2) @ gt_r[1:]
    relative_rotation_error = np.degrees(Rotation.from_matrix(
        np.swapaxes(gt_delta, -1, -2) @ pred_delta).magnitude())
    return centers, position_error, rotation_error, relative_rotation_error, {
        "scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist(),
        "covariance_singular_values": singular.tolist(),
        "ate_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "rotation_median_deg": float(np.median(rotation_error)),
        "rotation_p95_deg": float(np.percentile(rotation_error, 95)),
        "relative_rotation_median_deg": float(np.median(relative_rotation_error)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, nargs="?")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        centers = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        poses = Rotation.from_euler("xyz", [[0, 0, 0], [10, 5, 2], [0, 20, 0], [5, 10, 30]], degrees=True).as_matrix()
        r = Rotation.from_euler("xyz", [25, -17, 40], degrees=True).as_matrix()
        gt = 2.3 * centers @ r.T + [3, -2, 1]
        _, pe, re, rre, _ = evaluate(poses, centers, r @ poses, gt)
        assert max(pe) < 1e-10 and max(re) < 1e-8 and max(rre) < 1e-8
        wrong = poses.copy()
        wrong[-1] = wrong[-1] @ Rotation.from_euler("x", 10, degrees=True).as_matrix()
        _, _, re, rre, _ = evaluate(wrong, centers, r @ poses, gt)
        assert abs(re[-1] - 10) < 1e-8 and abs(rre[-1] - 10) < 1e-8
        print("ACCURACY SELF CHECK OK")
        return
    assert args.run is not None
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    out = args.run / "accuracy"
    out.mkdir(exist_ok=False)
    summaries, hashes = [], {}
    for seq in manifest["sequences"]:
        entries = seq["frames"][1:]
        assert len(entries) >= 3
        targets = {"physical": [], "training_symmetry": []}
        for entry in entries:
            path = args.run / entry["input"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == entry["sha256"], path
            hashes[str(path.relative_to(args.run))] = digest
            with np.load(path) as data:
                for label, key in [("physical", "source_extrinsics"), ("training_symmetry", "source_extrinsics_sym")]:
                    extrinsic = data[key].astype(np.float64)
                    assert extrinsic.shape == (1, 3, 4) and np.isfinite(extrinsic).all()
                    targets[label].append(extrinsic[0])
        for target, extrinsics in targets.items():
            e = np.stack(extrinsics)
            # Source rotations were archived as fp16. Explicitly project back to SO(3).
            u, _, vt = np.linalg.svd(e[:, :3, :3])
            signs = np.ones((len(e), 3))
            signs[:, -1] = np.linalg.det(u @ vt)
            gt_r = np.swapaxes((u * signs[:, None, :]) @ vt, -1, -2)
            gt_c = -(gt_r @ e[:, :3, 3, None])[..., 0]
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(131, projection="3d")
            posax, rotax = fig.add_subplot(132), fig.add_subplot(133)
            ax.plot(*gt_c.T, label="Annotation", color="black")
            rows = []
            for method in ("original", "readout", "cached"):
                encodings = []
                for index in range(len(entries)):
                    path = args.run / method / seq["directory"] / f"query{index:05d}.npz"
                    hashes[str(path.relative_to(args.run))] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path) as data:
                        pose = data["pose_enc"].astype(np.float64)
                        assert pose.shape == (1, 1, 9) and np.isfinite(pose).all()
                        encodings.append(pose[0, 0])
                enc = np.stack(encodings)
                assert np.all(np.linalg.norm(enc[:, 3:7], axis=1) > 0)
                pred_r = np.swapaxes(Rotation.from_quat(enc[:, 3:7]).as_matrix(), -1, -2)
                pred_c = -(pred_r @ enc[:, :3, None])[..., 0]
                centers, pe, re, rre, summary = evaluate(pred_r, pred_c, gt_r, gt_c)
                summaries.append({"sequence": seq["name"], "target": target, "method": method, **summary})
                ids = [entry["ids"][0] for entry in entries]
                for i, frame in enumerate(ids):
                    rows.append({"method": method, "frame": frame, "aligned_position_error_m": float(pe[i]),
                                 "aligned_rotation_error_deg": float(re[i]),
                                 "previous_query_rotation_error_deg": float(rre[i-1]) if i else None})
                ax.plot(*centers.T, label=method)
                posax.plot(ids, pe * 1000, label=method)
                rotax.plot(ids, re, label=method)
            ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="Query trajectories; per-method Sim(3)")
            posax.set(xlabel="Frame", ylabel="Position error (mm)")
            rotax.set(xlabel="Frame", ylabel="Rotation error (degrees)")
            for axis in (ax, posax, rotax):
                axis.legend()
            fig.suptitle(f"{seq['name']} | {target} annotations | alignment fitted on evaluated queries")
            fig.tight_layout()
            stem = f"{seq['directory']}_{target}"
            fig.savefig(out / f"{stem}.png", dpi=150)
            plt.close(fig)
            with (out / f"{stem}.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    (out / "provenance.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "input_sha256": hashes,
        "protocol": "Virtual crop cameras; xyzw world-to-camera predictions inverted. One Sim(3) fitted on ALL evaluated query centers separately for each method and target. No metric-scale recovery claim. GT fp16 rotations projected to SO(3). Physical and per-frame symmetry-adjusted targets reported separately; no confidence filtering.",
    }, indent=2))
    print(json.dumps(summaries, indent=2))
    print("ACCURACY OK")


if __name__ == "__main__":
    main()
