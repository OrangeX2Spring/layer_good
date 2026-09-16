"""CAMP CPU evaluation of saved STream3R ARCTIC poses against KV-Tracker.

No inference or model imports. Read existing archives without unpacking geometry.
Convention is fixed from code: invert STream3R camera-from-world extrinsics to
match the camera-to-world poses exported by KV-Tracker's pi3_utilts.py/main.py.
Masked RGB supplies the object-centric interpretation; original RGB is a control.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation


def read_array(archive, name, pickle=False):
    return np.load(io.BytesIO(archive.extractfile(name).read()), allow_pickle=pickle)


def ground_truth(archive, prefix, scene):
    # Exact algebra and units from kv_tracker/eval.py:load_gt_arctic.
    obj = read_array(archive, f"{prefix}/gt/{scene}.object.npy", True)[:, 1:]
    cam = read_array(archive, f"{prefix}/gt/{scene}.egocam.dist.npy", True).item()
    n = len(obj)
    assert obj.shape == (n, 6)
    camera = np.tile(np.eye(4), (n, 1, 1))
    camera[:, :3, :3] = cam["R_k_cam_np"]
    camera[:, :3, 3] = cam["T_k_cam_np"].squeeze()
    object_pose = np.tile(np.eye(4), (n, 1, 1))
    object_pose[:, :3, :3] = Rotation.from_rotvec(obj[:, :3]).as_matrix()
    object_pose[:, :3, 3] = obj[:, 3:] * 1e-3
    return np.linalg.inv(np.linalg.inv(camera) @ object_pose)[2:]


def evaluate(est, gt):
    assert est.shape == gt.shape and est.shape[1:] == (4, 4)
    assert np.isfinite(est).all() and np.isfinite(gt).all()
    assert np.allclose(est[:, 3], [0, 0, 0, 1])
    rotations = est[:, :3, :3]
    assert np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
    assert np.allclose(np.linalg.det(rotations), 1, atol=1e-4)
    x, y = est[:, :3, 3], gt[:, :3, 3]
    a, b = x - x.mean(0), y - y.mean(0)
    spread = np.linalg.svd(a, compute_uv=False) / np.sqrt(len(a))
    gt_spread = np.linalg.svd(b, compute_uv=False) / np.sqrt(len(b))
    stationary = float(np.sqrt(np.mean(np.sum(b * b, axis=1))))
    # A collapsed/collinear estimate cannot support evo's Sim(3) fit. Record
    # the failure rather than silently regularizing it or emitting a good ATE.
    covariance = b.T @ a / len(a)
    rank = int(np.linalg.matrix_rank(covariance))
    diagnostics = {"pred_spread_model_units": spread.tolist(),
                   "gt_spread_m": gt_spread.tolist(), "covariance_rank": rank,
                   "stationary_baseline_ate_m": stationary}
    if rank < 2:
        return {**diagnostics, "status": "degenerate_alignment", "ate_m": None}
    u, d, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    scale = np.sum(d * np.diag(sign)) / np.mean(np.sum(a * a, axis=1))
    assert scale > 0 and np.isfinite(scale)
    aligned = est.copy()
    aligned[:, :3, :3] = rotation @ rotations
    aligned[:, :3, 3] = scale * (a @ rotation.T) + y.mean(0)
    errors = np.linalg.norm(aligned[:, :3, 3] - y, axis=1)
    # evo RPE: inverse(reference relative motion) @ estimated relative motion.
    gt_delta = np.linalg.inv(gt[:-1]) @ gt[1:]
    pred_delta = np.linalg.inv(aligned[:-1]) @ aligned[1:]
    relative_error = np.linalg.inv(gt_delta) @ pred_delta
    rpe_rot = relative_error[:, :3, :3] - np.eye(3)
    return {**diagnostics, "status": "ok", "alignment_scale": float(scale),
            "aligned_pred_spread_m": (scale * spread).tolist(),
            "ate_m": float(np.sqrt(np.mean(errors ** 2))),
            "translation_errors_m": errors.tolist(),
            "rpe_t_m": float(np.sqrt(np.mean(np.sum(relative_error[:, :3, 3] ** 2, axis=1)))),
            "rpe_rot": float(np.sqrt(np.mean(np.sum(rpe_rot ** 2, axis=(1, 2)))))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", nargs="+", type=Path, required=True)
    parser.add_argument("--kvt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"metric": "Sim(3)-aligned translation ATE RMSE, metres",
              "pose_convention": "inverse of saved STream3R extrinsics; no GT-based convention selection",
              "rpe": "adjacent selected samples; rotation-part Frobenius norm, not degrees",
              "comparison": "KV-Tracker full-run poses sampled at identical timestamps; input histories differ for even sampling",
              "caveat": "Masked pose-head readout is the object-centric baseline; original RGB is a control. ATE alone does not validate shape or rigidity.",
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "kvt_archive": str(args.kvt), "self_checks": {}, "rows": []}
    seen = set()
    with tarfile.open(args.kvt) as kvt:
        baseline_manifest = json.load(kvt.extractfile("manifest.json"))
        baseline_metrics = json.load(kvt.extractfile("metrics.json"))
        recorded = {row["scene"]: row for row in baseline_metrics["rows"]}
        for path in args.archives:
            with tarfile.open(path) as archive:
                assert archive.extractfile("./exit_status.txt").read().strip() == b"0"
                for member in archive.getmembers():
                    if not member.name.endswith("/inputs.json"):
                        continue
                    prefix = member.name.rsplit("/", 1)[0]
                    sampling = prefix.split("/")[-2]
                    inputs = json.load(archive.extractfile(member))
                    scene = inputs["scene"]
                    assert (sampling, scene) not in seen, "Duplicate scene/sampling"
                    seen.add((sampling, scene))
                    archived_manifest = json.load(archive.extractfile(
                        prefix.rsplit("/", 1)[0] + "/kvt_manifest.json"))
                    assert archived_manifest == baseline_manifest
                    indices = np.array([frame["kvt_index"] for frame in inputs["frames"]])
                    assert np.all(np.diff(indices) > 0)
                    assert all(frame["gt_row"] == frame["kvt_index"] + 2 for frame in inputs["frames"])
                    traj = read_array(kvt, f"{scene}/results/traj.npy").astype(np.float64)
                    gt_full = ground_truth(archive, prefix, scene)
                    full = evaluate(traj, gt_full)
                    assert full["status"] == "ok"
                    for metric in ("ate_m", "rpe_t_m", "rpe_rot"):
                        assert abs(full[metric] - recorded[scene][metric]) < 1e-5, (scene, metric, full[metric], recorded[scene][metric])
                    report["self_checks"][scene] = {"status": "passed", "ate_m": full["ate_m"]}
                    gt = gt_full[indices]
                    rows = {"kv_tracker": evaluate(traj[indices], gt)}
                    for condition in ("original", "masked"):
                        with read_array(archive, f"{prefix}/{condition}.npz") as prediction:
                            extr = prediction["extrinsics"].astype(np.float64)
                        assert extr.shape == (len(indices), 3, 4)
                        poses = np.tile(np.eye(4), (len(indices), 1, 1))
                        poses[:, :3] = extr
                        rows[f"stream3r_{condition}"] = evaluate(np.linalg.inv(poses), gt)
                    report["rows"].append({"scene": scene, "sampling": sampling,
                                           "archive": str(path), "kvt_indices": indices.tolist(),
                                           "metrics": rows})
                    print(f"{sampling} {scene}", flush=True)
                    for label, metrics in rows.items():
                        print(f"  {label}: ATE={metrics['ate_m']} m; {metrics['status']}", flush=True)
    assert len(seen) == 6, f"Expected all three scenes and both samplings, got {seen}"
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"SELF CHECK OK: all three KV-Tracker full-run ATE/RPE metrics reproduced\nEVAL OK {args.out}")


if __name__ == "__main__":
    main()
