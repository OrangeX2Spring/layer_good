#!/usr/bin/env python3
"""Ground-truth metrics for STream3R on YCB-V, split object vs background.

    python tools/stream3r_ycbv_metrics.py \
        --scene-dir /tmp/data/ycbv/test/000051 \
        --pred-dir  /tmp/t32 \
        --out       /tmp/t32/metrics.json

Reads the .npz that stream3r_visualize.py saved per mode, so no model and no GPU.
Pure numpy + PIL; it runs on the Mac as happily as on a node.

WHY THIS SPLIT. The proposal is a semantic KV cache: keep the tracked object's
tokens, drop the rest. Token accounting (stream3r_ycbv_tokens.py) says the object
is ~6.85% of tokens, so the cache would be ~15x smaller. That is only worth having
if the model is actually good where we keep it and bad where we throw it away.
Reporting one whole-image error would average exactly that distinction out, so
every metric here is computed three times: over object pixels, over background
pixels, and over all valid pixels.

TWO CHOICES THAT DECIDE WHETHER THE NUMBERS MEAN ANYTHING.

1. Scale is fitted ONCE, GLOBALLY, over all frames pooled. STream3R's depth is up
   to scale, and the usual monocular convention of a per-frame median ratio would
   silently absorb drift - which, for a causal model over a sequence, is precisely
   the quantity of interest. A per-frame fit is also reported so the two can be
   compared, but the global one is the honest number.

2. Rotation error uses a SECOND, SEPARATELY FITTED gauge. Fitting a Sim(3) on
   camera centres alone leaves orientation undetermined when the trajectory is
   poorly conditioned, which is how the OPT-Pose evaluator reported 179 deg for a
   pose that was 2.26 deg wrong (CLAUDE.md, alignment-gauge). Translation keeps the
   centre-fitted Sim(3); rotation gets its own closed-form fit.

GT depth is downsampled to the model's grid by NEAREST index mapping, never
interpolated - averaging across a depth discontinuity invents a surface that is in
neither the prediction nor the sensor.
"""
import argparse
import glob
import json
import os
import numpy as np
from PIL import Image


def nearest_index_map(src, dst):
    """Index map for a nearest-neighbour resize of length src -> dst."""
    return np.clip(((np.arange(dst) + 0.5) * src / dst - 0.5).round(), 0, src - 1).astype(int)


def to_model_grid(img, hw):
    """Nearest-resize a HxW array onto the model's (h, w) grid."""
    h, w = hw
    return img[nearest_index_map(img.shape[0], h)][:, nearest_index_map(img.shape[1], w)]


def depth_metrics(pred, gt, mask):
    """AbsRel / RMSE / delta<1.25 over `mask`. pred is already scaled."""
    if mask.sum() < 100:
        return None
    p, g = pred[mask], gt[mask]
    ratio = np.maximum(p / g, g / p)
    return {
        "n": int(mask.sum()),
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta_1.25": float(np.mean(ratio < 1.25)),
        "delta_1.10": float(np.mean(ratio < 1.10)),
    }


def umeyama_sim3(src, dst):
    """Similarity transform mapping src -> dst (Nx3 each). Returns s, R, t."""
    assert src.shape == dst.shape and src.shape[1] == 3, (src.shape, dst.shape)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var = (S ** 2).sum() / len(src)
    s = float((sig * np.diag(W)).sum() / var) if var > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def rotation_angles_deg(R_pred, R_gt):
    """Per-frame angle after fitting one global rotation gauge (see module docstring).

    Verified synthetically: a constant offset applied to every frame - i.e. pure
    gauge - reports exactly 0. A genuine 5 deg error on one frame of twelve reports
    4.58 deg there and leaks 0.42 deg onto the rest, because one global gauge
    compromises across frames. So isolated errors read ~8% low, and the bias is
    identical between modes, which is what mode comparison needs.
    """
    M = np.zeros((3, 3))
    for a, b in zip(R_gt, R_pred):
        M += a @ b.T
    U, _, Vt = np.linalg.svd(M)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    Rg = U @ W @ Vt
    out = []
    for a, b in zip(R_gt, R_pred):
        c = (np.trace(Rg @ b @ a.T) - 1.0) / 2.0
        out.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--pred-dir", required=True, help="dir holding manifest.json and *_pred.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-depth", type=float, default=3.0, help="metres; YCB-V is a tabletop")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.pred_dir, "manifest.json")))
    im_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in manifest["images"]]
    scene_camera = json.load(open(os.path.join(args.scene_dir, "scene_camera.json")))
    scene_gt = json.load(open(os.path.join(args.scene_dir, "scene_gt.json")))

    results = {"scene_dir": args.scene_dir, "frames": len(im_ids), "modes": {}}

    for npz_path in sorted(glob.glob(os.path.join(args.pred_dir, "*_pred.npz"))):
        mode = os.path.basename(npz_path).replace("_pred.npz", "")
        z = np.load(npz_path)
        depth, extr = z["depth"], z["extrinsics"]
        assert depth.shape[0] == len(im_ids), (depth.shape, len(im_ids))
        hw = depth.shape[1:]

        # ---- gather GT aligned to the model grid -------------------------------
        gts, objs, valids = [], [], []
        for i, im_id in enumerate(im_ids):
            cam = scene_camera[str(im_id)]
            g = np.asarray(Image.open(
                os.path.join(args.scene_dir, "depth", f"{im_id:06d}.png"))).astype(np.float32)
            g = g * float(cam["depth_scale"]) / 1000.0          # -> metres
            g = to_model_grid(g, hw)

            obj = np.zeros(g.shape, bool)
            for gt_id in range(len(scene_gt[str(im_id)])):
                m = os.path.join(args.scene_dir, "mask_visib", f"{im_id:06d}_{gt_id:06d}.png")
                if not os.path.exists(m):
                    raise FileNotFoundError(m)
                obj |= to_model_grid(np.asarray(Image.open(m)) > 0, hw)

            gts.append(g)
            objs.append(obj)
            valids.append((g > 1e-3) & (g < args.max_depth))

        gt = np.stack(gts); obj = np.stack(objs); valid = np.stack(valids)

        # ---- scale: one factor for the whole sequence --------------------------
        pos = valid & (depth > 1e-6)
        scale_global = float(np.median(gt[pos] / depth[pos]))
        scaled = depth * scale_global

        per_frame = []
        for i in range(len(im_ids)):
            v = pos[i]
            per_frame.append({
                "im_id": im_ids[i],
                "object": depth_metrics(scaled[i], gt[i], v & obj[i]),
                "background": depth_metrics(scaled[i], gt[i], v & ~obj[i]),
                "all": depth_metrics(scaled[i], gt[i], v),
                # per-frame scale, for comparison only: its drift across the
                # sequence IS the drift a global fit is meant to expose.
                "scale_this_frame": float(np.median(gt[i][v] / depth[i][v])) if v.sum() else None,
            })

        def agg(region, key):
            xs = [f[region][key] for f in per_frame if f[region]]
            return float(np.mean(xs)) if xs else None

        sc = np.array([f["scale_this_frame"] for f in per_frame if f["scale_this_frame"]])

        # ---- camera trajectory -------------------------------------------------
        R_pred = extr[:, :, :3]
        C_pred = np.stack([-R.T @ t for R, t in zip(R_pred, extr[:, :, 3])])
        R_gt, C_gt = [], []
        for im_id in im_ids:
            cam = scene_camera[str(im_id)]
            R = np.array(cam["cam_R_w2c"], np.float64).reshape(3, 3)
            t = np.array(cam["cam_t_w2c"], np.float64) / 1000.0
            R_gt.append(R); C_gt.append(-R.T @ t)
        R_gt = np.stack(R_gt); C_gt = np.stack(C_gt)

        s, R_align, t_align = umeyama_sim3(C_pred.astype(np.float64), C_gt)
        C_aligned = (s * (R_align @ C_pred.T).T) + t_align
        ate = np.linalg.norm(C_aligned - C_gt, axis=1)
        rot_err = rotation_angles_deg(R_pred.astype(np.float64), R_gt)

        results["modes"][mode] = {
            "scale_global": scale_global,
            "scale_per_frame_spread": {
                "min": float(sc.min()), "max": float(sc.max()),
                "max_over_min": float(sc.max() / sc.min()),
            },
            "depth": {r: {k: agg(r, k) for k in
                          ("abs_rel", "rmse", "delta_1.25", "delta_1.10")}
                      for r in ("object", "background", "all")},
            "camera": {
                "ate_m": {"mean": float(ate.mean()), "median": float(np.median(ate)),
                          "max": float(ate.max())},
                "rot_deg": {"mean": float(rot_err.mean()),
                            "median": float(np.median(rot_err)),
                            "max": float(rot_err.max())},
                "gt_center_std_m": [float(v) for v in C_gt.std(0)],
            },
            "per_frame": per_frame,
            "ate_per_frame_m": [float(v) for v in ate],
            "rot_per_frame_deg": [float(v) for v in rot_err],
        }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nscene {os.path.basename(args.scene_dir)}, {len(im_ids)} frames")
    hdr = f"{'mode':8s} {'region':11s} {'AbsRel':>8s} {'RMSE(m)':>8s} {'d<1.25':>8s}"
    print(hdr); print("-" * len(hdr))
    for mode, m in results["modes"].items():
        for r in ("object", "background", "all"):
            d = m["depth"][r]
            print(f"{mode:8s} {r:11s} {d['abs_rel']:8.4f} {d['rmse']:8.4f} {d['delta_1.25']:8.4f}")
    print(f"\n{'mode':8s} {'ATE med (mm)':>13s} {'rot med (deg)':>14s} {'scale max/min':>14s}")
    for mode, m in results["modes"].items():
        print(f"{mode:8s} {m['camera']['ate_m']['median']*1000:13.1f} "
              f"{m['camera']['rot_deg']['median']:14.2f} "
              f"{m['scale_per_frame_spread']['max_over_min']:14.3f}")
    print("\nMETRICS OK", flush=True)


if __name__ == "__main__":
    main()
