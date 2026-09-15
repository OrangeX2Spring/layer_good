"""Track and evaluate the three-sequence ARCTIC S01 pilot with KV-Tracker.

Run inside a CAMP GPU allocation via `bash tools/kvt_run.sh arctic-run`, after
`arctic_prepare_subset.py` and after the `arctic_init_mask.py` overlays have been
reviewed. Stages the prepared TAR in job-local /tmp, tracks each sequence in object
mode from its reviewed initial mask, evaluates against the upstream ARCTIC ground
truth, and writes one archive back to project storage.

Frames are fed synchronously rather than through run_dataset.py's unbounded queue:
the same frames in the same order, but bounded memory and a recordable per-frame
mask. Timing under this harness is not the paper's FPS benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = ROOT / "kv_tracker"
sys.path.insert(0, str(CHECKOUT))
from kv_tracker.dataloaders.arctic_loader import arcticLoader
from kv_tracker.eval_tools.evo_utils import align_pair
import eval as kvt_eval
import main as tracker
import kvt_arctic_viz

SCENES = ("box_grab_01", "ketchup_grab_01", "espressomachine_grab_01")
# Paper Table 4, translation ATE RMSE in metres, for these three sequences only.
PAPER_ATE = {"box_grab_01": 0.200, "ketchup_grab_01": 0.249,
             "espressomachine_grab_01": 0.151}
OUT = Path("/mnt/projects/gr/3DRecon/kvt_arctic_out")
STAGE = Path("/tmp/data/arctic_data")
# The relative path eval.py and run_dataset.py hardcode, resolved from the checkout.
DATASET_DIR = Path("datasets/arctic_data/data/cropped_images_grab_only_subset/s01")


class ArcticFrames(arcticLoader):
    """The upstream loader, feeding run_track3r directly and recording its masks."""

    def __init__(self, device, mask_dir, **cfg):
        self.mask_dir = mask_dir
        self.index = 0
        super().__init__(device, **cfg)

    def get_rgb_frame(self, idx=0):
        self.index = idx
        return super().get_rgb_frame(idx)

    def get_segmentation(self, frame):
        mask = super().get_segmentation(frame)
        mask_np = mask.cpu().numpy()
        assert mask_np.any(), f"Empty object mask at frame {self.index}"
        cv2.imwrite(str(self.mask_dir / f"{self.index:05d}.png"),
                    mask_np.astype(np.uint8) * 255)
        return mask

    def __iter__(self):
        for index in range(self.length):
            frame = self.get_frame(index)
            frame["idx"] = index
            yield frame


class LatestKeyframes:
    """Keep only the newest keyframe reconstruction, which is the object cloud.

    Uncompressed: this is overwritten on every keyframe addition, and zlib on
    float32 pointmaps costs seconds for almost no saving.
    """

    def __init__(self, path):
        self.path = path
        self.count = 0

    def __call__(self, kind, frame_ids, xyz, poses, confidence, rgb, masks,
                 threshold, latest_rgb):
        if kind != "keyframes":
            return
        xyz = xyz[0].detach().float().cpu().numpy()
        confidence = confidence[0, ..., 0].detach().float().cpu().numpy()
        assert xyz.shape == rgb.shape and confidence.shape == masks.shape == xyz.shape[:-1]
        assert len(frame_ids) == len(xyz)
        np.savez(self.path, xyz=xyz, poses=poses[0].detach().float().cpu().numpy(),
                 confidence=confidence, rgb=rgb, masks=masks,
                 frame_ids=np.asarray(frame_ids), threshold=float(threshold))
        self.count += 1


def articulation_degrees(scene):
    """The GT column `load_gt_arctic` drops. A grab sequence should be near-rigid,
    which is what makes one whole-object mask the right initialization."""
    angles = np.rad2deg(np.load(
        f"datasets/arctic_data/data/raw_seqs/s01/{scene}.object.npy",
        allow_pickle=True)[:, 0])
    return {"min_deg": float(angles.min()), "max_deg": float(angles.max()),
            "range_deg": float(angles.max() - angles.min()),
            "std_deg": float(angles.std())}


def staged_images(scene):
    return STAGE / "data" / "cropped_images_grab_only_subset" / "s01" / scene / "0"


def stage_dataset(manifest):
    """Extract the prepared TAR into job-local /tmp; /tmp is per node, so verify."""
    expected = {scene: entry["frames_before_offset"]
                for scene, entry in manifest["scenes"].items()}
    counts = {scene: len(list(staged_images(scene).glob("*.jpg"))) for scene in expected}
    if counts != expected:
        print(f"STAGING: {counts} on disk against {expected} expected", flush=True)
        shutil.rmtree(STAGE, ignore_errors=True)
        STAGE.mkdir(parents=True)
        with tarfile.open(OUT / "prepared.tar") as prepared:
            prepared.extractall(STAGE)
        counts = {scene: len(list(staged_images(scene).glob("*.jpg"))) for scene in expected}
        assert counts == expected, (counts, expected)
    print("STAGED:", STAGE, expected, flush=True)

    datasets = CHECKOUT / "datasets"
    datasets.mkdir(exist_ok=True)
    link = datasets / "arctic_data"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(STAGE)


def track(scene, manifest, results_name, resize_dim):
    scene_dir = DATASET_DIR / scene
    reviewed = OUT / "masks" / scene / "init_mask.png"
    assert reviewed.is_file(), f"No reviewed initial mask for {scene}: {reviewed}"
    mask = cv2.imread(str(reviewed), cv2.IMREAD_GRAYSCALE)
    initial = scene_dir / "0" / Path(manifest["scenes"][scene]["initial_image"]).name
    first = cv2.imread(str(initial))
    assert mask is not None and first is not None, (reviewed, initial)
    assert mask.shape == first.shape[:2], (scene, mask.shape, first.shape)
    shutil.copyfile(reviewed, scene_dir / "init_mask.png")

    results_dir = scene_dir / results_name
    mask_dir = results_dir / "sam_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    source = ArcticFrames("cuda:0", mask_dir, scene_dir=scene_dir, offset=2,
                          obj_mode=True, resize_dim=resize_dim)
    expected = manifest["scenes"][scene]["tracking_frames"]
    assert source.length == expected, (scene, source.length, expected)

    recorder = LatestKeyframes(results_dir / "keyframes.npz")
    started = time.perf_counter()
    tracker.run_track3r(cfg={"results_path": str(results_dir), "que_size": 1},
                        args=["--obj_mode", "--resize_dim", str(resize_dim)],
                        frame_source=source, snapshot_callback=recorder)
    elapsed = time.perf_counter() - started
    assert recorder.count, f"{scene}: no keyframe reconstruction was recorded"

    traj = np.load(results_dir / "traj.npy")
    assert traj.shape == (expected, 4, 4), (scene, traj.shape, expected)
    masks_written = len(list(mask_dir.glob("*.png")))
    assert masks_written == expected, (scene, masks_written, expected)
    print(f"\nTRACKED {scene}: {expected} frames in {elapsed:.1f}s "
          f"({expected / elapsed:.2f} frames/s, synchronous export)", flush=True)
    return {"frames": expected, "seconds": round(elapsed, 1),
            "frames_per_second": round(expected / elapsed, 2)}


def evaluate(scenes, results_name):
    rows = []
    for scene in scenes:
        traj_gt = kvt_eval.load_gt_arctic(scene)
        traj_est = np.load(DATASET_DIR / scene / results_name / "traj.npy")
        assert traj_gt.shape == traj_est.shape, (scene, traj_gt.shape, traj_est.shape)
        ate, rpe_t, rpe_rot = align_pair(
            {"traj_gt": traj_gt, "traj_est": traj_est, "name": scene}, "traj", ret_np=False)
        rows.append({"scene": scene, "frames": int(traj_gt.shape[0]),
                     "ate_m": float(ate), "rpe_t_m": float(rpe_t),
                     # evo's dimensionless rotation-part norm, as eval.py reports it.
                     "rpe_rot": float(rpe_rot), "paper_ate_m": PAPER_ATE[scene]})

    print(f"\n=== ARCTIC S01 pilot ({len(rows)} of 10 paper sequences) ===")
    print(f"{'Scene':<28}{'frames':>8}{'ATE (m)':>10}{'paper':>8}{'RPE_t (m)':>11}{'RPE_R':>9}")
    print("-" * 74)
    for row in rows:
        print(f"{row['scene']:<28}{row['frames']:>8}{row['ate_m']:>10.3f}"
              f"{row['paper_ate_m']:>8.3f}{row['rpe_t_m']:>11.3f}{row['rpe_rot']:>9.3f}")
    print("-" * 74)
    print("Compare per sequence; this subset's mean is not the paper's ten-sequence mean.")
    return rows


def archive(scenes, results_name, provenance, metrics):
    archive_path = OUT / f"arctic_{results_name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.tar"
    staging = Path("/tmp") / archive_path.stem
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (staging / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    with tarfile.open(archive_path, "w") as bundle:
        bundle.add(staging / "manifest.json", arcname="manifest.json")
        bundle.add(staging / "metrics.json", arcname="metrics.json")
        for scene in scenes:
            bundle.add(DATASET_DIR / scene / results_name, arcname=f"{scene}/results")
            bundle.add(OUT / "masks" / scene, arcname=f"{scene}/initial")
            bundle.add(OUT / "initial_frames" / f"{scene}.jpg",
                       arcname=f"{scene}/initial/initial_frame.jpg")
    with tarfile.open(archive_path) as bundle:
        for scene in scenes:
            stored = bundle.extractfile(f"{scene}/results/traj.npy").read()
            assert stored == (DATASET_DIR / scene / results_name / "traj.npy").read_bytes()
        print(f"ARCHIVED {archive_path}: {len(bundle.getmembers())} members, "
              f"{archive_path.stat().st_size} bytes", flush=True)
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="pilot",
                        help="Results subdirectory name inside each scene folder")
    parser.add_argument("--resize-dim", type=int, default=308,
                        help="Pi3 input resolution; 308 is the resolution the TUM "
                             "baseline reproduced the paper at")
    parser.add_argument("--scenes", nargs="+", default=list(SCENES), choices=SCENES)
    parser.add_argument("--only-eval", action="store_true",
                        help="Re-evaluate and re-archive results already in /tmp")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "Tracking needs the GPU allocation"

    manifest = json.loads((OUT / "initial_frames" / "manifest.json").read_text())
    os.chdir(CHECKOUT)  # The SAM checkpoint and eval.py's GT paths are relative to it.
    stage_dataset(manifest)

    articulation = {scene: articulation_degrees(scene) for scene in args.scenes}
    for scene, stats in articulation.items():
        print(f"ARTICULATION {scene}: {stats['range_deg']:.2f} deg range, "
              f"std {stats['std_deg']:.2f} deg", flush=True)

    timings = {}
    for scene in args.scenes:
        if args.only_eval:
            assert (DATASET_DIR / scene / args.results / "traj.npy").is_file(), scene
        else:
            timings[scene] = track(scene, manifest, args.results, args.resize_dim)
    metrics = {"rows": evaluate(args.scenes, args.results), "timings": timings,
               "articulation_degrees": articulation}

    kvt_arctic_viz.visualize(args.scenes, args.results, metrics)

    provenance = {
        "scenes": args.scenes,
        "results": args.results,
        "resize_dim": args.resize_dim,
        "tracker_args": ["--obj_mode", "--resize_dim", str(args.resize_dim)],
        "offset": 2,
        "articulation_degrees": articulation,
        "prepare_manifest": manifest,
        "initial_masks": {scene: json.loads((OUT / "masks" / scene / "prompt.json").read_text())
                          for scene in args.scenes},
        "superproject_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "kv_tracker_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "kv_tracker_diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "script_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((ROOT / "tools").glob("*arctic*.py"))},
        "note": "Synchronous frame source; timing is not a throughput benchmark",
    }
    archive(args.scenes, args.results, provenance, metrics)
    print("ARCTIC RUN OK")


if __name__ == "__main__":
    main()
