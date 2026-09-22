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
import random
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
from kvt_online_selector import OnlineSelector

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


def track(scene, manifest, results_name, resize_dim, keyframe_indices=None,
          online_config=None, prefix_frames=None):
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
    assert not list(mask_dir.glob("*.png")), f"Results already exist: {results_dir}"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    source = ArcticFrames("cuda:0", mask_dir, scene_dir=scene_dir, offset=2,
                          obj_mode=True, resize_dim=resize_dim)
    expected = manifest["scenes"][scene]["tracking_frames"]
    assert source.length == expected, (scene, source.length, expected)

    if prefix_frames is not None:
        assert 2 <= prefix_frames <= expected
        source.length = expected = prefix_frames
    torch.cuda.reset_peak_memory_stats()
    selector = None
    cache_policy = None
    if online_config is not None:
        (results_dir / "online_config.json").write_text(json.dumps(online_config, indent=2) + "\n")
        decision_log = (results_dir / "decisions.jsonl").open("w")
        if 'cache_policy' in online_config:
            from kvt_correspondence_selector import CorrespondenceSelector
            selector = CorrespondenceSelector(online_config, decision_log, results_dir)
            cache_policy = selector.cache_policy
        else:
            selector = OnlineSelector(online_config, decision_log)
    recorder = LatestKeyframes(results_dir / "keyframes.npz")
    started = time.perf_counter()
    tracker.run_track3r(cfg={"results_path": str(results_dir), "que_size": 1},
                        args=["--obj_mode", "--resize_dim", str(resize_dim)],
                        frame_source=source, snapshot_callback=recorder,
                        keyframe_indices=keyframe_indices, keyframe_selector=selector,
                        keyframe_cache=cache_policy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if selector is not None:
        selector.close()
        decision_log.close()
        assert selector.last_index == expected - 1
        # Upstream only writes kf_idx.npy after its first insertion.
        if len(selector.inserted) == 1:
            np.save(results_dir / "kf_idx.npy", np.array([0]))
        if cache_policy is None:
            assert np.load(results_dir / "kf_idx.npy").tolist() == selector.inserted
        else:
            assert np.load(results_dir / "kf_idx.npy").tolist() == list(cache_policy.records)
            assert np.load(results_dir / "inserted_kf_idx.npy").tolist() == selector.inserted
        print(f"ONLINE {scene}: {len(selector.inserted)} keyframes, decisions verified", flush=True)
    assert recorder.count, f"{scene}: no keyframe reconstruction was recorded"

    if keyframe_indices is not None:
        # main.py:481 inserts on membership alone and its revert path is dead code
        # (:649 hardcodes revert_kf = False), so the run must have used exactly the
        # requested set. Anything else means the replay is not what was asked for.
        used = set(np.load(results_dir / "kf_idx.npy").tolist()) - {0}
        assert used == set(keyframe_indices), (
            scene, sorted(used ^ set(keyframe_indices)))
        print(f"REPLAYED {scene}: {len(used)} requested keyframes, all inserted",
              flush=True)

    traj = np.load(results_dir / "traj.npy")
    assert traj.shape == (expected, 4, 4), (scene, traj.shape, expected)
    masks_written = len(list(mask_dir.glob("*.png")))
    assert masks_written == expected, (scene, masks_written, expected)
    print(f"\nTRACKED {scene}: {expected} frames in {elapsed:.1f}s "
          f"({expected / elapsed:.2f} frames/s, synchronous export)", flush=True)
    return {"frames": expected, "seconds": round(elapsed, 1),
            "frames_per_second": round(expected / elapsed, 2),
            "keyframes": int(len(np.load(results_dir / "kf_idx.npy"))),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def evaluate(scenes, results_name, prefix_frames=None):
    rows = []
    for scene in scenes:
        traj_gt = kvt_eval.load_gt_arctic(scene)
        traj_est = np.load(DATASET_DIR / scene / results_name / "traj.npy")
        if prefix_frames is not None:
            traj_gt = traj_gt[:prefix_frames]
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
    parser.add_argument("--resize-dim", type=int, default=518,
                        help="Pi3 input resolution. 518 is main.py's default and the "
                             "settled protocol: at 308 box_grab_01 starves Pi3 of "
                             "confident points and lands 60%% above Table 4")
    parser.add_argument("--scenes", nargs="+", default=list(SCENES), choices=SCENES)
    parser.add_argument("--only-eval", action="store_true",
                        help="Re-evaluate and re-archive results already in /tmp")
    parser.add_argument("--keyframes-from", type=Path,
                        help="JSON from kvt_keyframe_sets.py. Replaces the tracker's "
                             "own keyframe decision with a fixed set, through "
                             "run_track3r's existing keyframe_indices argument")
    parser.add_argument("--policy",
                        help="Which policy in --keyframes-from to replay")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip rendering. For a policy sweep, where the comparison "
                             "is the metrics table and only the chosen runs are rendered")
    parser.add_argument("--online-policy", choices=["original", "angular", "interval", "random", "semantic"])
    parser.add_argument("--max-keyframes", type=int, default=32,
                        help="Fixed total cap including bootstrap; 0 only for original")
    parser.add_argument("--angle-degrees", type=float, default=25)
    parser.add_argument("--novelty-threshold", type=float, default=0.1)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix-frames", type=int)
    parser.add_argument("--compare-prefix", help="Earlier results name, same policy/config on a prefix")
    parser.add_argument("--expect-baseline", action="store_true")
    parser.add_argument("--check-masks-from", help="Earlier results name; compare every saved mask")
    parser.add_argument("--cache-policy", choices=["dense", "uniform", "correspondence",
                                                  "semantic_correspondence"],
                        help="Correspondence pilot; requires online interval selection")
    args = parser.parse_args()
    assert not (args.online_policy and args.keyframes_from)
    assert not (args.online_policy and args.only_eval), "Online provenance requires a fresh run"
    assert not args.cache_policy or (args.online_policy == 'interval' and args.no_viz)
    assert args.interval > 0 and 0 < args.angle_degrees < 180
    assert 0 <= args.novelty_threshold <= 2
    assert args.max_keyframes >= 2 or (args.online_policy == "original" and args.max_keyframes == 0)
    assert not args.prefix_frames or args.no_viz
    online_config = None
    if args.online_policy:
        online_config = dict(policy=args.online_policy, max_keyframes=args.max_keyframes,
                             angle_degrees=args.angle_degrees, novelty_threshold=args.novelty_threshold,
                             interval=args.interval, seed=args.seed)
        if args.cache_policy:
            online_config['cache_policy'] = args.cache_policy

    assert torch.cuda.is_available(), "Tracking needs the GPU allocation"

    keyframe_sets = None
    if args.keyframes_from is not None:
        assert args.policy, "--keyframes-from needs --policy"
        sets = json.loads(args.keyframes_from.read_text())
        assert args.policy in sets["policies"], (args.policy, sorted(sets["policies"]))
        keyframe_sets = sets["policies"][args.policy]
        for scene in args.scenes:
            assert scene in keyframe_sets, (args.policy, scene)
            print(f"POLICY {args.policy} {scene}: {len(keyframe_sets[scene])} "
                  f"keyframes to insert", flush=True)

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
            timings[scene] = track(
                scene, manifest, args.results, args.resize_dim,
                keyframe_indices=None if keyframe_sets is None else keyframe_sets[scene],
                online_config=online_config, prefix_frames=args.prefix_frames)
    metrics = {"rows": evaluate(args.scenes, args.results, args.prefix_frames), "timings": timings,
               "articulation_degrees": articulation}

    if not args.no_viz:
        kvt_arctic_viz.visualize(args.scenes, args.results, metrics)

    provenance = {
        "scenes": args.scenes,
        "results": args.results,
        "resize_dim": args.resize_dim,
        "tracker_args": ["--obj_mode", "--resize_dim", str(args.resize_dim)],
        "offset": 2,
        "keyframe_policy": args.online_policy or args.policy,
        "online_config": online_config,
        "prefix_frames": args.prefix_frames,
        "keyframe_sets": keyframe_sets,
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
                          for path in sorted([*(ROOT / "tools").glob("*arctic*.py"),
                                              ROOT / "tools/kvt_online_selector.py"])},
        "note": "Synchronous frame source; timing is not a throughput benchmark",
    }
    # Archive before validation: a failed gate must still leave reviewable evidence.
    archive(args.scenes, args.results, provenance, metrics)
    if args.expect_baseline:
        assert args.online_policy == "original" and args.max_keyframes == 0 and args.prefix_frames is None
        reference = dict(zip(SCENES, [0.1825401102245989, 0.29433964866875045, 0.15025625687661778]))
        for row in metrics["rows"]:
            assert abs(row["ate_m"] - reference[row["scene"]]) < 0.0005, row
        print("BASELINE GATE OK")
    for scene in args.scenes:
        current = DATASET_DIR / scene / args.results
        if args.check_masks_from:
            reference = DATASET_DIR / scene / args.check_masks_from / "sam_masks"
            for mask in (current / "sam_masks").glob("*.png"):
                assert mask.read_bytes() == (reference / mask.name).read_bytes(), (scene, mask.name)
            print(f"MASK GATE OK {scene}")
        if args.compare_prefix:
            reference = DATASET_DIR / scene / args.compare_prefix
            assert (reference / "online_config.json").read_bytes() == (current / "online_config.json").read_bytes()
            for mask in (reference / "sam_masks").glob("*.png"):
                assert mask.read_bytes() == (current / "sam_masks" / mask.name).read_bytes(), (scene, mask.name)
            prefix = np.load(reference / "traj.npy")
            full = np.load(current / "traj.npy")[:len(prefix)]
            assert np.allclose(full, prefix, rtol=1e-4, atol=1e-4), (scene, float(np.max(np.abs(full-prefix))))
            before = [json.loads(line) for line in (reference / "decisions.jsonl").read_text().splitlines()]
            after = [json.loads(line) for line in (current / "decisions.jsonl").read_text().splitlines()][:len(before)]
            assert len(before) == len(after) == len(prefix) - 1
            for a, b in zip(before, after):
                for key in ["frame", "cache_frame_ids", "candidate", "capped", "selected"]:
                    assert a[key] == b[key], (scene, key, a, b)
            print(f"PREFIX GATE OK {scene}: {len(prefix)} frames")
    print("ARCTIC RUN OK")


if __name__ == "__main__":
    main()
