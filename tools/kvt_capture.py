"""Record the existing KV-Tracker loop with a synchronous, finite RGB source."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kv_tracker"))
from kv_tracker.sam_interface import SAMInterface
import main as tracker


class OfflineFrames(SAMInterface):
    def __init__(self, manifest, mask_mode):
        super().__init__("cuda", resize_dim=manifest["resize_dim"], obj_mode=True)
        self.manifest = manifest
        self.mask_mode = mask_mode
        self.index = 0
        if mask_mode == "sam2":
            self.init_models()
            mask = cv2.imread(manifest["initial_mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(manifest["initial_mask"])
            self.init_SAM_w_mask(mask > 0)

    def get_rgb_frame(self, idx=0):
        self.index = idx
        path = self.manifest["frames"][idx]["rgb"]
        rgb = cv2.imread(path)
        if rgb is None:
            raise FileNotFoundError(path)
        return rgb[..., ::-1].copy()

    def get_segmentation(self, frame):
        if self.mask_mode == "sam2":
            mask = super().get_segmentation(frame)
        else:
            path = self.manifest["frames"][self.index]["instance"]
            instance = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if instance is None:
                raise FileNotFoundError(path)
            instance = instance[..., 2] if instance.ndim == 3 else instance
            mask = torch.tensor(instance == self.manifest["instance_id"], device=self.device)
        if not mask.any():
            raise ValueError(f"Empty object mask at frame {self.index}")
        return mask

    def __iter__(self):
        for index in range(len(self.manifest["frames"])):
            frame = self.get_frame(index)
            frame["idx"] = index
            yield frame


class Recorder:
    def __init__(self, out):
        self.out = out
        self.counts = {"keyframes": 0, "queries": 0}
        for kind in self.counts:
            (out / kind).mkdir()

    def __call__(self, kind, frame_ids, xyz, poses, confidence, rgb, masks, threshold, latest_rgb):
        xyz = xyz[0].detach().float().cpu().numpy()
        poses = poses[0].detach().float().cpu().numpy()
        confidence = confidence[0, ..., 0].detach().float().cpu().numpy()
        assert xyz.shape == rgb.shape and confidence.shape == masks.shape == xyz.shape[:-1]
        assert len(frame_ids) == len(xyz)
        path = self.out / kind / f"{self.counts[kind]:06d}.npz"
        np.savez_compressed(path, xyz=xyz, poses=poses, confidence=confidence,
                            rgb=rgb, masks=masks, frame_ids=np.asarray(frame_ids),
                            threshold=float(threshold), latest_rgb=latest_rgb)
        # Reload each archive while still on the allocation: detect incomplete exports.
        with np.load(path) as saved:
            for key in saved.files:
                saved[key]
            assert np.array_equal(saved["frame_ids"], frame_ids)
        self.counts[kind] += 1


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mask-mode", choices=["sam2", "annotation"], default="sam2")
    parser.add_argument("--keyframes-from", type=Path, help="Replay a completed baseline's keyframe schedule for a matched mask control")
    parser.add_argument("--keyframe-every", type=int, help="Force a keyframe every N frames instead of the tracker's own 10-degree view-change rule. Not the method; state it as a forced schedule")
    args = parser.parse_args()
    if args.keyframe_every is not None and args.keyframes_from is not None:
        raise ValueError("--keyframe-every and --keyframes-from both set the schedule")
    manifest = json.loads(args.input.read_text())
    if len(manifest["frames"]) < 2:
        raise ValueError("At least two input frames are required")
    keyframe_indices = None
    if args.keyframe_every is not None:
        assert args.keyframe_every > 0
        # Frame 0 is the bootstrap keyframe and is never re-added.
        keyframe_indices = set(range(args.keyframe_every, len(manifest["frames"]),
                                     args.keyframe_every))
        print(f"FORCED SCHEDULE: {len(keyframe_indices) + 1} keyframes "
              f"every {args.keyframe_every} frames")
    if args.keyframes_from is not None:
        baseline = json.loads((args.keyframes_from / "manifest.json").read_text())
        if baseline["input"] != manifest or not (args.keyframes_from / "capture_complete.json").is_file():
            raise ValueError("Matched control requires a complete capture of the identical input")
        last = sorted((args.keyframes_from / "keyframes").glob("*.npz"))[-1]
        with np.load(last) as data:
            keyframe_indices = set(data["frame_ids"].tolist()) - {0}
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT / "kv_tracker")  # SAM checkpoint path is relative to this checkout.
    provenance = {"input": manifest, "mask_mode": args.mask_mode,
                  "superproject_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                  "kv_tracker_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "kv_tracker_diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
                  "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                  "sim3": False, "capture": "all query maps and every keyframe reconstruction",
                  "forced_keyframe_indices": sorted(keyframe_indices) if keyframe_indices is not None else None,
                  "script_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in Path(__file__).parent.glob("kvt_*.py")},
                  "note": "Synchronous export; timing is not a throughput benchmark"}
    (args.out / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    recorder = Recorder(args.out)
    source = OfflineFrames(manifest, args.mask_mode)
    tracker.run_track3r(cfg={"results_path": str(args.out), "que_size": 1},
                       args=["--obj_mode", "--resize_dim", str(manifest["resize_dim"])],
                       frame_source=source, snapshot_callback=recorder,
                       keyframe_indices=keyframe_indices)
    if recorder.counts["queries"] != len(manifest["frames"]) - 1:
        raise RuntimeError(f"Incomplete capture: {recorder.counts}")
    (args.out / "capture_complete.json").write_text(json.dumps(recorder.counts) + "\n")
    print(f"CAPTURE OK: {recorder.counts}")


if __name__ == "__main__":
    main()
