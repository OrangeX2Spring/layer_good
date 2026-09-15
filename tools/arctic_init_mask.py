"""Generate KV-Tracker's initial SAM 2 masks for the ARCTIC S01 pilot, for review.

Run inside a CAMP GPU allocation via `bash tools/kvt_run.sh arctic-mask <prompt>...`.
Each prompt is `<scene>:point:<x>,<y>[:<label>]` (label 1 foreground, 0 background,
repeatable) or `<scene>:box:<x0>,<y0>,<x1>,<y1>`. The prompt is applied to the
initialization frame `arctic_prepare_subset.py` exported, which is the loader's
offset=2 frame, and the mask is produced by the same SAM 2 call the tracker makes.
Writes the `init_mask.png` the loader reads plus an overlay to confirm visually.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kv_tracker"))
from kv_tracker.sam_interface import SAMInterface

OUT = Path("/mnt/projects/gr/3DRecon/kvt_arctic_out")


class InitFrame(SAMInterface):
    """One still frame, so SAM 2 sees exactly the pixels the tracker starts from."""

    def __init__(self, path):
        super().__init__("cuda", obj_mode=True)
        bgr = cv2.imread(str(path))
        assert bgr is not None, f"Unreadable initialization frame: {path}"
        self.rgb = bgr[..., ::-1].copy()
        self.height, self.width = self.rgb.shape[:2]
        self.init_models()

    def get_rgb_frame(self, idx=0):
        return self.rgb


def parse_prompt(spec, prompts):
    fields = spec.split(":")
    assert len(fields) in (3, 4), f"Expected <scene>:point|box:<values>[:<label>]: {spec}"
    scene, kind = fields[0], fields[1]
    values = [int(v) for v in fields[2].split(",")]
    prompt = prompts.setdefault(scene, {"points": [], "labels": [], "box": None})
    if kind == "point":
        assert len(values) == 2, f"A point is <x>,<y>: {spec}"
        label = int(fields[3]) if len(fields) == 4 else 1
        assert label in (0, 1), f"A point label is 0 or 1: {spec}"
        prompt["points"].append(values)
        prompt["labels"].append(label)
    elif kind == "box":
        assert len(values) == 4, f"A box is <x0>,<y0>,<x1>,<y1>: {spec}"
        assert len(fields) == 3, f"A box takes no label: {spec}"
        assert values[0] < values[2] and values[1] < values[3], f"Empty box: {spec}"
        assert prompt["box"] is None, f"Two boxes for {scene}"
        prompt["box"] = values
    else:
        raise SystemExit(f"Unknown prompt kind {kind!r}: {spec}")


def segment(scene, prompt):
    frame = InitFrame(OUT / "initial_frames" / f"{scene}.jpg")
    if prompt["box"] is not None:
        assert not prompt["points"], f"{scene}: give either a box or points, not both"
        x0, y0, x1, y1 = prompt["box"]
        assert 0 <= x0 and 0 <= y0 and x1 < frame.width and y1 < frame.height, (
            f"{scene}: box outside the {frame.width}x{frame.height} frame")
        frame.init_SAM_w_bbox(np.array([[x0, y0], [x1, y1]], dtype=np.float32))
    else:
        assert prompt["points"], f"{scene}: no prompt given"
        assert 1 in prompt["labels"], f"{scene}: at least one foreground point is required"
        for x, y in prompt["points"]:
            assert 0 <= x < frame.width and 0 <= y < frame.height, (
                f"{scene}: point ({x},{y}) outside the {frame.width}x{frame.height} frame")
        frame.init_SAM(np.array(prompt["points"], dtype=np.float32),
                       np.array(prompt["labels"], dtype=np.int32), frame.rgb)
    # The same call the loader makes on its first frame, so the reviewed mask is
    # the mask the run starts from.
    mask = frame.get_segmentation(frame.rgb).cpu().numpy()
    assert mask.shape == (frame.height, frame.width), (mask.shape, frame.height, frame.width)
    assert mask.any(), f"{scene}: SAM 2 returned an empty mask for this prompt"
    return frame, mask


def write_review(scene, frame, mask, prompt):
    directory = OUT / "masks" / scene
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "init_mask.png"), mask.astype(np.uint8) * 255)

    overlay = frame.rgb.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    if prompt["box"] is not None:
        x0, y0, x1, y1 = prompt["box"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2)
    for (x, y), label in zip(prompt["points"], prompt["labels"]):
        cv2.circle(overlay, (x, y), 7, (255, 0, 0) if label else (0, 0, 255), -1)
    cv2.imwrite(str(directory / "init_overlay.png"), overlay[..., ::-1])

    rows, columns = np.where(mask)
    record = {
        "scene": scene,
        "prompt": prompt,
        "frame": str(OUT / "initial_frames" / f"{scene}.jpg"),
        "frame_size": [frame.width, frame.height],
        "mask_pixels": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
        "mask_bbox": [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (directory / "prompt.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="+",
                        help="<scene>:point:<x>,<y>[:<label>] or <scene>:box:<x0>,<y0>,<x1>,<y1>")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "Initial masks need the GPU allocation"

    prompts = {}
    for spec in args.prompts:
        parse_prompt(spec, prompts)

    for scene, prompt in prompts.items():
        frame, mask = segment(scene, prompt)
        record = write_review(scene, frame, mask, prompt)
        print(f"MASK {scene}: {record['mask_pixels']} px "
              f"({100 * record['mask_fraction']:.2f}% of {record['frame_size']}), "
              f"bbox={record['mask_bbox']}", flush=True)

    print("MASK OK:", OUT / "masks")
    print("Review init_overlay.png for every sequence before tracking.")


if __name__ == "__main__":
    main()
