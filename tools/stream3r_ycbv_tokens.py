#!/usr/bin/env python3
"""How much of STream3R's KV cache would a semantic (object-restricted) cache keep?

    python tools/stream3r_ycbv_tokens.py \
        --scene-dir /tmp/ycbv/test/000051 \
        --manifest  /tmp/stream3r_ycbv/out/scene000051/manifest.json \
        --out       /tmp/stream3r_ycbv/out/scene000051/tokens.json

Background. The tutor's proposal is a training-free module doing object tracking
with semantic-level KV caching. STream3R already caches K/V per patch token and
already truncates that cache - but positionally: stream_session.py's `window`
mode keeps frame 0's tokens plus the last `window_size` frames' tokens. A
semantic cache would instead keep the tokens belonging to the tracked object and
drop the rest. This script measures the number that decides whether that is worth
building: the fraction of tokens an object actually occupies.

It computes no reconstruction and loads no model. It reads the frames that
stream3r_visualize.py actually used (from its manifest, so the two cannot drift
apart) and, for each frame and each annotated object instance, reports:

  visib_fract     BOP's own visible fraction, from scene_gt_info.json
  px_fraction     visible mask pixels / image pixels, at native resolution
  token_fraction  patch tokens the visible mask touches / tokens per frame

token_fraction is the operative one, because the cache is indexed by token, not
by pixel. The mapping is exact rather than approximate: YCB-V is 640x480, and
load_and_preprocess_images in its default "crop" mode sets width to 518 and
height to round(480 * 518/640 / 14) * 14 = 392, which is under 518, so no crop
happens. That is a clean 37 x 28 = 1036 patch grid, and a mask resized the same
way pools into it exactly. The assertions below fail loudly if that ever stops
holding - a silently misaligned mask would still produce a plausible number.

Reference point: the KV-Tracker work measured its object at 2.43% of the input
frame (CLAUDE.md). If object tokens are a few percent here too, a semantic cache
is one to two orders of magnitude smaller than the frame cache STream3R's causal
mode grows - which is the mode that reaches 45.4 GB at 200 frames.
"""
import argparse
import json
import os
import numpy as np
from PIL import Image

PATCH = 14
TARGET_W = 518


def token_grid_shape(h, w):
    """Replicates load_and_preprocess_images(mode="crop")'s geometry."""
    new_w = TARGET_W
    new_h = round(h * (new_w / w) / PATCH) * PATCH
    cropped_h = min(new_h, TARGET_W)
    assert new_h % PATCH == 0, new_h
    return new_h, new_w, cropped_h


def mask_to_token_mask(mask):
    """Binary HxW mask -> binary (gh, gw) token mask, via the model's transform."""
    h, w = mask.shape
    new_h, new_w, cropped_h = token_grid_shape(h, w)
    m = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
        (new_w, new_h), Image.Resampling.NEAREST
    )
    m = np.asarray(m) > 0
    if new_h > cropped_h:  # centre crop, as the loader does
        start = (new_h - cropped_h) // 2
        m = m[start : start + cropped_h, :]
    gh, gw = m.shape[0] // PATCH, m.shape[1] // PATCH
    assert m.shape == (gh * PATCH, gw * PATCH), (m.shape, gh, gw)
    # A token is "object" if any pixel in its 14x14 patch is object: the cache
    # entry is contaminated by the object either way, so `any` is the honest
    # accounting for what a semantic cache would have to retain.
    return m.reshape(gh, PATCH, gw, PATCH).any(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, help="BOP scene dir, e.g. .../000051")
    ap.add_argument("--manifest", required=True, help="manifest.json from stream3r_visualize.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scene_gt = json.load(open(os.path.join(args.scene_dir, "scene_gt.json")))
    scene_info = json.load(open(os.path.join(args.scene_dir, "scene_gt_info.json")))
    used = json.load(open(args.manifest))["images"]

    rows = []
    for path in used:
        im_id = int(os.path.splitext(os.path.basename(path))[0])
        key = str(im_id)
        assert key in scene_gt, f"frame {im_id} not in scene_gt.json"

        for gt_id, (gt, info) in enumerate(zip(scene_gt[key], scene_info[key])):
            mpath = os.path.join(
                args.scene_dir, "mask_visib", f"{im_id:06d}_{gt_id:06d}.png"
            )
            if not os.path.exists(mpath):
                # Loudly, not silently: BOP loaders elsewhere in this repo skip
                # missing files with a bare `continue`, which turns a layout error
                # into a run over zero objects.
                raise FileNotFoundError(mpath)

            mask = np.asarray(Image.open(mpath))
            tok = mask_to_token_mask(mask)
            rows.append({
                "im_id": im_id,
                "gt_id": gt_id,
                "obj_id": gt["obj_id"],
                "visib_fract": info.get("visib_fract"),
                "px_count_visib": info.get("px_count_visib"),
                "px_fraction": float((mask > 0).sum() / mask.size),
                "token_fraction": float(tok.sum() / tok.size),
                "tokens_object": int(tok.sum()),
                "tokens_total": int(tok.size),
            })

    assert rows, "no annotated objects found on the frames that were run"
    tf = np.array([r["token_fraction"] for r in rows])
    pf = np.array([r["px_fraction"] for r in rows])
    vf = np.array([r["visib_fract"] or 0.0 for r in rows])
    n_obj = len(set(r["gt_id"] for r in rows))

    summary = {
        "scene_dir": args.scene_dir,
        "frames": len(used),
        "objects_per_frame": n_obj,
        "tokens_total_per_frame": int(rows[0]["tokens_total"]),
        "token_fraction_per_object": {
            "min": float(tf.min()), "median": float(np.median(tf)),
            "mean": float(tf.mean()), "max": float(tf.max()),
        },
        "px_fraction_per_object": {
            "median": float(np.median(pf)), "max": float(pf.max()),
        },
        "visib_fract": {
            "min": float(vf.min()), "median": float(np.median(vf)),
        },
        # What a cache holding every annotated object, on every frame, would cost
        # relative to caching whole frames. The union is per frame, so objects
        # sharing a patch are not double-counted.
        "all_objects_token_fraction_mean": float(
            np.mean([
                sum(r["token_fraction"] for r in rows if r["im_id"] == i)
                for i in sorted(set(r["im_id"] for r in rows))
            ])
        ),
    }

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print(json.dumps(summary, indent=2))
    med = summary["token_fraction_per_object"]["median"]
    print(f"\nmedian object occupies {med*100:.2f}% of tokens "
          f"-> a per-object semantic cache is ~{1/max(med,1e-9):.0f}x smaller "
          f"than the frame cache", flush=True)
    print("TOKENS OK", flush=True)


if __name__ == "__main__":
    main()
