"""Render three confidence diagnostics from one frozen KV-Tracker reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def render(points, colors, alpha, center, rotation, span, size=640):
    """One-pixel points, depth-sorted alpha compositing (nearest first)."""
    assert points.shape == colors.shape and points.shape[1] == 3
    assert alpha.shape == (len(points),)
    p = (points - center) @ rotation
    uv = np.floor((p[:, :2] / span + 0.5) * size).astype(int)
    valid = ((uv >= 0) & (uv < size)).all(1)
    indices = np.flatnonzero(valid)
    pixels = uv[valid, 1] * size + uv[valid, 0]
    order = np.lexsort((p[valid, 2], pixels))
    pixels, indices = pixels[order], indices[order]
    image = np.full((size * size, 3), 24., dtype=np.float64)
    if len(indices):
        unique, starts, counts = np.unique(pixels, return_index=True, return_counts=True)
        # Segmented exclusive product of (1-alpha): all deeper points remain visible
        # through translucent foreground points. Clipping avoids log(0) at opacity 1.
        a = alpha[indices].astype(np.float64)
        logs = np.log(np.maximum(1 - a, 1e-15))
        cumulative = np.concatenate(([0.], np.cumsum(logs)))
        before = cumulative[:-1] - np.repeat(cumulative[starts], counts)
        weights = a * np.exp(before)
        foreground = np.add.reduceat(colors[indices] * weights[:, None], starts)
        transmission = np.exp(cumulative[starts + counts] - cumulative[starts])
        image[unique] = foreground + 24 * transmission[:, None]
    return np.clip(image.reshape(size, size, 3), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--keyframes', type=Path, required=True)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.evaluation.read_text())
    with np.load(args.keyframes) as data, np.load(args.alignment) as al:
        _, first = np.unique(data['frame_ids'], return_index=True)
        first.sort()
        masks, confidence = data['masks'][first], data['confidence'][first]
        native = float(data['threshold'])
        keep = masks & (confidence > native)
        xyz = data['xyz'][first][keep]
        xyz = float(al['scale']) * xyz @ al['rotation'].T + al['translation']
        rgb = data['rgb'][first][keep]
        conf = confidence[keep]
        residual = float(al['residual_m'])
    assert np.isfinite(xyz).all() and np.isfinite(conf).all()
    assert ((conf >= 0) & (conf <= 1)).all()
    assert residual <= cfg['max_alignment_rmse_m'], 'Alignment gate failed'
    roi = np.asarray(cfg['roi_to_reference'])
    extent = np.asarray(cfg['roi_extent_m'])
    local = (xyz - roi[:3, 3]) @ roi[:3, :3]
    cavity = (np.abs(local) < extent / 2).all(1)
    removed = conf < 0.7
    sacrificed = removed & ~cavity
    counts = {'native_retained': len(xyz), 'cavity': int(cavity.sum()),
              'filtered_retained': int((~removed).sum()),
              'cavity_remaining': int((cavity & ~removed).sum()),
              'removed_non_cavity': int(sacrificed.sum())}
    # This visualization is specifically the measured 24-keyframe teapot result.
    assert len(first) == 24 and counts['native_retained'] == 653005, counts
    assert counts['cavity'] == 139 and counts['cavity_remaining'] == 6, counts
    args.out.mkdir(parents=True, exist_ok=False)
    bgr = rgb[:, ::-1]
    confidence_bgr = cv2.applyColorMap(np.round(conf * 255).astype(np.uint8),
                                      cv2.COLORMAP_TURBO).reshape(-1, 3)
    marked = bgr.copy()
    marked[sacrificed] = (0, 255, 0)
    marked[cavity] = (0, 0, 255)
    variants = [('01_filtered', ~removed, bgr, np.ones(len(xyz)),
                 'Confidence >= 0.70 | original object colors',
                 f"{counts['filtered_retained']:,} points kept | cavity: 139 -> 6"),
                ('02_confidence', np.ones(len(xyz), bool), confidence_bgr, conf,
                 'Confidence color + transparency | opacity = confidence',
                 'All originally retained points | overlapping points accumulate opacity'),
                ('03_removal_overlay', np.ones(len(xyz), bool), marked, np.ones(len(xyz)),
                 'RED: cavity | GREEN: removed non-cavity | original colors: kept',
                 f"139 red points | {counts['removed_non_cavity']:,} green points | red takes priority")]
    # Same camera for every variant; initial view looks along the cavity's axis.
    center = (xyz.min(0) + xyz.max(0)) / 2
    span = float(np.linalg.norm(xyz.max(0) - xyz.min(0)) * 1.08)
    fps, frames = 12, 180
    for name, selected, colors, opacity, title, subtitle in variants:
        path = args.out / f'{name}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 800))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot open video encoder: {path}')
        for frame in range(frames):
            angle = np.deg2rad(35) * np.sin(2 * np.pi * frame / (frames - 1))
            c, s = np.cos(angle), np.sin(angle)
            turn = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            rotation = roi[:3, :3] @ turn
            panel = np.full((800, 1280, 3), 24, np.uint8)
            for column, (focus, width) in enumerate(((center, span), (roi[:3, 3], 0.075))):
                panel[:640, column * 640:(column + 1) * 640] = render(
                    xyz[selected], colors[selected], opacity[selected], focus, rotation, width)
            for text, x, y, scale in [(title, 18, 676, 0.65), (subtitle, 18, 707, 0.56),
                                     ('Whole object', 18, 30, 0.65),
                                     ('Cavity close-up | 75 mm field of view', 658, 30, 0.55),
                                     ('Frozen 24-keyframe geometry | camera sweep +/-35 deg | one-pixel points',
                                      18, 787, 0.52)]:
                cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)
            if name == '02_confidence':
                bar = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_TURBO)
                panel[726:742, 18:530] = cv2.resize(bar, (512, 16))
                cv2.putText(panel, '0.0                 confidence                 1.0', (18, 760),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
            writer.write(panel)
            if frame in (0, 45, 135):
                assert cv2.imwrite(str(args.out / f'{name}_{frame:03d}.png'), panel)
            if frame % 30 == 0:
                print(f'{name}: {frame}/{frames}', flush=True)
        writer.release()
        reader = cv2.VideoCapture(str(path))
        decoded = 0
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            assert frame.shape == (800, 1280, 3)
            decoded += 1
        reader.release()
        assert decoded == frames, (path, decoded)
        print(f'VERIFIED: {path.name}: {decoded} frames', flush=True)
    provenance = {'counts': counts, 'native_threshold': native, 'new_threshold': 0.7,
                  'alignment_rmse_m': residual, 'opacity': 'confidence (linear)',
                  'fps': fps, 'frames': frames, 'evaluation': cfg,
                  'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (args.keyframes, args.alignment, args.evaluation, Path(__file__))}}
    (args.out / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('CONFIDENCE VIDEOS OK', counts, flush=True)


if __name__ == '__main__':
    main()
