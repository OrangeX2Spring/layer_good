"""Offline: how much ATE is left if every segment connection were exact (TUM runs).

CPU artifact analysis only; imports no project/model modules. GT replaces the
connections only, never the tracking. Per run directory of tools/kvt_reanchor_maps.py
(local_traj.npy, config.json boundaries, evaluation.npz):

- actual: the run's own global trajectory (traj.npy), one Sim(3) to GT.
- causal oracle: each segment's local poses are placed with what a connection could
  know at its commit time b+49: rotation and position pinned on GT at the first
  GT-valid frame >= b, scale from GT camera displacements over GT-valid frames in
  [b, b+49] (or to the segment end). Then one Sim(3) over the whole trajectory.
- fixed windows 100 / 200: the same pin, scale from GT over b..b+W (capped at the
  segment end; most office segments are shorter, so these approach the segment scale).
- progressive (causal): the same pin; frame f uses the scale fitted on GT over
  b..max(f, b+49), so each output uses only frames up to its own time (frames before
  b+49 are already 49-frame delayed in the design) and is never rewritten.
- pinned, segment scale (non-causal diagnostic): the same pin at b, scale from GT
  displacements over the whole segment; separates connection-scale noise from drift
  inside the segment after b.
- hindsight oracle: one Sim(3) per segment fitted on all its frames (the
  between/within split's "within"); a lower bound, not reachable causally.
ATE in metres over the run's GT-valid frames.
"""
import argparse
import json
from pathlib import Path

import numpy as np

COMMIT = 49  # connection commits at b+49 (kv_tracker.reanchor_maps)


def umeyama(x, y):
    """Sim(3) (scale, rotation, translation) mapping points x (n, 3) onto y (n, 3)."""
    assert x.shape == y.shape and x.ndim == 2 and x.shape[1] == 3 and len(x) >= 3
    mx, my = x.mean(0), y.mean(0)
    u, d, vt = np.linalg.svd((y - my).T @ (x - mx) / len(x))
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    rotation = (u * sign) @ vt
    scale = (d * sign).sum() / ((x - mx) ** 2).sum(1).mean()
    return scale, rotation, my - scale * rotation @ mx


def ate(positions, gt):
    s, r, t = umeyama(positions, gt)
    return float(np.sqrt(np.mean(np.sum((s * positions @ r.T + t - gt) ** 2, 1))))


def analyse(run):
    bounds = json.loads((run / 'config.json').read_text())['boundaries']
    local, estimate = np.load(run / 'local_traj.npy'), np.load(run / 'traj.npy')[:, :3, 3]
    with np.load(run / 'evaluation.npz') as q:
        index, reference = q['rgb_indices'], q['reference']
    assert local.shape == (bounds[-1], 4, 4) and reference.shape == (len(index), 4, 4)
    gt_of = dict(zip(index.tolist(), reference))
    names = ('commit', 'w100', 'w200', 'progressive', 'segment')
    placed = {name: np.full((len(local), 3), np.nan) for name in names}
    lengths = []
    within = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        frames = [f for f in range(start, end) if f in gt_of]
        pin = frames[0]
        lengths.append(end - start)
        rotation = gt_of[pin][:3, :3] @ local[pin, :3, :3].T
        moved_all = (local[:, :3, 3] - local[pin, :3, 3]) @ rotation.T

        def scale_until(last):
            window = [f for f in frames if f <= last]
            target = np.stack([gt_of[f][:3, 3] for f in window]) - gt_of[pin][:3, 3]
            scale = float((moved_all[window] * target).sum() / (moved_all[window] ** 2).sum())
            assert scale > 0, (start, last, scale)
            return scale

        fixed = dict(commit=start + COMMIT, w100=start + 100, w200=start + 200, segment=end)
        for name, last in fixed.items():
            placed[name][start:end] = scale_until(last) * moved_all[start:end] + gt_of[pin][:3, 3]
        for f in range(start, end):
            placed['progressive'][f] = scale_until(max(f, start + COMMIT)) * moved_all[f] + gt_of[pin][:3, 3]
        inside = (index >= start) & (index < end)
        s, r, t = umeyama(estimate[index[inside]], reference[inside, :3, 3])
        within.append(np.sum((s * estimate[index[inside]] @ r.T + t - reference[inside, :3, 3]) ** 2, 1))
    gt = reference[:, :3, 3]
    return dict(run=str(run), segments=len(bounds) - 1,
                actual_ate_m=ate(estimate[index], gt),
                segment_length_median=float(np.median(lengths)),
                segments_longer_than_100=int(sum(n > 100 for n in lengths)),
                causal_oracle_ate_m=ate(placed['commit'][index], gt),
                window100_ate_m=ate(placed['w100'][index], gt),
                window200_ate_m=ate(placed['w200'][index], gt),
                progressive_ate_m=ate(placed['progressive'][index], gt),
                pinned_segment_scale_ate_m=ate(placed['segment'][index], gt),
                hindsight_oracle_ate_m=float(np.sqrt(np.concatenate(within).mean())))


def main(args):
    rows = [analyse(run) for run in args.runs]
    for row in rows:
        print('CONNECTION ORACLE', row['run'], 'segments', row['segments'],
              'median length', row['segment_length_median'], '>100:', row['segments_longer_than_100'],
              'ATE m actual %.4f commit %.4f w100 %.4f w200 %.4f progressive %.4f segment %.4f hindsight %.4f' % (
                  row['actual_ate_m'], row['causal_oracle_ate_m'], row['window100_ate_m'],
                  row['window200_ate_m'], row['progressive_ate_m'], row['pinned_segment_scale_ate_m'],
                  row['hindsight_oracle_ate_m']))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='run directories with local_traj.npy')
    parser.add_argument('--out', type=Path)
    main(parser.parse_args())
