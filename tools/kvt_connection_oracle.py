"""Offline: how much ATE is left if every segment connection were exact (TUM and ARCTIC runs).

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
- design chain + GT scale (commit / progressive / segment): the run's own connection
  rotations and cut-frame positions (old map's pose of b, bridge_<b>.npz), with only each
  segment's scale replaced by its GT metric factor over the same windows; the target for
  any per-segment metric-scale signal. Re-chaining with the run's own scales must
  reproduce traj.npy (asserted).
- GT pin + design scale: the GT pin at b with the run's own cumulative segment scales
  (one global constant absorbed by the Sim(3)); isolates the connection pose.
- hindsight oracle: one Sim(3) per segment fitted on all its frames (the
  between/within split's "within"); a lower bound, not reachable causally.
ATE in metres over the run's GT-valid frames. ARCTIC runs (`--arctic-gt DIR` holding
<scene>.object.npy and <scene>.egocam.dist.npy) use KV-Tracker's eval.load_gt_arctic
convention, re-implemented here; the recomputed actual ATE must match summary.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

COMMIT = 49  # connection commits at b+49 (kv_tracker.reanchor_maps)
ARCTIC = ('box_grab_01', 'ketchup_grab_01', 'espressomachine_grab_01')


def rotvec_matrix(v):
    """Rodrigues: (n, 3) rotation vectors to (n, 3, 3) matrices (scipy from_rotvec)."""
    angle = np.linalg.norm(v, axis=1)[:, None, None]
    k = np.zeros((len(v), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    k = k - k.transpose(0, 2, 1)
    safe = np.where(angle > 1e-12, angle, 1.)
    return (np.eye(3) + np.sin(angle) / safe * k
            + (1 - np.cos(angle)) / safe ** 2 * k @ k)


def arctic_gt(directory, scene, frames):
    """kv_tracker eval.load_gt_arctic: object-to-camera poses, first two frames skipped."""
    obj = np.load(directory / f'{scene}.object.npy', allow_pickle=True)[:, 1:]
    cam = np.load(directory / f'{scene}.egocam.dist.npy', allow_pickle=True).item()
    w2c = np.tile(np.eye(4), (len(obj), 1, 1))
    w2c[:, :3, 3] = cam['T_k_cam_np'].squeeze()
    w2c[:, :3, :3] = cam['R_k_cam_np']
    w2obj = np.tile(np.eye(4), (len(obj), 1, 1))
    w2obj[:, :3, 3] = obj[:, 3:] * 1e-3
    w2obj[:, :3, :3] = rotvec_matrix(obj[:, :3].astype(float))
    gt = np.linalg.inv(np.linalg.inv(w2c) @ w2obj)[2:][:frames]
    assert len(gt) == frames
    return gt


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
    """None when a scale window was undefined (non-positive GT scale fit)."""
    if not np.isfinite(positions).all():
        return None
    s, r, t = umeyama(positions, gt)
    return float(np.sqrt(np.mean(np.sum((s * positions @ r.T + t - gt) ** 2, 1))))


def rechain(run, bounds, local, factor):
    """Global positions from the run's own connections with segment scale factor(s, f)."""
    positions = np.full((len(local), 3), np.nan)
    rotation, origin = np.eye(3), np.zeros(3)
    for s, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        if s > 0:
            with np.load(run / f'bridge_{start:04d}.npz') as bridge:
                old, new = bridge['old_pose'], bridge['new_pose']
            np.testing.assert_allclose(new, local[start], atol=1e-5)
            previous = bounds[s - 1]
            origin = factor(s - 1, start) * rotation @ (old[:3, 3] - local[previous, :3, 3]) + origin
            rotation = rotation @ old[:3, :3] @ new[:3, :3].T
        for f in range(start, end):
            positions[f] = factor(s, f) * rotation @ (local[f, :3, 3] - local[start, :3, 3]) + origin
    return positions


def analyse(run, gt_directory):
    bounds = json.loads((run / 'config.json').read_text())['boundaries']
    local, estimate = np.load(run / 'local_traj.npy'), np.load(run / 'traj.npy')[:, :3, 3]
    scene = [name for name in ARCTIC if run.name.endswith(name)]
    if scene:
        assert gt_directory is not None, 'ARCTIC runs need --arctic-gt'
        index, reference = np.arange(len(local)), arctic_gt(gt_directory, scene[0], len(local))
    else:
        with np.load(run / 'evaluation.npz') as q:
            index, reference = q['rgb_indices'], q['reference']
    assert local.shape == (bounds[-1], 4, 4) and reference.shape == (len(index), 4, 4)
    gt_of = dict(zip(index.tolist(), reference))
    names = ('commit', 'w100', 'w200', 'progressive', 'segment', 'design_scale')
    design = [t['scale'] for t in json.loads((run / 'transforms.json').read_text())]
    placed = {name: np.full((len(local), 3), np.nan) for name in names}
    lengths = []
    gt_scale = dict(commit=[], progressive=[], segment=[])
    undefined = set()
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
            if scale <= 0:  # tracked displacement opposes GT over the window (seen on ARCTIC)
                undefined.add((start, last))
                return np.nan
            return scale

        fixed = dict(commit=start + COMMIT, w100=start + 100, w200=start + 200, segment=end)
        for name, last in fixed.items():
            placed[name][start:end] = scale_until(last) * moved_all[start:end] + gt_of[pin][:3, 3]
        placed['design_scale'][start:end] = design[len(lengths) - 1] * moved_all[start:end] + gt_of[pin][:3, 3]
        for f in range(start, end):
            placed['progressive'][f] = scale_until(max(f, start + COMMIT)) * moved_all[f] + gt_of[pin][:3, 3]
        # Per frame of the segment; the next connection (at end) sees the whole segment.
        gt_scale['commit'].append(np.full(end - start + 1, scale_until(start + COMMIT)))
        gt_scale['segment'].append(np.full(end - start + 1, scale_until(end)))
        gt_scale['progressive'].append(np.array([scale_until(max(f, start + COMMIT))
                                                 for f in range(start, end + 1)]))
        inside = (index >= start) & (index < end)
        s, r, t = umeyama(estimate[index[inside]], reference[inside, :3, 3])
        within.append(np.sum((s * estimate[index[inside]] @ r.T + t - reference[inside, :3, 3]) ** 2, 1))
    gt = reference[:, :3, 3]
    recorded = json.loads((run / 'summary.json').read_text())['metrics']['ate_m']
    assert abs(ate(estimate[index], gt) - recorded) < 1e-4 * max(1, recorded), (ate(estimate[index], gt), recorded)
    # Validation: the run's own cumulative scales reproduce its trajectory up to one Sim(3).
    replay = rechain(run, bounds, local, lambda s, f: design[s])
    assert ate(replay, estimate) < 1e-6, 'offline re-chaining does not reproduce traj.npy'
    chained = {name: rechain(run, bounds, local, lambda s, f, name=name: gt_scale[name][s][f - bounds[s]])
               for name in ('commit', 'progressive', 'segment')}
    return dict(run=str(run), segments=len(bounds) - 1, undefined_scale_windows=len(undefined),
                actual_ate_m=ate(estimate[index], gt),
                segment_length_median=float(np.median(lengths)),
                segments_longer_than_100=int(sum(n > 100 for n in lengths)),
                causal_oracle_ate_m=ate(placed['commit'][index], gt),
                window100_ate_m=ate(placed['w100'][index], gt),
                window200_ate_m=ate(placed['w200'][index], gt),
                progressive_ate_m=ate(placed['progressive'][index], gt),
                pinned_segment_scale_ate_m=ate(placed['segment'][index], gt),
                gt_pin_design_scale_ate_m=ate(placed['design_scale'][index], gt),
                design_chain_gt_commit_scale_ate_m=ate(chained['commit'][index], gt),
                design_chain_gt_progressive_scale_ate_m=ate(chained['progressive'][index], gt),
                design_chain_gt_segment_scale_ate_m=ate(chained['segment'][index], gt),
                hindsight_oracle_ate_m=float(np.sqrt(np.concatenate(within).mean())))


def main(args):
    rows = [analyse(run, args.arctic_gt) for run in args.runs]
    for row in rows:
        row = {k: (float('nan') if v is None else v) for k, v in row.items()}
        print('CONNECTION ORACLE', row['run'], 'segments', row['segments'],
              'median length', row['segment_length_median'], '>100:', row['segments_longer_than_100'],
              'ATE m actual %.4f commit %.4f w100 %.4f w200 %.4f progressive %.4f segment %.4f hindsight %.4f' % (
                  row['actual_ate_m'], row['causal_oracle_ate_m'], row['window100_ate_m'],
                  row['window200_ate_m'], row['progressive_ate_m'], row['pinned_segment_scale_ate_m'],
                  row['hindsight_oracle_ate_m']),
              '| GT pin + design scale %.4f' % row['gt_pin_design_scale_ate_m'],
              '| design chain + GT scale: commit %.4f progressive %.4f segment %.4f' % (
                  row['design_chain_gt_commit_scale_ate_m'], row['design_chain_gt_progressive_scale_ate_m'],
                  row['design_chain_gt_segment_scale_ate_m']))
    if args.out:  # undefined variants are null
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='run directories with local_traj.npy')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--arctic-gt', type=Path, help='directory with ARCTIC raw GT .npy files')
    main(parser.parse_args())
