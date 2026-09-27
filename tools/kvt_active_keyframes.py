"""Bounded active-keyframe pilot or leave-one-out sweep; CAMP only, no full run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from kvt_tum_run import periodic_indices, prepare, write_json
from kvt_tum_sweep import archive_inputs

QUANTILES = (.25, .5, .6, .75)
SCENES = ('freiburg3_long_office_household', 'freiburg2_large_with_loop',
          'freiburg2_large_no_loop')


def audit(result):
    config = json.loads((result / 'config.json').read_text())
    rows = json.loads((result / 'active_keyframes.json').read_text())
    assert len(rows) == config['frames'] - 1
    previous = 1
    for i, row in enumerate(rows, 1):
        available = periodic_indices(i, 50, 20) if i >= 2 else [0]
        # Insertion at frame i takes effect only for query i+1.
        assert row['available_ids'] == available
        physical = [0, 0] if len(available) == 1 else available
        assert row['physical_ids'] == physical
        count = row['active_count']
        assert 1 <= count <= len(available) and row['bank_preserved']
        if row['mode'] == 'drop':
            expected = [j for j in available if j != row['drop']] if len(available) > 1 else available
        else:
            expected = [0] + (available[-(count - 1):] if count > 1 else [])
        assert row['selected_ids'] == expected and max(expected) < i
        assert row['selected_physical_ids'] == [j for j in physical if j in expected]
        assert row['previous_count'] == previous and row['frame'] == i
        assert row['decision'] == (i >= 8 and i % 8 == 0)
        threshold = None if i == 1 else float(np.quantile(
            [e['score'] for e in rows[max(0, i - 65):i - 1]], row['quantile']))
        assert row['threshold_start_frame'] == max(1, i - 64)
        assert row['threshold_end_frame'] == i - 1
        if threshold is None:
            assert row['threshold'] is None
        else:
            np.testing.assert_allclose(row['threshold'], threshold)
        mode = row['mode']
        if mode == 'all':
            target = len(available)
        elif mode == 'one':
            target = 1
        elif mode == 'half':
            target = (len(available) + 1) // 2
        elif mode == 'drop':
            target = len(expected)
        elif row['decision']:
            grow = row['score'] > threshold if mode == 'motion' else (i // 8) % 2 == 1
            target = min(len(available), max(1, previous + (1 if grow else -1)))
        else:
            target = previous
        assert count == target
        assert row['active_bytes'] * len(physical) == row['stored_bytes'] * len(row['selected_physical_ids'])
        previous = count
    return dict(mean_active_keyframes=float(np.mean([e['active_count'] for e in rows])),
        mean_available_keyframes=float(np.mean([e['available_count'] for e in rows])),
        active_counts=sorted(set(e['active_count'] for e in rows)),
        count_changes=sum(e['active_count'] != e['previous_count'] for e in rows),
        mean_active_bytes=float(np.mean([e['active_bytes'] for e in rows])),
        mean_stored_bytes=float(np.mean([e['stored_bytes'] for e in rows])),
        note='Active bytes are attention inputs, not resident memory; full bank retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--arms', choices=('pilot', 'loo'), default='pilot')
    args = parser.parse_args()
    assert sys.platform == 'linux' and args.work.is_relative_to('/tmp')
    runner = Path(__file__).with_name('kvt_tum_run.py')
    write_json(args.work / 'active_protocol.json', dict(scenes=SCENES, frames=512,
        quantiles=QUANTILES, score='mean absolute RGB channel-mean change, spatial stride16',
        decision_period=8, score_window=64, initial_count=1, count_step=1,
        admission='native interval50/cap20; rebuild all native keyframes',
        selection='anchor plus newest admitted keyframes; native bootstrap duplicate preserved',
        baseline='original native all available cache; no active controller',
        scope='heuristic access-controller pilot, not learned or GT-optimal; no full run'
        if args.arms == 'pilot' else 'query-only leave-one-keyframe-out headroom; no full run',
        arms=args.arms))
    results = {}
    for scene in SCENES:
        staged = args.work / 'inputs' / scene
        source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
        prepare(source, staged, 308, .02, count=512)
        h = hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        (staged / 'archive.sha256').write_text(f'{h.hexdigest()}  {source}\n')
        archive_inputs(staged, args.out / f'{args.tag}_inputs_{scene}.tar')
        base = dict(scene=scene, scene_dir=str(staged), frames=512, resize_dim=308,
            policy='original', interval=50, cap=20, max_gt_difference=.02,
            evaluate_trajectory=True, save_final_scene=False)

        def run(name, access=None, frames=512):
            target = args.work / 'runs' / scene / name
            target.mkdir(parents=True)
            write_json(target / 'config.json', dict(base, name=name, frames=frames, active_keyframes=access))
            print('RUN', scene, name, frames, flush=True)
            with (target / 'run.log').open('w') as log:
                completed = subprocess.run([sys.executable, str(runner), str(target / 'config.json')],
                                           stdout=log, stderr=subprocess.STDOUT)
            write_json(target / 'process.json', dict(returncode=completed.returncode))
            if completed.returncode:
                print((target / 'run.log').read_text()[-8000:], flush=True)
                raise RuntimeError(f'{scene}/{name} failed; see run.log')
            np.testing.assert_array_equal(np.load(target / 'kf_idx.npy'), periodic_indices(frames, 50, 20))
            metric = json.loads((target / 'metrics.json').read_text())
            if access is not None:
                metric.update(audit(target))
            with np.load(target / 'evaluation.npz') as e:
                metric['translation_rpe_p99_m'] = float(np.quantile(e['rpe_translation_per_pair_m'], .99))
                metric['alignment_scale'] = float(e['alignment_scale'])
            return target, metric

        native, baseline = run('native')
        all_run, all_metrics = run('all', dict(mode='all'))
        np.testing.assert_allclose(np.load(native / 'traj.npy'), np.load(all_run / 'traj.npy'), atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(np.load(native / 'kf_poses.npy'), np.load(all_run / 'kf_poses.npy'))
        if args.arms == 'loo':
            metrics = dict(native=baseline, all=all_metrics)
            for j in periodic_indices(512, 50, 20):
                target, metric = run(f'drop{j}', dict(mode='drop', drop=j))
                np.testing.assert_array_equal(np.load(native / 'kf_poses.npy'), np.load(target / 'kf_poses.npy'))
                assert metric['rebuild_calls'] == baseline['rebuild_calls']
                # Frames before j's first effective query (and the anchor's) are untouched.
                untouched = max(j, 49) + 1
                np.testing.assert_allclose(np.load(native / 'traj.npy')[:untouched],
                                           np.load(target / 'traj.npy')[:untouched], atol=1e-5, rtol=1e-4)
                with np.load(native / 'evaluation.npz') as a, np.load(target / 'evaluation.npz') as b:
                    for key in ('rgb_indices', 'rpe_pair_start_indices'):
                        np.testing.assert_array_equal(a[key], b[key])
                metrics[f'drop{j}'] = metric
            results[scene] = metrics
            write_json(args.work / 'comparison.json', results)
            print('LEAVE-ONE-OUT OK', scene, flush=True)
            continue
        forced, _ = run('forced_gate', dict(mode='alternate'), frames=128)
        forced_rows = json.loads((forced / 'active_keyframes.json').read_text())
        assert any(e['active_count'] > e['previous_count'] for e in forced_rows)
        assert any(e['active_count'] < e['previous_count'] for e in forced_rows)
        prefix, _ = run('q50_prefix', dict(mode='motion', quantile=.5), frames=128)
        print('ACTIVE KEYFRAME FIDELITY/RECOVERY GATE OK', scene, flush=True)
        metrics = dict(native=baseline, all=all_metrics)
        for name, access in [('one', dict(mode='one')), ('half', dict(mode='half'))] + [
                (f'q{int(q * 100)}', dict(mode='motion', quantile=q)) for q in QUANTILES]:
            target, metric = run(name, access)
            # Rebuilds use identical raw keyframes independently of query subset.
            np.testing.assert_array_equal(np.load(native / 'kf_poses.npy'), np.load(target / 'kf_poses.npy'))
            assert metric['rebuild_calls'] == baseline['rebuild_calls']
            if name == 'q50':
                np.testing.assert_allclose(np.load(prefix / 'traj.npy'), np.load(target / 'traj.npy')[:128], atol=1e-5, rtol=1e-4)
                assert json.loads((prefix / 'active_keyframes.json').read_text()) == json.loads((target / 'active_keyframes.json').read_text())[:127]
            with np.load(native / 'evaluation.npz') as a, np.load(target / 'evaluation.npz') as b:
                for key in ('rgb_indices', 'rpe_pair_start_indices'):
                    np.testing.assert_array_equal(a[key], b[key])
            metrics[name] = metric
        results[scene] = metrics
        write_json(args.work / 'comparison.json', results)
        print('ACTIVE KEYFRAME PILOT OK', scene, flush=True)
    (args.work / 'JOB_OK').write_text(f'Active-keyframe {args.arms} complete; artifact review required.\n')


if __name__ == '__main__':
    main()
