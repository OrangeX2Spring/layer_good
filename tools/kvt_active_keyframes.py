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
# Bursts each cover one native keyframe (149, 299, 449); visible fraction 0.6.
OCCLUDER = dict(fraction=.4, value=128, windows=[[100, 150], [250, 300], [400, 450]])
SUBSETS = [(f'{rank}_{size}', dict(mode=size, rank=rank))
           for size in ('two', 'half') for rank in ('newest', 'visible')]
SCENES = ('freiburg3_long_office_household', 'freiburg2_large_with_loop',
          'freiburg2_large_no_loop')


def audit(result, schedule):
    """Replay every query's selection from the native unique keyframe schedule."""
    config = json.loads((result / 'config.json').read_text())
    rows = json.loads((result / 'active_keyframes.json').read_text())
    assert len(rows) == config['frames'] - 1 and schedule[0] == 0
    previous = 1
    visible = {0: rows[0]['available_visible'][0]}
    for i, row in enumerate(rows, 1):
        available = [j for j in schedule if j < i]
        assert row['available_visible'] == [visible[j] for j in available]
        # Insertion at frame i takes effect only for query i+1.
        assert row['available_ids'] == available
        physical = [0, 0] if len(available) == 1 else available
        assert row['physical_ids'] == physical
        count = row['active_count']
        assert 1 <= count <= len(available) and row['bank_preserved']
        if row['mode'] == 'drop':
            expected = [j for j in available if j != row['drop']] if len(available) > 1 else available
        elif row['rank'] == 'visible':
            ranked = sorted(available[1:], key=lambda j: (-visible[j], -j))
            expected = [0] + sorted(ranked[:count - 1])
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
        elif mode == 'two':
            target = min(2, len(available))
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
        visible[i] = row['visible']
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
    parser.add_argument('--arms', choices=('pilot', 'loo', 'occlusion', 'shifted'), default='pilot')
    args = parser.parse_args()
    assert sys.platform == 'linux' and args.work.is_relative_to('/tmp')
    starts = (100, 200) if args.arms == 'shifted' else (0,)
    # no_loop has GT only on frames 0-228 and 2949-3358, so no shifted window is covered;
    # its first keyframe was also neutral in 25946.
    scenes = SCENES[:2] if args.arms == 'shifted' else SCENES
    runner = Path(__file__).with_name('kvt_tum_run.py')
    write_json(args.work / 'active_protocol.json', dict(scenes=scenes, frames=512,
        quantiles=QUANTILES, score='mean absolute RGB channel-mean change, spatial stride16',
        decision_period=8, score_window=64, initial_count=1, count_step=1,
        admission='native interval50/cap20; rebuild all native keyframes',
        selection='anchor plus newest admitted keyframes; native bootstrap duplicate preserved',
        baseline='original native all available cache; no active controller',
        scope='heuristic access-controller pilot, not learned or GT-optimal; no full run'
        if args.arms == 'pilot' else 'query-only leave-one-keyframe-out headroom; no full run',
        arms=args.arms, occluder=OCCLUDER if args.arms == 'occlusion' else None,
        starts=list(starts)))
    results = {}
    for scene, start in [(s, t) for s in scenes for t in starts]:
        # Shifted clips test whether the first keyframe matters by position or content.
        label = scene if start == 0 else f'{scene}_s{start}'
        staged = args.work / 'inputs' / label
        source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
        prepare(source, staged, 308, .02, start=start, count=512)
        h = hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        (staged / 'archive.sha256').write_text(f'{h.hexdigest()}  {source}\n')
        archive_inputs(staged, args.out / f'{args.tag}_inputs_{label}.tar')
        base = dict(scene=scene, start=start, scene_dir=str(staged), frames=512, resize_dim=308,
            policy='original', interval=50, cap=20, max_gt_difference=.02,
            evaluate_trajectory=True, save_final_scene=False,
            occluder=OCCLUDER if args.arms == 'occlusion' else None)

        def run(name, access=None, frames=512):
            target = args.work / 'runs' / label / name
            target.mkdir(parents=True)
            write_json(target / 'config.json', dict(base, name=name, frames=frames, active_keyframes=access))
            print('RUN', label, name, frames, flush=True)
            with (target / 'run.log').open('w') as log:
                completed = subprocess.run([sys.executable, str(runner), str(target / 'config.json')],
                                           stdout=log, stderr=subprocess.STDOUT)
            write_json(target / 'process.json', dict(returncode=completed.returncode))
            if completed.returncode:
                print((target / 'run.log').read_text()[-8000:], flush=True)
                raise RuntimeError(f'{label}/{name} failed; see run.log')
            np.testing.assert_array_equal(np.load(target / 'kf_idx.npy'), periodic_indices(frames, 50, 20))
            metric = json.loads((target / 'metrics.json').read_text())
            if access is not None:
                metric.update(audit(target, periodic_indices(frames, 50, 20)))
            with np.load(target / 'evaluation.npz') as e:
                metric['translation_rpe_p99_m'] = float(np.quantile(e['rpe_translation_per_pair_m'], .99))
                metric['alignment_scale'] = float(e['alignment_scale'])
            return target, metric

        native, baseline = run('native')
        all_run, all_metrics = run('all', dict(mode='all'))
        np.testing.assert_allclose(np.load(native / 'traj.npy'), np.load(all_run / 'traj.npy'), atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(np.load(native / 'kf_poses.npy'), np.load(all_run / 'kf_poses.npy'))
        if args.arms in ('loo', 'occlusion', 'shifted'):
            if args.arms == 'occlusion':
                for e in json.loads((all_run / 'active_keyframes.json').read_text()):
                    occluded = any(a <= e['frame'] < b for a, b in OCCLUDER['windows'])
                    assert (e['visible'] < 1) == occluded and e['available_visible'][0] == 1
            metrics = dict(native=baseline, all=all_metrics)
            for name, access in [(f'drop{j}', dict(mode='drop', drop=j))
                                 for j in periodic_indices(512, 50, 20)] + (
                                    SUBSETS if args.arms == 'occlusion' else []):
                target, metric = run(name, access)
                np.testing.assert_array_equal(np.load(native / 'kf_poses.npy'), np.load(target / 'kf_poses.npy'))
                assert metric['rebuild_calls'] == baseline['rebuild_calls']
                if access['mode'] == 'drop':
                    # Frames before j's first effective query (and the anchor's) are untouched.
                    untouched = max(access['drop'], 49) + 1
                    np.testing.assert_allclose(np.load(native / 'traj.npy')[:untouched],
                                               np.load(target / 'traj.npy')[:untouched], atol=1e-5, rtol=1e-4)
                with np.load(native / 'evaluation.npz') as a, np.load(target / 'evaluation.npz') as b:
                    for key in ('rgb_indices', 'rpe_pair_start_indices'):
                        np.testing.assert_array_equal(a[key], b[key])
                metrics[name] = metric
            results[label] = metrics
            write_json(args.work / 'comparison.json', results)
            print('OCCLUDED TUM OK' if args.arms == 'occlusion' else 'LEAVE-ONE-OUT OK', label, flush=True)
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
    # Occlusion continues with kvt_active_objects.py, which writes the job's JOB_OK.
    marker = 'TUM_OCCLUSION_OK' if args.arms == 'occlusion' else 'JOB_OK'
    (args.work / marker).write_text(f'Active-keyframe {args.arms} complete; artifact review required.\n')


if __name__ == '__main__':
    main()
