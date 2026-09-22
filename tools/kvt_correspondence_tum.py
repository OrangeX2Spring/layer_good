"""TUM scene correspondence gate and complete-sequence evaluation on CAMP."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from kvt_tum_run import prepare, write_json, periodic_indices
from kvt_tum_sweep import (LONG_SCENES, archive_directory, archive_inputs,
                           compare_prefix, release_page_cache)


def run_scene(args, scene):
    inputs = args.work / 'inputs' / scene
    runs = args.work / 'runs' / scene
    runs.mkdir(parents=True)
    archive = args.datasets / f'rgbd_dataset_{scene}.zip'
    manifest = prepare(archive, inputs, 308, .02)
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    (inputs / 'archive.sha256').write_text(f'{digest.hexdigest()}  {archive}\n')
    input_tar = args.out / f'{args.tag}_inputs_{scene}.tar'
    archive_inputs(inputs, input_tar)
    release_page_cache(inputs, input_tar)
    base = dict(scene=scene, scene_dir=str(inputs), resize_dim=308, interval=50,
                cap=20, layer='encoder', max_gt_difference=.02,
                evaluate_trajectory=False)
    write_json(args.work / f'protocol_{scene}.json', dict(base=base, stage=args.stage,
        variants=['dense', 'uniform', 'correspondence'], eviction='anchor + FIFO',
        masks='all true; no target or entity identity on TUM',
        geometry='model pointmaps from same dense rebuild',
        gate='native identity through 128 frames; prefix across replacements'))

    def run_one(name, policy, frames, patch_policy=None, evaluate=False):
        result = runs / name
        result.mkdir()
        config = dict(base, name=name, policy=policy, frames=frames,
                      evaluate_trajectory=evaluate)
        if patch_policy is not None:
            config['patch_policy'] = patch_policy
        write_json(result / 'config.json', config)
        print('RUN', scene, name, frames, flush=True)
        started = time.time()
        with (result / 'run.log').open('w') as stream:
            completed = subprocess.run([sys.executable,
                str(Path(__file__).with_name('kvt_tum_run.py')), str(result / 'config.json')],
                stdout=stream, stderr=subprocess.STDOUT)
        write_json(result / 'process.json', dict(start_unix=started,
            end_unix=time.time(), returncode=completed.returncode))
        if completed.returncode:
            print((result / 'run.log').read_text()[-8000:], flush=True)
            raise RuntimeError(f'{name} failed with exit {completed.returncode}')
        return result

    if args.stage == 'gate':
        reference = run_one('native_128', 'periodic', 128)
        unchanged = run_one('dense_128', 'correspondence', 128, 'dense')
        a, b = np.load(reference / 'traj.npy'), np.load(unchanged / 'traj.npy')
        np.testing.assert_array_equal(a, b)
        print('FULL RETENTION GATE OK', scene, flush=True)
    reports = {}
    for mode in ('dense', 'uniform', 'correspondence'):
        if args.stage == 'gate':
            prefix = run_one(f'{mode}_prefix', 'correspondence', 1100, mode)
            length = 1150
        else:
            length = manifest['frames']
        result = run_one(mode, 'correspondence', length, mode,
                         evaluate=args.stage == 'full')
        if args.stage == 'gate':
            compare_prefix(prefix, result)
        assert np.isfinite(np.load(result / 'traj.npy')).all()
        events = [json.loads(line) for line in (result / 'cache_events.jsonl').read_text().splitlines()]
        expected = periodic_indices(length, 50, length)
        assert [row['frame'] for row in events] == expected
        np.testing.assert_array_equal(np.load(result / 'inserted_kf_idx.npy'), expected)
        retained = [0] + expected[-19:] if len(expected) > 20 else expected
        assert np.load(result / 'kf_idx.npy').tolist() == retained
        assert all(len(row['retained_frame_ids']) <= 20 and row['retained_frame_ids'][0] == 0
                   for row in events)
        assert sum(len(row['evicted']) for row in events) == max(0, len(expected) - 20)
        patch_count = len(events[0]['patch_indices']['0'])
        for row in events:
            for frame, picked in row['patch_indices'].items():
                count = patch_count if frame == '0' or mode == 'dense' else (patch_count + 1) // 2
                assert len(picked) == len(set(picked)) == count
            if mode != 'dense' and row['frame']:
                assert row['query_cache_bytes'] < row['dense_cache_bytes']
            for link in row['links']:
                assert link['previous_frame'] < row['frame']
                assert link['previous_patch'] in row['patch_indices'][str(link['previous_frame'])]
        if args.stage == 'gate':
            before = [json.loads(line) for line in (prefix / 'cache_events.jsonl').read_text().splitlines()]
            for a, b in zip(before, events):
                for key in ('frame', 'retained_frame_ids', 'evicted', 'patch_indices', 'links'):
                    assert a[key] == b[key], (scene, mode, key)
        with np.load(result / 'final_scene.npz') as state:
            assert state['frame_ids'].tolist() == retained
        reports[mode] = json.loads((result / 'metrics.json').read_text())
        reports[mode]['matched_kept'] = sum(row['matched_kept'] for row in events)
    write_json(args.work / f'review_{scene}.json', dict(stage=args.stage,
        scene=scene, frames=length, conditions=reports,
        mechanism_active=reports['correspondence']['matched_kept'] > 0))
    for result in runs.iterdir():
        destination = args.out / f'{args.tag}_{scene}_{result.name}.tar'
        archive_directory(result, destination)
        release_page_cache(result, destination)
    write_json(args.work / f'GATES_OK_{scene}.json', dict(stage=args.stage,
        frames=length, full_retention_exact=args.stage == 'gate',
        bounded_retention=True, prefix_consistent=args.stage == 'gate',
        mechanism_active=reports['correspondence']['matched_kept'] > 0))
    print('SCENE GATES OK', scene, args.stage, flush=True)


def main(args):
    scenes = args.scene.split(',')
    assert all(scene in LONG_SCENES for scene in scenes)
    for scene in scenes:
        run_scene(args, scene)
    (args.work / 'JOB_OK').write_text('Correspondence conditions completed\n')
    print('CORRESPONDENCE GATES OK', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--stage', choices=('gate', 'full'), default='gate')
    parser.add_argument('--datasets', type=Path, default=Path('/mnt/datasets/tum-rgbd'))
    main(parser.parse_args())
