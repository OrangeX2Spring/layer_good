"""One combined-policy TUM run plus correctness gates; execute on CAMP only."""
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
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    (inputs / 'archive.sha256').write_text(f'{digest.hexdigest()}  {archive}\n')
    input_tar = args.out / f'{args.tag}_inputs_{scene}.tar'
    archive_inputs(inputs, input_tar)
    release_page_cache(inputs, input_tar)
    base = dict(scene=scene, scene_dir=str(inputs), resize_dim=308, interval=50,
                cap=20, layer='encoder', max_gt_difference=.02,
                evaluate_trajectory=False)
    write_json(args.work / f'protocol_{scene}.json', dict(base=base, stage=args.stage,
        evaluated_policy='redundancy eviction + confidence/novelty half patches',
        gating='full-retention fidelity; prefix across repeated evictions',
        geometry='native dense rebuild; persistent query KV pruning only',
        confidence='latest keyframe dense rebuild output, no extra query geometry',
        feature='arrival-time encoder; retained selected patches for novelty',
        sim3=False, anchor='frame zero complete; all five special tokens retained'))

    def run_one(name, policy, frames, fraction=.5, evaluate=False):
        result = runs / name
        result.mkdir()
        config = dict(base, name=name, policy=policy, frames=frames,
                      patch_fraction=fraction, evaluate_trajectory=evaluate)
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

    # These short runs are implementation gates, not extra experimental arms.
    reference = run_one('gate_periodic', 'periodic', 128)
    unchanged = run_one('gate_full_retention', 'combined', 128, fraction=1.)
    a, b = np.load(reference / 'traj.npy'), np.load(unchanged / 'traj.npy')
    np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(np.load(reference / 'kf_idx.npy'),
                                 np.load(unchanged / 'kf_idx.npy'))
    print('FULL RETENTION FIDELITY OK', scene, flush=True)

    prefix = run_one('combined_prefix', 'combined', 1100)
    length = 1150 if args.stage == 'pilot' else manifest['frames']
    full = run_one('combined', 'combined', length, evaluate=args.stage == 'full')
    assert np.isfinite(np.load(full / 'traj.npy')).all(), 'Nonfinite trajectory'
    compare_prefix(prefix, full)
    short_events = [json.loads(line) for line in (prefix / 'cache_events.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (full / 'cache_events.jsonl').read_text().splitlines()]
    assert short_events == events[:len(short_events)], 'Noncausal cache selection'
    assert sum(len(e['evicted']) for e in events) >= 3
    assert all(len(set(e['retained_frame_ids'])) <= 20 for e in events)
    assert all(e['retained_frame_ids'][0] == 0 for e in events)
    assert all(e['query_cache_bytes'] < e['dense_cache_bytes'] for e in events[1:])
    expected = periodic_indices(length, 50, length)
    np.testing.assert_array_equal(np.load(full / 'inserted_kf_idx.npy'), expected)
    assert [e['frame'] for e in events] == expected
    retained = np.load(full / 'kf_idx.npy').tolist()
    assert retained == events[-1]['retained_frame_ids']
    with np.load(full / 'final_scene.npz') as scene_data:
        assert scene_data['frame_ids'].tolist() == retained
        assert np.isfinite(scene_data['poses']).all()
    write_json(args.work / f'GATES_OK_{scene}.json', dict(full_retention_max_error=float(np.max(np.abs(a-b))),
        prefix_frames=1100, frames=length, evictions=sum(len(e['evicted']) for e in events),
        passed=True))
    for result in runs.iterdir():
        destination = args.out / f'{args.tag}_{scene}_{result.name}.tar'
        archive_directory(result, destination)
        release_page_cache(result, destination)
    print('SCENE GATES OK', scene, flush=True)


def main(args):
    # Scenes run sequentially inside one job: Slurm counts submissions, not loops,
    # and each scene keeps its own inputs, gates and per-run archives.
    scenes = args.scene.split(',')
    assert all(scene in LONG_SCENES for scene in scenes), scenes
    for scene in scenes:
        run_scene(args, scene)
    (args.work / 'JOB_OK').write_text('Combined policy and correctness gates passed\n')
    print('COMBINED GATES OK', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--scene', required=True,
                        help=f'one scene, or several comma separated, from {LONG_SCENES}')
    parser.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--datasets', type=Path, default=Path('/mnt/datasets/tum-rgbd'))
    main(parser.parse_args())
