"""Full-office stock-ID replay fidelity gate. Execute inside CAMP's KVT image."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from kvt_tum_run import prepare, write_json
from kvt_tum_sweep import archive_directory, archive_inputs, release_page_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    scene = 'freiburg3_long_office_household'
    source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
    staged = args.work / 'inputs' / scene
    manifest = prepare(source, staged, 308, .02)
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    (staged / 'archive.sha256').write_text(f'{digest.hexdigest()}  {source}\n')
    inputs = args.out / f'{args.tag}_inputs_{scene}.tar'
    archive_inputs(staged, inputs)
    release_page_cache(staged, inputs)
    base = dict(scene=scene, scene_dir=str(staged), frames=manifest['frames'],
                resize_dim=308, interval=50, cap=20, max_gt_difference=.02,
                evaluate_trajectory=True)
    write_json(args.work / 'protocol.json', dict(base=base, stage='stock-ID fidelity',
        trajectory_atol=1e-4, trajectory_rtol=1e-4,
        comparison='unmodified stock decisions versus arrival-time fixed replay',
        seed=0, next_gate='human review before LLM/uniform/random evaluation'))
    results = {}
    for name, policy in (('stock', 'bare'), ('stock_replay', 'fixed')):
        result = args.work / 'runs' / name
        result.mkdir(parents=True)
        config = dict(base, name=name, policy=policy)
        if policy == 'fixed':
            config['insertion_indices'] = np.load(results['stock'] / 'kf_idx.npy').tolist()[1:]
        write_json(result / 'config.json', config)
        print('RUN', name, base['frames'], flush=True)
        with (result / 'run.log').open('w') as log:
            completed = subprocess.run([sys.executable,
                str(Path(__file__).with_name('kvt_tum_run.py')), str(result / 'config.json')],
                stdout=log, stderr=subprocess.STDOUT)
        write_json(result / 'process.json', dict(returncode=completed.returncode))
        archive = args.out / f'{args.tag}_{scene}_{name}.tar'
        archive_directory(result, archive)
        release_page_cache(result, archive)
        if completed.returncode:
            print((result / 'run.log').read_text()[-8000:], flush=True)
            raise RuntimeError(f'{name} failed: {completed.returncode}; see archived run.log')
        results[name] = result
    stock, replay = results['stock'], results['stock_replay']
    stock_ids = np.load(stock / 'kf_idx.npy')
    np.testing.assert_array_equal(stock_ids, np.load(replay / 'kf_idx.npy'))
    assert len(stock_ids) == 20
    original = np.load(stock / 'traj.npy')
    reproduced = np.load(replay / 'traj.npy')
    assert original.shape == reproduced.shape == (manifest['frames'], 4, 4)
    np.testing.assert_allclose(original, reproduced, atol=1e-4, rtol=1e-4)
    write_json(args.work / 'fidelity.json', dict(passed=True,
        frames=manifest['frames'], keyframe_ids=stock_ids.tolist(),
        max_pose_difference=float(np.max(np.abs(original - reproduced))),
        atol=1e-4, rtol=1e-4,
        metrics={name: json.loads((path / 'metrics.json').read_text())
                 for name, path in results.items()}))
    print('STOCK-ID FIDELITY GATE OK', flush=True)
    (args.work / 'JOB_OK').write_text('Stock replay only; comparison awaits review.\n')


if __name__ == '__main__':
    main()
