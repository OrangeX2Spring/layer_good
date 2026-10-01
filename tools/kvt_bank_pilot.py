"""Bounded office whole-keyframe pilot. Execute only on CAMP."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from kvt_tum_run import prepare, periodic_indices, write_json
from kvt_tum_sweep import archive_inputs, archive_directory, compare_prefix


def main(args):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    checkpoint = {}
    for path in sorted(snapshot.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            checkpoint[str(path.relative_to(snapshot))] = digest.hexdigest()
    assert checkpoint
    write_json(args.work / 'checkpoint.json', dict(snapshot=str(snapshot), files=checkpoint))
    scene = 'freiburg3_long_office_household'
    inputs = args.work / 'inputs' / scene
    runs = args.work / 'runs' / scene
    runs.mkdir(parents=True)
    source_archive = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
    prepare(source_archive, inputs, 308, .02, count=950)
    digest = hashlib.sha256()
    with source_archive.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    (inputs / 'archive.sha256').write_text(f'{digest.hexdigest()}  {source_archive}\n')
    archive_inputs(inputs, args.out / f'{args.tag}_bank_inputs.tar')
    base = dict(scene=scene, scene_dir=str(inputs), resize_dim=308, interval=50,
                cap=8, layer='encoder', max_gt_difference=.02,
                evaluate_trajectory=False, save_final_scene=True)

    def run(name, arm, frames, evaluate=False):
        result = runs / name
        result.mkdir()
        config = dict(base, name=name, frames=frames, evaluate_trajectory=evaluate)
        if arm == 'native':
            config.update(policy='original', cap=20)
        else:
            config.update(policy='combined', bank_policy=arm, patch_fraction=1.)
        write_json(result / 'config.json', config)
        print('RUN', name, flush=True)
        with (result / 'run.log').open('w') as log:
            completed = subprocess.run([sys.executable,
                str(Path(__file__).with_name('kvt_tum_run.py')), str(result / 'config.json')],
                stdout=log, stderr=subprocess.STDOUT)
        write_json(result / 'process.json', dict(returncode=completed.returncode))
        archive_directory(result, args.out / f'{args.tag}_bank_{name}.tar')
        if completed.returncode:
            print((result / 'run.log').read_text()[-8000:], flush=True)
            raise RuntimeError(f'{name} failed: {completed.returncode}')
        poses = np.load(result / 'traj.npy')
        assert poses.shape == (frames, 4, 4) and np.isfinite(poses).all()
        insertion = 'kf_idx.npy' if arm == 'native' else 'inserted_kf_idx.npy'
        expected = periodic_indices(frames, 50, 20)
        assert np.load(result / insertion).tolist() == expected
        if arm != 'native':
            events = [json.loads(line) for line in (result / 'cache_events.jsonl').read_text().splitlines()]
            assert [e['frame'] for e in events] == expected
            for event in events:
                ids = event['retained_frame_ids']
                assert ids[0] == 0 and ids[-1] == event['frame']
                assert len(ids) <= 8
                assert set(event['protected']) <= set(ids)
                assert event['query_cache_bytes'] == event['dense_cache_bytes']
            assert np.load(result / 'kf_idx.npy').tolist() == events[-1]['retained_frame_ids']
            with np.load(result / 'final_scene.npz') as final:
                assert final['frame_ids'].tolist() == events[-1]['retained_frame_ids']
                assert np.isfinite(final['poses']).all()
            if frames > 400:
                assert sum(len(e['evicted']) for e in events) >= 3
        return result

    reference = run('fidelity_native', 'native', 128)
    for arm in ('recent', 'relevance'):
        candidate = run('fidelity_' + arm, arm, 128)
        np.testing.assert_allclose(np.load(reference / 'traj.npy'),
                                   np.load(candidate / 'traj.npy'), atol=1e-5, rtol=1e-4)
    summary = {}
    for arm in ('native', 'recent', 'relevance'):
        prefix = run(arm + '_prefix', arm, 850)
        result = run(arm, arm, 950, evaluate=True)
        compare_prefix(prefix, result)
        if arm != 'native':
            short = (prefix / 'cache_events.jsonl').read_text().splitlines()
            long = (result / 'cache_events.jsonl').read_text().splitlines()
            assert short == long[:len(short)], 'Noncausal bank choices'
        metrics = json.loads((result / 'metrics.json').read_text())
        with np.load(result / 'evaluation.npz') as quality:
            metrics['translation_p99_m'] = float(np.quantile(quality['rpe_translation_per_pair_m'], .99))
        if arm != 'native':
            events = [json.loads(line) for line in (result / 'cache_events.jsonl').read_text().splitlines()]
            metrics['max_bank_feature_bytes'] = max(e['feature_bytes'] for e in events)
            metrics['max_rebuild_frames'] = max(e['rebuild_input_frames'] for e in events)
        summary[arm] = metrics
    keys = ('ate_m', 'rpe_translation_m', 'translation_p99_m')
    for arm in ('recent', 'relevance'):
        summary[arm]['within_native_5pct'] = all(
            summary[arm][key] <= 1.05 * summary['native'][key] for key in keys)
    summary['relevance']['no_worse_than_recent_all_three'] = all(
        summary['relevance'][key] <= summary['recent'][key] for key in keys)
    a = (runs / 'recent' / 'cache_events.jsonl').read_text().splitlines()
    b = (runs / 'relevance' / 'cache_events.jsonl').read_text().splitlines()
    summary['different_retention_events'] = sum(
        json.loads(x)['retained_frame_ids'] != json.loads(y)['retained_frame_ids'] for x, y in zip(a, b))
    summary['relevance']['selection_evidence_gate'] = (
        summary['different_retention_events'] > 0
        and summary['relevance']['no_worse_than_recent_all_three']
        and any(summary['relevance'][key] < summary['recent'][key] for key in keys))
    write_json(args.work / 'bank_summary.json', summary)
    for result in runs.iterdir():
        archive_directory(result, args.out / f'{args.tag}_bank_{result.name}.tar')
    archives = sorted(args.out.glob(f'{args.tag}_bank_*.tar'))
    hashes = {}
    for path in archives:
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    write_json(args.work / 'bank_archives.json', hashes)
    (args.work / 'JOB_OK').write_text('Bank implementation gates passed; scientific review pending\n')
    print('BANK PILOT COMPLETE: 3 fidelity + 3 prefixes + 3 main runs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    main(parser.parse_args())
