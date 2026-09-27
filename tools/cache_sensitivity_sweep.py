"""Bounded three-host compression diagnostic. Execute on CAMP, never locally."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

GROUPS = {'layer': 6, 'head': 4, 'channel': 4}
SCENES = [('calibration', 'freiburg3_long_office_household', 256),
          ('with_loop', 'freiburg2_large_with_loop', 512),
          ('no_loop', 'freiburg2_large_no_loop', 512)]
REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pose_score(poses, native):
    """No GT or candidate alignment: relative-pose distortion in native gauge."""
    assert poses.shape == native.shape and poses.shape[1:] == (4, 4)
    assert np.isfinite(poses).all() and np.isfinite(native).all()
    reference = np.linalg.inv(native[:-1]) @ native[1:]
    candidate = np.linalg.inv(poses[:-1]) @ poses[1:]
    error = np.linalg.inv(reference) @ candidate
    unit = float(np.sqrt(np.mean(np.sum(reference[:, :3, 3] ** 2, axis=1))))
    assert unit > 1e-8, 'Native reference has insufficient translation for this fixed score'
    translation = np.linalg.norm(error[:, :3, 3], axis=1) / unit
    angle = np.arccos(np.clip((np.trace(error[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1))
    return dict(score=float(np.mean(translation ** 2 + (angle / np.deg2rad(1.)) ** 2)),
                translation_unit=unit, translation_normalized_rmse=float(np.sqrt(np.mean(translation ** 2))),
                rotation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(angle ** 2)))))


def condition(name, bits, family=None, group=None, profile=False):
    return dict(name=name, compression=dict(bits=bits, family=family, group=group, profile=profile))


def calibration_conditions():
    rows = [dict(name='reference', compression=None), condition('native', 0, profile=True),
            condition('int4', 4), condition('int2', 2)]
    rows += [condition(f'{family}_{group}', 2, family, group)
             for family, count in GROUPS.items() for group in range(count)]
    return rows


def camera(path, host):
    if host == 'kvt':
        return np.load(path / 'traj.npy')
    ex = np.load(path / 'camera.npz')['extrinsics']
    w2c = np.tile(np.eye(4), (len(ex), 1, 1))
    w2c[:, :3] = ex[:, :3]
    return np.linalg.inv(w2c)


def verify_events(path, native_path, host, compression):
    """Retention, real bytes and budgets must not drift between conditions."""
    if host == 'kvt':
        report = json.loads((path / 'compression.json').read_text())
        native = json.loads((native_path / 'compression.json').read_text())
        rows, baseline = report['events'], native['events']
        assert len(rows) == len(baseline) and rows
        for a, b in zip(rows, baseline):
            assert a['frame_ids'] == b['frame_ids'] and a['layer_ids'] == b['layer_ids']
            assert a['actual_bytes'] == b['actual_bytes'] and a['dtype'] == b['dtype']
        actual = [r['actual_bytes'] for r in rows]
        packed = [r['analytical_payload_bytes'] for r in rows]
        changed = any(r['changed'] for r in rows)
    else:
        rows = [json.loads(t) for t in (path / 'events.jsonl').read_text().splitlines()]
        baseline = [json.loads(t) for t in (native_path / 'events.jsonl').read_text().splitlines()]
        assert len(rows) == len(baseline) and rows
        for a, b in zip(rows, baseline):
            for key in ('retained_frames', 'refresh', 'aggregator_bytes', 'aggregator_dtype',
                        'camera_bytes', 'relative_pose_bytes', 'reference_bytes'):
                assert a[key] == b[key], (path, key)
            if a['refresh']:
                assert a['refresh_seed']['retained_frames'] == b['refresh_seed']['retained_frames']
                assert a['post_refresh_memory'] == b['post_refresh_memory']
        actual = [r['aggregator_bytes'] for r in rows]
        packed = [r['analytical_compression_bytes'] for r in rows]
        for row in rows:
            if row['refresh']:
                actual.append(row['post_refresh_memory']['aggregator_bytes'])
                packed.append(row['refresh_seed']['analytical_compression_bytes'])
        changed = any(r['compressed_entry']['changed'] for r in rows)
    assert (changed if compression['bits'] else not changed)
    if compression['bits']:
        assert all(0 < a < b for a, b in zip(packed, actual))
    else:
        assert packed == actual
    return dict(max_actual_bytes=max(actual), max_analytical_bytes=max(packed),
                analytical_bytes=packed)


def execute_clip(args, role, scene, length, rows):
    inputs = args.work / 'inputs' / role
    source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
    inputs.parent.mkdir(exist_ok=True)
    target = args.work / 'runs' / role
    target.parent.mkdir(exist_ok=True)
    if args.host == 'kvt':
        from kvt_tum_run import prepare, periodic_indices
        from kvt_tum_sweep import release_page_cache
        prepare(source, inputs, 308, .02, start=0, count=length)
        with tarfile.open(args.out / f'{args.tag}_inputs_{role}.tar', 'w') as archive:
            archive.add(inputs, arcname='.')
        release_page_cache(inputs, args.out / f'{args.tag}_inputs_{role}.tar')
        target.mkdir()
        ids = periodic_indices(length, 50, 20)
        for row in rows:
            run = target / row['name']
            run.mkdir()
            config = dict(name=row['name'], scene=scene, scene_dir=str(inputs), frames=length,
                resize_dim=308, interval=50, cap=len(ids), policy='fixed', insertion_indices=ids[1:],
                evaluate_trajectory=True, max_gt_difference=.02, save_final_scene=False,
                compression=row['compression'])
            write(run / 'config.json', config)
            print('SENSITIVITY RUN', args.host, role, row['name'], flush=True)
            with (run / 'run.log').open('w') as log:
                completed = subprocess.run([sys.executable, str(REPO / 'tools/kvt_tum_run.py'),
                    str(run / 'config.json')], stdout=log, stderr=subprocess.STDOUT)
            write(run / 'process.json', dict(returncode=completed.returncode))
            if completed.returncode:
                print((run / 'run.log').read_text()[-8000:], flush=True)
                completed.check_returncode()
    else:
        extracted = args.work / ('extracted_' + role)
        subprocess.run([sys.executable, 'tools/stream_cache_tum.py', '--zip', str(source),
            '--out', str(extracted), '--start', '0', '--count', str(length)], cwd=REPO, check=True)
        subprocess.run([sys.executable, 'tools/stream_cache_sweep.py', 'prepare', '--manifest',
            str(extracted / 'manifest.json'), '--out', str(inputs), '--width', '308'], cwd=REPO, check=True)
        specification = dict(max_frames=length, isolate_conditions=True, geometry_export='final',
            conditions=[dict(name=r['name'], policy=dict(frame_budget=32, compression=r['compression']))
                        for r in rows])
        spec_path = args.work / f'{role}_sweep.json'
        write(spec_path, specification)
        command = [sys.executable, 'tools/stream_cache_sweep.py', 'run', '--host', args.host,
            '--checkpoint', str(args.checkpoint), '--inputs', str(inputs), '--sweep', str(spec_path),
            '--out', str(target), '--git-provenance', str(args.work / 'git_provenance')]
        if args.host == 'longstream':
            command += ['--model-config', str(REPO / 'longstream/configs/longstream_infer.yaml')]
        subprocess.run(command, cwd=REPO, check=True)
    # Source ZIP and selected image identity survive the allocation.
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    write(target / 'input_identity.json', dict(scene=scene, source_zip=str(source),
        source_zip_sha256=digest.hexdigest(), start=0, frames=length, width=308))
    native = camera(target / 'native', args.host)
    reports = {}
    for row in rows:
        name = row['name']
        poses = camera(target / name, args.host)
        report = pose_score(poses, native)
        if name == 'reference':
            np.testing.assert_allclose(poses, native, atol=1e-4, rtol=1e-4)
            print('SENSITIVITY NATIVE FIDELITY OK', args.host, flush=True)
        else:
            report.update(verify_events(target / name, target / 'native', args.host, row['compression']))
        if args.host == 'kvt':
            metrics = json.loads((target / name / 'metrics.json').read_text())
            with np.load(target / name / 'evaluation.npz') as e:
                metrics['rpe_translation_p99_m'] = float(np.quantile(e['rpe_translation_per_pair_m'], .99))
                metrics['fitted_scale'] = float(e['alignment_scale'])
        else:
            metrics = json.loads((target / name / 'camera_metrics.json').read_text())
            metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
        report['metrics'] = metrics
        reports[name] = report
    # Every group in a family has the exact same representational budget at each prefix.
    for family in GROUPS:
        names = [r['name'] for r in rows if r['compression'] is not None
                 and r['compression']['family'] == family]
        assert names
        assert all(reports[n]['analytical_bytes'] == reports[names[0]]['analytical_bytes'] for n in names)
    write(target / 'diagnostic.json', reports)
    print('SENSITIVITY CONTRACTS OK', args.host, role, flush=True)
    return target, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('kvt', 'longstream', 'streamvggt'), required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--tag')
    args = parser.parse_args()
    assert sys.platform == 'linux' and args.work.is_dir()
    if args.host == 'kvt':
        assert args.out is not None and args.tag is not None
    else:
        assert args.checkpoint is not None
    source = args.work / 'sensitivity_source'
    source.mkdir()
    for name in ('cache_sensitivity_sweep.py', 'stream_cache_sensitivity.py',
                 'stream_cache_quantization.py', 'test_cache_sensitivity.py',
                 'cache_sensitivity.json', 'cache_sensitivity_overnight.sbatch',
                 'CACHE_SENSITIVITY.md'):
        shutil.copy2(REPO / 'tools' / name, source)
    write(args.work / 'sensitivity_protocol.json', dict(host=args.host, groups=GROUPS, scenes=SCENES,
        seed=0, calibration='relative pose distortion vs native; no candidate alignment or GT selection',
        score='mean((translation_error/native_RMS_step)^2 + (rotation_error/1degree)^2)',
        protection='one contiguous layer/head/channel group, K and V together, native dtype',
        bytes='full quantized payload + native overwrite values + 8-byte group descriptor',
        evaluation='frozen calibration choices; clips are diagnostics, not full-sequence evidence'))
    path, reports = execute_clip(args, *SCENES[0], calibration_conditions())
    profile = json.loads((path / 'native/compression.json').read_text())
    assert profile['profile_calls'] > 0
    rng = np.random.default_rng(0)
    choices = {}
    for family, count in GROUPS.items():
        scores = [reports[f'{family}_{i}']['score'] for i in range(count)]
        choices[family] = dict(sensitivity=int(np.argmin(scores)),
            variance=int(np.argmax(profile['variance'][family])), random=int(rng.integers(count)),
            calibration_scores=scores, variance_scores=profile['variance'][family])
    write(args.work / 'frozen_choices.json', choices)
    choice_hash = hashlib.sha256((args.work / 'frozen_choices.json').read_bytes()).hexdigest()
    evaluation = [dict(name='reference', compression=None), condition('native', 0),
                  condition('int4', 4), condition('int2', 2)]
    evaluation += [condition(f'{family}_{method}', 2, family, choices[family][method])
                   for family in GROUPS for method in ('sensitivity', 'variance', 'random')]
    for clip in SCENES[1:]:
        execute_clip(args, *clip, evaluation)
        assert hashlib.sha256((args.work / 'frozen_choices.json').read_bytes()).hexdigest() == choice_hash
    (args.work / 'JOB_OK').write_text('Bounded sensitivity diagnostic complete; full run requires review.\n')
    print('SENSITIVITY DIAGNOSTIC OK', args.host, flush=True)


if __name__ == '__main__':
    main()
