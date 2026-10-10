"""CAMP-only full-office gates: native/two-/three-image history, or pinned refreshes.

Stage 'gate' is the reviewed 26476 matrix. Stage 'pinned' runs native, the unpinned
three-image control (must reproduce 26476) and three-image with pinned rebuilds.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_reanchor_maps import SOURCES
from kvt_tum_run import CHECKOUT, write_json
from kvt_tum_sweep import archive_directory, release_page_cache

REFERENCE = 'tum_25680_freiburg3_long_office_household_original.tar'
REFERENCE_SHA256 = 'e3d56f66e411d93621e0d3bbc5b150a5d3c52bfdb09afdea9a564f7cbd95e125'
# Reviewed 26476 three-image arm (Mac audit, archive_inventory.json).
CONTROL = 'tum_26476_camera_history_three_frame.tar'
CONTROL_SHA256 = '79256c7f09ff56a2e694d1cbce02ffcf5f4f11afab70532076f1985c120d0590'
ARMS = dict(gate=('native', 'no_retirement', 'three_frame'),
            pinned=('native', 'three_frame', 'three_frame_pinned'))


def refresh_pairs(result):
    """Share of squared translation RPE in pairs spanning a scheduled 50-frame refresh."""
    with np.load(result / 'evaluation.npz') as q:
        error = q['rpe_translation_per_pair_m'] ** 2
        refresh = q['rpe_pair_start_indices'] % 50 == 49
    return dict(pairs=int(refresh.sum()), squared_error_share=float(error[refresh].sum() / error.sum()),
                other_rms_m=float(np.sqrt(error[~refresh].mean())))


def main(args):
    from huggingface_hub import snapshot_download

    tools = Path(__file__).resolve().parent
    comparison = args.work / 'runs' / 'camera_history'
    comparison.mkdir(parents=True)
    tracker_commit = subprocess.check_output(
        ['git', '-C', str(CHECKOUT), 'rev-parse', 'HEAD'], text=True).strip()
    assert tracker_commit == subprocess.check_output(
        ['git', '-C', str(tools.parent), 'rev-parse', 'HEAD:kv_tracker'], text=True).strip()
    sources = dict(SOURCES, **{REFERENCE: REFERENCE_SHA256})
    if args.stage == 'pinned':
        sources[CONTROL] = CONTROL_SHA256
    for name, digest in sources.items():
        assert sha256(args.out / name) == digest, name
    with tarfile.open(args.out / 'tum_26127_context.tar') as packed:
        checkpoint = json.load(packed.extractfile('./oracle_sources.json'))['checkpoint']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    assert {name: sha256(snapshot / name) for name in checkpoint} == checkpoint
    with tarfile.open(args.out / 'tum_26127_oracle_maps_inputs.tar') as packed:
        packed.extractall(args.work / 'inputs')
    inputs = args.work / 'inputs' / 'office'
    reference = args.work / 'inputs' / 'reference'
    reference.mkdir()
    # Only the fidelity/evaluation evidence is needed, not the large final scene.
    with tarfile.open(args.out / REFERENCE) as packed:
        for name in ('traj.npy', 'evaluation.npz', 'metrics.json'):
            (reference / name).write_bytes(packed.extractfile('original/' + name).read())
    release_page_cache(args.out / REFERENCE, inputs)
    if args.stage == 'pinned':
        with tarfile.open(args.out / CONTROL) as packed:
            control_trajectory = np.load(packed.extractfile('camera_history_three_frame/traj.npy'))
    manifest = json.loads((inputs / 'manifest.json').read_text())
    frames = manifest['frames']
    assert frames == len(manifest['inputs']) == 2585 and manifest['resize_dim'] == 308
    report = dict(complete=False, stage=args.stage, task='camera', scene=manifest['scene'], frames=frames,
        duration_seconds=manifest['duration_seconds'], sources=sources, checkpoint=checkpoint,
        input_manifest_sha256=sha256(inputs / 'manifest.json'), tracker_commit=tracker_commit,
        implementation={name: sha256(tools / name) for name in (
            'kvt_camera_history.py', 'kvt_segment_arctic.py', 'kvt_reanchor_maps.py',
            'kvt_segment_detector.py', 'kvt_inference_timing.py', 'kvt_tum_run.py')}, runs=[],
        caveat='One reused office sequence, one repeat; profiled saved-pixel pipeline. '
               'Frozen-cache head timing is not online FPS. Quality failure is recorded, '
               'not a runtime failure. Review before expansion.')
    write_json(comparison / 'comparison.json', report)
    policy = comparison / 'policy.json'
    write_json(policy, dict(detector='orb', minimum_inliers=25, maximum_segment_frames=200))
    native_metrics = comparison / 'native_metrics.json'
    durations = []
    decisions = None
    environment = None
    keys = ('ate_m', 'rpe_translation_m', 'translation_p99_m')
    for arm in ARMS[args.stage]:
        remaining = args.deadline - time.time() - 1800
        if remaining < max(1800, 2 * max(durations, default=0)):
            report['stop_reason'] = 'Insufficient time for another arm plus packaging reserve'
            write_json(comparison / 'comparison.json', report)
            archive_directory(comparison, args.out / f'{args.tag}_camera_history_comparison.tar')
            raise SystemExit(3)
        name = 'camera_history_' + arm
        result = args.work / 'runs' / name
        if arm == 'native':
            result.mkdir()
            write_json(result / 'config.json', dict(task='camera', inputs=str(inputs),
                reference_trajectory=str(reference / 'traj.npy'), profile_forward=True))
            command = [sys.executable, str(tools / 'kvt_segment_arctic.py'),
                       '--native-replay', str(result / 'config.json')]
        else:
            command = [sys.executable, str(tools / 'kvt_reanchor_maps.py'),
                '--work', str(args.work), '--out', str(args.out), '--tag', args.tag,
                '--inputs', str(inputs), '--native-metrics', str(native_metrics),
                '--detector-config', str(policy), '--name', name, '--disable-retirement',
                '--local-keyframe-cap', '2' if arm == 'no_retirement' else '3', '--profile-forward']
            if arm == 'three_frame_pinned':
                command.append('--pin-rebuilds')
        report['active_run'] = dict(name=name, command=command)
        write_json(comparison / 'comparison.json', report)
        print('CAMERA HISTORY ARM', arm, frames, flush=True)
        started = time.monotonic()
        subprocess.run(command, check=True, timeout=remaining)
        durations.append(time.monotonic() - started)
        trajectory = np.load(result / 'traj.npy')
        assert trajectory.shape == (frames, 4, 4) and np.isfinite(trajectory).all()
        if args.stage == 'pinned' and arm == 'three_frame':
            np.testing.assert_allclose(trajectory, control_trajectory, rtol=1e-4, atol=1e-4)
        with np.load(result / 'evaluation.npz') as current, np.load(reference / 'evaluation.npz') as saved:
            for key in ('rgb_indices', 'gt_indices', 'timestamps', 'rpe_pair_start_indices', 'reference'):
                np.testing.assert_array_equal(current[key], saved[key])
        summary = json.loads((result / 'summary.json').read_text())
        benchmark = json.loads((result / 'cached_head_benchmark.json').read_text())
        current_environment = {key: benchmark[key] for key in (
            'shape', 'gpu', 'torch_version', 'cuda_version', 'autocast_dtype',
            'matmul_tf32', 'cudnn_tf32')}
        if environment is not None:
            assert current_environment == environment
        environment = current_environment
        if arm == 'native':
            write_json(native_metrics, summary['metrics'])
        else:
            config = json.loads((result / 'config.json').read_text())
            assert config['boundaries'] == [0, frames] and not summary['events']
            assert bool(config['pin_rebuilds']) == (arm == 'three_frame_pinned')
            current_decisions = [{k: v for k, v in json.loads(line).items() if k != 'seconds'}
                for line in (result / 'segmentation_live.jsonl').read_text().splitlines()]
            assert len(current_decisions) == frames
            if decisions is not None:
                assert decisions == current_decisions
            decisions = current_decisions
        baseline = json.loads(native_metrics.read_text())
        ratios = {key: summary['metrics'][key] / baseline[key] for key in keys}
        row = dict(arm=arm, summary=summary, subprocess_seconds=durations[-1],
                   environment=environment, native_ratios=ratios,
                   quality_pass=all(r <= 1.05 for r in ratios.values()))
        if args.stage == 'pinned':
            row['refresh_pairs'] = refresh_pairs(result)
        if arm == 'three_frame' and args.stage == 'gate':
            control = report['runs'][1]['summary']['metrics']
            row['two_image_ratios'] = {key: summary['metrics'][key] / control[key] for key in keys}
            row['improves_two_image'] = all(r < 1 for r in row['two_image_ratios'].values())
        if arm == 'three_frame_pinned':
            control = report['runs'][1]['summary']['metrics']
            row['unpinned_ratios'] = {key: summary['metrics'][key] / control[key] for key in keys}
            row['improves_unpinned'] = all(r < 1 for r in row['unpinned_ratios'].values())
            row['pins'] = [{k: v for k, v in json.loads(line).items()
                            if k in ('frame', 'pin_rotation_deg', 'pin_position_step')}
                           for line in (result / 'inference.jsonl').read_text().splitlines()
                           if json.loads(line)['kind'] == 'update_total']
            assert len(row['pins']) == (frames - 1) // 50 and all('pin_rotation_deg' in x for x in row['pins'])
        archive = args.out / f'{args.tag}_{name}.tar'
        archive_directory(result, archive)
        row['archive_sha256'] = sha256(archive)
        report['runs'].append(row)
        del report['active_run']
        write_json(comparison / 'comparison.json', report)
        archive_directory(comparison, args.out / f'{args.tag}_camera_history_comparison.tar')
    report['complete'] = True
    write_json(comparison / 'comparison.json', report)
    archive_directory(comparison, args.out / f'{args.tag}_camera_history_comparison.tar')
    (args.work / 'JOB_OK').write_text('Camera fidelity/execution complete; review quality before expansion\n')
    print('CAMERA HISTORY COMPLETE', *(['pinned'] if args.stage == 'pinned' else []), 3, frames, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--stage', choices=tuple(ARMS), default='gate')
    main(parser.parse_args())
