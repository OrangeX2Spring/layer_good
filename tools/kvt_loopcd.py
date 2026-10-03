"""LoopCD-inspired geometry pilot; five independent fixed-depth query arms."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_reanchor_maps import SOURCES, NATIVE
from kvt_tum_run import write_json
from kvt_tum_sweep import archive_directory

ARMS = dict(native=None, full=dict(depth=36), full_guided=dict(depth=36, reference_depth=12, guidance=.5),
            reduced=dict(depth=24), reduced_guided=dict(depth=24, reference_depth=12, guidance=.5))


def main(args):
    from huggingface_hub import snapshot_download
    for name, expected in SOURCES.items():
        assert sha256(args.out / name) == expected
    with tarfile.open(args.out / 'tum_26127_context.tar') as packed:
        checkpoint = json.load(packed.extractfile('./oracle_sources.json'))['checkpoint']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    assert {name: sha256(snapshot / name) for name in checkpoint} == checkpoint
    with tarfile.open(args.out / 'tum_26127_oracle_maps_inputs.tar') as packed:
        packed.extractall(args.work / 'inputs')
    inputs = args.work / 'inputs' / 'office'
    manifest = json.loads((inputs / 'manifest.json').read_text())
    assert len(manifest['inputs']) == 2585
    frame_count = 2585 if args.full else 256
    arms = {name: ARMS[name] for name in ("native", "full", "full_guided")} if args.full else ARMS
    paths, metrics = {}, {}
    summary = args.work / 'runs' / 'loopcd_comparison'
    summary.mkdir(parents=True)
    for name, method in arms.items():
        for frames in (frame_count, 32):
            run_name = name if frames == frame_count else name + '_prefix'
            result = args.work / 'runs' / ('loopcd_' + run_name)
            result.mkdir()
            write_json(result / 'config.json', dict(mode='native', scene_dir=str(inputs),
                frames=frames, evaluate=True, transition_windows=False, depth_contrast=method,
                sources=SOURCES, checkpoint=checkpoint))
            print('LOOPCD RUN', run_name, frames, 'frames', flush=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name('kvt_handoff_pilot.py')),
                            '--run', str(result / 'config.json')], check=True)
            archive_directory(result, args.out / f'{args.tag}_loopcd_{run_name}.tar')
            if frames == frame_count:
                paths[name] = result
                metrics[name] = json.loads((result / 'metrics.json').read_text())
            else:
                np.testing.assert_allclose(np.load(result / 'traj.npy'),
                    np.load(paths[name] / 'traj.npy')[:32], rtol=1e-4, atol=1e-4)
        result = paths[name]
        if name == 'native' and args.full:
            assert (metrics[name]['evaluated_frames'], metrics[name]['rpe_pairs']) == (2583, 2582)
            for key in ('ate_m', 'rpe_translation_m', 'translation_p99_m'):
                assert abs(metrics[name][key] - NATIVE[key]) <= 1e-4
        if name != 'native':
            with np.load(paths['native'] / 'evaluation.npz') as reference, np.load(result / 'evaluation.npz') as current:
                for key in ('rgb_indices', 'gt_indices', 'timestamps', 'rpe_pair_start_indices', 'reference'):
                    np.testing.assert_array_equal(current[key], reference[key])
            np.testing.assert_array_equal(np.load(result / 'kf_idx.npy'),
                                          np.load(paths['native'] / 'kf_idx.npy'))
            assert metrics[name]['max_cache_bytes'] == metrics['native']['max_cache_bytes']
            assert json.loads((result / 'environment.json').read_text()) == json.loads(
                (paths['native'] / 'environment.json').read_text())
        if name == 'full':
            np.testing.assert_allclose(np.load(result / 'traj.npy'),
                np.load(paths['native'] / 'traj.npy'), rtol=1e-4, atol=1e-4)
        write_json(summary / 'comparison.json', dict(metrics=metrics, complete=False))
        print('LOOPCD COMPARE', name, 'fps', round(frame_count / metrics[name]['tracking_seconds'], 2),
              'ATE/RPE/p99 cm', [round(100 * metrics[name][key], 3) for key in
                  ('ate_m', 'rpe_translation_m', 'translation_p99_m')], flush=True)
    keys = ('ate_m', 'rpe_translation_m', 'translation_p99_m')
    comparisons = {}
    for name in (('full_guided',) if args.full else ('full_guided', 'reduced', 'reduced_guided')):
        baseline = 'full' if name == 'full_guided' else ('native' if name == 'reduced' else 'reduced')
        native_ratios = {key: metrics[name][key] / metrics['native'][key] for key in keys}
        control_ratios = {key: metrics[name][key] / metrics[baseline][key] for key in keys}
        comparisons[name] = dict(native_ratios=native_ratios, control=baseline,
            control_ratios=control_ratios, within_5_percent=all(v <= 1.05 for v in native_ratios.values()),
            speedup=metrics['native']['tracking_seconds'] / metrics[name]['tracking_seconds'])
    write_json(summary / 'comparison.json', dict(metrics=metrics, comparisons=comparisons,
        frames=frame_count, complete=True, reference_depth=12, guidance=.5,
        reduced_guidance_improves_all=None if args.full else all(
            metrics['reduced_guided'][key] <= metrics['reduced'][key] for key in keys)
            and any(metrics['reduced_guided'][key] < metrics['reduced'][key] for key in keys),
        interpretation='untrained layer-depth analogue, not a recurrent LoopCD reproduction',
        full_sequence_pending=not args.full,
        caveat='one office sequence; first256 frames informed selection; one ordered timing observation per arm'
            if args.full else 'one office prefix, one ordered timing observation per arm'))
    archive_directory(summary, args.out / f'{args.tag}_loopcd_comparison.tar')
    (args.work / 'JOB_OK').write_text('Depth-contrast comparison complete; review results\n')
    print('LOOPCD FULL COMPLETE: 3 arms, 2585 frames, prefix fidelity' if args.full else
          'LOOPCD PILOT COMPLETE: 5 arms, prefix fidelity; full evaluation pending', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    main(parser.parse_args())
