"""Same-batch full-office native and five live segmentation/reanchor comparisons."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_reanchor_maps import SOURCES, NATIVE, MARGIN
from kvt_segment_detector import INLIER_THRESHOLDS
from kvt_tum_run import write_json
from kvt_tum_sweep import archive_directory


def main(args):
    from huggingface_hub import snapshot_download
    (args.work / 'JOB_OK').unlink()  # Detector completion is not comparison completion.
    for name, expected in SOURCES.items():
        assert sha256(args.out / name) == expected, name
    with tarfile.open(args.out / 'tum_26127_context.tar') as packed:
        expected = json.load(packed.extractfile('./oracle_sources.json'))['checkpoint']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    assert {name: sha256(snapshot / name) for name in expected} == expected
    with tarfile.open(args.out / 'tum_26127_oracle_maps_inputs.tar') as packed:
        packed.extractall(args.work / 'inputs')
    inputs = args.work / 'inputs' / 'office'
    assert json.loads((inputs / 'manifest.json').read_text())['frames'] == 2585
    tools = Path(__file__).parent
    native = args.work / 'runs' / 'segment_native'
    native.mkdir()
    write_json(native / 'config.json', dict(mode='native', scene_dir=str(inputs),
        frames=2585, evaluate=True, sources=SOURCES, checkpoint=expected))
    print('SEGMENT TRACKING RUN native: 2585 frames', flush=True)
    subprocess.run([sys.executable, str(tools / 'kvt_handoff_pilot.py'),
                    '--run', str(native / 'config.json')], check=True)
    native_metrics = json.loads((native / 'metrics.json').read_text())
    archive_directory(native, args.out / f'{args.tag}_segment_native.tar')
    keys = ('ate_m', 'rpe_translation_m', 'translation_p99_m')
    assert (native_metrics['evaluated_frames'], native_metrics['rpe_pairs']) == (2583, 2582)
    for key in keys:
        assert abs(native_metrics[key] - NATIVE[key]) <= 1e-4, (key, native_metrics[key], NATIVE[key])
    native_environment = json.loads((native / 'environment.json').read_text())
    records = []
    comparison = args.work / 'runs' / 'segment_comparison'
    comparison.mkdir()
    policies = [('manual', None), ('flow', args.work / 'runs' / 'segment_detector_flow' / 'summary.json')] \
        if args.flow_compare else [(threshold, args.work / 'runs' / f'segment_detector_inliers{threshold}' / 'summary.json')
                                  for threshold in INLIER_THRESHOLDS]
    for policy, segmentation in policies:
        name = f'segment_reanchor_{policy}' if args.flow_compare else f'segment_reanchor_inliers{policy}'
        print('SEGMENT TRACKING RUN', name, ': 2585 frames', flush=True)
        command = [sys.executable, str(tools / 'kvt_reanchor_maps.py'),
            '--work', str(args.work), '--out', str(args.out), '--tag', args.tag,
            '--inputs', str(inputs), '--name', name]
        if segmentation is not None:
            command += ['--segmentation', str(segmentation)]
        subprocess.run(command, check=True)
        result = args.work / 'runs' / name
        summary = json.loads((result / 'summary.json').read_text())
        metrics, timing = summary['metrics'], summary['timing']
        assert json.loads((result / 'environment.json').read_text()) == native_environment
        # Verify identical GT associations and RPE pairs, not just equal counts.
        with np.load(native / 'evaluation.npz') as reference, np.load(result / 'evaluation.npz') as current:
            for key in ('rgb_indices', 'gt_indices', 'timestamps', 'rpe_pair_start_indices', 'reference'):
                np.testing.assert_array_equal(current[key], reference[key])
        ratios = {key: metrics[key] / native_metrics[key] for key in keys}
        checks = {key: ratio <= MARGIN for key, ratio in ratios.items()}
        record = dict(policy=policy, minimum_inliers=None if args.flow_compare else policy, metrics=metrics, timing=timing,
            native_ratios=ratios, native_checks=checks, objective_achieved=all(checks.values()),
            speedup=native_metrics['tracking_seconds'] / timing['tracking_seconds'],
            cuts=len(summary['events']), live_segmentation_verified=segmentation is not None,
            manual_boundaries_are_offline_reference=segmentation is None)
        records.append(record)
        write_json(comparison / 'comparison.json', dict(native=native_metrics,
            historical_native=NATIVE, margin=MARGIN, records=records, complete=False))
        print('SEGMENT COMPARE', policy, 'ATE/RPE/p99 cm',
              [round(metrics[key] * 100, 3) for key in keys],
              'fps', round(timing['synchronous_frames_per_second'], 2),
              'within 5%', record['objective_achieved'], flush=True)
    passing = [record for record in records if record['objective_achieved']]
    best = min(passing, key=lambda record: record['timing']['tracking_seconds']) if passing else None
    write_json(comparison / 'comparison.json', dict(native=native_metrics, historical_native=NATIVE,
        margin=MARGIN, records=records, complete=True,
        best_passing_threshold=best['minimum_inliers'] if best else None,
        best_passing_policy=best['policy'] if best else None,
        selection='fastest observed live detector+tracking loop among arms meeting all three 5% gates',
        caveat='one sequence, one timing observation per arm; no held-out validation or statistical speed claim'))
    archive_directory(comparison, args.out / f'{args.tag}_segment_comparison.tar')
    sweep_path = args.work / ('segment_detector_flow_comparison.json' if args.flow_compare else 'segment_detector_sweep.json')
    sweep = json.loads(sweep_path.read_text())
    sweep['tracking_test_pending'] = False
    sweep['best_segmentation'] = best['policy'] if best else None
    write_json(sweep_path, sweep)
    (args.work / 'JOB_OK').write_text('Full tracking comparisons completed; read verdicts\n')
    print('SEGMENT TRACKING COMPARISON COMPLETE: native +', len(policies),
          'policies, 2585 frames each; best passing policy', best['policy'] if best else None, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--flow-compare', action='store_true')
    main(parser.parse_args())
