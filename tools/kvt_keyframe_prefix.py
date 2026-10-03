"""CAMP-only full-office ablation of the frame range allowed to populate KV."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_reanchor_maps import SOURCES, NATIVE
from kvt_tum_run import periodic_indices, write_json
from kvt_tum_sweep import archive_directory


def main(args):
    from huggingface_hub import snapshot_download
    for name, expected in SOURCES.items():
        assert sha256(args.out / name) == expected
    with tarfile.open(args.out / 'tum_26127_context.tar') as packed:
        checkpoint = json.load(packed.extractfile('./oracle_sources.json'))['checkpoint']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    assert {name: sha256(snapshot / name) for name in checkpoint} == checkpoint
    with tarfile.open(args.out / 'tum_26127_oracle_maps_inputs.tar') as packed:
        packed.extractall(args.work / 'prefix_inputs')
    inputs = args.work / 'prefix_inputs' / 'office'
    assert len(json.loads((inputs / 'manifest.json').read_text())['inputs']) == 2585
    summary = args.work / 'runs' / 'keyframe_prefix_comparison'
    summary.mkdir(parents=True)
    results, baseline_path, baseline_metrics = {}, None, None
    keys = ('ate_m', 'rpe_translation_m', 'translation_p99_m')
    # Baseline first. Continued-admission caps exceed every eligible insertion.
    for name, stop, cap in (('native', 2585, 20), ('first200', 200, 20),
                            ('first500', 500, 20), ('first750', 750, 20),
                            ('through1200', 1200, 25), ('through1600', 1600, 33),
                            ('through2000', 2000, 41)):
        result = args.work / 'runs' / ('keyframe_prefix_' + name)
        result.mkdir()
        write_json(result / 'config.json', dict(mode='native', scene_dir=str(inputs),
            frames=2585, keyframe_stop_frame=stop, evaluate=True,
            native_keyframe_cap=cap, sources=SOURCES, checkpoint=checkpoint))
        print('KEYFRAME PREFIX RUN', name, '2585 frames; admission before', stop, flush=True)
        command = [sys.executable, str(Path(__file__).with_name('kvt_handoff_pilot.py')),
                   '--run', str(result / 'config.json'), '--report-oom']
        with (result / 'run.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        # Preserve completed inference even if a subsequent comparison gate fails.
        archive_directory(result, args.out / f'{args.tag}_keyframe_prefix_{name}.tar')
        if completed.returncode == 75:
            failure = json.loads((result / 'oom.json').read_text())
            results[name] = dict(failure, keyframe_stop_frame=stop, native_keyframe_cap=cap)
            write_json(summary / 'comparison.json', dict(results=results, complete=False))
            archive_directory(summary, args.out / f'{args.tag}_keyframe_prefix_comparison.tar')
            print('KEYFRAME PREFIX OOM', name, failure, flush=True)
            if name == 'native':
                completed.check_returncode()
            continue
        if completed.returncode:
            print((result / 'run.log').read_text()[-8000:], flush=True)
            completed.check_returncode()
        metrics = json.loads((result / 'metrics.json').read_text())
        assert (metrics['frames'], metrics['evaluated_frames'], metrics['rpe_pairs']) == (2585, 2583, 2582)
        expected_ids = periodic_indices(stop, 50, cap)
        np.testing.assert_array_equal(np.load(result / 'kf_idx.npy'), expected_ids)
        rows = [json.loads(line) for line in (result / 'inference.jsonl').read_text().splitlines()]
        assert [r['frame'] for r in rows if r['kind'] == 'rebuild'] == expected_ids[1:]
        queries = [r for r in rows if r['kind'] == 'query']
        assert [r['frame'] for r in queries] == list(range(1, 2585))
        assert all(max(r['bank_ids']) < r['frame'] for r in queries)
        assert all(r['bank_ids'] == expected_ids for r in queries if r['frame'] >= stop)
        trajectory = np.load(result / 'traj.npy')
        if name == 'native':
            baseline_path, baseline_metrics = result, metrics
            for key in keys:
                assert abs(metrics[key] - NATIVE[key]) <= 1e-4
        else:
            shared_prefix = min(stop, 1000)
            np.testing.assert_allclose(trajectory[:shared_prefix],
                np.load(baseline_path / 'traj.npy')[:shared_prefix], rtol=1e-4, atol=1e-4)
            assert json.loads((result / 'environment.json').read_text()) == json.loads(
                (baseline_path / 'environment.json').read_text())
        windows = {}
        with np.load(result / 'evaluation.npz') as quality:
            if name != 'native':
                with np.load(baseline_path / 'evaluation.npz') as reference:
                    for key in ('rgb_indices', 'gt_indices', 'timestamps', 'rpe_pair_start_indices', 'reference'):
                        np.testing.assert_array_equal(quality[key], reference[key])
            # Slice saved full-run residuals; do not independently fit each window.
            for start, end in ((0, 1000), (1000, 2500), (1025, 1376), (1025, 2026)):
                frame_mask = (quality['rgb_indices'] >= start) & (quality['rgb_indices'] < end)
                pair_mask = (quality['rpe_pair_start_indices'] >= start) & (
                    quality['rpe_pair_start_indices'] + 1 < end)
                ate = quality['ate_per_frame_m'][frame_mask]
                rpe = quality['rpe_translation_per_pair_m'][pair_mask]
                windows[f'{start}_{end - 1}'] = dict(evaluated_frames=len(ate), rpe_pairs=len(rpe),
                    ate_m=float(np.sqrt(np.mean(ate ** 2))),
                    rpe_translation_m=float(np.sqrt(np.mean(rpe ** 2))),
                    translation_p99_m=float(np.quantile(rpe, .99)))
            alignment_scale = float(quality['alignment_scale'])
        results[name] = dict(metrics=metrics, keyframe_ids=expected_ids, windows=windows,
            alignment_scale=alignment_scale, fps=2585 / metrics['tracking_seconds'],
            speedup=baseline_metrics['tracking_seconds'] / metrics['tracking_seconds'],
            native_ratios={key: metrics[key] / baseline_metrics[key] for key in keys})
        write_json(summary / 'comparison.json', dict(results=results, complete=False))
        archive_directory(summary, args.out / f'{args.tag}_keyframe_prefix_comparison.tar')
        print('KEYFRAME PREFIX COMPARE', name, 'keys', len(expected_ids),
              'fps', round(results[name]['fps'], 2), 'ATE/RPE/p99 cm',
              [round(100 * metrics[key], 3) for key in keys], flush=True)
        print('KEYFRAME PREFIX LATE', name, windows['1025_2025'], flush=True)
    write_json(summary / 'comparison.json', dict(results=results, complete=True,
        scope='fresh native, first200/500/750, continued admission through1200/1600/2000; all2585 frames',
        alignment='one full-valid-trajectory Sim(3) per arm; windows retain that transform',
        caveat='one sequence and ordered timing observation; no no-cache control; '
               'tests early-bank sufficiency, not causal importance of individual keys; '
               'keyframe count, coverage and reconstruction gauge change together',
        gt_used_by_tracker=False))
    archive_directory(summary, args.out / f'{args.tag}_keyframe_prefix_comparison.tar')
    (args.work / 'JOB_OK').write_text('Keyframe prefix ablation complete; inspect accuracy and timing\n')
    print('KEYFRAME PREFIX ABLATION COMPLETE: 7 arms; inspect any OOM and accuracy verdicts', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    main(parser.parse_args())
