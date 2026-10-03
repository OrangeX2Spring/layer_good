"""Independent native-cache graph and query-only half-token office experiments."""
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


def main(args):
    from huggingface_hub import snapshot_download
    for name, expected in SOURCES.items():
        assert sha256(args.out / name) == expected
    with tarfile.open(args.out / 'tum_26127_context.tar') as packed:
        expected = json.load(packed.extractfile('./oracle_sources.json'))['checkpoint']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    assert {name: sha256(snapshot / name) for name in expected} == expected
    with tarfile.open(args.out / 'tum_26127_oracle_maps_inputs.tar') as packed:
        packed.extractall(args.work / 'query_inputs')
    inputs = args.work / 'query_inputs' / 'office'
    assert len(json.loads((inputs / 'manifest.json').read_text())['inputs']) == 2585
    results, paths = {}, {}
    summary = args.work / 'runs' / 'query_comparison'
    summary.mkdir(parents=True)
    # Independent methods only. half_graph is available but not dispatched here.
    for name, method, checked in (('native', None, False), ('graph_checked', 'graph', True),
                                  ('graph', 'graph', False), ('half', 'half', False)):
        result = args.work / 'runs' / ('query_' + name)
        result.mkdir()
        write_json(result / 'config.json', dict(mode='native', scene_dir=str(inputs),
            frames=2585, evaluate=True, query_method=method, query_checked=checked,
            checkpoint=expected, sources=SOURCES))
        print('QUERY TRACKING RUN', name, '2585 frames', flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('kvt_handoff_pilot.py')),
                        '--run', str(result / 'config.json')], check=True)
        archive_directory(result, args.out / f'{args.tag}_query_{name}.tar')
        paths[name] = result
        metrics = json.loads((result / 'metrics.json').read_text())
        results[name] = metrics
        assert (metrics['evaluated_frames'], metrics['rpe_pairs']) == (2583, 2582)
        if name == 'native':
            for key in ('ate_m', 'rpe_translation_m', 'translation_p99_m'):
                assert abs(metrics[key] - NATIVE[key]) <= 1e-4
        else:
            with np.load(paths['native'] / 'evaluation.npz') as reference, np.load(result / 'evaluation.npz') as current:
                for key in ('rgb_indices', 'gt_indices', 'timestamps', 'rpe_pair_start_indices', 'reference'):
                    np.testing.assert_array_equal(current[key], reference[key])
            assert json.loads((result / 'environment.json').read_text()) == json.loads(
                (paths['native'] / 'environment.json').read_text())
            np.testing.assert_array_equal(np.load(result / 'kf_idx.npy'),
                                          np.load(paths['native'] / 'kf_idx.npy'))
            assert metrics['max_cache_bytes'] == results['native']['max_cache_bytes']
            if method == 'graph':
                np.testing.assert_allclose(np.load(result / 'traj.npy'),
                    np.load(paths['native'] / 'traj.npy'), rtol=1e-4, atol=1e-4)
            if name == 'graph':
                np.testing.assert_allclose(np.load(result / 'traj.npy'),
                    np.load(paths['graph_checked'] / 'traj.npy'), rtol=1e-4, atol=1e-4)
        write_json(summary / 'comparison.json', dict(results=results, complete=False))
        print('QUERY COMPARE', name, 'fps', round(2585 / metrics['tracking_seconds'], 2),
              'ATE/RPE/p99 cm', [round(100 * metrics[k], 3) for k in
                  ('ate_m', 'rpe_translation_m', 'translation_p99_m')], flush=True)
    records = {}
    for name in ('graph', 'half'):
        metrics = results[name]
        speedup = results['native']['tracking_seconds'] / metrics['tracking_seconds']
        ratios = {key: metrics[key] / results['native'][key] for key in
                  ('ate_m', 'rpe_translation_m', 'translation_p99_m')}
        records[name] = dict(speedup=speedup, doubled_fps=speedup >= 2.,
            native_ratios=ratios, within_5_percent=all(v <= 1.05 for v in ratios.values()))
    write_json(summary / 'comparison.json', dict(results=results, records=records, complete=True,
        scope='native bank and dense rebuilds; no segmentation; independent graph and query-half',
        timing='loop includes upload, selection, capture/validation, IO and logging; excludes load/evaluation',
        caveat='one sequence, one ordered timing observation per arm; checked arm is fidelity only'))
    archive_directory(summary, args.out / f'{args.tag}_query_comparison.tar')
    print('QUERY COMPARISON COMPLETE: independent graph and half; inspect accuracy and doubled_fps', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    main(parser.parse_args())
