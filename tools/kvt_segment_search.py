"""CAMP-only bounded calibration of causal segmentation plus fixed reanchoring."""
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

KEYS = ('ate_m', 'rpe_translation_m', 'translation_p99_m')
PROFILES = ([dict(detector='orb', minimum_inliers=x) for x in (25, 30, 35)]
            + [dict(detector='flow', surviving_fraction=x) for x in (.35, .5, .65)])
CANDIDATES = [dict(profile, maximum_segment_frames=age)
              for age in (None, 100, 150, 75, 200) for profile in PROFILES]


def main(args):
    from huggingface_hub import snapshot_download
    assert 1 <= args.max_trials <= 30
    for name, expected in SOURCES.items():
        assert sha256(args.out / name) == expected
    with tarfile.open(args.out / 'tum_26127_context.tar') as packed:
        checkpoint = json.load(packed.extractfile('./oracle_sources.json'))['checkpoint']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    assert {name: sha256(snapshot / name) for name in checkpoint} == checkpoint
    with tarfile.open(args.out / 'tum_26127_oracle_maps_inputs.tar') as packed:
        packed.extractall(args.work / 'inputs')
    inputs = args.work / 'inputs' / 'office'
    assert len(json.loads((inputs / 'manifest.json').read_text())['inputs']) == 2585
    tools = Path(__file__).parent
    native = args.work / 'runs' / 'search_native'
    native.mkdir(parents=True)
    write_json(native / 'config.json', dict(mode='native', scene_dir=str(inputs),
        frames=2585, evaluate=True, sources=SOURCES, checkpoint=checkpoint))
    subprocess.run([sys.executable, str(tools / 'kvt_handoff_pilot.py'),
                    '--run', str(native / 'config.json')], check=True)
    baseline = json.loads((native / 'metrics.json').read_text())
    assert (baseline['evaluated_frames'], baseline['rpe_pairs']) == (2583, 2582)
    for key in KEYS:
        assert abs(baseline[key] - NATIVE[key]) <= 1e-4
    archive_directory(native, args.out / f'{args.tag}_search_native.tar')
    summary = args.work / 'runs' / 'segment_search'
    summary.mkdir()
    records, passing, best = [], None, None
    for index, policy in enumerate(CANDIDATES[:args.max_trials], 1):
        name = f'search_trial_{index:02d}'
        config = summary / (name + '.json')
        write_json(config, policy)
        print('SEGMENT SEARCH TRIAL', index, policy, flush=True)
        subprocess.run([sys.executable, str(tools / 'kvt_reanchor_maps.py'),
            '--work', str(args.work), '--out', str(args.out), '--tag', args.tag,
            '--inputs', str(inputs), '--detector-config', str(config), '--name', name], check=True)
        result = args.work / 'runs' / name
        run = json.loads((result / 'summary.json').read_text())
        assert json.loads((result / 'environment.json').read_text()) == json.loads(
            (native / 'environment.json').read_text())
        with np.load(native / 'evaluation.npz') as reference, np.load(result / 'evaluation.npz') as current:
            for key in ('rgb_indices', 'gt_indices', 'timestamps', 'rpe_pair_start_indices', 'reference'):
                np.testing.assert_array_equal(current[key], reference[key])
        ratios = {key: run['metrics'][key] / baseline[key] for key in KEYS}
        score = max(ratios.values())
        record = dict(trial=index, policy=policy, metrics=run['metrics'], timing=run['timing'],
            ratios=ratios, worst_ratio=score, passed=score <= 1.5,
            cuts=len(run['events']), boundaries=json.loads((result / 'config.json').read_text())['boundaries'])
        records.append(record)
        if best is None or score < best['worst_ratio']:
            best = record
        if record['passed']:
            passing = record
        write_json(summary / 'comparison.json', dict(native=baseline, records=records,
            best=best, passing=passing, margin=1.5, max_trials=args.max_trials,
            complete=False, gt_used_for_offline_selection=True, gt_used_by_tracker=False))
        archive_directory(summary, args.out / f'{args.tag}_segment_search.tar')
        print('SEGMENT SEARCH RESULT', index, 'ratios', ratios, 'passed', record['passed'], flush=True)
        if passing is not None:
            break
    write_json(summary / 'comparison.json', dict(native=baseline, records=records,
        best=best, passing=passing, margin=1.5, max_trials=args.max_trials, complete=True,
        stop_reason='accuracy_target' if passing else 'trial_limit',
        gt_used_for_offline_selection=True, gt_used_by_tracker=False,
        caveat='same-sequence calibration; no held-out claim; fixed reanchor mechanism'))
    archive_directory(summary, args.out / f'{args.tag}_segment_search.tar')
    (args.work / 'JOB_OK').write_text('Bounded search completed; inspect stop reason and accuracy verdict\n')
    print('SEGMENT SEARCH COMPLETE:', len(records), 'trials; target met', passing is not None, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--max-trials', type=int, default=30)
    main(parser.parse_args())
