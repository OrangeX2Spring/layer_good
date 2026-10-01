"""Independent fixed diagnostics; full validation requires a reviewed anchor SHA."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from stream_depth_ablation import REPO, reference26089
from stream_cache_sweep import prepare, write_json, digest


def prepare_inputs(args, scene, start, count, label):
    source = args.work / 'source' / label
    subprocess.run([sys.executable, str(REPO / 'tools/stream_cache_tum.py'),
                    '--zip', str(args.dataset / f'rgbd_dataset_{scene}.zip'),
                    '--out', str(source), '--start', str(start), '--count', str(count)], check=True)
    prepare(SimpleNamespace(manifest=source / 'manifest.json', out=args.work / 'inputs' / label, width=308))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('diagnostics', 'full'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-archive', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('/mnt/datasets/tum-rgbd'))
    parser.add_argument('--reviewed-anchor-archive', type=Path)
    parser.add_argument('--reviewed-anchor-sha256')
    args = parser.parse_args()
    assert sys.platform.startswith('linux')
    reference = reference26089(args)
    selection = json.loads((args.work / 'selection.json').read_text())
    frozen = selection['omitted_layers']
    modes = ('native', 'omit', 'uniform_special')
    plan = []
    if args.stage == 'diagnostics':
        reverse = sorted(row['layer'] for row in sorted(selection['scores'],
            key=lambda row: (-row['score'], row['layer']))[:8])
        null = sorted(random.Random(17).sample(range(24), 8))
        for scene in ('office', 'with_loop'):
            for label, layers in (('reverse', reverse), ('random17', null)):
                plan.append(dict(group='specificity', scene=scene, condition=f'specificity_{label}',
                                 mode='omit', omitted_layers=layers, capacity=32, history_policy='fifo'))
        order = random.Random(29)
        for repeat in range(3):
            for scene in ('office', 'with_loop'):
                shuffled = list(modes)
                order.shuffle(shuffled)
                for mode in shuffled:
                    plan.append(dict(group='repeatability', scene=scene,
                        condition=f'repeat{repeat+1}_{mode}', mode=mode, omitted_layers=frozen,
                        capacity=32, history_policy='fifo'))
        for scene, dataset_scene in (('office', 'freiburg3_long_office_household'),
                                     ('with_loop', 'freiburg2_large_with_loop')):
            label = f'scale_{scene}'
            prepare_inputs(args, dataset_scene, 0, 512, label)
            for capacity in (8, 32, 64):
                for mode in modes:
                    plan.append(dict(group='history_scaling', scene=label,
                        condition=f'capacity{capacity}_{mode}', mode=mode, omitted_layers=frozen,
                        capacity=capacity, history_policy='fifo'))
        assert len(plan) == 40
    else:
        # This manually supplied SHA is the explicit review gate. No diagnostic
        # or pilot can automatically populate it or submit this stage.
        assert args.reviewed_anchor_archive is not None and args.reviewed_anchor_sha256 is not None
        assert len(args.reviewed_anchor_sha256) == 64
        assert digest(args.reviewed_anchor_archive) == args.reviewed_anchor_sha256
        import tarfile
        with tarfile.open(args.reviewed_anchor_archive) as packed:
            assert packed.extractfile('./exit_status.txt').read().strip() == b'0'
            reviewed = json.loads(packed.extractfile('./summary.json').read())
            assert reviewed['complete'] and reviewed['history_policy'] == 'anchor'
            assert len(reviewed['records']) == len(reviewed['comparisons']) == 6
            assert all(row['prefix_pose_equivalence_pass'] and row['equal_per_frame_storage_pass']
                       for row in reviewed['comparisons'])
        write_json(args.work / 'reviewed_anchor.json', dict(archive=str(args.reviewed_anchor_archive),
            sha256=args.reviewed_anchor_sha256, approval='user supplied SHA after manual scientific review'))
        for scene, count in (('freiburg3_long_office_household', 2585),
                             ('freiburg2_large_with_loop', 5182), ('freiburg2_large_no_loop', 3359)):
            prepare_inputs(args, scene, 0, count, scene)
            for mode in modes:
                plan.append(dict(group='full_sequence', scene=scene, condition=mode, mode=mode,
                                 omitted_layers=frozen, capacity=32, history_policy='anchor'))
        assert len(plan) == 9
    write_json(args.work / 'overnight_plan.json', dict(stage=args.stage, cases=plan,
        seed_policy='all model runs seed17; null17; repeat order29',
        scope='fixed cases; no result-driven selection, calibration refit or successor'))
    records, failures = [], []
    for index, case in enumerate(plan):
        result = args.work / 'runs' / case['scene'] / case['condition']
        result.mkdir(parents=True)
        print('OVERNIGHT RUN', index+1, '/', len(plan), case, flush=True)
        with (result / 'run.log').open('w') as log:
            process = subprocess.run([sys.executable, str(REPO / 'tools/stream_depth_ablation.py'),
                'arm', '--work', str(args.work), '--checkpoint', str(args.checkpoint),
                '--scene', case['scene'], '--condition', case['condition'], '--mode', case['mode'],
                '--capacity', str(case['capacity']), '--history-policy', case['history_policy'],
                '--omitted-layers', ','.join(str(i) for i in case['omitted_layers'])],
                stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            failures.append(dict(case=case, returncode=process.returncode, log=str(result / 'run.log')))
            print('OVERNIGHT CASE FAILED', case, process.returncode, flush=True)
            print('\n'.join((result / 'run.log').read_text().splitlines()[-40:]), flush=True)
        else:
            metrics = json.loads((result / 'metrics.json').read_text())
            events = [json.loads(line) for line in (result / 'inference.jsonl').read_text().splitlines()]
            assert events[-1]['frame']+1 == metrics['frames']
            for event in events:
                expected = event['dense_equivalent_bytes'] if case['mode'] == 'native' else event['dense_equivalent_bytes']*2//3
                assert event['persistent_bytes'] == expected
            if case['group'] == 'repeatability':
                np.testing.assert_allclose(np.load(result / 'traj.npy'),
                    np.load(reference / f'runs/{case["scene"]}/{case["mode"]}/traj.npy'), atol=1e-5, rtol=1e-5)
            with np.load(result / 'evaluation.npz') as evaluation:
                pair_ids = evaluation['rpe_pair_start_indices']
                error = evaluation['rpe_translation_per_pair_m']
                first_eviction = case['capacity']*4
                mask = pair_ids == first_eviction
                eviction = dict(pair_start=first_eviction, valid=bool(mask.any()),
                    translation_rpe_m=float(error[mask][0]) if mask.any() else None,
                    squared_error_fraction=float(error[mask][0]**2 / np.sum(error**2)) if mask.any() else None)
            records.append(dict(case=case, metrics=metrics, first_eviction=eviction))
            print('OVERNIGHT RUN OK', case, metrics, eviction, flush=True)
        write_json(args.work / 'overnight_summary.json', dict(complete=False, stage=args.stage,
            planned_cases=len(plan), records=records, failures=failures,
            reference_records=json.loads((reference / 'summary.json').read_text())['records'],
            next='manual review; independent case failures remain explicit'))
    repeats = []
    for scene in ('office', 'with_loop'):
        for mode in modes:
            values = [r['metrics']['steady_query_plus_write_p50_seconds'] for r in records
                      if r['case']['group']=='repeatability' and r['case']['scene']==scene and r['case']['mode']==mode]
            if values:
                repeats.append(dict(scene=scene, mode=mode, process_count=len(values),
                    process_p50_seconds=values, median=float(np.median(values)),
                    min=float(min(values)), max=float(max(values))))
    write_json(args.work / 'overnight_summary.json', dict(complete=not failures, stage=args.stage,
        planned_cases=len(plan), records=records, failures=failures, repeatability=repeats,
        reference_records=json.loads((reference / 'summary.json').read_text())['records'],
        limits='single GPU; historical FIFO timing descriptive; no tuning or training',
        next='manual review; no automatic full-stage or training submission'))
    if failures:
        raise SystemExit('OVERNIGHT SUITE FAILED: inspect recorded independent failures')
    print('STREAM DEPTH OVERNIGHT COMPLETE', args.stage, len(records), flush=True)


if __name__ == '__main__':
    main()
