"""A1: stock-ID append-only KV-Tracker comparison on TUM office. CAMP only."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
from pi3.models.pi3 import Pi3

from kvt_tum_run import periodic_indices, prepare, write_json
from kvt_tum_sweep import archive_directory, archive_inputs, release_page_cache


def insertion_metrics(result):
    """Step k joins the insertion frame k (old cache) to query k+1 (new cache)."""
    ids = np.load(result / 'kf_idx.npy')[1:]
    with np.load(result / 'evaluation.npz') as evaluation:
        starts = evaluation['rpe_pair_start_indices']
        at_insertion = np.isin(starts, ids)
        rpe = evaluation['rpe_translation_per_pair_m']
        positions = evaluation['full_aligned_positions']
        steps = np.linalg.norm(positions[starts + 1] - positions[starts], axis=1)
        assert at_insertion.any() and (~at_insertion).any()
        return dict(insertion_pair_count=int(at_insertion.sum()),
            insertion_step_rms_m=float(np.sqrt(np.mean(steps[at_insertion] ** 2))),
            insertion_rpe_t_rms_m=float(np.sqrt(np.mean(rpe[at_insertion] ** 2))),
            noninsertion_rpe_t_rms_m=float(np.sqrt(np.mean(rpe[~at_insertion] ** 2))),
            alignment_scale=float(evaluation['alignment_scale']),
            insertion_pair_start_indices=starts[at_insertion].tolist())


def refresh_metrics(result, common_scale):
    """Within-window RPE; use baseline scale for both arms, no per-window fit."""
    with np.load(result / 'evaluation.npz') as e:
        indices = e['rgb_indices']
        adjacent = ((np.diff(indices) == 1) & (np.diff(e['timestamps']) <= .1))
        starts = indices[:-1][adjacent]
        np.testing.assert_array_equal(starts, e['rpe_pair_start_indices'])
        predicted = (np.linalg.inv(e['aligned'][:-1]) @ e['aligned'][1:])[adjacent]
        reference = (np.linalg.inv(e['reference'][:-1]) @ e['reference'][1:])[adjacent]
        predicted[:, :3, 3] *= common_scale / float(e['alignment_scale'])
        errors = np.linalg.inv(reference) @ predicted
        translation = np.linalg.norm(errors[:, :3, 3], axis=1)
        rotation = e['rpe_rotation_per_pair_deg']
        insertion = np.isin(starts, np.load(result / 'kf_idx.npy')[1:])
        windows = []
        for lo, hi in ((450, 700), (700, 750), (750, 800), (800, 850),
                       (850, 900), (900, 950), (700, 950)):
            selected = (starts >= lo) & (starts + 1 < hi) & ~insertion
            assert selected.any()
            windows.append(dict(start=lo, end_exclusive=hi, pairs=int(selected.sum()),
                translation_rpe_m=float(np.sqrt(np.mean(translation[selected] ** 2))),
                rotation_rpe_deg=float(np.sqrt(np.mean(rotation[selected] ** 2)))))
        crossing = starts == 699
        assert crossing.sum() == 1
        return dict(common_scale=common_scale, noninsertion_windows=windows,
                    refresh_crossing_translation_m=float(translation[crossing][0]),
                    refresh_crossing_rotation_deg=float(rotation[crossing][0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--stage', choices=('pilot', 'full', 'refresh'), default='pilot')
    parser.add_argument('--reviewed-pilot', type=Path,
                        help='Successful pilot context tar, explicitly reviewed before full run')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__).resolve(), repo / 'tools/kvt_tum_run.py',
                  repo / 'tools/kvt_tum_selector.py', repo / 'tools/test_kvt_append.py',
                  repo / 'kv_tracker/main.py', repo / 'kv_tracker/kv_tracker/append_cache.py',
                  repo / 'kv_tracker/kv_tracker/pi3_utilts.py']
    pi3_path = Path(inspect.getfile(Pi3))
    code_paths += [pi3_path, pi3_path.parent / 'layers/block.py',
                   pi3_path.parent / 'layers/attention.py']
    code_hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in code_paths}
    container_sha = (args.work / 'context/container_tar.sha256').read_text().split()[0]
    if args.stage == 'full':
        assert args.reviewed_pilot is not None, 'Review a successful pilot before full evaluation'
        with tarfile.open(args.reviewed_pilot) as archive:
            gate = json.load(archive.extractfile('./pilot_gate.json'))
            assert archive.extractfile('./context/exit_status.txt').read().strip() == b'0'
            assert archive.extractfile('./JOB_OK').read().strip()
        assert gate['passed'] and gate['code_sha256'] == code_hashes
        assert gate['container_sha256'] == container_sha
    else:
        assert args.reviewed_pilot is None

    scene = 'freiburg3_long_office_household'
    source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
    staged = args.work / 'inputs' / scene
    manifest = prepare(source, staged, 308, .02)
    source_digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            source_digest.update(chunk)
    source_sha = source_digest.hexdigest()
    rgb_digest = hashlib.sha256(''.join(
        r['model_rgb_sha256'] for r in manifest['inputs']).encode()).hexdigest()
    if args.stage == 'full':
        assert (source_sha, rgb_digest) == (gate['source_zip_sha256'], gate['rgb_digest'])
    (staged / 'archive.sha256').write_text(f'{source_sha}  {source}\n')
    inputs_archive = args.out / f'{args.tag}_inputs_{scene}.tar'
    archive_inputs(staged, inputs_archive)
    release_page_cache(staged, inputs_archive)
    length = {'pilot': 256, 'refresh': 950}.get(args.stage, manifest['frames'])
    ids = periodic_indices(length, 50, 20)
    base = dict(scene=scene, scene_dir=str(staged), frames=length, resize_dim=308,
                interval=50, cap=len(ids), max_gt_difference=.02, evaluate_trajectory=True,
                save_final_scene=False)
    conditions = [('stock_replay', dict(policy='fixed', insertion_indices=ids[1:])),
                  ('append_only', dict(policy='fixed', insertion_indices=ids[1:],
                                       append_only=True, verify_append=args.stage == 'pilot'))]
    if args.stage == 'refresh':
        conditions = [(name, dict(policy='fixed', insertion_indices=ids[1:],
                       append_only=True, verify_append=True, **options))
                      for name, options in [('append_only', {}),
                                            ('append_refresh', dict(refresh_frame=699))]]
    if args.stage == 'pilot':
        conditions.insert(0, ('stock_native', dict(policy='original', cap=20)))
    write_json(args.work / 'protocol.json', dict(stage=args.stage, base=base,
        ids=ids, conditions=[dict(name=name, **options) for name, options in conditions],
        code_sha256=code_hashes, container_sha256=container_sha,
        source_zip_sha256=source_sha, rgb_digest=rgb_digest,
        gauge='Both arms sim3=False; append freezes bootstrap origin_offset and scene_origin',
        bootstrap='Native duplicate retained in append: one extra physical slot, same unique IDs',
        reviewed_pilot=str(args.reviewed_pilot) if args.reviewed_pilot else None,
        next_gate='Human review; no automatic full run or accuracy acceptance'))

    results, metrics = {}, {}
    for name, options in conditions:
        result = args.work / 'runs' / name
        result.mkdir(parents=True)
        write_json(result / 'config.json', dict(base, name=name, **options))
        print('RUN', name, length, flush=True)
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
        np.testing.assert_array_equal(np.load(result / 'kf_idx.npy'), ids)
        results[name] = result
        metrics[name] = json.loads((result / 'metrics.json').read_text())
        metrics[name].update(insertion_metrics(result))
        inference = [json.loads(line) for line in (result / 'inference.jsonl').read_text().splitlines()]
        query_seconds = {row['frame']: row['seconds'] for row in inference if row['kind'] == 'query'}
        if options.get('append_only', False):
            events = json.loads((result / 'append_events.json').read_text())
            np.testing.assert_array_equal(np.load(result / 'kf_poses.npy'),
                                          np.load(result / 'traj.npy')[ids])
            assert [e['frame'] for e in events] == ids[1:]
            expected_refresh = [699] if name == 'append_refresh' else []
            refreshes = json.loads((result / 'refresh_events.json').read_text())
            assert [e['frame'] for e in refreshes] == expected_refresh
            assert [r['frame'] for r in inference if r['kind'] == 'rebuild'] == expected_refresh
            assert metrics[name]['bootstrap_calls'] == 1
            assert metrics[name]['rebuild_calls'] == len(expected_refresh)
            for event in refreshes:
                assert event['cache_frame_ids'] == [0] + [i for i in ids if i <= 699]
                assert event['cache_shapes_preserved'] and event['changed_layers']
                assert event['origin_offset'] == events[0]['origin_offset']
                assert event['scene_origin'] == events[0]['scene_origin']
            assert metrics[name]['query_calls'] == length - 1
            for event in events:
                assert event['origin_offset'] == events[0]['origin_offset']
                assert event['scene_origin'] == events[0]['scene_origin']
                assert not event['sim3_enabled']
                if args.stage in ('pilot', 'refresh'):
                    assert event['old_cache_prefix_verified'] and event['old_pose_prefix_verified']
            updates = [dict(frame=e['frame'], update_seconds=e['commit_seconds'],
                            query_plus_update_seconds=query_seconds[e['frame']] + e['commit_seconds'])
                       for e in events]
        else:
            rebuilds = [row for row in inference if row['kind'] == 'rebuild']
            assert metrics[name]['bootstrap_calls'] == 1
            assert metrics[name]['query_calls'] == length - 1
            assert [r['frame'] for r in rebuilds] == ids[1:]
            updates = [dict(frame=r['frame'], update_seconds=r['seconds'],
                            query_plus_update_seconds=query_seconds[r['frame']] + r['seconds'])
                       for r in rebuilds]
        metrics[name]['insertion_timings'] = updates
        print('RUN OK', name, json.dumps(metrics[name]), flush=True)

    append = np.load(results['append_only'] / 'traj.npy')
    if args.stage == 'refresh':
        refreshed = np.load(results['append_refresh'] / 'traj.npy')
        np.testing.assert_allclose(refreshed[:700], append[:700], atol=1e-4, rtol=1e-4)
        baseline_events = json.loads((results['append_only'] / 'append_events.json').read_text())
        refreshed_events = json.loads((results['append_refresh'] / 'append_events.json').read_text())
        for baseline, refreshed_event in zip(baseline_events, refreshed_events):
            for key in ('cache_frame_ids', 'tokens_per_frame', 'origin_offset', 'scene_origin'):
                assert baseline[key] == refreshed_event[key]
        common_scale = metrics['append_only']['alignment_scale']
        for name in results:
            metrics[name]['refresh_diagnostic'] = refresh_metrics(results[name], common_scale)
            print('REFRESH DIAGNOSTIC', name,
                  json.dumps(metrics[name]['refresh_diagnostic']), flush=True)
        write_json(args.work / 'refresh_gate.json', dict(passed=True,
            prefix_through_frame=699, refresh_frame=699, frames=length,
            same_physical_history=True, frozen_bootstrap_gauge=True,
            code_sha256=code_hashes, container_sha256=container_sha,
            source_zip_sha256=source_sha, rgb_digest=rgb_digest,
            accuracy_gate='Review post-refresh non-insertion rotation RPE; no automatic acceptance'))
        print('APPEND REFRESH CONTRACT GATE OK', flush=True)
    else:
        stock = np.load(results['stock_replay'] / 'traj.npy')
        # Insertion frame prediction precedes the cache update.
        np.testing.assert_allclose(append[:ids[1]+1], stock[:ids[1]+1], atol=1e-4, rtol=1e-4)
    if args.stage == 'pilot':
        np.testing.assert_allclose(np.load(results['stock_native'] / 'traj.npy'), stock,
                                   atol=1e-4, rtol=1e-4)
        write_json(args.work / 'pilot_gate.json', dict(passed=True,
            code_sha256=code_hashes, container_sha256=container_sha,
            source_zip_sha256=source_sha, rgb_digest=rgb_digest, frames=length, ids=ids,
            stock_replay_fidelity=True, before_first_insertion_fidelity=True,
            cache_pose_prefix_checks=True, frozen_bootstrap_gauge=True,
            zero_rebuilds=True, accuracy_gate='Not automatic: review comparison.json'))
        print('APPEND PILOT CONTRACT GATE OK', flush=True)
    write_json(args.work / 'comparison.json', dict(stage=args.stage, conditions=metrics,
        note='Insertion RMS is a pose step, not GT-subtracted error; insertion RPE also reported. '
             'Times are instrumented; pilot prefix equality checks add overhead. '
             'Append uses one extra physical bootstrap slot. No full-map geometry comparison.'))
    (args.work / 'JOB_OK').write_text(f'A1 {args.stage} complete; review before advancing.\n')


if __name__ == '__main__':
    main()
