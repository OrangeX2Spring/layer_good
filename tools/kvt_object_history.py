"""CAMP-only fixed ketchup history diagnostics; pilot before overnight repeats."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_tum_run import CHECKOUT, write_json
from kvt_tum_sweep import archive_directory

SCENE = 'ketchup_grab_01'
SOURCES = {
    'tum_26458_segment_arctic_ketchup_grab_01_inputs.tar': '9ca7598fe6ddaa1bc8d1cc308329d3d022e5acb8d957bd448c7bcb03d21a345b',
    'tum_26458_arctic_native_replay_ketchup_grab_01.tar': '238682d077a9324be442dacbadd04e428db96fe0525dd9f7f4be475342c32fcd',
    'tum_26458_arctic_reanchor_anchor_support_ketchup_grab_01.tar': '984298b035b097479395c39abe7427bb42714e7cb819bc9dddcb7759c15182c8',
    'arctic_tum_26458_segment_native_ketchup_grab_01_20261009T202717Z.tar': 'a05cf214a138c54fd1ce6a8fbd0a663d499971300b3f4f7bec1e9e24e5c4b391',
}
# Stage 'pinned': unpinned three-image control (must reproduce 26472), pinned rebuilds,
# and pinned rebuilds with refresh-frame scale.
PINNED = ('native', 'three_frame', 'three_frame_pinned', 'three_frame_pinned_scale')
# Stage 'novelty': pinned control (must reproduce 26488) and pinned rebuilds that also
# refresh when native's object keyframe rule fires.
NOVELTY = ('native', 'three_frame_pinned', 'three_frame_pinned_novelty')
# Stage 'retrieval': the same pinned control and a four-image GPU bank retrieved from a
# CPU memory of all keyframes (native's object rule over the whole memory, fixed poses).
RETRIEVAL = ('native', 'three_frame_pinned', 'retrieval_bank')
# Stage 'two-way': the user's segmented reanchor (26458 policy; must reproduce it) and the
# same cuts with the two-frame confidence-weighted connection scale.
TWO_WAY = ('native', 'reanchor', 'reanchor_twoway')
# Stage 'latest-anchor': the same control, three-image segment banks, and three-image banks
# with each connection pinned on the old map's latest rebuild keyframe instead of b.
LATEST = ('native', 'reanchor', 'reanchor_three', 'reanchor_three_latest')
COMPARE = dict(three_frame_pinned='three_frame', three_frame_pinned_scale='three_frame_pinned',
               three_frame_pinned_novelty='three_frame_pinned', retrieval_bank='three_frame_pinned',
               reanchor_twoway='reanchor', reanchor_three='reanchor',
               reanchor_three_latest='reanchor_three')
ORDERS = (('native', 'reanchor', 'no_retirement'),
          ('reanchor', 'no_retirement', 'native'),
          ('no_retirement', 'native', 'reanchor'))


def error_groups(result, reference, boundaries):
    """Descriptive paired errors; no realignment or causal attribution."""
    with np.load(result / 'evaluation.npz') as q, np.load(reference / 'evaluation.npz') as n:
        ends = q['pair_end_frames']
        np.testing.assert_array_equal(ends, n['pair_end_frames'])
        squared, baseline = q['translation_m'] ** 2, n['translation_m'] ** 2
    events = json.loads((result / 'events.json').read_text())
    rows = [json.loads(s) for s in (result / 'inference.jsonl').read_text().splitlines()]
    membership = np.full(len(ends), 'other', dtype='<U10')
    membership[np.isin(ends, [r['global_frame'] for r in rows if r['kind'] == 'update_total'])] = 'refresh'
    membership[np.isin(ends, [e['decision_frame'] for e in events])] = 'connection'
    membership[np.isin(ends, boundaries[1:-1])] = 'cut'
    groups = {name: membership == name for name in ('cut', 'connection', 'refresh', 'other')}
    groups.update({f'segment_{start}_{end}': (ends > start) & (ends < end)
                   for start, end in zip(boundaries[:-1], boundaries[1:])})
    return {name: dict(pairs=int(mask.sum()), squared_error_sum=float(squared[mask].sum()),
                       native_squared_error_sum=float(baseline[mask].sum()),
                       excess_squared_error_sum=float((squared[mask] - baseline[mask]).sum()))
            for name, mask in groups.items()}


def main(args):
    import cv2
    from huggingface_hub import snapshot_download
    from kvt_arctic_run import OUT as ARCTIC_OUT, stage_dataset
    from kvt_reanchor_maps import SOURCES as REANCHOR_SOURCES

    tools = Path(__file__).resolve().parent
    os.chdir(CHECKOUT)
    summary = args.work / 'runs' / 'object_history'
    summary.mkdir(parents=True)
    implementation = {name: sha256(tools / name) for name in (
        'kvt_object_history.py', 'kvt_reanchor_maps.py', 'kvt_segment_arctic.py',
        'kvt_segment_detector.py', 'kvt_patch_select.py', 'kvt_inference_timing.py')}
    tracker_commit = subprocess.check_output(['git', '-C', str(CHECKOUT), 'rev-parse', 'HEAD'], text=True).strip()
    expected_tracker = subprocess.check_output(
        ['git', '-C', str(tools.parent), 'rev-parse', 'HEAD:kv_tracker'], text=True).strip()
    assert tracker_commit == expected_tracker, 'Tracker must match the published parent gitlink'
    orders = (('native', 'no_retirement', 'three_frame'),) if args.stage == 'three-frame' else (
        (PINNED,) if args.stage == 'pinned' else (NOVELTY,) if args.stage == 'novelty' else
        (RETRIEVAL,) if args.stage == 'retrieval' else (TWO_WAY,) if args.stage == 'two-way' else
        (LATEST,) if args.stage == 'latest-anchor' else
        ORDERS[:1] if args.stage == 'pilot' else ORDERS)
    profiled = args.stage in ('three-frame', 'pinned', 'novelty', 'retrieval', 'two-way', 'latest-anchor')
    source_archives = dict(SOURCES)
    if args.stage == 'three-frame':
        source_archives['tum_26471_object_history_r0_no_retirement.tar'] = (
            '8674e84bf060091188910360ca607f9e2d746b866bc72bc415a2483bf6a68094')
    if args.stage == 'pinned':
        source_archives['tum_26472_object_history_r0_three_frame.tar'] = (
            '881e503ace949234ea7843d2dbd5cde6f7e097acc95147dd98615d53180975f4')
    if args.stage in ('novelty', 'retrieval'):
        source_archives['tum_26488_object_history_r0_three_frame_pinned.tar'] = (
            'abfa9ebbc2973b41111c73872657046b9e9b59ea9a43f81acfdd7e979f47e0d5')
    report = dict(stage=args.stage, scene=SCENE, sources=source_archives, complete=False,
                  orders=orders, runs=[],
                  implementation=implementation, tracker_commit=tracker_commit,
                  interpretation='No-retirement changes resets and anchor age; not a pure capacity test.')
    for name, expected in REANCHOR_SOURCES.items():
        assert sha256(args.out / name) == expected, name
    report['reanchor_prerequisites'] = REANCHOR_SOURCES
    if args.stage == 'overnight':
        assert args.reviewed_pilot is not None and args.reviewed_sha256 is not None
        assert sha256(args.reviewed_pilot) == args.reviewed_sha256
        with tarfile.open(args.reviewed_pilot) as packed:
            prior = json.load(packed.extractfile('object_history/comparison.json'))
        assert prior['stage'] == 'pilot' and prior['complete'] and prior['sources'] == SOURCES
        assert prior['implementation'] == implementation and prior['tracker_commit'] == tracker_commit
        assert [r['arm'] for r in prior['runs']] == list(ORDERS[0])
        report['reviewed_pilot'] = dict(path=str(args.reviewed_pilot), sha256=args.reviewed_sha256)
    write_json(summary / 'comparison.json', report)
    sources = args.work / 'inputs' / 'history_sources'
    for name, expected in source_archives.items():
        archive = (ARCTIC_OUT if name.startswith('arctic_') else args.out) / name
        assert sha256(archive) == expected, name
        destination = sources / Path(name).stem
        destination.mkdir(parents=True)
        with tarfile.open(archive) as packed:
            for member in packed.getmembers():
                assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                assert member.isfile() or member.isdir()
            packed.extractall(destination)
    inputs = sources / 'tum_26458_segment_arctic_ketchup_grab_01_inputs' / SCENE
    reference = sources / 'tum_26458_arctic_native_replay_ketchup_grab_01' / f'arctic_native_replay_{SCENE}'
    reanchor = sources / 'tum_26458_arctic_reanchor_anchor_support_ketchup_grab_01' / f'arctic_reanchor_anchor_support_{SCENE}'
    native_name = next(name for name in SOURCES if name.startswith('arctic_'))
    live = sources / Path(native_name).stem / SCENE / 'results'
    manifest = json.loads((inputs / 'manifest.json').read_text())
    config = json.loads((reanchor / 'config.json').read_text())
    assert manifest['native_archive_sha256'] == SOURCES[native_name]
    assert sha256(ARCTIC_OUT / 'prepared.tar') == manifest['prepared_archive_sha256']
    prepared = json.loads((ARCTIC_OUT / 'initial_frames' / 'manifest.json').read_text())
    stage_dataset(prepared)
    checkpoint = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    assert {name: sha256(checkpoint / name) for name in config['checkpoint']} == config['checkpoint']
    for index, row in enumerate(manifest['inputs']):
        rgb = cv2.imread(str(inputs / 'model_rgb' / row['file']))
        mask = cv2.imread(str(inputs / 'model_masks' / row['file']), cv2.IMREAD_GRAYSCALE)
        assert rgb is not None and mask is not None
        assert hashlib.sha256(rgb[:, :, ::-1].copy().tobytes()).hexdigest() == row['model_rgb_sha256']
        assert hashlib.sha256((mask > 127).tobytes()).hexdigest() == row['model_mask_sha256']
        assert sha256(Path(row['source'])) == row['source_sha256']
        assert sha256(live / 'sam_masks' / f'{index:05d}.png') == row['mask_sha256']
    np.testing.assert_array_equal(np.load(live / 'traj.npy'), np.load(reference / 'traj.npy'))
    bridge_support = []
    for event in json.loads((reanchor / 'events.json').read_text()):
        frame = event['boundary']
        mask = cv2.imread(str(inputs / 'model_masks' / manifest['inputs'][frame]['file']), cv2.IMREAD_GRAYSCALE)
        with np.load(reanchor / f'bridge_{frame:04d}.npz') as evidence:
            y, x = evidence['sampled_pixels'].T
            foreground = mask[y, x] > 127
            row = dict(boundary=frame, accepted=event['accepted'])
            for subset in ('fit_mask', 'validation_mask', 'robust_fit_mask'):
                selected = evidence[subset]
                row[subset] = dict(total=int(selected.sum()), foreground=int((selected & foreground).sum()))
            bridge_support.append(row)
    report['audit'] = dict(frames=manifest['frames'], checkpoint=config['checkpoint'],
        input_manifest_sha256=sha256(inputs / 'manifest.json'),
        error_groups=error_groups(reanchor, reference, config['boundaries']),
        bridge_support=bridge_support,
        caveat='Foreground support counts describe the saved fit; no alternative connection is applied.')
    write_json(summary / 'comparison.json', report)
    archive_directory(summary, args.out / f'{args.tag}_object_history_comparison.tar')
    policy = summary / 'policy.json'
    # The fixed 26458 ORB control saved thresholds without the CLI detector tag.
    write_json(policy, dict(config['segmentation_policy'], detector='orb'))
    native_metrics = summary / 'native_metrics.json'
    write_json(native_metrics, json.loads((reference / 'summary.json').read_text())['metrics'])
    durations = []
    for repeat, order in enumerate(report['orders']):
        for arm in order:
            # All subprocesses have a hard timeout; leave packaging time to the wrapper.
            remaining = args.deadline - time.time() - 1800
            required = max(1800, 2 * max(durations, default=0))
            if remaining < required:
                report['stop_reason'] = 'Insufficient time for another arm plus packaging reserve'
                write_json(summary / 'comparison.json', report)
                archive_directory(summary, args.out / f'{args.tag}_object_history_comparison.tar')
                raise SystemExit(3)
            name = f'object_history_r{repeat}_{arm}'
            result = args.work / 'runs' / name
            if arm == 'native':
                result.mkdir()
                write_json(result / 'config.json', dict(inputs=str(inputs), scene=SCENE,
                    reference_trajectory=str(reference / 'traj.npy'),
                    profile_forward=profiled))
                command = [sys.executable, str(tools / 'kvt_segment_arctic.py'),
                           '--native-replay', str(result / 'config.json')]
            else:
                command = [sys.executable, str(tools / 'kvt_reanchor_maps.py'),
                    '--work', str(args.work), '--out', str(args.out), '--tag', args.tag,
                    '--inputs', str(inputs), '--native-metrics', str(native_metrics),
                    '--detector-config', str(policy), '--object-scene', SCENE, '--name', name]
                if not arm.startswith('reanchor'):
                    command.append('--disable-retirement')
                if arm == 'reanchor_twoway':
                    command.append('--two-way-bridge')
                if arm.endswith('_latest'):
                    command.append('--latest-anchor')
                if arm.startswith('three_frame') or '_three' in arm:
                    command.extend(['--local-keyframe-cap', '3'])
                if 'pinned' in arm:
                    command.append('--pin-rebuilds')
                if arm.endswith('_scale'):
                    command.append('--pin-scale')
                if arm.endswith('_novelty'):
                    command.append('--novelty-refresh')
                if arm == 'retrieval_bank':
                    command.extend(['--retrieval-bank', '4'])
                if profiled:
                    command.append('--profile-forward')
            report['active_run'] = dict(name=name, command=command)
            write_json(summary / 'comparison.json', report)
            started = time.monotonic()
            subprocess.run(command, check=True, timeout=remaining)
            durations.append(time.monotonic() - started)
            trajectory = np.load(result / 'traj.npy')
            assert trajectory.shape == (manifest['frames'], 4, 4) and np.isfinite(trajectory).all()
            with np.load(result / 'evaluation.npz') as q, np.load(reference / 'evaluation.npz') as n:
                np.testing.assert_array_equal(q['pair_end_frames'], n['pair_end_frames'])
            metrics = json.loads((result / 'summary.json').read_text())
            baseline = json.loads(native_metrics.read_text())
            ratios = {key: metrics['metrics'][key] / baseline[key]
                      for key in ('ate_m', 'rpe_translation_m', 'translation_p99_m')}
            row = dict(name=name, arm=arm, repeat=repeat, wall_seconds=durations[-1],
                       summary=metrics, native_ratios=ratios,
                       quality_pass=all(value <= 1.05 for value in ratios.values()))
            if arm == 'reanchor':
                np.testing.assert_allclose(trajectory, np.load(reanchor / 'traj.npy'), rtol=1e-4, atol=1e-4)
            if args.stage == 'three-frame' and arm == 'no_retirement':
                control = sources / 'tum_26471_object_history_r0_no_retirement' / 'object_history_r0_no_retirement'
                np.testing.assert_allclose(trajectory, np.load(control / 'traj.npy'), rtol=1e-4, atol=1e-4)
            if args.stage == 'pinned' and arm == 'three_frame':
                control = sources / 'tum_26472_object_history_r0_three_frame' / 'object_history_r0_three_frame'
                np.testing.assert_allclose(trajectory, np.load(control / 'traj.npy'), rtol=1e-4, atol=1e-4)
            if args.stage in ('novelty', 'retrieval') and arm == 'three_frame_pinned':
                control = sources / 'tum_26488_object_history_r0_three_frame_pinned' / 'object_history_r0_three_frame_pinned'
                np.testing.assert_allclose(trajectory, np.load(control / 'traj.npy'), rtol=1e-4, atol=1e-4)
            compared = [r for r in report['runs'] if r['arm'] == COMPARE.get(arm)]
            if compared:
                control = compared[0]['summary']['metrics']
                row['compared_arm'] = compared[0]['arm']
                row['compared_ratios'] = {key: metrics['metrics'][key] / control[key] for key in ratios}
                row['improves_compared'] = all(value < 1 for value in row['compared_ratios'].values())
            if 'pinned' in arm:
                row['pins'] = [{k: v for k, v in json.loads(line).items()
                                if k in ('frame', 'pin_rotation_deg', 'pin_position_step', 'rebuild_scale',
                                         'trigger')}
                               for line in (result / 'inference.jsonl').read_text().splitlines()
                               if json.loads(line)['kind'] == 'update_total']
                scheduled = (manifest['frames'] - 1) // 50
                assert len(row['pins']) >= scheduled if arm.endswith('_novelty') else len(row['pins']) == scheduled
                row['novelty_refreshes'] = sum(x.get('trigger') == 'novelty' for x in row['pins'])
                assert all('pin_rotation_deg' in x for x in row['pins'])
            if arm == 'retrieval_bank':
                row['admissions'] = [{k: v for k, v in json.loads(line).items()
                                      if k in ('frame', 'bank_ids', 'scores', 'fit_scale', 'fit_accepted',
                                               'validation_median', 'camera_rotation_deg')}
                                     for line in (result / 'inference.jsonl').read_text().splitlines()
                                     if json.loads(line)['kind'] == 'update_total']
            if arm == 'three_frame' and args.stage == 'three-frame':
                control = next(r for r in report['runs'] if r['arm'] == 'no_retirement')
                row['two_image_ratios'] = {key: metrics['metrics'][key] / control['summary']['metrics'][key]
                                           for key in ratios}
                row['improves_two_image'] = all(value < 1 for value in row['two_image_ratios'].values())
            if arm != 'native':
                actual = json.loads((result / 'config.json').read_text())['boundaries']
                assert actual == (config['boundaries'] if arm.startswith('reanchor') else [0, manifest['frames']])
                decisions = [json.loads(s) for s in (result / 'segmentation_live.jsonl').read_text().splitlines()]
                expected = [json.loads(s) for s in (reanchor / 'segmentation_live.jsonl').read_text().splitlines()]
                assert [{k: v for k, v in r.items() if k != 'seconds'} for r in decisions] == [
                    {k: v for k, v in r.items() if k != 'seconds'} for r in expected]
                if not arm.startswith('reanchor'):
                    assert not metrics['events']
                actual_config = json.loads((result / 'config.json').read_text())
                assert bool(actual_config['pin_rebuilds']) == ('pinned' in arm)
                assert bool(actual_config['pin_scale']) == arm.endswith('_scale')
                assert bool(actual_config['novelty_refresh']) == arm.endswith('_novelty')
                assert actual_config['retrieval_bank'] == (4 if arm == 'retrieval_bank' else None)
                assert bool(actual_config['two_way_bridge']) == (arm == 'reanchor_twoway')
                assert bool(actual_config['latest_anchor']) == arm.endswith('_latest')
                assert actual_config['local_keyframe_cap'] == (3 if (arm.startswith('three_frame') or '_three' in arm) else 2)
                if arm.endswith('_latest'):
                    row['connections'] = [{k: e.get(k) for k in ('boundary', 'scale', 'latest_frame',
                                                                  'anchor_disagreement_deg', 'validation_median')}
                                          for e in metrics['events']]
                    assert all(c['anchor_disagreement_deg'] is not None for c in row['connections'])
                if arm == 'reanchor_twoway':
                    row['connections'] = [{k: e.get(k) for k in (
                        'boundary', 'point_fit_scale', 'two_way_scale', 'frame_b_scale', 'latest_frame',
                        'latest_frame_scale', 'two_way_inliers', 'validation_median')} for e in metrics['events']]
                    assert all(c['two_way_scale'] is not None for c in row['connections'])
                row['error_groups'] = error_groups(result, reference, actual)
            archive_directory(result, args.out / f'{args.tag}_{name}.tar')
            row['archive_sha256'] = sha256(args.out / f'{args.tag}_{name}.tar')
            report['runs'].append(row)
            del report['active_run']
            write_json(summary / 'comparison.json', report)
            archive_directory(summary, args.out / f'{args.tag}_object_history_comparison.tar')
            print('OBJECT HISTORY ARM', name, 'quality_pass', row['quality_pass'], flush=True)
    report['complete'] = True
    write_json(summary / 'comparison.json', report)
    archive_directory(summary, args.out / f'{args.tag}_object_history_comparison.tar')
    (args.work / 'JOB_OK').write_text('Fixed object history diagnostics complete; inspect quality separately\n')
    print('OBJECT HISTORY COMPLETE', args.stage, len(report['runs']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--stage', choices=('pilot', 'overnight', 'three-frame', 'pinned', 'novelty',
                                            'retrieval', 'two-way', 'latest-anchor'),
                        required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--reviewed-pilot', type=Path)
    parser.add_argument('--reviewed-sha256')
    main(parser.parse_args())
