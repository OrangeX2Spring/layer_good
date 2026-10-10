"""CAMP-only evaluation of the user's segmentation + reanchor design on every tracked sequence.

Groups: 'tum-short' (five fr1 sequences and fr3 office), 'tum-long' (fr2 large with/no
loop), 'arctic' (box, ketchup, espressomachine), and 'tum-long-arctic' (both of the
latter in one job, since one running job per user is allowed). Arms per sequence:
- native: KV-Tracker unchanged, fresh in-job run on the same saved pixels;
- segmented: the current design (live ORB cuts at 25 inliers, two-image segment banks,
  pose-pinned point-fit connection; objects use the 26458 anchor-support policy);
- anchor_schedule: anchor-only segments cut every 48 frames, no detector;
- anchor_signal: anchor-only segments cut by a tracker-side signal (anchor covisibility
  below 0.7 for TUM; viewing direction about the object centre beyond 20 degrees for
  ARCTIC), no detector.
The thresholds are fixed, uncalibrated choices (FINDINGS 2026-10-10 detector analysis).
No GT in tracking. The deadline is the job's own Slurm end time.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_tum_run import prepare, write_json
from kvt_tum_sweep import archive_directory, archive_inputs, release_page_cache

GROUPS = {'tum-short': ('freiburg1_xyz', 'freiburg1_rpy', 'freiburg1_desk2', 'freiburg1_desk',
                        'freiburg1_room', 'freiburg3_long_office_household'),
          'tum-long': ('freiburg2_large_with_loop', 'freiburg2_large_no_loop'),
          'arctic': ('box_grab_01', 'ketchup_grab_01', 'espressomachine_grab_01')}
GROUPS['tum-long-arctic'] = GROUPS['tum-long'] + GROUPS['arctic']
FRAMES = dict(freiburg1_xyz=798, freiburg1_rpy=723, freiburg1_desk2=640, freiburg1_desk=613,
              freiburg1_room=1362, freiburg3_long_office_household=2585,
              freiburg2_large_with_loop=5182, freiburg2_large_no_loop=3359)
# Job 25680 native ATE in metres (FINDINGS), recorded beside the fresh native, not asserted.
NATIVE_25680_ATE = dict(freiburg1_xyz=0.02106388381982558, freiburg1_rpy=0.045306488509958096,
                        freiburg1_desk2=0.08293419670248443, freiburg1_desk=0.05904974919224374,
                        freiburg1_room=0.3613912169161852, freiburg3_long_office_household=0.061902558893707854,
                        freiburg2_large_with_loop=1.1417495447202815, freiburg2_large_no_loop=2.1168791505126197)
TUM_POLICY = dict(detector='orb', minimum_inliers=25)  # 26159 live office policy, no maximum length
ARMS = ('native', 'segmented', 'anchor_schedule', 'anchor_signal')
KEYS = ('ate_m', 'rpe_translation_m', 'translation_p99_m')
RESERVE_SECONDS = 900  # left for archiving after the last run


def main(args):
    tools = Path(__file__).resolve().parent
    comparison = args.work / 'runs' / f'segment_eval_{args.group}'
    comparison.mkdir(parents=True)
    report = dict(group=args.group, sequences=GROUPS[args.group], arms=ARMS, complete=False, runs=[],
                  thresholds=dict(schedule_frames=48, covisibility=.7, object_view_degrees=20.),
                  deadline=args.deadline, gt_used_by_tracker=False,
                  caveat='One run per arm; thresholds uncalibrated; timings exclude model load and evaluation.')
    durations = []

    def save():
        write_json(comparison / 'comparison.json', report)
        archive_directory(comparison, args.out / f'{args.tag}_segment_eval_{args.group}_comparison.tar')

    def run(command):
        remaining = args.deadline - time.time() - RESERVE_SECONDS
        if remaining < max(600, 2 * max(durations, default=0)):
            report['stop_reason'] = 'Insufficient time before the Slurm end time for another run'
            save()
            raise SystemExit(3)
        started = time.monotonic()
        subprocess.run(command, check=True, timeout=remaining)
        durations.append(time.monotonic() - started)
        (args.work / 'JOB_OK').unlink(missing_ok=True)  # only this driver declares completion

    def tracker(inputs, native_metrics, name, scene, extra):
        command = [sys.executable, str(tools / 'kvt_reanchor_maps.py'), '--work', str(args.work),
                   '--out', str(args.out), '--tag', args.tag, '--inputs', str(inputs),
                   '--native-metrics', str(native_metrics), '--name', name, *extra]
        run(command + (['--object-scene', scene] if scene in GROUPS['arctic'] else []))
        return args.work / 'runs' / name

    def record(scene, arm, result, native_dir, frames):
        objects = scene in GROUPS['arctic']
        summary = json.loads((result / 'summary.json').read_text())
        if arm != 'native':
            with np.load(result / 'evaluation.npz') as current, np.load(native_dir / 'evaluation.npz') as base:
                for key in (('pair_end_frames',) if objects else
                            ('rgb_indices', 'gt_indices', 'timestamps', 'rpe_pair_start_indices')):
                    np.testing.assert_array_equal(current[key], base[key])
        metrics, timing = summary['metrics'], summary['timing']
        row = dict(scene=scene, arm=arm, frames=frames, metrics=metrics, timing=timing,
                   fps=frames / timing['tracking_seconds'],
                   peak_allocated_gib=timing['peak_allocated_bytes'] / 2 ** 30,
                   cuts=None if arm == 'native' else len(summary['cut_frames']))
        base = [r for r in report['runs'] if r['scene'] == scene and r['arm'] == 'native']
        if base:
            row['native_ratios'] = {k: metrics[k] / base[0]['metrics'][k] for k in KEYS}
        control = [r for r in report['runs'] if r['scene'] == scene and r['arm'] == 'segmented']
        if control and arm.startswith('anchor'):
            row['segmented_ratios'] = {k: metrics[k] / control[0]['metrics'][k] for k in KEYS}
        if arm == 'native' and not objects:
            row['native_25680_ate_m'] = NATIVE_25680_ATE[scene]
        report['runs'].append(row)
        save()
        print('SEGMENT EVAL', scene, arm, 'ATE/RPE/p99 m', [round(metrics[k], 4) for k in KEYS],
              'fps', round(row['fps'], 2), 'peak GiB', round(row['peak_allocated_gib'], 2),
              'cuts', row['cuts'], flush=True)

    for scene in GROUPS[args.group]:
        objects = scene in GROUPS['arctic']
        if scene == GROUPS['arctic'][0]:
            # Live native with SAM masks, saved masked inputs, matched native replay and the
            # current design's anchor-support control, exactly as job 26458 for ketchup.
            run([sys.executable, str(tools / 'kvt_segment_arctic.py'), '--work', str(args.work),
                 '--out', str(args.out), '--tag', args.tag, '--object-coverage',
                 '--coverages', 'anchor_support', '--scenes', *GROUPS['arctic']])
        if objects:
            inputs = args.work / 'inputs' / scene
            native_dir = args.work / 'runs' / f'arctic_native_replay_{scene}'
            frames = json.loads((inputs / 'manifest.json').read_text())['frames']
            control = args.work / 'runs' / f'arctic_reanchor_anchor_support_{scene}'
            signal = 'object_view'
        else:
            source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
            inputs = args.work / 'inputs' / scene
            prepare(source, inputs, 308, .02)
            (inputs / 'archive.sha256').write_text(f'{sha256(source)}  {source}\n')
            frames = len(json.loads((inputs / 'manifest.json').read_text())['inputs'])
            assert frames == FRAMES[scene], (scene, frames)
            packed = args.out / f'{args.tag}_segment_eval_{scene}_inputs.tar'
            archive_inputs(inputs, packed)
            release_page_cache(source, inputs, packed)
            native_dir = args.work / 'runs' / f'eval_{scene}_native'
            native_dir.mkdir()
            write_json(native_dir / 'config.json', dict(task='camera', inputs=str(inputs)))
            run([sys.executable, str(tools / 'kvt_segment_arctic.py'), '--native-replay',
                 str(native_dir / 'config.json')])
            archive_directory(native_dir, args.out / f'{args.tag}_eval_{scene}_native.tar')
            signal = 'covisibility'
        native_metrics = native_dir / 'native_metrics.json'
        write_json(native_metrics, json.loads((native_dir / 'summary.json').read_text())['metrics'])
        record(scene, 'native', native_dir, native_dir, frames)
        if not objects:
            policy = comparison / 'segmented_policy.json'
            write_json(policy, TUM_POLICY)
            control = tracker(inputs, native_metrics, f'eval_{scene}_segmented', scene,
                              ['--detector-config', str(policy)])
        record(scene, 'segmented', control, native_dir, frames)
        for arm, cut in (('anchor_schedule', 'schedule'), ('anchor_signal', signal)):
            result = tracker(inputs, native_metrics, f'eval_{scene}_{arm}', scene, ['--anchor-segments', cut])
            record(scene, arm, result, native_dir, frames)
    report['complete'] = True
    save()
    (args.work / 'JOB_OK').write_text(f'Segment evaluation {args.group} complete; read quality per arm\n')
    print('SEGMENT EVAL COMPLETE', args.group, len(report['runs']), 'runs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--group', choices=tuple(GROUPS), required=True)
    parser.add_argument('--deadline', type=float, required=True, help='Unix time of the Slurm end time')
    main(parser.parse_args())
