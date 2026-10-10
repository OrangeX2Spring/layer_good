"""One full-office pass: manual segments, reanchoring and immediate history deletion.

Scored as one continuous trajectory against native KV-Tracker. CAMP only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_office_segments import BOUNDARIES
from kvt_tum_run import evaluate, write_json
from kvt_tum_sweep import archive_directory

# Verified on the Mac: job 26127 inputs (the same pixels as 26115) and its context.
SOURCES = {
    'tum_26127_oracle_maps_inputs.tar': '9ecd7546a56f2135bfbfc2d33a1be4de45c1670cb8cf56e1bedf2639a2933b33',
    'tum_26127_context.tar': 'e7c21e5b9505e1d96945fb6243dc3703d8a2c1c0a5975f2593c13ef72d9db39d',
}
# Native 25680 office, one alignment over all frames, same evaluator and inputs.
# Mac archive tum_25680_freiburg3_long_office_household_original.tar, SHA256 e3d56f66…e125.
NATIVE = dict(ate_m=0.061902558893707854, rpe_translation_m=0.011884587661546165,
              translation_p99_m=0.03256165204535927, evaluated_frames=2583, rpe_pairs=2582)
MARGIN = 1.05


def main(args):
    import cv2
    import torch
    from huggingface_hub import snapshot_download
    from kv_tracker.map_handoff import transform_pose
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
    from kv_tracker.reanchor_maps import ReanchorMaps
    from kv_tracker.retrieval_bank import RetrievalBank

    for name, expected in SOURCES.items():
        assert sha256(args.out / name) == expected, f'{name} differs from the reviewed archive'
    with tarfile.open(args.out / 'tum_26127_context.tar') as packed:
        expected_checkpoint = json.load(packed.extractfile('./oracle_sources.json'))['checkpoint']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    checkpoint = {name: sha256(snapshot / name) for name in expected_checkpoint}
    assert checkpoint == expected_checkpoint, 'Checkpoint differs from job 26127'
    if args.inputs is None:
        with tarfile.open(args.out / 'tum_26127_oracle_maps_inputs.tar') as packed:
            packed.extractall(args.work / 'inputs')
    inputs = args.inputs if args.inputs is not None else args.work / 'inputs' / 'office'
    manifest = json.loads((inputs / 'manifest.json').read_text())
    frames = len(manifest['inputs'])
    native = json.loads(args.native_metrics.read_text()) if args.native_metrics else NATIVE
    sources = {str(inputs): sha256(inputs / 'manifest.json')} if args.native_metrics else SOURCES
    if not args.native_metrics:
        assert frames == 2585
    if args.object_scene:
        assert args.inputs is not None and args.native_metrics is not None
    if args.disable_retirement:
        assert args.detector_config is not None and args.segmentation is None
    assert args.disable_retirement or not args.pin_rebuilds
    assert args.pin_rebuilds or not args.pin_scale
    assert not args.shared_scale or (args.pin_rebuilds and not args.pin_scale and args.local_keyframe_cap == 3)
    assert args.shared_scale or not args.fuse_scale
    assert not args.novelty_refresh or (args.object_scene and args.disable_retirement)
    assert args.retrieval_bank is None or (args.disable_retirement and args.local_keyframe_cap == 2 and not (
        args.pin_rebuilds or args.shared_scale or args.novelty_refresh))
    assert not args.overlap_bridge or not (args.disable_retirement or args.retrieval_bank)
    assert not args.two_way_bridge or not (args.disable_retirement or args.retrieval_bank or args.overlap_bridge)
    assert not args.latest_anchor or not (args.disable_retirement or args.retrieval_bank or args.overlap_bridge
                                          or args.two_way_bridge)
    cap = args.retrieval_bank or args.local_keyframe_cap
    if args.native_metrics:
        assert args.detector_config is not None, 'Non-office runs require a live detector policy'

    boundaries = list(BOUNDARIES)
    detector = None
    if args.segmentation is not None or args.detector_config is not None:
        from kvt_segment_detector import SegmentDetector, FlowSegmentDetector, POLICY, FLOW_POLICY
        segmentation = json.loads((args.segmentation or args.detector_config).read_text())
        expected_rows = None
        if args.segmentation is not None:
            expected_rows = [json.loads(line) for line in
                             args.segmentation.with_name('scores.jsonl').read_text().splitlines()]
            assert segmentation['frames'] == len(expected_rows) == frames
        flow = segmentation['detector'] == 'flow'
        assert segmentation['detector'] in ('flow', 'orb')
        maximum = segmentation.get('maximum_segment_frames')
        coverage_mode = segmentation.get('coverage_mode', 'image')
        if coverage_mode == 'anchor_support':
            assert args.object_scene and not flow, 'Anchor-support coverage requires masked object inputs'
        detector = FlowSegmentDetector(segmentation.get('surviving_fraction', .25), maximum) \
            if flow else SegmentDetector(segmentation['minimum_inliers'], maximum,
                coverage_mode, segmentation.get('minimum_anchor_coverage', .5))
        detector_policy = dict(FLOW_POLICY, surviving_fraction=detector.surviving_fraction) if flow \
            else dict(POLICY, minimum_inliers=segmentation['minimum_inliers'])
        detector_policy['maximum_segment_frames'] = maximum
        if coverage_mode == 'anchor_support':
            detector_policy.update(coverage_mode=coverage_mode,
                                   minimum_anchor_coverage=detector.minimum_anchor_coverage)
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        boundaries = [0, frames]  # Append only when the live detector triggers.
    result = args.work / 'runs' / args.name
    result.mkdir(parents=True)
    write_json(result / 'config.json', dict(boundaries=boundaries, sources=sources,
        checkpoint=checkpoint, local_keyframe_cap=args.local_keyframe_cap,
        keyframes=(f'CPU memory of all admitted keyframes (fixed poses); GPU bank of the newest plus '
                   f'the {args.retrieval_bank - 1} best-covering stored keyframes; one Sim(3) over the '
                   'shared members per rebuild' if args.retrieval_bank else
                   ('segment first frame + previous + latest' if args.local_keyframe_cap == 3
                    else 'segment first frame + latest') + ', rebuilt every 50; '
                   'later rebuilds rescaled to the first via the anchor pointmap'),
        connection='none' if args.disable_retirement else
                   'pose-anchored at the old latest keyframe L, point-fit scale at b, always committed, '
                   '49-frame delay' if args.latest_anchor else
                   'pose-anchored at b, ' + ('scale from old/new camera displacements on every 5th frame of '
                   'b..b+48 (old map kept until b+49)' if args.overlap_bridge else
                   'confidence-weighted Sim(3) scale over b and the old latest keyframe (two-way)'
                   if args.two_way_bridge else 'point-fit scale') +
                   ', always committed, 49-frame delay',
        overlap_bridge=args.overlap_bridge, two_way_bridge=args.two_way_bridge, latest_anchor=args.latest_anchor,
        history=f'first anchor with bounded {args.local_keyframe_cap}-image bank; no map retirement' if args.disable_retirement else
                'old KV and images deleted at b', native=native, margin=MARGIN,
        short_tail='at EOF rebuild [anchor,last] and commit pending connection',
        disable_retirement=args.disable_retirement,
        pin_rebuilds=args.pin_rebuilds and 'each rebuild rigidly pinned to the outgoing pose of its frame',
        pin_scale=args.pin_scale and 'rebuild scale from the outgoing bank depths of the refresh frame',
        shared_scale=args.shared_scale and 'rebuild scale from the previous keyframe shared by both rebuilds',
        fuse_scale=args.fuse_scale and 'geometric mean of the shared-keyframe chain and the anchor scale',
        novelty_refresh=args.novelty_refresh and 'extra rebuild after frame 49 when native check_if_keyframe fires',
        retrieval_bank=args.retrieval_bank,
        admission=args.retrieval_bank and ('native check_if_keyframe against all stored keyframes'
                                           if args.object_scene else 'every 50 frames, uncapped'),
        segmentation_policy=detector_policy if detector is not None else None,
        gt_used_by_tracker=False, seed=0))
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())
    if args.profile_forward:
        from kvt_inference_timing import InferenceTiming
        forward_timing = InferenceTiming(model)
    if detector is not None:
        cv2.setRNGSeed(0)

    def save_bridge(event, evidence):
        name = f'bridge_{event["boundary"]:04d}'
        write_json(result / f'{name}.json', event)
        np.savez(result / f'{name}.npz', **evidence)
        print('REANCHOR', event['boundary'], 'scale', round(event['scale'], 4),
              *(['overlap scale', round(event['overlap_scale'], 4)] if 'overlap_scale' in event else []),
              *(['two-way scale', round(event['two_way_scale'], 4), 'b/L', round(event['frame_b_scale'], 4),
                 round(event['latest_frame_scale'], 4)] if 'two_way_scale' in event else []),
              *(['L', event['latest_frame'], 'anchor disagreement deg', round(event['anchor_disagreement_deg'], 2)]
                if 'anchor_disagreement_deg' in event else []),
              'point-fit checks', event['checks'], flush=True)

    with (result / 'inference.jsonl').open('w') as stream, torch.inference_mode():
        def log(row):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        novelty = None
        if args.novelty_refresh or (args.retrieval_bank and args.object_scene):
            from main import check_if_keyframe  # native object keyframe rule, unchanged thresholds
            novelty = check_if_keyframe
        if args.retrieval_bank:
            tracker = RetrievalBank(model, lambda row: log(dict(row, segment_start=0, global_frame=row['frame'])),
                                    capacity=args.retrieval_bank, novelty=novelty)
        else:
            tracker = ReanchorMaps(model, boundaries, log, save_bridge,
                                   local_keyframe_cap=args.local_keyframe_cap,
                                   pin_rebuilds=args.pin_rebuilds, pin_scale=args.pin_scale,
                                   shared_scale=args.shared_scale, fuse_scale=args.fuse_scale,
                                   novelty_refresh=novelty, overlap_stride=5 if args.overlap_bridge else None,
                                   two_way=args.two_way_bridge, latest_anchor=args.latest_anchor)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        local_poses = []
        detector_rows = []
        for frame, row in enumerate(manifest['inputs']):
            image = cv2.imread(str(inputs / 'model_rgb' / Path(row['file']).name))
            assert image is not None
            image = image[:, :, ::-1].copy()
            assert hashlib.sha256(image.tobytes()).hexdigest() == row['model_rgb_sha256'], frame
            if detector is not None:
                decision, _ = detector.step(image, frame)
                # Saved decisions are assertions only, never inputs to the live policy.
                if expected_rows is not None:
                    assert {k: v for k, v in decision.items() if k != 'seconds'} == {
                        k: v for k, v in expected_rows[frame].items() if k not in ('seconds', 'timestamp')}, frame
                detector_rows.append(decision)
                if decision['boundary'] and not args.disable_retirement:
                    boundaries.insert(-1, frame)
            mask = None
            if novelty is not None:
                mask = cv2.imread(str(inputs / 'model_masks' / Path(row['file']).name), cv2.IMREAD_GRAYSCALE) > 127
                assert hashlib.sha256(mask.tobytes()).hexdigest() == row['model_mask_sha256'], frame
                mask = torch.from_numpy(mask)
            local_poses.append(tracker.step(image, frame, mask))
        tracker.finish(image, frame)
        torch.cuda.synchronize()
        timing = dict(tracking_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        timing['detector_seconds'] = sum(row['seconds'] for row in detector_rows)
        timing['synchronous_frames_per_second'] = len(local_poses) / timing['tracking_seconds']
        timing['includes'] = 'pixels, hashing, live detector, tracking, geometry, bridge and logging; excludes load/evaluation'
    if detector is not None:
        if expected_rows is not None:
            assert boundaries == segmentation['boundaries'] + [frames]
        with (result / 'segmentation_live.jsonl').open('w') as stream:
            for row in detector_rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    config = json.loads((result / 'config.json').read_text())
    config['boundaries'] = boundaries
    write_json(result / 'config.json', config)
    local_poses = np.asarray(local_poses)
    assert local_poses.shape == (frames, 4, 4) and np.isfinite(local_poses).all()
    assert tracker.pending is None and len(tracker.transforms) == len(boundaries) - 1
    assert [e['boundary'] for e in tracker.events] == boundaries[1:-1]
    rows = [json.loads(x) for x in (result / 'inference.jsonl').read_text().splitlines()]
    assert all(r['input_images'] <= cap for r in rows if 'input_images' in r)
    assert all(len(r['bank_ids']) <= cap for r in rows if r['kind'] == 'query')
    np.save(result / 'local_traj.npy', local_poses)
    global_poses = np.concatenate([np.stack([
        transform_pose(torch.from_numpy(pose), transform).numpy()
        for pose in local_poses[start:end]])
        for start, end, transform in zip(boundaries[:-1], boundaries[1:], tracker.transforms, strict=True)])
    np.save(result / 'traj.npy', global_poses)
    write_json(result / 'events.json', tracker.events)
    write_json(result / 'transforms.json', [dict(start=start, scale=float(s),
        rotation=r.tolist(), translation=t.tolist())
        for start, (s, r, t) in zip(boundaries[:-1], tracker.transforms, strict=True)])

    if args.object_scene:
        from kvt_patch_select import evaluate_object
        metrics = evaluate_object(args.object_scene, result, frames)
    else:
        metrics = evaluate(inputs, result, .02)
    with np.load(result / 'evaluation.npz') as q:
        rpe = q['translation_m'] if args.object_scene else q['rpe_translation_per_pair_m']
        pair_starts = q['pair_end_frames'] - 1 if args.object_scene else q['rpe_pair_start_indices']
        metrics['translation_p99_m'] = float(np.quantile(rpe, .99))
        seams = np.isin(pair_starts, np.array(boundaries[1:-1]) - 1)
        metrics['seams'] = dict(starts=pair_starts[seams].tolist(),
                                translation_m=rpe[seams].tolist())
    assert (metrics['evaluated_frames'], metrics['rpe_pairs']) == (
        native['evaluated_frames'], native['rpe_pairs']), 'GT association differs from native'
    checks = {key: metrics[key] <= MARGIN * native[key]
              for key in ('ate_m', 'rpe_translation_m', 'translation_p99_m')}
    timing['max_cache_bytes'] = max(row['cache_bytes'] for row in rows)
    for kind in ('bootstrap', 'query', 'rebuild', 'shared_geometry', 'update_total'):
        timing[kind + '_seconds'] = sum(row['seconds'] for row in rows if row['kind'] == kind)
    write_json(result / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), seed=0, resize_dim=518 if args.object_scene else 308))
    if args.profile_forward:
        write_json(result / 'forward_timing.json', forward_timing.finish())
        from kvt_inference_timing import benchmark_cached_heads
        write_json(result / 'cached_head_benchmark.json', benchmark_cached_heads(model, image))
    write_json(result / 'summary.json', dict(metrics=metrics, native=native, margin=MARGIN,
        native_checks=checks, objective_achieved=all(checks.values()), timing=timing,
        events=tracker.events))
    archive_directory(result, args.out / f'{args.tag}_{args.name}.tar')
    write_json(args.work / f'{args.name}_archive.json',
               {f'{args.tag}_{args.name}.tar': sha256(args.out / f'{args.tag}_{args.name}.tar')})
    if detector is None:
        (args.work / 'JOB_OK').write_text('Execution passed; read native_checks in summary.json\n')
    print('REANCHOR MAPS COMPLETE: ATE %.4f m (native %.4f), RPE %.4f (%.4f), p99 %.4f (%.4f), '
          'within 5%% of native: %s' % (metrics['ate_m'], native['ate_m'],
          metrics['rpe_translation_m'], native['rpe_translation_m'],
          metrics['translation_p99_m'], native['translation_p99_m'], all(checks.values())),
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--native-metrics', type=Path)
    parser.add_argument('--object-scene', choices=('box_grab_01', 'ketchup_grab_01',
                                                  'espressomachine_grab_01'))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--segmentation', type=Path)
    selection.add_argument('--detector-config', type=Path)
    parser.add_argument('--name', default='reanchor_maps')
    parser.add_argument('--local-keyframe-cap', type=int, choices=(2, 3), default=2)
    parser.add_argument('--profile-forward', action='store_true')
    parser.add_argument('--disable-retirement', action='store_true',
                        help='Diagnostic: keep the first local map; still log the live detector')
    parser.add_argument('--pin-rebuilds', action='store_true',
                        help='Keep each refresh frame at its outgoing pose; requires --disable-retirement')
    parser.add_argument('--pin-scale', action='store_true',
                        help='Take rebuild scale from the refresh frame, not the anchor; requires --pin-rebuilds')
    parser.add_argument('--shared-scale', action='store_true',
                        help='Take rebuild scale from the shared previous keyframe; cap 3 with --pin-rebuilds')
    parser.add_argument('--fuse-scale', action='store_true',
                        help='Geometric mean of the shared-keyframe and anchor scales; requires --shared-scale')
    parser.add_argument('--novelty-refresh', action='store_true',
                        help='Object scenes: also rebuild when native check_if_keyframe fires after frame 49')
    parser.add_argument('--overlap-bridge', action='store_true',
                        help='Keep the old map until b+49 and take connection scale from shared displacements')
    parser.add_argument('--latest-anchor', action='store_true',
                        help='Pin connection rotation/position on the old latest keyframe instead of b')
    parser.add_argument('--two-way-bridge', action='store_true',
                        help='Connection scale from b and the old latest keyframe, each seen by both maps')
    parser.add_argument('--retrieval-bank', type=int, choices=(3, 4, 5),
                        help='GPU bank size retrieved from a CPU keyframe memory; requires --disable-retirement')
    main(parser.parse_args())
