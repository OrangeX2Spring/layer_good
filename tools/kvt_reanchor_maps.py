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
    assert len(manifest['inputs']) == 2585

    boundaries = list(BOUNDARIES)
    detector = None
    if args.segmentation is not None:
        from kvt_segment_detector import SegmentDetector, POLICY
        segmentation = json.loads(args.segmentation.read_text())
        expected_rows = [json.loads(line) for line in
                         args.segmentation.with_name('scores.jsonl').read_text().splitlines()]
        assert segmentation['frames'] == len(expected_rows) == 2585
        detector = SegmentDetector(segmentation['minimum_inliers'])
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        boundaries = [0, 2585]  # Append only when the live detector triggers.
    result = args.work / 'runs' / args.name
    result.mkdir(parents=True)
    write_json(result / 'config.json', dict(boundaries=boundaries, sources=SOURCES,
        checkpoint=checkpoint, keyframes='segment first frame + latest, rebuilt every 50; '
        'later rebuilds rescaled to the first via the anchor pointmap',
        connection='pose-anchored at b, point-fit scale, always committed, 49-frame delay',
        history='old KV and images deleted at b', native=NATIVE, margin=MARGIN,
        short_tail='at EOF rebuild [anchor,last] and commit pending connection',
        segmentation_policy=dict(POLICY, minimum_inliers=segmentation['minimum_inliers'])
            if detector is not None else None,
        gt_used_by_tracker=False, seed=0))
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())
    if detector is not None:
        cv2.setRNGSeed(0)

    def save_bridge(event, evidence):
        name = f'bridge_{event["boundary"]:04d}'
        write_json(result / f'{name}.json', event)
        np.savez(result / f'{name}.npz', **evidence)
        print('REANCHOR', event['boundary'], 'scale', round(event['scale'], 4),
              'point-fit checks', event['checks'], flush=True)

    with (result / 'inference.jsonl').open('w') as stream, torch.inference_mode():
        def log(row):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        tracker = ReanchorMaps(model, boundaries, log, save_bridge)
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
                assert {k: v for k, v in decision.items() if k != 'seconds'} == {
                    k: v for k, v in expected_rows[frame].items() if k not in ('seconds', 'timestamp')}, frame
                detector_rows.append(decision)
                if decision['boundary']:
                    boundaries.insert(-1, frame)
            local_poses.append(tracker.step(image, frame))
        tracker.finish(image, frame)
        torch.cuda.synchronize()
        timing = dict(tracking_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        timing['detector_seconds'] = sum(row['seconds'] for row in detector_rows)
        timing['synchronous_frames_per_second'] = len(local_poses) / timing['tracking_seconds']
        timing['includes'] = 'pixels, hashing, live detector, tracking, geometry, bridge and logging; excludes load/evaluation'
    if detector is not None:
        assert boundaries == segmentation['boundaries'] + [2585]
        with (result / 'segmentation_live.jsonl').open('w') as stream:
            for row in detector_rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    config = json.loads((result / 'config.json').read_text())
    config['boundaries'] = boundaries
    write_json(result / 'config.json', config)
    local_poses = np.asarray(local_poses)
    assert local_poses.shape == (2585, 4, 4) and np.isfinite(local_poses).all()
    assert tracker.pending is None and len(tracker.transforms) == len(boundaries) - 1
    assert [e['boundary'] for e in tracker.events] == boundaries[1:-1]
    rows = [json.loads(x) for x in (result / 'inference.jsonl').read_text().splitlines()]
    assert all(r['input_images'] <= 2 for r in rows if 'input_images' in r)
    assert all(len(r['bank_ids']) <= 2 for r in rows if r['kind'] == 'query')
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

    metrics = evaluate(inputs, result, .02)
    with np.load(result / 'evaluation.npz') as q:
        rpe = q['rpe_translation_per_pair_m']
        metrics['translation_p99_m'] = float(np.quantile(rpe, .99))
        seams = np.isin(q['rpe_pair_start_indices'], np.array(boundaries[1:-1]) - 1)
        metrics['seams'] = dict(starts=q['rpe_pair_start_indices'][seams].tolist(),
                                translation_m=rpe[seams].tolist())
    assert (metrics['evaluated_frames'], metrics['rpe_pairs']) == (
        NATIVE['evaluated_frames'], NATIVE['rpe_pairs']), 'GT association differs from native'
    checks = {key: metrics[key] <= MARGIN * NATIVE[key]
              for key in ('ate_m', 'rpe_translation_m', 'translation_p99_m')}
    timing['max_cache_bytes'] = max(row['cache_bytes'] for row in rows)
    for kind in ('bootstrap', 'query', 'rebuild', 'shared_geometry', 'update_total'):
        timing[kind + '_seconds'] = sum(row['seconds'] for row in rows if row['kind'] == kind)
    write_json(result / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), seed=0, resize_dim=308))
    write_json(result / 'summary.json', dict(metrics=metrics, native=NATIVE, margin=MARGIN,
        native_checks=checks, objective_achieved=all(checks.values()), timing=timing,
        events=tracker.events))
    archive_directory(result, args.out / f'{args.tag}_{args.name}.tar')
    write_json(args.work / f'{args.name}_archive.json',
               {f'{args.tag}_{args.name}.tar': sha256(args.out / f'{args.tag}_{args.name}.tar')})
    if detector is None:
        (args.work / 'JOB_OK').write_text('Execution passed; read native_checks in summary.json\n')
    print('REANCHOR MAPS COMPLETE: ATE %.4f m (native %.4f), RPE %.4f (%.4f), p99 %.4f (%.4f), '
          'within 5%% of native: %s' % (metrics['ate_m'], NATIVE['ate_m'],
          metrics['rpe_translation_m'], NATIVE['rpe_translation_m'],
          metrics['translation_p99_m'], NATIVE['translation_p99_m'], all(checks.values())),
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--segmentation', type=Path)
    parser.add_argument('--name', default='reanchor_maps')
    main(parser.parse_args())
