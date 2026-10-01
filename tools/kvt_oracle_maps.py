"""One full-office pass with oracle banks and five delayed handoffs. CAMP only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

import numpy as np

from kvt_office_segments import BOUNDARIES
from kvt_tum_run import prepare, evaluate, write_json
from kvt_tum_sweep import archive_inputs, archive_directory


def main(args):
    import cv2
    import torch
    from huggingface_hub import snapshot_download
    from kv_tracker.oracle_maps import OracleMaps
    from kv_tracker.map_handoff import transform_pose
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32

    # Read archived references only; no baseline inference or GT enters tracking.
    references, provenance = [], {}
    for start, end in zip(BOUNDARIES, BOUNDARIES[1:]):
        name = f'office_{start:04d}_{end - 1:04d}'
        reference = {}
        for suffix in ('', '_inputs'):
            path = args.out / f'tum_26115_{name}{suffix}.tar'
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8388608), b''):
                    digest.update(chunk)
            provenance[path.name] = digest.hexdigest()
            with tarfile.open(path) as packed:
                if suffix:
                    reference['manifest'] = json.load(packed.extractfile(f'{name}/manifest.json'))
                else:
                    reference['traj'] = np.load(io.BytesIO(packed.extractfile(f'{name}/traj.npy').read()))
                    reference['quality'] = dict(np.load(io.BytesIO(
                        packed.extractfile(f'{name}/evaluation.npz').read())))
        references.append(reference)
    with tarfile.open(args.out / 'tum_26115_context.tar') as packed:
        expected_checkpoint = json.load(packed.extractfile('./checkpoint.json'))['files']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    checkpoint = {}
    for name in expected_checkpoint:
        digest = hashlib.sha256()
        with (snapshot / name).open('rb') as source:
            for chunk in iter(lambda: source.read(8388608), b''):
                digest.update(chunk)
        checkpoint[name] = digest.hexdigest()
    assert checkpoint == expected_checkpoint, 'Checkpoint differs from oracle'
    write_json(args.work / 'oracle_sources.json', dict(archives=provenance, checkpoint=checkpoint))
    inputs = args.work / 'inputs' / 'office'
    archive = Path('/mnt/datasets/tum-rgbd/rgbd_dataset_freiburg3_long_office_household.zip')
    manifest = prepare(archive, inputs, 308, .02, count=2585)
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for chunk in iter(lambda: source.read(8388608), b''):
            digest.update(chunk)
    (inputs / 'archive.sha256').write_text(f'{digest.hexdigest()}  {archive}\n')
    for reference, start, end in zip(references, BOUNDARIES, BOUNDARIES[1:]):
        for actual, expected in zip(manifest['inputs'][start:end],
                                    reference['manifest']['inputs'], strict=True):
            for key in ('model_rgb_sha256', 'timestamp', 'source_sha256'):
                assert actual[key] == expected[key], (start, key)
    archive_inputs(inputs, args.out / f'{args.tag}_oracle_maps_inputs.tar')
    result = args.work / 'runs' / 'oracle_maps'
    result.mkdir(parents=True)
    write_json(result / 'config.json', dict(boundaries=BOUNDARIES, local_keyframes=[0, 49],
        overlap_delay_frames=49, geometry='old-bank dense query of new anchor', seed=0,
        oracle_job=26115, inference_passes=1, gt_used_by_tracker=False))
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())

    def save_bridge(event, evidence):
        name = f'bridge_{event["boundary"]:04d}'
        write_json(result / f'{name}.json', event)
        np.savez(result / f'{name}.npz', **evidence)
        print('BRIDGE', event['boundary'], 'accepted=', event['accepted'], flush=True)

    with (result / 'inference.jsonl').open('w') as stream, torch.inference_mode():
        def log(row):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        tracker = OracleMaps(model, BOUNDARIES, log, save_bridge)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        local_poses = []
        for frame, row in enumerate(manifest['inputs']):
            image = cv2.imread(str(inputs / 'model_rgb' / Path(row['file']).name))
            assert image is not None
            image = image[:, :, ::-1].copy()
            assert hashlib.sha256(image.tobytes()).hexdigest() == row['model_rgb_sha256']
            local_poses.append(tracker.step(image, frame))
        torch.cuda.synchronize()
        timing = dict(tracking_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    local_poses = np.asarray(local_poses)
    assert local_poses.shape == (2585, 4, 4) and np.isfinite(local_poses).all()
    assert len(tracker.events) == 5 and tracker.pending is None
    assert [event['boundary'] for event in tracker.events] == list(BOUNDARIES[1:-1])
    assert all(event['decision_frame'] == event['boundary'] + 49 and
               event['old_kv_released'] and event['old_images_released'] for event in tracker.events)
    np.save(result / 'local_traj.npy', local_poses)
    write_json(result / 'events.json', tracker.events)
    write_json(result / 'transforms.json', [dict(start=start, scale=float(s),
        rotation=r.tolist(), translation=t.tolist())
        for start, (s, r, t) in zip(BOUNDARIES[:-1], tracker.transforms, strict=True)])
    rows = [json.loads(x) for x in (result / 'inference.jsonl').read_text().splitlines()]
    summaries = []
    for reference, start, end in zip(references, BOUNDARIES, BOUNDARIES[1:]):
        scoped = [r for r in rows if r['segment_start'] == start]
        assert [r['frame'] for r in scoped if r['kind'] == 'rebuild'] == [49]
        assert len([r for r in scoped if r['kind'] == 'bootstrap']) == 1
        assert len([r for r in scoped if r['kind'] == 'query']) == end - start - 1 + (
            2 if end < BOUNDARIES[-1] else 0)
        assert len([r for r in scoped if r['kind'] == 'shared_geometry']) == (
            1 if end < BOUNDARIES[-1] else 0)
        assert all(r['input_images'] <= 2 for r in scoped if 'input_images' in r)
        assert all(r['bank_ids'] == ([0] if r['frame'] <= 49 else [0, 49])
                   for r in scoped if r['kind'] == 'query')
        segment = result / f'segment_{start:04d}'
        segment.mkdir()
        segment_inputs = args.work / 'inputs' / f'segment_{start:04d}'
        segment_inputs.mkdir()
        write_json(segment_inputs / 'manifest.json', dict(manifest, inputs=manifest['inputs'][start:end]))
        (segment_inputs / 'groundtruth.txt').symlink_to(inputs / 'groundtruth.txt')
        np.save(segment / 'traj.npy', local_poses[start:end])
        metrics = evaluate(segment_inputs, segment, .02)
        quality = dict(np.load(segment / 'evaluation.npz'))
        old = reference['quality']
        for key in ('rgb_indices', 'gt_indices', 'rpe_pair_start_indices', 'reference'):
            np.testing.assert_array_equal(quality[key], old[key])
        oracle = dict(ate_m=float(np.sqrt(np.mean(old['ate_per_frame_m'] ** 2))),
                      rpe_translation_m=float(np.sqrt(np.mean(old['rpe_translation_per_pair_m'] ** 2))),
                      translation_p99_m=float(np.quantile(old['rpe_translation_per_pair_m'], .99)))
        metrics['translation_p99_m'] = float(np.quantile(quality['rpe_translation_per_pair_m'], .99))
        checks = {key: metrics[key] <= value * 1.0001 + 1e-6 for key, value in oracle.items()}
        summaries.append(dict(start=start, end=end, metrics=metrics, oracle=oracle,
            no_worse_than_oracle=checks,
            raw_pose_max_difference=float(np.abs(local_poses[start:end] - reference['traj']).max())))
        write_json(segment / 'metrics.json', summaries[-1])
    local_pass = all(all(s['no_worse_than_oracle'].values()) for s in summaries)
    continuous = None
    continuous_pass = False
    if tracker.connected:
        global_poses = np.concatenate([np.stack([
            transform_pose(torch.from_numpy(pose), transform).numpy()
            for pose in local_poses[start:end]])
            for start, end, transform in zip(BOUNDARIES, BOUNDARIES[1:], tracker.transforms)])
        np.save(result / 'traj.npy', global_poses)
        continuous = evaluate(inputs, result, .02)
        with np.load(result / 'evaluation.npz') as q:
            continuous_checks = []
            for segment in summaries:
                start, end = segment['start'], segment['end']
                frames = (q['rgb_indices'] >= start) & (q['rgb_indices'] < end)
                pairs = (q['rpe_pair_start_indices'] >= start) & (q['rpe_pair_start_indices'] < end - 1)
                values = dict(ate_m=float(np.sqrt(np.mean(q['ate_per_frame_m'][frames] ** 2))),
                    rpe_translation_m=float(np.sqrt(np.mean(q['rpe_translation_per_pair_m'][pairs] ** 2))),
                    translation_p99_m=float(np.quantile(q['rpe_translation_per_pair_m'][pairs], .99)))
                continuous_checks.append(dict(start=start, metrics=values, passed=all(
                    values[k] <= v * 1.0001 + 1e-6 for k, v in segment['oracle'].items())))
            boundary_pairs = np.isin(q['rpe_pair_start_indices'], np.array(BOUNDARIES[1:-1]) - 1)
            continuous['boundary_pairs'] = dict(starts=q['rpe_pair_start_indices'][boundary_pairs].tolist(),
                translation_m=q['rpe_translation_per_pair_m'][boundary_pairs].tolist(),
                rotation_deg=q['rpe_rotation_per_pair_deg'][boundary_pairs].tolist())
            limits = [max(a['oracle']['translation_p99_m'], b['oracle']['translation_p99_m'])
                      for a, b in zip(summaries, summaries[1:])]
            seams = continuous['boundary_pairs']
            seams['translation_limits_m'] = limits
            seams['passed'] = (seams['starts'] == [b - 1 for b in BOUNDARIES[1:-1]] and
                all(error <= limit * 1.0001 + 1e-6
                    for error, limit in zip(seams['translation_m'], limits)))
        continuous['segments_under_single_alignment'] = continuous_checks
        continuous_pass = all(x['passed'] for x in continuous_checks) and seams['passed']
    write_json(result / 'summary.json', dict(segments=summaries, events=tracker.events,
        local_oracle_gate=local_pass, all_bridges_accepted=tracker.connected,
        continuous=continuous, continuous_oracle_gate=continuous_pass, timing=timing,
        objective_achieved=local_pass and tracker.connected and continuous_pass,
        limitation='Boundary gate uses max adjacent oracle p99; a new engineering criterion, not an oracle measurement.'))
    archive_directory(result, args.out / f'{args.tag}_oracle_maps.tar')
    archives = {}
    for path in args.out.glob(f'{args.tag}_oracle_maps*.tar'):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8388608), b''):
                digest.update(chunk)
        archives[path.name] = digest.hexdigest()
    write_json(args.work / 'oracle_maps_archives.json', archives)
    assert local_pass, 'Local oracle quality regressed; inspect archived segment metrics'
    (args.work / 'JOB_OK').write_text('Execution/local oracle gate passed; review continuous gate separately\n')
    print('ORACLE MAPS COMPLETE: local gate passed; connected=', tracker.connected,
          'continuous oracle gate=', continuous_pass, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    main(parser.parse_args())
