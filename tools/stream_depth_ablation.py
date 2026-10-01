"""Bounded CAMP-only second-backbone historical-KV allocation pilot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'streamvggt/src'))
from streamvggt.models.depth_cache import DepthCache, storage_bytes
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from stream_cache_sweep import prepare, load_model, load_frame, write_json, digest


def step(model, cache, camera, image, frame, geometry=False):
    tokens, special = cache.forward(image, frame)
    poses, camera = model.camera_head(tokens, past_key_values_camera=camera, use_cache=True)
    if geometry:
        depth, confidence = model.depth_head(tokens, images=image, patch_start_idx=special)
        points, point_confidence = model.point_head(tokens, images=image, patch_start_idx=special)
    pose = poses[-1]
    extrinsics, _ = pose_encoding_to_extri_intri(pose, image.shape[-2:])
    transform = torch.eye(4, device=image.device)[None].repeat(1, 1, 1)
    transform[:, :3] = extrinsics[:, 0]
    return torch.linalg.inv(transform)[0], camera


def native_fidelity(model, inputs, records):
    """All intermediates compared to untouched cached attention before calibration."""
    reference = []
    bank = [None] * model.aggregator.depth
    for frame, record in enumerate(records[:3]):
        image, _ = load_frame(inputs, record)
        outputs, _, bank = model.aggregator(image, past_key_values=bank, use_cache=True,
                                          past_frame_idx=frame)
        reference.append([value.cpu() for value in outputs])
    del bank, outputs
    cache = DepthCache(model.aggregator)
    differences = []
    for frame, record in enumerate(records[:3]):
        image, _ = load_frame(inputs, record)
        outputs, _ = cache.forward(image, frame)
        for expected, actual in zip(reference[frame], outputs):
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
            differences.append(float((actual.cpu() - expected).abs().max()))
        cache.retain(True)
    cache.close()
    return dict(frames=3, intermediates=len(differences), max_abs=max(differences),
                atol=1e-5, rtol=1e-5, reference='untouched upstream cached aggregator')


@torch.inference_mode()
def calibrate(args):
    inputs = args.work / 'inputs/calibration'
    records = json.loads((inputs / 'manifest.json').read_text())['frames']
    model = load_model(SimpleNamespace(host='streamvggt', checkpoint=args.checkpoint))
    torch.manual_seed(17)
    write_json(args.work / 'native_fidelity.json', native_fidelity(model, inputs, records))
    print('STREAM DEPTH NATIVE FIDELITY OK', flush=True)
    cache = DepthCache(model.aggregator)
    camera = [None] * model.camera_head.trunk_depth
    bank_poses = []
    for frame, record in enumerate(records[:32]):
        image, _ = load_frame(inputs, record)
        pose, camera = step(model, cache, camera, image, frame)
        bank_poses.append(pose.cpu().numpy())
        cache.retain(True)
    # Held only by the teacher stage; every intervention restores the same state.
    dense = (cache.cache, cache.positions, cache.ids, camera)
    frames = list(cache.frames)
    radius = float(np.sqrt(np.mean(np.sum((np.asarray(bank_poses)[:, :3, 3]
                                           - bank_poses[0][:3, 3]) ** 2, axis=-1))))
    assert radius > 0

    def query(omitted):
        cache.mode = 'omit' if omitted else 'native'
        cache.omitted = tuple(omitted)
        cache.cache, cache.positions, cache.ids = [list(value) for value in dense[:3]]
        cache.frames = list(frames)
        cache.frame = 31
        cache.retain(False)
        current_camera = list(dense[3])
        results = []
        for frame in range(32, 40):
            image, _ = load_frame(inputs, records[frame])
            pose, current_camera = step(model, cache, current_camera, image, frame)
            results.append(pose.cpu().numpy())
            cache.retain(False)
        return np.asarray(results)

    native = query(())
    saved = dict(bank=np.asarray(bank_poses), native=native)
    scores = []
    for layer in range(model.aggregator.depth):
        poses = query((layer,))
        saved[f'omit_{layer}'] = poses
        translation = np.sum((poses[:, :3, 3] - native[:, :3, 3]) ** 2, axis=-1) / radius ** 2
        rotation = np.sum((poses[:, :3, :3] - native[:, :3, :3]) ** 2, axis=(-1, -2)) / 6
        score = float(np.mean(translation + rotation))
        scores.append(dict(layer=layer, score=score,
                           normalized_translation_mse=float(translation.mean()),
                           rotation_matrix_mse=float(rotation.mean())))
        print('STREAM DEPTH CALIBRATION', layer, score, flush=True)
    np.testing.assert_allclose(query(()), native, rtol=1e-5, atol=1e-5)
    chosen = sorted(row['layer'] for row in sorted(scores, key=lambda row: (row['score'], row['layer']))[:8])
    assert model.aggregator.depth == 24
    np.savez(args.work / 'calibration_poses.npz', **saved)
    write_json(args.work / 'selection.json', dict(omitted_layers=chosen, scores=scores,
        bank_frames=list(range(32)), query_frames=list(range(32, 40)), bank_translation_radius=radius,
        criterion='mean native c2w translation squared/radius squared plus rotation Frobenius squared/6',
        mapping='zero-based StreamVGGT global blocks; all 24 global and 24 frame blocks execute',
        source='freiburg1_desk source 0..39; no GT labels used',
        budget='omit eight of 24 historical-KV layers; single frozen choice'))
    print('STREAM DEPTH SELECTION FROZEN', chosen, flush=True)


def quality(poses, records, result):
    from stream3r_ycbv_metrics import umeyama_sim3
    valid = np.asarray([i for i, row in enumerate(records) if row.get('gt_c2w') is not None
                       and abs(row['timestamp'] - row['gt_timestamp']) <= .02])
    assert len(valid) >= 3
    gt = np.asarray([records[i]['gt_c2w'] for i in valid], dtype=np.float64)
    estimated = poses[valid].astype(np.float64)
    assert np.isfinite(estimated).all()
    assert np.linalg.matrix_rank(estimated[:, :3, 3] - estimated[:, :3, 3].mean(0), tol=1e-8) >= 2
    assert np.linalg.matrix_rank(gt[:, :3, 3] - gt[:, :3, 3].mean(0), tol=1e-8) >= 2
    scale, rotation, translation = umeyama_sim3(estimated[:, :3, 3], gt[:, :3, 3])
    aligned = estimated.copy()
    aligned[:, :3, :3] = rotation @ estimated[:, :3, :3]
    aligned[:, :3, 3] = scale * estimated[:, :3, 3] @ rotation.T + translation
    errors = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    times = np.asarray([records[i]['timestamp'] for i in valid])
    pairs = (np.diff(valid) == 1) & (np.diff(times) <= .1)
    assert pairs.any()
    relative = np.linalg.inv(np.linalg.inv(gt[:-1]) @ gt[1:]) @ (
        np.linalg.inv(aligned[:-1]) @ aligned[1:])
    rpe = np.linalg.norm(relative[pairs, :3, 3], axis=1)
    angles = np.degrees(np.arccos(np.clip((np.trace(relative[pairs, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    np.savez(result / 'evaluation.npz', rgb_indices=valid, aligned=aligned, reference=gt,
             timestamps=times, ate_per_frame_m=errors, rpe_translation_per_pair_m=rpe,
             rpe_rotation_per_pair_deg=angles, rpe_pair_start_indices=valid[:-1][pairs],
             alignment_scale=scale, alignment_rotation=rotation, alignment_translation=translation)
    steady_frames = valid >= 128
    steady_pairs = valid[:-1][pairs] >= 128
    assert steady_frames.any() and steady_pairs.any()
    return dict(ate_m=float(np.sqrt(np.mean(errors ** 2))), translation_p99_m=float(np.quantile(errors, .99)),
                rpe_translation_m=float(np.sqrt(np.mean(rpe ** 2))),
                rpe_rotation_deg=float(np.sqrt(np.mean(angles ** 2))),
                evaluated_frames=len(valid), rpe_pairs=int(pairs.sum()),
                steady_ate_m=float(np.sqrt(np.mean(errors[steady_frames] ** 2))),
                steady_translation_p99_m=float(np.quantile(errors[steady_frames], .99)),
                steady_rpe_translation_m=float(np.sqrt(np.mean(rpe[steady_pairs] ** 2))),
                steady_rpe_rotation_deg=float(np.sqrt(np.mean(angles[steady_pairs] ** 2))),
                alignment='per-arm full-window Sim(3); timestamp-valid adjacent RPE only')


@torch.inference_mode()
def arm(args):
    inputs = args.work / 'inputs' / args.scene
    records = json.loads((inputs / 'manifest.json').read_text())['frames']
    chosen = json.loads((args.work / 'selection.json').read_text())['omitted_layers']
    result = args.work / 'runs' / args.scene / args.mode
    write_json(result / 'config.json', dict(mode=args.mode, omitted_layers=chosen,
        history_capacity=32, admission_stride=4, frames=len(records), dtype='float32',
        checkpoint=str(args.checkpoint), input_manifest_sha256=digest(inputs / 'manifest.json'),
        camera_policy='unchanged upstream camera-head cache', seed=17))
    model = load_model(SimpleNamespace(host='streamvggt', checkpoint=args.checkpoint))
    torch.manual_seed(17)
    cache = DepthCache(model.aggregator, args.mode, chosen, capacity=32)
    camera = [None] * model.camera_head.trunk_depth
    # Unscored warmup has no influence on the subsequently reset stream.
    image, _ = load_frame(inputs, records[0])
    step(model, cache, camera, image, 0, True)
    cache.reset()
    camera = [None] * model.camera_head.trunk_depth
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    poses, events = [], []
    frame_calls, global_calls = [], []
    hooks = [block.register_forward_hook(lambda module, inputs, output, i=i: frame_calls.append(i))
             for i, block in enumerate(model.aggregator.frame_blocks)]
    hooks += [block.register_forward_hook(lambda module, inputs, output, i=i: global_calls.append(i))
              for i, block in enumerate(model.aggregator.global_blocks)]
    for frame, record in enumerate(records):
        image, _ = load_frame(inputs, record)
        frame_calls.clear()
        global_calls.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        pose, camera = step(model, cache, camera, image, frame, True)
        torch.cuda.synchronize()
        query_seconds = time.perf_counter() - start
        assert torch.isfinite(pose).all()
        assert frame_calls == global_calls == list(range(model.aggregator.depth))
        poses.append(pose.cpu().numpy())
        start = time.perf_counter()
        event = cache.retain(frame % 4 == 0)
        torch.cuda.synchronize()
        event.update(query_seconds=query_seconds, write_seconds=time.perf_counter() - start,
                     camera_bytes=storage_bytes(camera))
        events.append(event)
        with (result / 'inference.jsonl').open('a') as log:
            log.write(json.dumps(event) + '\n')
    for hook in hooks:
        hook.remove()
    poses = np.asarray(poses)
    np.save(result / 'traj.npy', poses)
    metrics = quality(poses, records, result)
    steady = [event for event in events if event['frame'] >= 128]
    assert len(steady) == 128 and all(len(event['history_frames']) == 32 for event in steady)
    metrics.update(mode=args.mode, frames=len(records), omitted_layers=chosen,
        max_persistent_bytes=max(event['persistent_bytes'] for event in events),
        max_kv_bytes=max(event['kv_bytes'] for event in events),
        max_camera_bytes=max(event['camera_bytes'] for event in events),
        steady_query_p50_seconds=float(np.median([event['query_seconds'] for event in steady])),
        steady_query_p95_seconds=float(np.quantile([event['query_seconds'] for event in steady], .95)),
        steady_write_p50_seconds=float(np.median([event['write_seconds'] for event in steady])),
        steady_query_plus_write_p50_seconds=float(np.median([
            event['query_seconds'] + event['write_seconds'] for event in steady])),
        query_seconds=sum(event['query_seconds'] for event in events),
        write_seconds=sum(event['write_seconds'] for event in events),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        dtype='FP32 weights, activations and KV; autocast disabled',
        timing='synchronized aggregator + camera/depth/point heads + c2w conversion; retention timed separately')
    write_json(result / 'metrics.json', metrics)
    print('STREAM DEPTH RUN OK', args.scene, args.mode, metrics, flush=True)


def pilot(args):
    for scene, start, count, label in (
            ('freiburg1_desk', 0, 40, 'calibration'),
            ('freiburg3_long_office_household', 256, 256, 'office'),
            ('freiburg2_large_with_loop', 256, 256, 'with_loop')):
        source = args.work / 'source' / label
        archive = args.dataset / f'rgbd_dataset_{scene}.zip'
        subprocess.run([sys.executable, str(REPO / 'tools/stream_cache_tum.py'),
                        '--zip', str(archive), '--out', str(source), '--start', str(start),
                        '--count', str(count)], check=True)
        prepare(SimpleNamespace(manifest=source / 'manifest.json', out=args.work / 'inputs' / label, width=308))
        if label != 'calibration':
            frames = json.loads((source / 'manifest.json').read_text())['frames']
            valid = [i for i, row in enumerate(frames) if i >= 128 and row['gt_c2w'] is not None]
            assert any(b == a + 1 and frames[b]['timestamp'] - frames[a]['timestamp'] <= .1
                       for a, b in zip(valid, valid[1:])), 'Steady-history window needs valid adjacent GT'
    write_json(args.work / 'identity.json', dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=digest(args.checkpoint), seed=17, history_capacity=32, admission_stride=4))

    def child(stage, scene='', mode='native'):
        log_path = args.work / (f'runs/{scene}/{mode}/run.log' if scene else 'calibration.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as log:
            process = subprocess.run([sys.executable, __file__, stage, '--work', str(args.work),
                '--checkpoint', str(args.checkpoint), '--scene', scene, '--mode', mode],
                stdout=log, stderr=subprocess.STDOUT)
        print('\n'.join(log_path.read_text().splitlines()[-35:]), flush=True)
        process.check_returncode()
    child('calibrate')
    records = []
    for scene in ('office', 'with_loop'):
        for mode in ('native', 'omit', 'uniform_special'):
            child('arm', scene, mode)
            metrics = json.loads((args.work / f'runs/{scene}/{mode}/metrics.json').read_text())
            records.append(dict(scene=scene, **metrics))
        events = {mode: [json.loads(line) for line in (args.work / f'runs/{scene}/{mode}/inference.jsonl').read_text().splitlines()]
                  for mode in ('native', 'omit', 'uniform_special')}
        for native, omitted, uniform in zip(*events.values()):
            assert native['history_frames'] == omitted['history_frames'] == uniform['history_frames']
            assert omitted['persistent_bytes'] == uniform['persistent_bytes']
            assert 3 * omitted['persistent_bytes'] == 2 * native['persistent_bytes']
            assert native['camera_bytes'] == omitted['camera_bytes'] == uniform['camera_bytes']
    write_json(args.work / 'summary.json', dict(complete=True, records=records,
        next='manual review only; no automatic full evaluation or training'))
    print('STREAM DEPTH PILOT COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('pilot', 'calibrate', 'arm'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('/mnt/datasets/tum-rgbd'))
    parser.add_argument('--scene', default='')
    parser.add_argument('--mode', choices=('native', 'omit', 'uniform_special'), default='native')
    args = parser.parse_args()
    assert sys.platform.startswith('linux') and torch.cuda.is_available()
    {'pilot': pilot, 'calibrate': calibrate, 'arm': arm}[args.stage](args)


if __name__ == '__main__':
    main()
