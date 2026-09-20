"""Three-host cache sweep. Execute on remote Linux; see STREAM_CACHE_SWEEP.md."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

from kvcache_policy import pool_confidence, pool_mask
from stream_cache_adapters import StreamAdapter
from stream_cache_policy import CachePolicy, PolicyConfig

REPO = Path(__file__).resolve().parents[1]
PREDICTIONS = ('pose_enc', 'rel_pose_enc', 'depth', 'depth_conf',
               'world_points', 'world_points_conf', 'predicted_scale_factor')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def prepare(args):
    source = json.loads(args.manifest.read_text())
    frames = source['frames']
    assert frames and args.width % 14 == 0
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'rgb').mkdir()
    (args.out / 'masks').mkdir()
    records = []
    for index, frame in enumerate(frames):
        path = args.manifest.parent / frame['rgb']
        with Image.open(path) as image:
            image = image.convert('RGB')
            width, height = image.size
            resized = (args.width, max(14, round(height * args.width / width / 14) * 14))
            top = max(0, (resized[1] - args.width) // 2)
            crop = (0, top, args.width, top + min(resized[1], args.width))
            image = image.resize(resized, Image.Resampling.BICUBIC).crop(crop)
            rgb_path = f'rgb/{index:06d}.png'
            image.save(args.out / rgb_path)
        row = dict(frame, rgb=rgb_path, source_rgb=str(path.resolve()),
                   source_sha256=digest(path), rgb_sha256=digest(args.out / rgb_path),
                   source_hw=[height, width], model_hw=[image.height, image.width],
                   pixel_transform=[[resized[0] / width, 0, (resized[0] / width - 1) / 2],
                                    [0, resized[1] / height, (resized[1] / height - 1) / 2 - top],
                                    [0, 0, 1]])
        if 'mask' in frame:
            mask_path = args.manifest.parent / frame['mask']
            with Image.open(mask_path) as mask:
                assert mask.size == (width, height)
                mask = mask.convert('L').resize(resized, Image.Resampling.NEAREST).crop(crop)
                row['mask'] = f'masks/{index:06d}.png'
                mask.save(args.out / row['mask'])
                row['mask_sha256'] = digest(args.out / row['mask'])
                row['source_mask_sha256'] = digest(mask_path)
        if 'intrinsics' in frame:
            row['model_intrinsics'] = (np.asarray(row['pixel_transform']) @
                                       np.asarray(frame['intrinsics'])).tolist()
        records.append(row)
    write_json(args.out / 'manifest.json', dict(source, frames=records,
               preprocessing='shared RGB width resize / center-height crop, patch 14; no input masking'))
    print(f'PREPARED {len(records)} frames: {args.out}', flush=True)


def load_model(args):
    sys.path.insert(0, str(REPO / ('streamvggt/src' if args.host == 'streamvggt' else args.host)))
    if args.host == 'stream3r':
        from stream3r.models.stream3r import STream3R
        model = STream3R.from_pretrained(str(args.checkpoint), local_files_only=True)
    elif args.host == 'streamvggt':
        from streamvggt.models.streamvggt import StreamVGGT
        model = StreamVGGT()
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    else:
        import yaml
        from longstream.core.model import LongStreamModel
        config = yaml.safe_load(args.model_config.read_text())['model']
        config['checkpoint'] = str(args.checkpoint)
        config['strict_load'] = True
        config['hf'] = None
        model = LongStreamModel(config)
    return model.cuda().eval()


def load_frame(root, record):
    with Image.open(root / record['rgb']) as source:
        image = torch.from_numpy(np.array(source.convert('RGB'))).permute(2, 0, 1).float() / 255
    mask = None
    if 'mask' in record:
        with Image.open(root / record['mask']) as source:
            mask = torch.from_numpy(np.array(source.convert('L')) > 127)
        assert mask.shape == image.shape[-2:]
    assert image.shape[-1] % 14 == image.shape[-2] % 14 == 0
    return image[None, None].cuda(), mask


def cpu_predictions(output):
    result = {key: output[key].detach().float().cpu() for key in PREDICTIONS if key in output}
    for key, value in result.items():
        assert torch.isfinite(value).all(), key
    return result


def native_predictions(args, model, root, frames):
    """Independent oracle through each host's unmodified streaming entry point."""
    if args.host == 'stream3r':
        from stream3r.stream_session import StreamSession
        session = StreamSession(model, mode='causal')
        result = []
        for frame in frames:
            image, _ = load_frame(root, frame)
            session.forward_stream(image)
            result.append(cpu_predictions(session.get_last_prediction()))
        return result
    if args.host == 'longstream':
        from longstream.streaming.refresh import run_streaming_refresh
        from longstream.streaming.keyframe_selector import KeyframeSelector
        images = torch.cat([load_frame(root, frame)[0].cpu() for frame in frames], dim=1)
        flags, indices = KeyframeSelector(min_interval=args.keyframe_stride,
                                          max_interval=args.keyframe_stride).select_keyframes(len(frames))
        output = run_streaming_refresh(model, images, flags, indices, 'causal', 48,
                                        args.refresh, {'num_iterations': 4})
        return [{key: output[key][:, index:index + 1].float().cpu()
                 for key in PREDICTIONS if key in output and key != 'predicted_scale_factor'}
                for index in range(len(frames))]
    views = [{'img': load_frame(root, frame)[0][0]} for frame in frames]
    output = model.inference(views)
    names = {'camera_pose': 'pose_enc', 'pts3d_in_other_view': 'world_points',
             'conf': 'world_points_conf', 'depth': 'depth', 'depth_conf': 'depth_conf'}
    return [{target: row[source][:, None].float().cpu() for source, target in names.items()}
            for row in output.ress]


def fidelity(args, model, root, frames):
    """Compare unmodified incremental execution, twice, with all-retained adapter."""
    # Include at least one LongStream refresh so its duplicate seed frame is tested.
    count = min(len(frames), max(args.gate_frames,
                args.keyframe_stride * (args.refresh - 1) + 2 if args.host == 'longstream' else 2))
    references = []
    reports = []
    for run in range(3):
        torch.manual_seed(args.seed)
        if run < 2:
            with torch.no_grad():
                predictions = native_predictions(args, model, root, frames[:count])
        else:
            adapter = StreamAdapter(model, args.host,
                                    keyframe_stride=args.keyframe_stride, refresh=args.refresh)
            predictions = []
            for index, frame in enumerate(frames[:count]):
                image, _ = load_frame(root, frame)
                with torch.no_grad():
                    output = adapter.forward(image, index)
                    prediction = cpu_predictions(output)
                    prediction.pop('predicted_scale_factor', None)
                    predictions.append(prediction)
                    adapter.prune(torch.arange(adapter.features.shape[0]), list(range(index + 1)))
                    if adapter.needs_refresh(index):
                        adapter.reset_segment(index)
                        adapter.forward(image, index, reference_self=True)
                        adapter.prune(torch.arange(adapter.features.shape[0]), [index])
                del output, image
            adapter.close()
            del adapter
        if run == 0:
            references = predictions
        else:
            worst = {}
            for reference, prediction in zip(references, predictions):
                assert reference.keys() == prediction.keys()
                for key in reference:
                    error = float((reference[key] - prediction[key]).abs().max())
                    worst[key] = max(worst.get(key, 0.), error)
                    torch.testing.assert_close(prediction[key], reference[key],
                                               atol=args.gate_atol, rtol=args.gate_rtol)
            reports.append(dict(run='repeat' if run == 1 else 'all_retained', max_abs=worst))
        del predictions
    report = dict(frames=count, atol=args.gate_atol, rtol=args.gate_rtol, comparisons=reports)
    write_json(args.out / 'fidelity.json', report)
    print('FIDELITY GATE OK', flush=True)


def pose_arrays(host, predictions, hw, stride):
    if host == 'longstream':
        from longstream.utils.vendor.models.components.utils.pose_enc import pose_encoding_to_extri_intri
        from longstream.utils.camera import compose_abs_from_rel
        from longstream.streaming.keyframe_selector import KeyframeSelector
        if 'rel_pose_enc' in predictions:
            relative = predictions['rel_pose_enc'][0]
            _, indices = KeyframeSelector(min_interval=stride, max_interval=stride).select_keyframes(len(relative))
            poses = compose_abs_from_rel(relative, indices[0])[None]
        else:
            poses = predictions['pose_enc']
    elif host == 'stream3r':
        from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
        poses = predictions['pose_enc']
    else:
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        poses = predictions['pose_enc']
    extrinsics, intrinsics = pose_encoding_to_extri_intri(poses, hw)
    return dict(extrinsics=extrinsics[0].numpy(), intrinsics=intrinsics[0].numpy())


def run_condition(args, model, root, frames, name, config):
    target = args.out / name
    target.mkdir()
    write_json(target / 'config.json', asdict(config))
    policy = CachePolicy(config)
    adapter = StreamAdapter(model, args.host, feature=config.feature,
                            keyframe_stride=args.keyframe_stride, refresh=args.refresh)
    torch.manual_seed(config.seed)
    pose_chunks = {}
    latencies = []
    admission_scores = []
    admitted = 0
    with (target / 'events.jsonl').open('w') as log:
        for index, frame in enumerate(frames):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            image, mask = load_frame(root, frame)
            grid = tuple(size // adapter.patch_size for size in image.shape[-2:])
            with torch.no_grad():
                output = adapter.forward(image, index)
                torch.cuda.synchronize()
                prediction_end = time.perf_counter()
                confidence = pool_confidence(output['depth_conf'][0].float().cpu(), grid, 0)[0]
                token_mask = None if mask is None else pool_mask(mask[None], grid, 0)[0]
                picked, kept, event = policy.update(index, adapter.features.float().cpu(), grid,
                                                    confidence, token_mask)
                if event['score'] is not None:
                    admission_scores.append(event['score'])
                    admitted += int(event['accepted'])
                adapter.prune(picked, kept)
                event.update(adapter.memory())
                event['feature_bytes'] = policy.feature_bytes()
                torch.cuda.synchronize()
                selection_end = time.perf_counter()
                predictions = cpu_predictions(output)
                # The source output contains full, unpruned caches; release before refresh.
                del output
                event['refresh'] = adapter.needs_refresh(index)
                if event['refresh']:
                    adapter.reset_segment(index)
                    policy.clear()
                    replay = adapter.forward(image, index, reference_self=True)
                    confidence = pool_confidence(replay['depth_conf'][0].float().cpu(), grid, 0)[0]
                    picked, kept, refresh_event = policy.update(
                        index, adapter.features.float().cpu(), grid, confidence, token_mask)
                    adapter.prune(picked, kept)
                    event['refresh_seed'] = refresh_event
                    event['post_refresh_memory'] = adapter.memory()
                    del replay
                torch.cuda.synchronize()
                end = time.perf_counter()
            event.update(prediction_seconds=prediction_end - start,
                         selection_seconds=selection_end - prediction_end,
                         export_transfer_refresh_seconds=end - selection_end,
                         seconds=end - start,
                         peak_allocated=torch.cuda.max_memory_allocated(),
                         peak_reserved=torch.cuda.max_memory_reserved())
            latencies.append(event['seconds'])
            for key in ('pose_enc', 'rel_pose_enc'):
                if key in predictions:
                    pose_chunks.setdefault(key, []).append(predictions[key])
            export_start = time.perf_counter()
            if args.geometry_export == 'all' or index == len(frames) - 1:
                np.savez_compressed(target / f'frame_{index:06d}.npz',
                                    **{key: value.numpy() for key, value in predictions.items()})
            event['disk_seconds'] = time.perf_counter() - export_start
            log.write(json.dumps(event, allow_nan=False) + '\n')
            log.flush()
            del image, predictions
    poses = {key: torch.cat(value, dim=1) for key, value in pose_chunks.items()}
    np.savez_compressed(target / 'pose_encodings.npz',
                        **{key: value.numpy() for key, value in poses.items()})
    arrays = pose_arrays(args.host, poses, frames[0]['model_hw'], args.keyframe_stride)
    np.savez_compressed(target / 'camera.npz', **arrays)
    write_json(target / 'summary.json', dict(frames=len(frames),
               p50_seconds=float(np.median(latencies)), p95_seconds=float(np.quantile(latencies, .95)),
               admission_diagnostics=dict(
                   scored_frames=len(admission_scores), admitted=admitted,
                   rejected=len(admission_scores) - admitted,
                   nontrivial=0 < admitted < len(admission_scores),
                   score_min=min(admission_scores) if admission_scores else None,
                   score_max=max(admission_scores) if admission_scores else None),
               total_seconds=sum(latencies), inference='single-frame causal, fixed head histories',
               geometry_export=args.geometry_export,
               quality='camera.npz and selected geometry exports; no object-pose claim'))
    adapter.close()
    print(f'RUN OK {name}', flush=True)


def evaluate(args):
    """Timestamp-valid camera diagnostics; never label these object tracking."""
    from stream3r_ycbv_metrics import umeyama_sim3
    frames = json.loads((args.out / 'inputs/manifest.json').read_text())['frames']
    valid = [i for i, row in enumerate(frames) if row.get('gt_c2w') is not None
             and abs(row['timestamp'] - row['gt_timestamp']) <= args.max_gt_delta]
    if len(valid) < 3:
        raise ValueError('Need at least three timestamp-valid gt_c2w entries for camera evaluation')
    gt = np.asarray([frames[i]['gt_c2w'] for i in valid], dtype=np.float64)
    assert gt.shape == (len(valid), 4, 4) and np.isfinite(gt).all()
    times = np.asarray([frames[i]['timestamp'] for i in valid])
    assert (np.diff(times) > 0).all()
    # Do not bridge missing GT, dropped source frames, or large time gaps.
    pairs = (np.diff(valid) == 1) & (np.diff(times) <= args.max_pair_gap)
    summaries = {}
    for condition in json.loads((args.out / 'sweep.json').read_text())['conditions']:
        name = condition['name']
        extrinsics = np.load(args.out / name / 'camera.npz')['extrinsics'][valid]
        world_to_camera = np.tile(np.eye(4), (len(valid), 1, 1))
        world_to_camera[:, :3] = extrinsics[:, :3]
        estimated = np.linalg.inv(world_to_camera)
        centers = estimated[:, :3, 3]
        gt_centers = gt[:, :3, 3]
        if (np.linalg.matrix_rank(centers - centers.mean(0), tol=1e-8) < 2
                or np.linalg.matrix_rank(gt_centers - gt_centers.mean(0), tol=1e-8) < 2):
            summaries[name] = {'status': 'degenerate camera-centre path; no Sim(3) accuracy claim'}
            continue
        scale, rotation, translation = umeyama_sim3(centers, gt[:, :3, 3])
        aligned_centers = scale * centers @ rotation.T + translation
        errors = np.linalg.norm(aligned_centers - gt[:, :3, 3], axis=1)
        aligned = estimated.copy()
        aligned[:, :3, :3] = rotation @ estimated[:, :3, :3]
        aligned[:, :3, 3] = aligned_centers
        relative_estimated = np.linalg.inv(aligned[:-1]) @ aligned[1:]
        relative_gt = np.linalg.inv(gt[:-1]) @ gt[1:]
        relative_error = np.linalg.inv(relative_gt) @ relative_estimated
        translation_error = np.linalg.norm(relative_error[:, :3, 3], axis=1)
        rotation_error = np.degrees(np.arccos(np.clip(
            (np.trace(relative_error[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
        report = dict(status='ok' if pairs.any() else 'ATE only: no eligible RPE pairs',
                      rpe_pair_count=int(pairs.sum()), max_pair_gap=args.max_pair_gap,
                      max_gt_delta=args.max_gt_delta, valid_frames=valid, aligned_center_error_m=errors.tolist(),
                      full_trajectory_sim3_ate_m=float(np.sqrt(np.mean(errors ** 2))),
                      fitted_scale=scale, rpe_pair_end_frames=np.asarray(valid)[1:][pairs].tolist(),
                      rpe_translation_m=translation_error[pairs].tolist(),
                      rpe_rotation_deg=rotation_error[pairs].tolist(),
                      rpe_translation_rmse_m=float(np.sqrt(np.mean(translation_error[pairs] ** 2))) if pairs.any() else None,
                      rpe_rotation_rmse_deg=float(np.sqrt(np.mean(rotation_error[pairs] ** 2))) if pairs.any() else None,
                      alignment='full-trajectory camera-centre Sim(3), offline diagnostic; no absolute rotation/object pose claim')
        write_json(args.out / name / 'camera_metrics.json', report)
        summaries[name] = {key: value for key, value in report.items() if not isinstance(value, list)}
    write_json(args.out / 'camera_metrics.json', summaries)
    print('CAMERA EVALUATION OK' if pairs.any() else
          'CAMERA EVALUATION INCOMPLETE: no eligible RPE pairs; inspect timestamps and --max-pair-gap',
          flush=True)


def run(args):
    assert sys.platform == 'linux' and torch.cuda.is_available(), 'Remote Linux GPU required'
    assert args.checkpoint.exists()
    if args.host == 'longstream' and args.model_config is None:
        raise ValueError('--model-config is required for LongStream')
    specification = json.loads(args.sweep.read_text())
    args.geometry_export = specification.get('geometry_export', 'all')
    assert args.geometry_export in ('all', 'final')
    configs = [(row['name'], PolicyConfig(**row['policy'])) for row in specification['conditions']]
    names = [name for name, _ in configs]
    assert len(names) == len(set(names)) and all(name and Path(name).name == name for name in names)
    assert all(name not in ('.', '..', 'inputs', '__fidelity__') for name in names)
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    frames = manifest['frames']
    assert len(frames) >= 2
    if 'max_frames' in specification and len(frames) > specification['max_frames']:
        raise ValueError('Prepared clip exceeds this sweep protocol max_frames; no silent truncation')
    assert len({tuple(frame['model_hw']) for frame in frames}) == 1
    for frame in frames:
        assert digest(args.inputs / frame['rgb']) == frame['rgb_sha256']
        if 'mask' in frame:
            assert digest(args.inputs / frame['mask']) == frame['mask_sha256']
    if any(config.patch_policy == 'mask' for _, config in configs):
        assert all('mask' in frame for frame in frames)
    if args.worker is not None:
        model = load_model(args)
        if args.worker == '__fidelity__':
            fidelity(args, model, args.out / 'inputs', frames)
        else:
            run_condition(args, model, args.out / 'inputs', frames,
                          args.worker, dict(configs)[args.worker])
        return
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.inputs, args.out / 'inputs')
    shutil.copy2(args.sweep, args.out / 'sweep.json')
    if args.model_config:
        shutil.copy2(args.model_config, args.out / 'model_config.yaml')
    source = args.out / 'source'
    source.mkdir()
    for path in Path(__file__).parent.glob('stream_cache_*.py'):
        shutil.copy2(path, source / path.name)
    shutil.copy2(Path(__file__).with_name('kvcache_policy.py'), source)
    shutil.copy2(Path(__file__).with_name('stream3r_ycbv_metrics.py'), source)
    shutil.copy2(Path(__file__).with_name('test_stream_cache.py'), source)
    shutil.copy2(Path(__file__).with_name('test_stream_cache_workers.py'), source)
    shutil.copy2(Path(__file__).with_name('test_kvcache_policy.py'), source)
    checkpoint_files = sorted(args.checkpoint.rglob('*')) if args.checkpoint.is_dir() else [args.checkpoint]
    provenance = dict(host=args.host, checkpoint={str(p): digest(p) for p in checkpoint_files if p.is_file()},
                      torch=torch.__version__, device=torch.cuda.get_device_name(),
                      arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    for name in ('superproject', 'model'):
        provenance[name + '_commit'] = (args.git_provenance / f'{name}_commit.txt').read_text().strip()
        shutil.copy2(args.git_provenance / f'{name}.patch', source)
    write_json(args.out / 'provenance.json', provenance)
    if specification.get('isolate_conditions', False):
        # Fidelity and each condition release their CUDA context at process exit.
        command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]]
        for worker in ['__fidelity__', *names]:
            print(f'WORKER {worker}', flush=True)
            subprocess.run([*command, '--worker', worker], check=True)
    else:
        model = load_model(args)
        fidelity(args, model, args.out / 'inputs', frames)
        for name, config in configs:
            print(f'RUN {name}', flush=True)
            run_condition(args, model, args.out / 'inputs', frames, name, config)
    if any(frame.get('gt_c2w') is not None for frame in frames):
        evaluate(args)
    else:
        write_json(args.out / 'camera_metrics.json',
                   {'status': 'not evaluated: no camera GT supplied in manifest'})
    print('SWEEP OK', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--manifest', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--width', type=int, default=518)
    sweep = commands.add_parser('run')
    sweep.add_argument('--host', choices=('stream3r', 'streamvggt', 'longstream'), required=True)
    sweep.add_argument('--checkpoint', type=Path, required=True)
    sweep.add_argument('--model-config', type=Path)
    sweep.add_argument('--inputs', type=Path, required=True)
    sweep.add_argument('--sweep', type=Path, required=True)
    sweep.add_argument('--out', type=Path, required=True)
    sweep.add_argument('--git-provenance', type=Path, required=True)
    sweep.add_argument('--keyframe-stride', type=int, default=8)
    sweep.add_argument('--refresh', type=int, default=4)
    sweep.add_argument('--gate-frames', type=int, default=4)
    sweep.add_argument('--gate-atol', type=float, default=1e-5)
    sweep.add_argument('--gate-rtol', type=float, default=1e-4)
    sweep.add_argument('--seed', type=int, default=0)
    sweep.add_argument('--worker', help=argparse.SUPPRESS)
    sweep.add_argument('--max-gt-delta', type=float, default=.02)
    sweep.add_argument('--max-pair-gap', type=float, default=.1)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--out', type=Path, required=True)
    evaluation.add_argument('--max-gt-delta', type=float, default=.02)
    evaluation.add_argument('--max-pair-gap', type=float, default=.1)
    args = parser.parse_args()
    if args.command == 'prepare':
        assert sys.platform == 'linux', 'Dataset preparation runs on remote Linux'
        prepare(args)
    elif args.command == 'run':
        run(args)
    else:
        assert sys.platform == 'linux', 'Run evaluation on remote Linux'
        evaluate(args)


if __name__ == '__main__':
    main()
