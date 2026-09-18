"""TUM staging, one isolated KV-Tracker run, and timestamp-valid evaluation.

Execute on CAMP only. kvt_tum_sweep.py coordinates fresh processes for each run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = ROOT / 'kv_tracker'
sys.path.insert(0, str(CHECKOUT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def periodic_indices(length, interval, cap):
    """Stock counter starts at 2 for source frame 1 (not frame 2)."""
    assert length >= 2 and interval >= 2 and cap >= 2
    return [0] + list(range(interval - 1, length, interval))[:cap - 1]


def associate_gt(rgb_times, gt_times, max_difference):
    """Nearest timestamp, rejecting missing-GT intervals; no extrapolated poses."""
    assert rgb_times.ndim == gt_times.ndim == 1
    assert np.all(np.diff(rgb_times) > 0) and np.all(np.diff(gt_times) > 0)
    right = np.searchsorted(gt_times, rgb_times).clip(0, len(gt_times) - 1)
    left = (right - 1).clip(0)
    use_left = np.abs(gt_times[left] - rgb_times) < np.abs(gt_times[right] - rgb_times)
    nearest = np.where(use_left, left, right)
    difference = np.abs(gt_times[nearest] - rgb_times)
    return nearest, difference <= max_difference, difference


def prepare(archive, destination, resize_dim, max_difference):
    import cv2
    from kv_tracker.image import pi3_resize_image

    destination.mkdir(parents=True)
    # Only RGB and small metadata, never depth. Select paths before extraction.
    scene = archive.stem.removeprefix('rgbd_dataset_')
    prefix = f'rgbd_dataset_{scene}/'
    with zipfile.ZipFile(archive) as packed:
        for info in packed.infolist():
            if not info.filename.startswith(prefix) or info.is_dir():
                continue
            relative = Path(info.filename[len(prefix):])
            assert '..' not in relative.parts and not relative.is_absolute()
            if not ((len(relative.parts) == 2 and relative.parts[0] == 'rgb'
                     and relative.suffix == '.png')
                    or relative.as_posix() in ('groundtruth.txt', 'rgb.txt')):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with packed.open(info) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    paths = sorted((destination / 'rgb').glob('*.png'))
    assert len(paths) >= 128, (scene, len(paths))
    times = np.array([float(p.stem) for p in paths])
    gt = np.loadtxt(destination / 'groundtruth.txt')
    _, valid, difference = associate_gt(times, gt[:, 0], max_difference)
    model_dir = destination / 'model_rgb'
    model_dir.mkdir()
    records = []
    for index, path in enumerate(paths):
        bgr = cv2.imread(str(path))
        assert bgr is not None, path
        rgb = bgr[:, :, ::-1]
        resized = pi3_resize_image(rgb.copy(), (resize_dim, resize_dim))
        assert cv2.imwrite(str(model_dir / path.name), resized[:, :, ::-1])
        records.append(dict(index=index, file=str(path.relative_to(destination)),
                            timestamp=float(times[index]), shape=list(resized.shape),
                            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            model_rgb_sha256=hashlib.sha256(resized.tobytes()).hexdigest()))
    manifest = dict(scene=scene, archive=str(archive), resize_dim=resize_dim,
                    frames=len(paths), duration_seconds=float(times[-1] - times[0]),
                    mask='all true at model resolution; scene mode', offset=0,
                    max_gt_difference_seconds=max_difference,
                    gt_valid_frames=int(valid.sum()), gt_valid_fraction=float(valid.mean()),
                    max_nearest_gt_gap_seconds=float(difference.max()), inputs=records)
    write_json(destination / 'manifest.json', manifest)
    print('PREPARED', scene, len(paths), 'GT-valid', int(valid.sum()), flush=True)
    return manifest


def evaluate(scene_dir, result, max_difference):
    from scipy.spatial.transform import Rotation
    from kv_tracker.eval_tools.evo_utils import align_pair
    from kv_tracker.geometry import umeyama_alignment

    manifest = json.loads((scene_dir / 'manifest.json').read_text())
    estimate = np.load(result / 'traj.npy')
    assert estimate.ndim == 3 and estimate.shape[1:] == (4, 4)
    assert np.isfinite(estimate).all()
    timestamps = np.array([r['timestamp'] for r in manifest['inputs'][:len(estimate)]])
    gt = np.loadtxt(scene_dir / 'groundtruth.txt')
    nearest, valid, difference = associate_gt(timestamps, gt[:, 0], max_difference)
    indices = np.flatnonzero(valid)
    assert len(indices) >= 3, 'Insufficient timestamp-valid GT for alignment'
    poses = np.tile(np.eye(4), (len(indices), 1, 1))
    matched = gt[nearest[valid]]
    poses[:, :3, :3] = Rotation.from_quat(matched[:, 4:8]).as_matrix()
    poses[:, :3, 3] = matched[:, 1:4]
    aligned, reference = align_pair(dict(traj_gt=poses, traj_est=estimate[valid]))
    rotation, translation_offset, scale = umeyama_alignment(
        estimate[valid, :3, 3].T, poses[:, :3, 3].T, with_scale=True)
    full_positions = scale * (estimate[:, :3, 3] @ rotation.T) + translation_offset
    np.testing.assert_allclose(full_positions[valid], aligned[:, :3, 3], atol=1e-6, rtol=1e-6)
    translation = np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    # No adjacent-pose errors across a missing GT interval or dropped RGB frame.
    adjacent = (np.diff(indices) == 1) & (np.diff(timestamps[valid]) <= 0.1)
    assert adjacent.any(), 'No adjacent timestamp-valid RGB pairs'
    ref_relative = np.linalg.inv(reference[:-1]) @ reference[1:]
    est_relative = np.linalg.inv(aligned[:-1]) @ aligned[1:]
    error = (np.linalg.inv(ref_relative) @ est_relative)[adjacent]
    angles = Rotation.from_matrix(error[:, :3, :3]).magnitude()
    relative_translation = np.linalg.norm(error[:, :3, 3], axis=1)
    np.savez(result / 'evaluation.npz', rgb_indices=indices,
             gt_indices=nearest[valid], timestamps=timestamps[valid],
             timestamp_difference=difference[valid], aligned=aligned, reference=reference,
             rpe_pair_start_indices=indices[:-1][adjacent], ate_per_frame_m=translation,
             alignment_rotation=rotation, alignment_translation=translation_offset,
             alignment_scale=scale, full_aligned_positions=full_positions,
             rpe_translation_per_pair_m=relative_translation,
             rpe_rotation_per_pair_deg=np.rad2deg(angles))
    return dict(ate_m=float(np.sqrt(np.mean(translation ** 2))),
                rpe_translation_m=float(np.sqrt(np.mean(relative_translation ** 2))),
                rpe_rotation_deg=float(np.rad2deg(np.sqrt(np.mean(angles ** 2)))),
                evaluated_frames=len(indices), evaluated_fraction=float(valid.mean()),
                rpe_pairs=int(adjacent.sum()), alignment='one full-valid-trajectory Sim(3)',
                max_gt_difference_seconds=max_difference)


class FinalScene:
    """Keep the latest joint reconstruction on CPU; write once, even on failure."""
    def __init__(self):
        self.data = None
        self.seconds = 0.

    def __call__(self, kind, frame_ids, xyz, poses, confidence, rgb, masks, threshold, latest_rgb):
        if kind != 'keyframes':
            return
        started = time.perf_counter()
        assert masks.all()
        self.data = dict(xyz=xyz[0].detach().float().cpu().numpy(),
                         confidence=confidence[0, ..., 0].detach().float().cpu().numpy(),
                         poses=poses[0].detach().float().cpu().numpy(),
                         frame_ids=np.asarray(frame_ids), threshold=float(threshold))
        assert self.data['xyz'].shape == rgb.shape
        assert self.data['confidence'].shape == masks.shape
        self.seconds += time.perf_counter() - started


def run(config_path):
    import torch
    from kv_tracker.dataloaders.tum import TUMLoader
    import main as tracker
    from kvt_tum_selector import TumSelector

    config = json.loads(config_path.read_text())
    scene_dir, result = Path(config['scene_dir']), config_path.parent
    manifest = json.loads((scene_dir / 'manifest.json').read_text())
    length = config['frames']
    assert 2 <= length <= manifest['frames']
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    class Frames(TUMLoader):
        def __iter__(self):
            for index in range(length):
                torch.cuda.synchronize()
                started = time.perf_counter()
                frame = self.get_frame(index)
                frame['idx'] = index
                pixels = frame['resized_rgb_masked_np']
                assert frame['resized_mask_np'].all()
                assert hashlib.sha256(pixels.tobytes()).hexdigest() == (
                    manifest['inputs'][index]['model_rgb_sha256'])
                yield frame
                torch.cuda.synchronize()
                timing.write(json.dumps(dict(frame=index,
                    seconds=time.perf_counter() - started)) + '\n')
                timing.flush()

    with (result / 'frame_times.jsonl').open('w') as timing, \
            (result / 'decisions.jsonl').open('w') as decisions, \
            (result / 'inference.jsonl').open('w') as inference:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        source = Frames('cuda:0', scene_dir=scene_dir, obj_mode=False,
                        resize_dim=config['resize_dim'], offset=0)
        selector = None if config['policy'] == 'bare' else TumSelector(config, decisions, inference)
        recorder = FinalScene()
        try:
            tracker.run_track3r(cfg=dict(results_path=str(result), que_size=1),
                args=['--cam_only', '--resize_dim', str(config['resize_dim']),
                      '--kf_auto', str(config['interval'])], frame_source=source,
                keyframe_selector=selector, snapshot_callback=recorder)
        finally:
            if selector is not None:
                selector.close(result)
            if recorder.data is not None:
                np.savez(result / 'final_scene.npz', **recorder.data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if selector is not None:
            assert selector.last_index == length - 1
            if len(selector.inserted) == 1:
                np.save(result / 'kf_idx.npy', np.array([0]))
            assert np.load(result / 'kf_idx.npy').tolist() == selector.inserted
        trajectory = np.load(result / 'traj.npy')
        assert trajectory.shape == (length, 4, 4)
        selected = np.load(result / 'kf_idx.npy').tolist()
        if config['policy'] in ('bare', 'original', 'periodic'):
            cap = 20 if config['policy'] in ('bare', 'original') else config['cap']
            assert selected == periodic_indices(length, config['interval'], cap)
        free, total = torch.cuda.mem_get_info()
        metrics = dict(frames=length, keyframes=len(selected), last_keyframe=selected[-1],
                       snapshot_copy_seconds=recorder.seconds,
                       tail_frames=length - 1 - selected[-1],
                       tail_seconds=manifest['inputs'][length-1]['timestamp']
                           - manifest['inputs'][selected[-1]]['timestamp'],
                       seconds=elapsed, synchronous_frames_per_second=length / elapsed,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       device_total_bytes=total, device_free_bytes_at_end=free,
                       cap_reached=len(selected) >= config['cap'])
        # Peak reservation plus end-of-run non-allocator use is a conservative
        # preflight estimate; sampled nvidia-smi device usage is archived separately.
        metrics['estimated_peak_device_bytes'] = (metrics['peak_reserved_bytes']
            + total - free - torch.cuda.memory_reserved())
        # Prefixes verify causality and probes measure memory. Neither requires
        # a nondegenerate GT alignment on its short initial motion segment.
        if config['evaluate_trajectory']:
            metrics.update(evaluate(scene_dir, result, config['max_gt_difference']))
        rows = [json.loads(line) for line in (result / 'inference.jsonl').read_text().splitlines()]
        for kind in ('bootstrap', 'query', 'rebuild'):
            values = [row['seconds'] for row in rows if row['kind'] == kind]
            metrics[f'{kind}_calls'] = len(values)
            metrics[f'{kind}_seconds'] = float(sum(values))
            metrics[f'{kind}_p50_seconds'] = float(np.median(values)) if values else None
            metrics[f'{kind}_p95_seconds'] = float(np.quantile(values, .95)) if values else None
        decisions.flush()
        choices = [json.loads(line) for line in (result / 'decisions.jsonl').read_text().splitlines()]
        metrics['cap_blocked_frames'] = sum(row['cap_blocked'] for row in choices)
        metrics['selector_seconds'] = sum(row['selector_seconds'] for row in choices)
        metrics['feature_export_seconds'] = sum(row['feature_export_seconds'] for row in choices)
        metrics['max_cache_bytes'] = max((row['cache_bytes'] for row in rows), default=0)
        metrics['final_feature_bytes'] = 0 if selector is None else sum(
            value.numel() * value.element_size() for value in selector.retained)
        metrics['diagnostic_cpu_feature_bytes'] = 0 if selector is None else sum(
            value.nbytes for value in selector.frame_features + selector.patch_maps)
        write_json(result / 'environment.json', dict(torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            seed=0, timing='synchronous instrumented wall times, no FPS benchmark',
            resize_dim=config['resize_dim']))
        write_json(result / 'metrics.json', metrics)
        print('RUN OK', config['name'], metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    # A failed preflight is handled explicitly by the sweep, in a fresh process.
    import torch
    try:
        run(args.config)
    except torch.cuda.OutOfMemoryError:
        import traceback
        traceback.print_exc()
        (args.config.parent / 'CUDA_OOM').write_text('CUDA out of memory\n')
        raise SystemExit(42)
