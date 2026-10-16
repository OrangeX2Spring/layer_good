"""One-clip InfiniteVGGT dynamic-cache pilot. Execute on allocated CAMP Linux.

See DYNAMIC_CACHE_BIAS.md for the input contract and staged submission.
"""

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'infinitevggt/src'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def prepare(manifest_path, destination, predicted_person_masks=False):
    """Archive source pixels and labels; apply one recorded RGB/mask transform."""
    manifest = json.loads(manifest_path.read_text())
    kind = manifest['mask_source']['kind']
    if predicted_person_masks and kind != 'predicted_person':
        raise ValueError('Person diagnostic requires predicted_person provenance')
    if not predicted_person_masks and kind not in ('gt_motion', 'gt_derived_motion'):
        raise ValueError('Pilot requires documented GT or GT-derived motion masks, not semantic/predicted masks')
    if not manifest['mask_source']['description'] or not manifest['mask_source']['reference']:
        raise ValueError('Mask description and provenance reference are required')
    frames = manifest['frames']
    if not 16 <= len(frames) <= 64:
        raise ValueError('Freeze one 16..64-frame clip before running; no automatic truncation')
    times = np.asarray([row['timestamp'] for row in frames])
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Frame timestamps must be finite and strictly increasing')
    if not manifest['pose_units'] == 'metres':
        raise ValueError('Convert GT camera-to-world translations to metres explicitly')
    destination.mkdir()
    for name in ('source_rgb', 'source_masks', 'rgb', 'masks'):
        (destination / name).mkdir()
    coverage, records, shapes = [], [], set()
    for index, row in enumerate(frames):
        rgb_path = manifest_path.parent / row['rgb']
        mask_path = manifest_path.parent / row['motion_mask']
        source_rgb = destination / 'source_rgb' / f'{index:06d}{rgb_path.suffix}'
        source_mask = destination / 'source_masks' / f'{index:06d}{mask_path.suffix}'
        shutil.copyfile(rgb_path, source_rgb)
        shutil.copyfile(mask_path, source_mask)
        with Image.open(rgb_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size or mask.mode != 'L':
                raise ValueError('Motion mask must be an L-mode PNG at original RGB resolution')
            raw_mask = np.asarray(mask)
            if not np.isin(raw_mask, (0, 128, 255)).all():
                raise ValueError('Mask encoding: 0 static, 255 moving, 128 unknown')
            width, height = image.size
            resized = (308, max(14, round(height * 308 / width / 14) * 14))
            top = max(0, (resized[1] - 308) // 2)
            crop = (0, top, 308, top + min(308, resized[1]))
            image = image.convert('RGB').resize(resized, Image.Resampling.BICUBIC).crop(crop)
            mask = mask.resize(resized, Image.Resampling.NEAREST).crop(crop)
            image.save(destination / 'rgb' / f'{index:06d}.png')
            mask.save(destination / 'masks' / f'{index:06d}.png')
            labels = np.asarray(mask)
            ph, pw = labels.shape[0] // 14, labels.shape[1] // 14
            patches = labels.reshape(ph, 14, pw, 14).transpose(0, 2, 1, 3).reshape(-1, 196)
            # A partially unknown patch is not silently treated as fully labelled.
            known = (patches != 128).all(axis=1)
            fractions = np.where(known, (patches == 255).mean(axis=1), -1).astype(np.float32)
            coverage.append(fractions)
            shapes.add(image.size)
        pose = row.get('gt_c2w')
        if pose is not None:
            pose = np.asarray(pose, dtype=np.float64)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                    or not np.allclose(pose[3], [0, 0, 0, 1])
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
                raise ValueError('Invalid GT camera-to-world rigid transform')
            if not np.isfinite(row['gt_timestamp']) or abs(row['gt_timestamp'] - row['timestamp']) > .02:
                raise ValueError('GT association exceeds 20 ms; mark this GT null instead')
        records.append(dict(row, rgb=f'rgb/{index:06d}.png', motion_mask=f'masks/{index:06d}.png',
                            original_rgb=str(rgb_path.resolve()), original_mask=str(mask_path.resolve()),
                            source_rgb_sha256=digest(source_rgb), source_mask_sha256=digest(source_mask),
                            model_rgb_sha256=digest(destination / 'rgb' / f'{index:06d}.png'),
                            model_mask_sha256=digest(destination / 'masks' / f'{index:06d}.png'),
                            source_wh=[width, height], resize_wh=list(resized), crop=list(crop)))
    if len(shapes) != 1:
        raise ValueError('All clip frames must have the same processed dimensions')
    if sum(row.get('gt_c2w') is not None for row in records) < 3:
        raise ValueError('Need at least three associated camera GT poses')
    coverage = np.stack(coverage)
    if manifest['role'] not in ('primary_dynamic', 'static_control'):
        raise ValueError('role must be primary_dynamic or static_control')
    if manifest['role'] == 'primary_dynamic' and (not (coverage > 0).any() or not (coverage == 0).any()):
        raise ValueError('The primary diagnostic needs both labelled moving and static patches')
    if manifest['role'] == 'static_control' and ((coverage > 0).any() or not (coverage == 0).any()):
        raise ValueError('Static control needs verified static labels and no moving patches')
    np.save(destination / 'coverage.npy', coverage)
    write_json(destination / 'manifest.json', dict(manifest, frames=records,
               preprocessing='308-wide bicubic RGB, nearest mask, center-height crop; patch=14',
               unknown_patch_rule='any unknown pixel makes the patch unknown',
               source_manifest_sha256=digest(manifest_path)))
    shutil.copyfile(manifest_path, destination / 'source_manifest.json')
    print(f'INPUTS OK sequence={manifest["sequence"]} frames={len(frames)}', flush=True)


def load_model(checkpoint, budget):
    from streamvggt.models.streamvggt import StreamVGGT
    model = StreamVGGT(total_budget=budget)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.cuda().eval()


def worker(args):
    from streamvggt.models.dynamic_cache_probe import DynamicCacheProbe
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    inputs = args.work / 'inputs'
    frames = json.loads((inputs / 'manifest.json').read_text())['frames']
    coverage = np.load(inputs / 'coverage.npy')
    tokens = coverage.shape[1] + 5
    if args.arm == 'prefix':
        frames = frames[:max(12, len(frames) // 2)]
        coverage = coverage[:len(frames)]
    budget = 24 * tokens * (len(frames) + 1 if args.arm == 'full' else 8)
    output = args.work / args.arm
    output.mkdir()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.checkpoint, budget)
    assert model.aggregator.depth == 24 and model.aggregator.patch_start_idx == 5
    references, oracles = [], []
    intervention = args.arm == 'oracle' or args.arm.startswith('random')
    if intervention:
        references = [json.loads((args.work / 'native' / f'layer_{layer:02d}.json').read_text())
                      for layer in range(24)]
        model.aggregator.cache_budget_schedule = np.asarray([r['budgets'] for r in references]).T.tolist()
        if args.arm.startswith('random'):
            oracles = [json.loads((args.work / 'oracle' / f'layer_{layer:02d}.json').read_text())
                       for layer in range(24)]
    probes = []
    if args.arm != 'native_unlogged':
        for layer, block in enumerate(model.aggregator.global_blocks):
            arm = 'random' if args.arm.startswith('random') else args.arm
            arm = 'native' if arm == 'prefix' else arm
            probe = DynamicCacheProbe(layer, coverage, 5, arm, output,
                                      seed=int(args.arm[-1]) if arm == 'random' else 0,
                                      reference=references[layer]['events'] if intervention else None,
                                      oracle=oracles[layer]['events'] if oracles else None)
            block.attn.cache_probe = probe
            probes.append(probe)
    images = []
    for row in frames:
        path = inputs / row['rgb']
        if digest(path) != row['model_rgb_sha256']:
            raise ValueError('Prepared RGB hash changed')
        with Image.open(path) as image:
            images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255)
    poses = []
    def save_pose(index, frame, result):
        pose = result['camera_pose'].float()
        if not torch.isfinite(pose).all():
            raise ValueError(f'Nonfinite pose at frame {index}')
        poses.append(pose[0].numpy())
        np.save(output / 'pose_enc.npy', np.stack(poses))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        model.inference(({'img': image[None].cuda()} for image in images),
                        frame_writer=save_pose, cache_results=False)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    report = dict(arm=args.arm, frames=len(poses), budget_sum_tokens_per_head=budget,
                  dtype='float32', total_seconds=seconds,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  timing_scope='official inference incl CPU output copies, IO and diagnostic overhead; not a speed claim',
                  valid_pose_fraction=len(poses) / len(frames))
    layers = [probe.save() for probe in probes]
    if layers:
        report['global_kv_bytes_by_frame'] = np.sum([r['kv_bytes'] for r in layers], axis=0).tolist()
        report['cpu_token_id_bytes_by_frame'] = np.sum([r['id_bytes'] for r in layers], axis=0).tolist()
        report['cpu_coverage_bytes'] = coverage.nbytes
        report['eviction_events'] = sum(len(r['events']) for r in layers)
    if args.arm == 'full':
        assert report['eviction_events'] == 0, 'Full cache must never evict'
    extrinsics, _ = pose_encoding_to_extri_intri(torch.from_numpy(np.stack(poses))[None], images[0].shape[-2:])
    c2w = np.tile(np.eye(4), (len(poses), 1, 1))
    c2w[:, :3] = extrinsics[0].numpy()
    np.save(output / 'c2w.npy', np.linalg.inv(c2w))
    write_json(output / 'resources.json', report)
    if args.arm not in ('full', 'native_unlogged') and not report['eviction_events']:
        raise ValueError('Budget never bound: this is not an eviction diagnostic')
    print(f'ARM OK {args.arm} frames={len(poses)}', flush=True)


def metrics(work, arms):
    from stream3r_ycbv_metrics import umeyama_sim3
    manifest = json.loads((work / 'inputs/manifest.json').read_text())
    frames = manifest['frames']
    indices = np.array([i for i, row in enumerate(frames) if row.get('gt_c2w') is not None])
    gt = np.array([frames[i]['gt_c2w'] for i in indices], dtype=np.float64)
    times = np.array([frames[i]['timestamp'] for i in indices])
    pairs = (np.diff(indices) == 1) & (np.diff(times) <= .1)
    summaries = {}
    for arm in arms:
        estimated = np.load(work / arm / 'c2w.npy')[indices]
        centres, target = estimated[:, :3, 3], gt[:, :3, 3]
        if (np.linalg.matrix_rank(centres - centres.mean(0), tol=1e-8) < 2
                or np.linalg.matrix_rank(target - target.mean(0), tol=1e-8) < 2):
            raise ValueError('Degenerate trajectory: cannot claim Sim(3) ATE; retain raw outputs')
        scale, rotation, translation = umeyama_sim3(centres, target)
        aligned = estimated.copy()
        aligned[:, :3, 3] = scale * centres @ rotation.T + translation
        aligned[:, :3, :3] = rotation @ estimated[:, :3, :3]
        error = np.linalg.norm(aligned[:, :3, 3] - target, axis=1)
        relative = np.linalg.inv(np.linalg.inv(gt[:-1]) @ gt[1:]) @ (np.linalg.inv(aligned[:-1]) @ aligned[1:])
        rpe_t = np.linalg.norm(relative[:, :3, 3], axis=1)[pairs]
        rpe_r = np.degrees(np.arccos(np.clip((np.trace(relative[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))[pairs]
        enrichments, excesses = [], []
        for path in sorted((work / arm).glob('events_*.jsonl')):
            for line in path.read_text().splitlines():
                for head in json.loads(line)['heads']:
                    row = head['historical_patches']
                    if row['enrichment'] is not None:
                        enrichments.append(row['enrichment'])
                        excesses.append(row['excess'])
        summaries[arm] = dict(ate_m=float(np.sqrt(np.mean(error ** 2))), fitted_scale=scale,
                             gt_valid_frames=len(indices), gt_valid_fraction=len(indices) / len(frames),
                             rpe_pairs=int(pairs.sum()),
                             rpe_translation_rmse_m=float(np.sqrt(np.mean(rpe_t ** 2))) if len(rpe_t) else None,
                             rpe_rotation_rmse_deg=float(np.sqrt(np.mean(rpe_r ** 2))) if len(rpe_r) else None,
                             historical_enrichment_median=float(np.median(enrichments)) if enrichments else None,
                             historical_excess_median=float(np.median(excesses)) if excesses else None)
    native, full = summaries['native']['ate_m'], summaries['full']['ate_m']
    comparison = dict(native_minus_full_ate_m=native - full,
                      oracle_gain_fraction=(native - summaries['oracle']['ate_m']) / native if native > 1e-8 else None,
                      random_median_ate_m=float(np.median([summaries[f'random{s}']['ate_m'] for s in range(3)])),
                      interpretation='one-clip screening only; raw per-layer/head events retained; no independent-token significance')
    if manifest['mask_source']['kind'] == 'predicted_person':
        comparison['interpretation'] = ('Predicted person/background diagnostic only; oracle is a '
            'hindsight person-suppression intervention, not a motion-GT oracle. '
            'Background is not verified static; segmentation errors confound retention.')
    write_json(work / 'summary.json', dict(arms=summaries, comparison=comparison,
               mask_source=manifest['mask_source'],
               alignment='one sequence-wide Sim(3), no window alignment',
               rpe='adjacent selected frames, <=100 ms, no bridge over missing GT'))
    print('METRICS OK ' + json.dumps(comparison), flush=True)


def smoke(args):
    """Synthetic environment/fidelity check; never evidence for H1/H2."""
    from streamvggt.models.dynamic_cache_probe import DynamicCacheProbe
    torch.manual_seed(0)
    model = load_model(args.checkpoint, 24 * (4 * 4 + 5) * 8)
    images = torch.rand(12, 1, 3, 56, 56, device='cuda')
    with torch.inference_mode():
        result = model.inference([{'img': image} for image in images])
    assert len(result.ress) == 12
    assert all(torch.isfinite(row['camera_pose']).all() for row in result.ress)
    expected = torch.stack([row['camera_pose'] for row in result.ress])
    del model, result
    torch.cuda.empty_cache()
    model = load_model(args.checkpoint, 24 * (4 * 4 + 5) * 8)
    output = args.work / 'synthetic_logging_fidelity'
    output.mkdir()
    coverage = np.tile(np.arange(16) % 2, (12, 1)).astype(np.float32)
    probes = []
    for layer, block in enumerate(model.aggregator.global_blocks):
        probe = DynamicCacheProbe(layer, coverage, 5, 'native', output)
        block.attn.cache_probe = probe
        probes.append(probe)
    with torch.inference_mode():
        result = model.inference([{'img': image} for image in images])
    actual = torch.stack([row['camera_pose'] for row in result.ress])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert sum(len(probe.events) for probe in probes) > 0
    for probe in probes:
        probe.save()
    write_json(args.work / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(), python=sys.version, dtype='float32',
               synthetic_frames=12, input_hw=[56, 56], checkpoint_strict_load=True,
               logging_fidelity_atol=1e-6, logging_fidelity_rtol=1e-5,
               peak_allocated_bytes=torch.cuda.max_memory_allocated(), evidence='environment only'))
    print('ENVIRONMENT SMOKE OK (synthetic; not experimental evidence)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'pilot', 'worker'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--predicted-person-masks', action='store_true')
    parser.add_argument('--arm', choices=('full', 'native_unlogged', 'native', 'prefix', 'oracle', 'random0', 'random1', 'random2'))
    args = parser.parse_args()
    assert sys.platform == 'linux' and torch.cuda.is_available(), 'CAMP GPU execution only'
    assert args.work.resolve().is_relative_to('/tmp')
    if args.mode == 'smoke':
        smoke(args)
        return
    if args.mode == 'worker':
        worker(args)
        return
    if args.manifest is None:
        parser.error('pilot requires --manifest; prepare/inventory does not establish data readiness')
    prepare(args.manifest, args.work / 'inputs', args.predicted_person_masks)
    arms = ['full', 'native_unlogged', 'native', 'prefix', 'oracle', 'random0', 'random1', 'random2']
    for arm in arms:
        subprocess.run([sys.executable, __file__, 'worker', '--checkpoint', str(args.checkpoint),
                        '--work', str(args.work), '--arm', arm], check=True)
        if arm in ('native', 'prefix'):
            expected = np.load(args.work / 'native_unlogged/pose_enc.npy')
            actual = np.load(args.work / arm / 'pose_enc.npy')
            np.testing.assert_allclose(actual, expected[:len(actual)], atol=1e-6, rtol=1e-5)
            if arm == 'prefix':
                for layer in range(24):
                    short = json.loads((args.work / 'prefix' / f'layer_{layer:02d}.json').read_text())
                    long = json.loads((args.work / 'native' / f'layer_{layer:02d}.json').read_text())
                    assert short['events'] == {k: v for k, v in long['events'].items() if int(k) < len(actual)}
            print(f'FIDELITY OK {arm}', flush=True)
    metrics(args.work, ['full', 'native', 'oracle', 'random0', 'random1', 'random2'])
    print('DYNAMIC CACHE PILOT COMPLETE one clip; review before expansion', flush=True)


if __name__ == '__main__':
    main()
