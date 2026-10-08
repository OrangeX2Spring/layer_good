"""CAMP-only bounded training and separately reviewed loop-transformer pilots."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from kvt_tum_run import prepare, evaluate, write_json
from kvt_tum_sweep import release_page_cache
from kv_tracker.loop_models import LoopConfig, LoopedPi3, VARIANTS, load_looped, pose_distance
from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32

SPLITS = dict(train='freiburg1_desk', development='freiburg3_structure_texture_far')
QUERY_FRAMES = (1, 8, 16, 32, 48, 50, 56, 63)
TEST_SCENE = 'freiburg3_long_office_household'
SEED = 17
STEPS = 20
RESIZE = 308


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity():
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    files = sorted(snapshot.glob('*.safetensors')) + sorted(snapshot.glob('pytorch_model*.bin'))
    assert files, 'Offline Pi3 weights missing'
    return {p.name: sha256(p) for p in files}


def load_images(inputs, count, pixels=False):
    manifest = json.loads((inputs / 'manifest.json').read_text())
    assert count <= manifest['frames']
    images = []
    for row in manifest['inputs'][:count]:
        bgr = cv2.imread(str(inputs / 'model_rgb' / Path(row['file']).name))
        assert bgr is not None
        rgb = bgr[:, :, ::-1].copy()
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['model_rgb_sha256']
        images.append(rgb if pixels else torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.))
    return manifest, images


def calibration(scene, height, width, device):
    if scene.startswith('freiburg1_'):
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
        coefficients = (.2624, -.9531, -.0054, .0026, 1.1633)
    elif scene.startswith('freiburg2_'):
        fx, fy, cx, cy = 520.9, 521., 325.1, 249.7
        coefficients = (.2312, -.7849, -.0033, -.0001, .9172)
    else:
        assert scene.startswith('freiburg3_')
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
        coefficients = None
    k = torch.tensor([[fx * width / 640, 0, (cx + .5) * width / 640 - .5],
                      [0, fy * height / 480, (cy + .5) * height / 480 - .5],
                      [0, 0, 1]], device=device, dtype=torch.float32)
    distortion = None if coefficients is None else torch.tensor(coefficients, device=device)
    return k, distortion


def build_model(variant, saved=None):
    native = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())
    native.requires_grad_(False)
    if saved is not None:
        model, provenance = load_looped(native, saved)
        assert provenance['checkpoint_weights'] == checkpoint_identity()
        assert provenance['source_sha256'] == sha256(inspect.getfile(LoopedPi3))
        assert provenance['driver_sha256'] == sha256(__file__)
        assert model.config.variant == variant
    else:
        model, provenance = LoopedPi3(native, LoopConfig(variant)), None
    del native
    gc.collect()
    return model, provenance


def teacher_labels(inputs, output):
    torch.manual_seed(SEED)
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())
    model.requires_grad_(False)
    manifest, images = load_images(inputs, 64)
    captured = []
    hook = model.camera_decoder.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    labels = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for frame in QUERY_FRAMES:
            bank = [0, 0] if frame < 49 else [0, 49]
            captured.clear()
            reconstruction = model(torch.stack([images[i] for i in bank]).cuda()[None], store_cache=True)
            origin = torch.linalg.inv(reconstruction['camera_poses'][0, 0].float())
            points = reconstruction['points'][0, :, 7::14, 7::14].reshape(-1, 3).float()
            scale = (points - points.mean(0)).norm(dim=-1).median().clamp_min(1e-6)
            captured.clear()
            result = model(images[frame].cuda()[None, None], use_cache=True)
            assert len(captured) == 1
            labels.append(dict(frame=frame, bank=bank, origin=origin.cpu(), scale=scale.cpu(),
                pose=result['camera_poses'].float().cpu(), features=captured[0].cpu(),
                local_points=result['local_points'].float().cpu(),
                confidence=result['conf'].float().sigmoid().cpu()))
    hook.remove()
    torch.save(dict(labels=labels, scene=manifest['scene'], manifest_sha256=sha256(inputs / 'manifest.json'),
                    checkpoint_weights=checkpoint_identity()), output)
    print('LOOP TEACHER LABELS OK', manifest['scene'], len(labels), flush=True)


def rebuild(model, images, bank):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(torch.stack([images[i] for i in bank]).cuda()[None], store_cache=True)
        return torch.linalg.inv(result['camera_poses'][0, 0].float()).detach()


def distillation_loss(output, label, origin, scale):
    with torch.autocast('cuda', enabled=False):
        pose = output['camera_poses'][0, 0].float()
        predicted = origin @ pose
        expected = label['origin'].cuda() @ label['pose'][0, 0].cuda()
        loss = F.mse_loss(predicted[:3, :3], expected[:3, :3])
        loss = loss + F.mse_loss(predicted[:3, 3] / scale, expected[:3, 3] / label['scale'].cuda())
        if 'local_points' in output:
            target = label['local_points'].cuda()[:, :, 7::14, 7::14]
            current = output['local_points'][:, :, 7::14, 7::14].float()
            weight = label['confidence'].cuda()[:, :, 7::14, 7::14]
            loss = loss + ((current / scale - target / label['scale'].cuda()).square() * weight).mean()
        return loss


def query_loss(model, images, label, budget):
    model.query_loops = budget
    origin = rebuild(model, images, label['bank'])
    if model.config.variant == 'refiner':
        # Match deployment history; truncate gradients at each predicted pose.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for frame in range(label['bank'][-1] + 1, label['frame']):
                model(images[frame].cuda()[None, None], cam_only=True, use_cache=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(images[label['frame']].cuda()[None, None], use_cache=True,
                       cam_only=model.config.variant == 'refiner')
        loss = distillation_loss(output, label, origin, model.map_scale)
        if model.config.variant != 'refiner':
            target = label['features'].cuda().float()
            norm = target.square().mean().sqrt().clamp_min(1e-6)
            loss = loss + F.mse_loss(model.last_features.float() / norm, target / norm)
            for features in model.exit_features.values():
                # Supervise every prefix eligible for adaptive early stopping.
                loss = loss + F.mse_loss(features.float() / norm, target / norm)
                h, w = images[0].shape[-2:]
                pos = model.position_getter(1, h // 14, w // 14, target.device) + 1
                pos = torch.cat((pos.new_zeros(1, 5, 2), pos), 1)
                short = dict(camera_poses=model.read_pose(features, pos, h, w).reshape(1, 1, 4, 4),
                             local_points=model.read_local(features, pos, h, w)[None])
                loss = loss + distillation_loss(short, label, origin, model.map_scale)
            if model.routing_scores is not None:
                # Detached per-patch feature error supervises discrete top/spread routing.
                difficulty = (model.routing_features.detach().float() - target[:, 5:, :model.dec_embed_dim]).square().mean(-1)
                difficulty = difficulty / difficulty.mean().clamp_min(1e-6)
                loss = loss + .1 * F.mse_loss(model.routing_scores.float(), difficulty.flatten())
    assert torch.isfinite(loss)
    return loss


def calibrate_halting(model, images, labels, result):
    model.eval()
    model.collect_exits = True
    signals, safe, records = [], [], []
    calibration_frames = {label['frame'] for label in labels[:len(labels) // 2]}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for label in labels:
            origin = rebuild(model, images, label['bank'])
            model(images[label['frame']].cuda()[None, None], cam_only=True, use_cache=True)
            with torch.autocast('cuda', enabled=False):
                target = label['origin'].cuda() @ label['pose'][0, 0].cuda()
            for row in model.exit_records:
                if not model.config.short_loops <= row['loop'] < model.config.full_loops:
                    continue
                with torch.autocast('cuda', enabled=False):
                    predicted = origin @ row['pose'].float()
                # Both translations normalized by their bank scale, fixed for the episode.
                normalized = predicted.clone()
                normalized[:3, 3] /= model.map_scale
                expected = target.clone()
                expected[:3, 3] /= label['scale'].cuda()
                error = pose_distance(normalized, expected, normalized.new_tensor(1.))
                good = bool((error <= .05).all())
                signals.append(row['signals'].float())
                safe.append(good)
                records.append(dict(frame=label['frame'], loop=row['loop'], safe=good,
                    partition='fit' if label['frame'] in calibration_frames else 'validation',
                    error=error.cpu().tolist(),
                    signals=[float(v) if torch.isfinite(v) else None for v in row['signals']]))
    x = torch.stack(signals)
    good = torch.tensor(safe, device=x.device) & torch.isfinite(x).all(1)
    passed = False
    thresholds = None
    fit = torch.tensor([row['partition'] == 'fit' for row in records], device=x.device)
    if (good & fit).any():
        candidate = torch.quantile(x[good & fit], .25, dim=0)
        accepted = (x <= candidate).all(1)
        passed = bool((accepted & fit).any() and (accepted & ~fit).any()
                      and not (accepted & ~good).any())
        if passed:
            model.halt_thresholds.copy_(candidate)
            thresholds = candidate.cpu().tolist()
    model.collect_exits = False
    write_json(result / 'halting_calibration.json', dict(passed=passed, thresholds=thresholds,
        rule='Fit on first four development queries; require safe acceptance on disjoint last four',
        limitation='Same-scene frame holdout, not independent-scene or trajectory quality', records=records))
    return passed


def train_one(work, variant):
    torch.manual_seed(SEED)
    model, _ = build_model(variant)
    data = {}
    for split, scene in SPLITS.items():
        inputs = work / 'inputs' / split
        manifest, images = load_images(inputs, 64)
        saved = torch.load(work / 'labels' / f'{split}.pt', weights_only=True)
        assert saved['scene'] == scene and saved['manifest_sha256'] == sha256(inputs / 'manifest.json')
        assert saved['checkpoint_weights'] == checkpoint_identity()
        data[split] = images, saved['labels']
    result = work / 'models' / variant
    result.mkdir(parents=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    budgets = ([model.config.full_loops, model.config.short_loops]
               if variant in ('elastic', 'shared_kv', 'combined') else [model.config.full_loops])
    images, labels = data['train']
    model.geometry_calibration = calibration(SPLITS['train'], *images[0].shape[-2:], 'cuda')
    model.train()
    model.collect_exits = False
    with torch.no_grad():
        initial = float(query_loss(model, images, labels[0], budgets[0]))
    updates = []
    for step in range(STEPS):
        label = labels[step % len(labels)]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for budget in budgets:
            loss = query_loss(model, images, label, budget) / len(budgets)
            loss.backward()
            losses.append(float(loss.detach()))
            # Keep no old graph live across budgets or optimizer steps.
            model.last_features = model.routing_features = model.routing_scores = None
            model.exit_features = {}
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.))
        optimizer.step()
        updates.append(dict(step=step + 1, frame=label['frame'], loss=sum(losses), gradient_norm=grad_norm))
        write_json(result / 'updates.json', updates)
    with torch.no_grad():
        final = float(query_loss(model, images, labels[0], budgets[0]))
    model.eval()
    images, labels = data['development']
    model.geometry_calibration = calibration(SPLITS['development'], *images[0].shape[-2:], 'cuda')
    development = []
    # Train-mode without gradients suppresses deployment halting until calibrated.
    model.train()
    with torch.no_grad():
        for label in labels:
            development.append(float(query_loss(model, images, label, budgets[0])))
    calibration_pass = calibrate_halting(model, images, labels, result) if variant == 'adaptive' else None
    model.eval()
    provenance = dict(seed=SEED, steps=STEPS, learning_rate=1e-4, splits=SPLITS,
        checkpoint_weights=checkpoint_identity(), training_loss_decreased=final < initial,
        halting_calibration_pass=calibration_pass, engineering_smoke_only=True,
        training_protocol_sha256=sha256(work / 'protocol.json'),
        source_sha256=sha256(inspect.getfile(LoopedPi3)), driver_sha256=sha256(__file__),
        label_sha256={s: sha256(work / 'labels' / f'{s}.pt') for s in SPLITS})
    path = result / 'model.pt'
    model.save(path, provenance)
    # Exact state round-trip, followed by live output equality in the GPU contracts.
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = model.state_dict()
    for key, value in saved['state_dict'].items():
        assert torch.equal(value, current[key].detach().cpu())
    del saved, current
    summary = dict(variant=variant, config=asdict(model.config), initial_loss=initial, final_loss=final,
        development_losses=development, calibration_pass=calibration_pass,
        training_loss_decreased=final < initial, checkpoint_sha256=sha256(path),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        storage=model.storage_report(), eligible_for_full_evaluation=False,
        next_gate='review smoke and calibration before an isolated tracking pilot')
    write_json(result / 'summary.json', summary)
    print('LOOP TRAINING SMOKE COMPLETE', variant, json.dumps(summary), flush=True)


class PhaseRecorder:
    """CUDA-event profiling only; profiled runs are separate from timed pilots."""
    def __init__(self, model):
        self.pending, self.handles = [], []
        self.model = model
        model.profile_compact = True
        groups = [('encoder', model.encoder), ('camera_decoder', model.camera_decoder),
                  ('camera_head', model.camera_head), ('point_decoder', model.point_decoder),
                  ('point_head', model.point_head), ('conf_decoder', model.conf_decoder),
                  ('conf_head', model.conf_head)]
        if model.config.variant in ('native', 'native_compact', 'refiner'):
            groups += [('global' if i % 2 else 'local', b) for i, b in enumerate(model.decoder)]
        else:
            groups += [('global' if i % 2 else 'local', b)
                       for collection in (model.entry, model.exit, model.core, model.depth_blocks)
                       for i, b in enumerate(collection)]
        if model.config.variant == 'refiner':
            groups.append(('refiner', model.refiner))
        for name, module in groups:
            def before(mod, args, name=name):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                self.pending.append((name, start, end))
                mod._loop_profile_end = end
            def after(mod, args, output):
                mod._loop_profile_end.record()
            self.handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]

    def drain(self):
        torch.cuda.synchronize()
        total = {}
        for name, start, end in self.pending + self.model.compact_events:
            total[name] = total.get(name, 0.) + start.elapsed_time(end) / 1000
        self.pending.clear()
        self.model.compact_events.clear()
        return total

    def close(self):
        self.model.profile_compact = False
        for hook in self.handles:
            hook.remove()


def arm(config_path):
    from kv_tracker.map_handoff import MapHandoff
    config = json.loads(config_path.read_text())
    torch.manual_seed(SEED)
    result = config_path.parent
    inputs = Path(config['inputs'])
    manifest, images = load_images(inputs, config['frames'], pixels=True)
    checkpoint_path = None if config['variant'] in ('native', 'native_compact') else Path(config['checkpoint'])
    model, provenance = build_model(config['variant'], checkpoint_path)
    model.eval()
    model.record_details = config['details']
    model.query_loops = config['budget']
    model.geometry_calibration = calibration(manifest['scene'], *images[0].shape[:2], 'cuda')
    if config['variant'] == 'adaptive':
        assert provenance['halting_calibration_pass'] and (model.halt_thresholds >= 0).all()
    if provenance is not None:
        assert provenance['training_loss_decreased'], 'Failed smoke is not eligible for a tracking pilot'
    profiler = PhaseRecorder(model) if config['profile'] else None
    rows, poses = [], []
    def log(row):
        if profiler is not None and row['kind'] in ('bootstrap', 'query', 'rebuild'):
            row['gpu_phase_seconds'] = profiler.drain()
        if row['kind'] == 'query' and config['variant'] not in ('native', 'native_compact'):
            row['execution'] = dict(model.last_execution)
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
    with (result / 'inference.jsonl').open('w') as stream, torch.inference_mode():
        tracker = MapHandoff(model, 'native', log, None)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            for frame, pixels in enumerate(images):
                poses.append(tracker.bootstrap(pixels) if frame == 0 else tracker.step(pixels, frame))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
        finally:
            write_json(result / 'tracking_progress.json', dict(completed_frames=len(poses),
                       requested_frames=len(images), complete=len(poses) == len(images)))
            if poses:
                filename = 'traj.npy' if len(poses) == len(images) else 'partial_traj.npy'
                np.save(result / filename, np.asarray(poses))
        np.save(result / 'kf_idx.npy', np.asarray(tracker.ids))
        metrics = dict(frames=len(poses), seconds=seconds, fps=len(poses) / seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), storage=model.storage_report(),
            profile=config['profile'], details=config['details'], checkpoint_provenance=provenance,
            timing='upload + query + full bank updates + CPU trajectory/logging; preloaded RGB, scene only')
        if config['frames'] >= 128:
            metrics.update(evaluate(inputs, result, .02))
            with np.load(result / 'evaluation.npz') as saved:
                metrics['translation_p99_m'] = float(np.quantile(saved['rpe_translation_per_pair_m'], .99))
        free, total = torch.cuda.mem_get_info()
        metrics.update(device_total_bytes=total, device_free_bytes_at_end=free,
                       gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda)
        write_json(result / 'metrics.json', metrics)
    if profiler is not None:
        profiler.close()
    print('LOOP ARM COMPLETE', config['variant'], config['budget'], config['repeat'], flush=True)


def invoke(*arguments):
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *map(str, arguments)], check=True)


def comparison(paths, output):
    records = {name: json.loads((p / 'metrics.json').read_text()) for name, p in paths.items()}
    comparisons = {}
    for name, metrics in records.items():
        if name == 'native':
            continue
        repeat = int(name.rsplit('_r', 1)[1])
        reference_name = ('native' if repeat == 0 or name == f'native_b4_r{repeat}'
                          else f'native_b4_r{repeat}')
        reference = records[reference_name]
        with np.load(paths[reference_name] / 'evaluation.npz') as a, np.load(paths[name] / 'evaluation.npz') as b:
            for key in ('rgb_indices', 'gt_indices', 'rpe_pair_start_indices'):
                np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(np.load(paths[reference_name] / 'kf_idx.npy'), np.load(paths[name] / 'kf_idx.npy'))
        ratios = {key: metrics[key] / reference[key] for key in ('ate_m', 'rpe_translation_m', 'translation_p99_m')}
        quality = all(v <= 1.05 for v in ratios.values())
        timed = not (metrics['profile'] or reference['profile']
                     or metrics['details'] or reference['details'])
        comparisons[name] = dict(reference=reference_name, quality_ratios=ratios, quality_pass=quality,
            speedup=metrics['fps'] / reference['fps'], memory_ratio=metrics['peak_allocated_bytes'] / reference['peak_allocated_bytes'],
            doubled_fps=timed and quality and metrics['fps'] >= 2 * reference['fps'],
            half_memory=timed and quality and metrics['peak_allocated_bytes'] <= .5 * reference['peak_allocated_bytes'])
    write_json(output, dict(runs=records, comparisons=comparisons,
        limitation='scene-only pilot; no ARCTIC/SAM or unobserved-GT claims; no automatic full evaluation'))


def main(args):
    assert sys.platform == 'linux' and torch.cuda.is_available(), 'CAMP GPU only'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.arm:
        return arm(args.arm)
    assert args.work is not None and args.work.is_relative_to('/tmp')
    if args.labels:
        return teacher_labels(args.labels, args.label_output)
    if args.train_one:
        return train_one(args.work, args.train_one)
    protocol = dict(variants=VARIANTS, splits=SPLITS, seed=SEED, steps=STEPS,
        query_frames=QUERY_FRAMES, resize=RESIZE, stage=args.stage,
        checkpoint_weights=checkpoint_identity(), source_sha256=sha256(inspect.getfile(LoopedPi3)),
        scope='bounded engineering training or separately reviewed scene evaluation',
        calibration_source='TUM official RGB camera calibration; exact resize with half-pixel centers',
        calibration_url='https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats')
    write_json(args.work / 'protocol.json', protocol)
    if args.stage == 'train':
        (args.work / 'labels').mkdir()
        for split, scene in SPLITS.items():
            source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
            inputs = args.work / 'inputs' / split
            prepare(source, inputs, RESIZE, .02, count=64)
            (inputs / 'archive.sha256').write_text(f'{sha256(source)}  {source}\n')
            invoke('--work', args.work, '--labels', inputs, '--label-output', args.work / 'labels' / f'{split}.pt')
            release_page_cache(inputs)
        variants = VARIANTS if args.variant == 'all' else (args.variant,)
        for variant in variants:
            invoke('--work', args.work, '--train-one', variant)
        write_json(args.work / 'training_complete.json', dict(variants=variants, steps_per_variant=STEPS,
            summaries={v: json.loads((args.work / 'models' / v / 'summary.json').read_text()) for v in variants},
            eligible_for_full_evaluation=False))
        print('LOOP TRAINING ENGINEERING COMPLETE: review each smoke/calibration; no automatic tracking', flush=True)
        return
    assert args.checkpoint is not None, 'Supply the reviewed training archive'
    assert args.variant in VARIANTS or args.variant == 'all'
    assert args.stage != 'full' or args.variant in VARIANTS, 'Full evaluation selects one reviewed variant'
    if args.stage == 'full':
        assert args.reviewed_pilot is not None
        pilot = json.loads(args.reviewed_pilot.read_text())
        assert pilot['variant'] == args.variant and pilot['checkpoint_archive_sha256'] == sha256(args.checkpoint)
        assert pilot['quality_pass'], 'Full evaluation requires a reviewed passing pilot'
        assert pilot['driver_sha256'] == sha256(__file__)
        assert pilot['source_sha256'] == sha256(inspect.getfile(LoopedPi3))
    training = args.work / 'reviewed_training'
    training.mkdir()
    with tarfile.open(args.checkpoint) as packed:
        # External archive is a filesystem boundary; reject traversal and links.
        for member in packed.getmembers():
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            assert member.isfile() or member.isdir()
        packed.extractall(training)
    summaries = json.loads((training / 'training_complete.json').read_text())['summaries']
    inputs = args.work / 'inputs' / 'office'
    count = None if args.stage == 'full' else 256
    source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{TEST_SCENE}.zip'
    manifest = prepare(source, inputs, RESIZE, .02, count=count)
    (inputs / 'archive.sha256').write_text(f'{sha256(source)}  {source}\n')
    variants = VARIANTS if args.variant == 'all' else (args.variant,)
    blocked = {}
    for variant in variants:
        assert variant in summaries
        assert sha256(training / 'models' / variant / 'model.pt') == summaries[variant]['checkpoint_sha256']
        if not summaries[variant]['training_loss_decreased']:
            blocked[variant] = 'Training smoke did not decrease the fixed-example loss'
        elif variant == 'adaptive' and not summaries[variant]['calibration_pass']:
            blocked[variant] = 'Halting calibration failed; adaptive deployment is unavailable'
    write_json(args.work / 'blocked_variants.json', blocked)
    if args.variant != 'all':
        assert not blocked, blocked
    for variant, reason in blocked.items():
        print('LOOP VARIANT BLOCKED', variant, reason, flush=True)
    eligible = tuple(v for v in variants if v not in blocked)
    paths = {}
    # Two ordered paired repetitions; profiling is a separate diagnostic stage.
    for repeat in range(2 if args.stage != 'profile' else 1):
        for variant in ('native', 'native_compact') + eligible:
            if variant not in ('native', 'native_compact'):
                assert summaries[variant]['training_loss_decreased']
                assert variant != 'adaptive' or summaries[variant]['calibration_pass']
            budgets = (2, 4) if variant in ('elastic', 'shared_kv', 'combined') else (4,)
            for budget in budgets:
                name = f'{variant}_b{budget}_r{repeat}'
                result = args.work / 'runs' / name
                result.mkdir(parents=True)
                path = training / 'models' / variant / 'model.pt'
                config = dict(variant=variant, checkpoint=None if variant in ('native', 'native_compact') else str(path),
                    inputs=str(inputs), frames=manifest['frames'], profile=args.stage == 'profile',
                    details=args.stage == 'profile', budget=budget, repeat=repeat)
                write_json(result / 'config.json', config)
                invoke('--arm', result / 'config.json')
                paths['native' if variant == 'native' and repeat == 0 else name] = result
    comparison(paths, args.work / 'comparison.json')
    if args.stage == 'pilot':
        compared = json.loads((args.work / 'comparison.json').read_text())['comparisons']
        gates = args.work / 'pilot_gates'
        gates.mkdir()
        for variant in eligible:
            selected = [row for key, row in compared.items() if key.startswith(variant + '_')]
            assert len(selected) == (4 if variant in ('elastic', 'shared_kv', 'combined') else 2)
            gate = dict(variant=variant, checkpoint_archive_sha256=sha256(args.checkpoint),
                        driver_sha256=sha256(__file__), source_sha256=sha256(inspect.getfile(LoopedPi3)),
                        quality_pass=all(r['quality_pass'] for r in selected),
                        comparison_sha256=sha256(args.work / 'comparison.json'),
                        next_gate='manual artifact review before full stage')
            write_json(gates / f'{variant}.json', gate)
            if args.variant != 'all':
                write_json(args.work / 'pilot_gate.json', gate)
    print('LOOP', args.stage.upper(), 'COMPLETE: inspect comparison and provenance', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--stage', choices=('train', 'pilot', 'profile', 'full'), default='train')
    parser.add_argument('--variant', choices=VARIANTS + ('all',), default='all')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reviewed-pilot', type=Path)
    parser.add_argument('--arm', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--label-output', type=Path)
    parser.add_argument('--train-one', choices=VARIANTS)
    main(parser.parse_args())
