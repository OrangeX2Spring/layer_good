"""ATD feasibility/oracle gate; CAMP only. No scorer training or automatic next stage."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch

from kvt_patch_select import (CLIPS, RESIZE, TOLERANCE, TUM_ZIPS, MAX_GT_DIFFERENCE,
    CHECKOUT, evaluate_object, evaluate_scene, prepare, archive_inputs, sha256, write_json)
from kv_tracker.adaptive_tokens import AdaptiveTokens, ARMS, keep_set
from kv_tracker.patch_select import SEEDS
from kv_tracker.token_drop import forward_kept

CONTROLS = ('uniform',) + tuple(f'random{s}' for s in SEEDS)


def run(path):
    import main as tracker
    config = json.loads(path.read_text())
    target = path.parent
    os.chdir(CHECKOUT)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    task, frames = config['task'], config['frames']
    if task == 'object':
        from kvt_arctic_run import ArcticFrames, DATASET_DIR
        (target / 'sam_masks').mkdir()
        source = ArcticFrames('cuda:0', target / 'sam_masks',
            scene_dir=DATASET_DIR / config['scene'], offset=2, obj_mode=True, resize_dim=518)
        assert source.length == config['source_frames']
        source.length = frames
        arguments = ['--obj_mode', '--resize_dim', '518']
    else:
        from kv_tracker.dataloaders.tum import TUMLoader
        scene_dir = Path(config['scene_dir'])
        manifest = json.loads((scene_dir / 'manifest.json').read_text())

        class Frames(TUMLoader):
            def __iter__(self):
                for index in range(frames):
                    frame = self.get_frame(index)
                    frame['idx'] = index
                    import hashlib
                    assert hashlib.sha256(frame['resized_rgb_masked_np'].tobytes()).hexdigest() == (
                        manifest['inputs'][index]['model_rgb_sha256'])
                    yield frame

        source = Frames('cuda:0', scene_dir=scene_dir, obj_mode=False, resize_dim=308, offset=0)
        arguments = ['--cam_only', '--resize_dim', '308', '--kf_auto', '50']
    rows = []
    with (target / 'tokens.jsonl').open('w') as log:
        policy = None if config['policy'] == 'native' else AdaptiveTokens(config['policy'], task, log)
        original = tracker.pi3_inference
        started = None

        def measured(model, *args, **kwargs):
            nonlocal started
            torch.cuda.synchronize()
            if started is None:
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
            before = time.perf_counter()
            result = original(model, *args, **kwargs)
            torch.cuda.synchronize()
            rows.append(dict(kind='query' if kwargs.get('use_cache') else 'rebuild',
                seconds=time.perf_counter() - before,
                cache_bytes=sum(t.numel() * t.element_size() for v in model.cache.values() for t in v.values())))
            return result

        tracker.pi3_inference = measured
        schedule = config['schedule']
        tracker.run_track3r(cfg=dict(results_path=str(target), que_size=1), args=arguments,
            frame_source=source, token_policy=policy,
            keyframe_indices=None if schedule is None else schedule[1:])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    trajectory = np.load(target / 'traj.npy')
    assert trajectory.shape == (frames, 4, 4) and np.isfinite(trajectory).all()
    if not (target / 'kf_idx.npy').exists():
        np.save(target / 'kf_idx.npy', np.array([0]))
    keyframes = np.load(target / 'kf_idx.npy').tolist()
    if schedule is not None:
        assert keyframes == schedule
    assert sum(r['kind'] == 'query' for r in rows) == frames - 1
    assert sum(r['kind'] == 'rebuild' for r in rows) == len(keyframes)
    (target / 'pcd.npy').unlink(missing_ok=True)
    metrics = dict(task=task, policy=config['policy'], frames=frames, keyframes=keyframes,
        tracking_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
        max_cache_bytes=max(r['cache_bytes'] for r in rows),
        timing='synchronized wall time from first inference, including teacher/selection when present',
        teacher_seconds=0 if policy is None else policy.teacher_seconds,
        selection_seconds=0 if policy is None else policy.selection_seconds,
        student_forward_seconds=None if policy is None else policy.student_seconds,
        teacher_state_in_peak=config['policy'] in ('oracle', 'come'))
    if policy is not None:
        metrics.update(mean_kept_patches=float(np.mean([e['budget'] for e in policy.events])),
            mismatched_budget_frames=sum(e['budget'] != e['target_budget'] for e in policy.events))
    metrics.update(evaluate_object(config['scene'], target, frames) if task == 'object'
                   else evaluate_scene(Path(config['scene_dir']), target, frames))
    write_json(target / 'inference.json', rows)
    write_json(target / 'events.json', [] if policy is None else policy.events)
    write_json(target / 'metrics.json', metrics)
    print('ATD RUN OK', config['clip'], config['name'], json.dumps(metrics), flush=True)


def benchmark(target):
    """Fresh process: synthetic 4:3-ish grids matching tracker resize dimensions."""
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
    torch.manual_seed(0)
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained('cuda').eval())
    model.requires_grad_(False)
    results = []
    # Actual model-input grids are recorded by the all-kept tracking runs.
    config = json.loads(target.read_text())
    for spec in config['shapes']:
        height, width = spec['shape']
        task, resize_dim = spec['task'], spec['resize_dim']
        grid = (height // 14, width // 14)
        p = math.prod(grid)
        for bank in (1, 10, 20):
            imgs = torch.rand(1, bank, 3, height, width, device='cuda')
            query = torch.rand(1, 1, 3, height, width, device='cuda')
            for fraction in (1., .75, .5, .25):
                model.cache = {}
                model.kept_cache = None
                torch.cuda.empty_cache()
                keep = keep_set(torch.zeros(p, dtype=torch.bool, device='cuda'), math.ceil(p * fraction), grid)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    for kind in ('rebuild', 'query'):
                        times, peaks = [], []
                        for repeat in range(7):
                            if kind == 'rebuild':
                                model.cache = {}
                                model.kept_cache = None
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            started = time.perf_counter()
                            output = forward_kept(model, imgs if kind == 'rebuild' else query,
                                keep[None].expand(bank if kind == 'rebuild' else 1, -1),
                                cam_only=(kind == 'query' and task == 'scene'),
                                store_cache=kind == 'rebuild', use_cache=kind == 'query')
                            torch.cuda.synchronize()
                            seconds = time.perf_counter() - started
                            if repeat >= 2:
                                times.append(seconds)
                                peaks.append(torch.cuda.max_memory_allocated())
                            del output
                        results.append(dict(shape=[height, width], task=task, resize_dim=resize_dim,
                            bank=bank, fraction=fraction,
                            kind=kind, seconds=times, median_seconds=float(np.median(times)),
                            peak_allocated_bytes=max(peaks), kept=int(keep.sum()), patches=p))
                        write_json(target.parent / 'benchmark.json', results)
            del imgs, query
    print('ATD BENCHMARK OK', flush=True)



def profile_scene(target):
    """Diagnose scene query latency without changing the policy or timing gate."""
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
    from torch.profiler import profile, ProfilerActivity
    target.mkdir()
    torch.manual_seed(0)
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained('cuda').eval())
    model.requires_grad_(False)
    height, width = 266, 350  # Recorded office model-input grid, resize_dim=308.
    grid = (height // 14, width // 14)
    patches = math.prod(grid)
    records = []
    for bank in (1, 10, 20):
        imgs = torch.rand(1, bank, 3, height, width, device='cuda')
        query = torch.rand(1, 1, 3, height, width, device='cuda')
        for fraction in (1., .5):
            model.cache = {}
            model.kept_cache = None
            keep = keep_set(torch.zeros(patches, dtype=torch.bool, device='cuda'),
                            math.ceil(patches * fraction), grid)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                forward_kept(model, imgs, keep[None].expand(bank, -1), store_cache=True)
                times = []
                for repeat in range(7):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = forward_kept(model, query, keep[None], cam_only=True, use_cache=True)
                    torch.cuda.synchronize()
                    if repeat >= 2:
                        times.append(time.perf_counter() - started)
                expected = output['camera_poses'].clone()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=True, with_stack=True) as prof:
                    output = forward_kept(model, query, keep[None], cam_only=True, use_cache=True)
                    torch.cuda.synchronize()
                torch.testing.assert_close(output['camera_poses'], expected, rtol=0, atol=0)
            name = f'bank{bank}_fraction{fraction}'
            prof.export_chrome_trace(str(target / f'{name}.json'))
            averages = prof.key_averages(group_by_input_shape=True)
            (target / f'{name}_cpu.txt').write_text(averages.table(
                sort_by='self_cpu_time_total', row_limit=40))
            (target / f'{name}_cuda.txt').write_text(averages.table(
                sort_by='self_cuda_time_total', row_limit=40))
            record = dict(bank=bank, fraction=fraction, kept=int(keep.sum()),
                shape=[height, width], seconds=times, median_seconds=float(np.median(times)),
                gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                timing='uninstrumented synchronized wall time; separate one-query profiler trace')
            records.append(record)
            write_json(target / 'summary.json', records)
            print('ATD SCENE PROFILE', json.dumps(record), flush=True)
    print('ATD SCENE PROFILE OK', flush=True)


def decide(results, timing):
    speeds = []
    for bank in (1, 10, 20):
        rows = [r for r in timing if r['task'] == 'scene' and r['resize_dim'] == 308 and r['kind'] == 'query' and r['bank'] == bank]
        dense = next(r for r in rows if r['fraction'] == 1.)
        half = next(r for r in rows if r['fraction'] == .5)
        speeds.append(half['median_seconds'] / dense['median_seconds'])
    output = dict(scene_query_ratios=speeds, scene_speed_pass=all(r <= .8 for r in speeds), tasks={})
    for task in ('object', 'scene'):
        checks = {}
        for clip, spec in CLIPS.items():
            if spec['task'] != task:
                continue
            r = results[clip]
            oracle = r['oracle']
            checks[clip] = dict(
                headroom=all(oracle['rpe_translation_m'] <= .95 * r[c]['rpe_translation_m']
                             and oracle['translation_p99_m'] <= r[c]['translation_p99_m'] for c in CONTROLS),
                oracle_to_controls={c: oracle['rpe_translation_m'] / r[c]['rpe_translation_m'] for c in CONTROLS},
                random_rms_range=[min(r[c]['rpe_translation_m'] for c in CONTROLS[1:]),
                                  max(r[c]['rpe_translation_m'] for c in CONTROLS[1:])],
                come_comparison={name: {key: r[name][key] for key in
                    ('rpe_translation_m', 'translation_p99_m', 'tracking_seconds',
                     'mean_kept_patches', 'mismatched_budget_frames')}
                    for name in ('heuristic', 'oracle', 'come')})
        passed = sum(c['headroom'] for c in checks.values()) >= 2
        output['tasks'][task] = dict(clips=checks, oracle_headroom_pass=passed,
            eligible_for_training_review=passed and (task == 'object' or output['scene_speed_pass']),
            next='manual archive review; never automatic training')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--profile-scene', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--hours', type=float, default=22.)
    parser.add_argument('--stage', choices=('gate', 'prefix'), default='gate')
    args = parser.parse_args()
    assert sys.platform == 'linux', 'CAMP only'
    if args.run:
        return run(args.run)
    if args.benchmark:
        return benchmark(args.benchmark)
    if args.profile_scene:
        return profile_scene(args.profile_scene)
    assert args.work.is_relative_to('/tmp')
    os.chdir(CHECKOUT)
    started = time.monotonic()
    deadline = started + args.hours * 3600
    from kvt_arctic_run import DATASET_DIR, OUT as ARCTIC_OUT, stage_dataset
    manifest = json.loads((ARCTIC_OUT / 'initial_frames' / 'manifest.json').read_text())
    stage_dataset(manifest)
    clips = {c: dict(s) for c, s in CLIPS.items()
             if args.stage == 'gate' or c in ('espresso', 'office')}
    inputs = dict(arctic_prepared_sha256=sha256(ARCTIC_OUT / 'prepared.tar'))
    # Preserve exact source pixels alongside per-run SAM masks and model-input hashes.
    shutil.copyfile(ARCTIC_OUT / 'prepared.tar', args.out / f'{args.tag}_atd_arctic_inputs.tar')
    for clip, spec in clips.items():
        if spec['task'] == 'object':
            mask = ARCTIC_OUT / 'masks' / spec['scene'] / 'init_mask.png'
            shutil.copyfile(mask, DATASET_DIR / spec['scene'] / 'init_mask.png')
            inputs[clip] = dict(init_mask_sha256=sha256(mask))
            shutil.copyfile(mask, args.work / 'context' / f'{clip}_init_mask.png')
            spec['frames'] = spec['source_frames'] = manifest['scenes'][spec['scene']]['tracking_frames']
        else:
            staged = args.work / 'inputs' / clip
            source = TUM_ZIPS / f"rgbd_dataset_{spec['scene']}.zip"
            prepare(source, staged, 308, MAX_GT_DIFFERENCE, count=spec['frames'])
            spec['scene_dir'] = str(staged)
            digest = sha256(source)
            inputs[clip] = dict(zip_sha256=digest)
            (staged / 'archive.sha256').write_text(f'{digest}  {source}\n')
            archive_inputs(staged, args.out / f'{args.tag}_atd_inputs_{clip}.tar')
    checkpoints = Path(os.environ['HF_HOME']) / 'hub' / 'models--yyfz233--Pi3' / 'snapshots'
    weights = list(checkpoints.glob('**/*.safetensors')) + list(checkpoints.glob('**/pytorch_model.bin'))
    assert weights, f'No cached Pi3 checkpoint found at {checkpoints}'
    inputs['pi3_weights'] = {str(w): sha256(w) for w in weights}
    inputs['pi3_refs'] = {str(ref): ref.read_text() for ref in
                          (checkpoints.parent / 'refs').rglob('*') if ref.is_file()}
    sam = CHECKOUT / 'thirdparty/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt'
    inputs['sam_checkpoint'] = dict(path=str(sam), sha256=sha256(sam))
    write_json(args.work / 'protocol.json', dict(stage=args.stage, clips=clips,
        arms=ARMS if args.stage == 'gate' else ('native', 'oracle', 'oracle_prefix'), inputs=inputs,
        seeds=SEEDS, scene_fraction=.5, object_background=64,
        admission='native schedule replay; arrival choices frozen and reused at rebuild',
        heuristic='scene: half of kept budget highest gradient energy, rest spread; object: bg64',
        oracle='six local body-pose gate derivatives; independent dense teacher cache; not deployable',
        scene_speed='at least 20% faster at all three cache sizes', deadline_hours=args.hours))
    results = {c: {} for c in clips}
    schedules = {}

    def condition(clip, name, length=None):
        assert time.monotonic() < deadline, 'ATD deadline; incomplete, not a negative result'
        spec = clips[clip]
        frames = spec['frames'] if length is None else length
        policy = name.removesuffix('_repeat').removesuffix('_prefix')
        schedule = None if policy == 'native' else [f for f in schedules[clip] if f < frames]
        target = args.work / 'runs' / clip / name
        target.mkdir(parents=True)
        write_json(target / 'config.json', dict(spec, frames=frames, clip=clip, name=name,
                                              policy=policy, schedule=schedule))
        print('ATD RUN', clip, name, frames, flush=True)
        with (target / 'run.log').open('w') as log:
            process = subprocess.run([sys.executable, __file__, '--run', str(target / 'config.json')],
                                     stdout=log, stderr=subprocess.STDOUT)
        write_json(target / 'process.json', dict(returncode=process.returncode))
        if process.returncode:
            print((target / 'run.log').read_text()[-10000:], flush=True)
            raise RuntimeError(f'{clip}/{name} failed; partial run preserved by wrapper')
        metric = json.loads((target / 'metrics.json').read_text())
        if policy == 'native':
            schedules[clip] = metric['keyframes']
        else:
            native = args.work / 'runs' / clip / 'native'
            if spec['task'] == 'object':
                for mask in (target / 'sam_masks').glob('*.png'):
                    assert mask.read_bytes() == (native / 'sam_masks' / mask.name).read_bytes()
            if policy == 'all':
                # Per-call fidelity is tested on GPU at token-drop's established
                # 1e-2 bound; full trajectories can amplify bf16 kernel differences.
                metric['native_trajectory_max_abs_difference'] = float(np.abs(
                    np.load(native / 'traj.npy')[:frames] - np.load(target / 'traj.npy')).max())
                metric['native_keyframe_pose_max_abs_difference'] = float(np.abs(
                    np.load(native / 'kf_poses.npy') - np.load(target / 'kf_poses.npy')).max())
                print('ATD DENSE COMPARISON', clip, name,
                      metric['native_trajectory_max_abs_difference'], flush=True)
            if name.endswith('_repeat'):
                np.testing.assert_array_equal(np.load(target / 'traj.npy'),
                    np.load(target.parent / policy / 'traj.npy'))
                np.testing.assert_array_equal(np.load(target / 'kf_poses.npy'),
                    np.load(target.parent / policy / 'kf_poses.npy'))
                print('ATD DENSE REPEAT OK', clip, flush=True)
            if name.endswith('_prefix'):
                full = target.parent / policy
                np.testing.assert_allclose(np.load(full / 'traj.npy')[:frames], np.load(target / 'traj.npy'),
                                           **TOLERANCE[spec['task']])
                original = json.loads((full / 'events.json').read_text())
                short = json.loads((target / 'events.json').read_text())
                assert short == [e for e in original if e['frame'] < frames]
                print('ATD CAUSAL PREFIX OK', clip, policy, flush=True)
        write_json(target / 'metrics.json', metric)
        archive = args.out / f'{args.tag}_atd_{clip}_{name}.tar'
        with tarfile.open(archive.with_suffix('.tar.partial'), 'w') as bundle:
            bundle.add(target, arcname=f'{clip}/{name}')
        archive.with_suffix('.tar.partial').replace(archive)
        with tarfile.open(archive) as bundle:
            assert bundle.extractfile(f'{clip}/{name}/traj.npy').read() == (target / 'traj.npy').read_bytes()
        results[clip][name] = metric
        write_json(args.work / 'comparison.json', results)
        print('ATD ARCHIVE OK', clip, name, flush=True)

    if args.stage == 'prefix':
        # Fresh processes and real inputs; cover the first native cache rebuild.
        # No scientific headroom decision or automatic full evaluation follows.
        for clip in clips:
            condition(clip, 'native')
            condition(clip, 'oracle')
            assert len(schedules[clip]) >= 2
            length = min(clips[clip]['frames'] - 1, schedules[clip][1] + 2)
            condition(clip, 'oracle_prefix', length)
        (args.work / 'JOB_OK').write_text('ATD oracle prefixes passed; headroom not evaluated.\n')
        print('ATD ORACLE PREFIX PILOT OK', flush=True)
        return

    # GPU contracts precede this driver; establish native schedules and dense references.
    for clip in clips:
        condition(clip, 'native')
        condition(clip, 'all')
        condition(clip, 'all_repeat')
    # Exact model-input grids, recorded by the all-kept path.
    shapes = []
    for clip in ('office', 'espresso'):
        event = json.loads((args.work / 'runs' / clip / 'all' / 'events.json').read_text())[0]
        height, width = [14 * d for d in event['grid']]
        task = clips[clip]['task']
        resize_dim = RESIZE[task]
        # resize_dim defines a pixel-area budget, not the longest edge.
        assert height > 0 and width > 0 and height * width <= resize_dim ** 2
        shapes.append(dict(shape=[height, width], task=task, resize_dim=resize_dim))
    bench_config = args.work / 'benchmark_config.json'
    write_json(bench_config, dict(shapes=shapes))
    subprocess.run([sys.executable, __file__, '--benchmark', str(bench_config)], check=True)
    for policy in ARMS[1:]:
        for clip in clips:
            condition(clip, policy)
    for clip in ('espresso', 'office'):
        # Prefix extends beyond first real admission, exercising rebuilt sparse caches.
        assert len(schedules[clip]) >= 2
        length = min(clips[clip]['frames'] - 1, schedules[clip][1] + 2)
        for policy in ('heuristic', 'oracle', 'random17'):
            condition(clip, policy + '_prefix', length)
    timing = json.loads((args.work / 'benchmark.json').read_text())
    decision = decide(results, timing)
    write_json(args.work / 'decision.json', decision)
    (args.work / 'JOB_OK').write_text('ATD gate complete; manual archive review required before training.\n')
    print('ATD GATE OK', json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
