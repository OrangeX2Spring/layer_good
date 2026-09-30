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
    observer = None
    if config.get('observe', False):
        from kv_tracker.scene_routing import SceneRoutingObserver
        assert task == 'scene' and config['policy'] == 'native'
        assert (source.height, source.width) == (480, 640)
        h, w, _ = manifest['inputs'][0]['shape']
        if config['scene'].startswith('freiburg3_'):
            fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
            distortion = None  # Freiburg3 RGB is already undistorted.
        else:
            assert config['scene'].startswith('freiburg2_')
            fx, fy, cx, cy = 520.9, 521., 325.1, 249.7
            distortion = torch.tensor([.2312, -.7849, -.0033, -.0001, .9172], device='cuda')
        k = torch.tensor([[fx*w/640, 0, (cx+.5)*w/640-.5],
                          [0, fy*h/480, (cy+.5)*h/480-.5], [0, 0, 1]], device='cuda')
        observer = SceneRoutingObserver(target, k, h, w, distortion)
        write_json(target / 'calibration.json', dict(K=k.cpu().tolist(),
            distortion=None if distortion is None else distortion.cpu().tolist(),
            source_size=[480, 640], model_size=[h, w],
            source='https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats',
            rgb='original RGB; cv2 linear resize; no crop or undistortion',
            pose='last estimated camera-to-world; canonical tracker gauge',
            sampling='source pixel centers 7::14; patch samples, not dense visibility'))
    with (target / 'tokens.jsonl').open('w') as log:
        executor = None
        if config.get('graph', False):
            from kv_tracker.graph_query import GraphQueries
            executor = GraphQueries(os.environ['KVT_GRAPH_ROPE_BUILD'], config['graph_checked'])
        policy = None if config['policy'] == 'native' else AdaptiveTokens(
            config['policy'], task, log, query_executor=executor)
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
            frame_source=source, token_policy=policy, scene_observer=observer,
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
        timing='synchronized wall time from first inference; includes teacher/selection and observer serialization when present',
        routing_observer=observer is not None,
        teacher_seconds=0 if policy is None else policy.teacher_seconds,
        selection_seconds=0 if policy is None else policy.selection_seconds,
        student_forward_seconds=None if policy is None else policy.student_seconds,
        teacher_state_in_peak=config['policy'] in ('oracle', 'oracle_spread', 'come'))
    if policy is not None:
        metrics.update(mean_kept_patches=float(np.mean([e['budget'] for e in policy.events])),
            mismatched_budget_frames=sum(e['budget'] != e['target_budget'] for e in policy.events))
    metrics.update(evaluate_object(config['scene'], target, frames) if task == 'object'
                   else evaluate_scene(Path(config['scene_dir']), target, frames))
    if executor is not None:
        assert executor.queries == frames - 1
        write_json(target / 'graph.json', dict(checked=executor.checked,
            captures=executor.rows, queries=executor.queries,
            max_checked_pose_error=executor.max_pose_error,
            backend_load_seconds=executor.backend_load_seconds,
            timing='tracking includes graph setup/validation; backend load precedes tracking'))
        print('ATD TRACKING GRAPH RUN OK', config['name'], flush=True)
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


def decide(results, timing, clips=CLIPS):
    speeds = []
    if timing is not None:
        for bank in (1, 10, 20):
            rows = [r for r in timing if r['task'] == 'scene' and r['resize_dim'] == 308 and r['kind'] == 'query' and r['bank'] == bank]
            dense = next(r for r in rows if r['fraction'] == 1.)
            half = next(r for r in rows if r['fraction'] == .5)
            speeds.append(half['median_seconds'] / dense['median_seconds'])
    output = dict(scene_query_ratios=speeds,
                  scene_speed_pass=None if timing is None else all(r <= .8 for r in speeds), tasks={})
    for task in ('object', 'scene'):
        checks = {}
        for clip, spec in clips.items():
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
        if not checks:
            continue
        passed = sum(c['headroom'] for c in checks.values()) >= 2
        output['tasks'][task] = dict(clips=checks, oracle_headroom_pass=passed,
            eligible_for_training_review=passed and (task == 'object' or output['scene_speed_pass'] is True),
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
    parser.add_argument('--stage', choices=('gate', 'prefix', 'scenes', 'graph-track', 'spread-headroom'), default='gate')
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
    clips = {c: dict(s) for c, s in CLIPS.items()
             if args.stage == 'gate' or
             (args.stage == 'prefix' and c in ('espresso', 'office')) or
             (args.stage in ('scenes', 'graph-track', 'spread-headroom') and s['task'] == 'scene')}
    reviewed = None
    if args.stage == 'spread-headroom':
        # This stage reuses exactly the controls already audited locally, not an arbitrary run.
        source = args.out / 'tum_26044_context.tar'
        context_digest = sha256(source)
        assert context_digest == '92b47204bd5cd6fb4d7b2653b7994561a5565f5fa77a28b63fcf97a667648d8c'
        with tarfile.open(source) as bundle:
            assert bundle.extractfile('./context/exit_status.txt').read().strip() == b'0'
            reviewed = json.load(bundle.extractfile('./protocol.json'))
            reviewed_results = json.load(bundle.extractfile('./comparison.json'))
            previous_container = bundle.extractfile('./context/container_tar.sha256').read().split()[0]
        assert previous_container == (args.work / 'context/container_tar.sha256').read_bytes().split()[0]
        write_json(args.work / 'context/reviewed_comparison.json', reviewed_results)
        reviewed_runs = args.work / 'inputs/reviewed_runs'
        reviewed_runs.mkdir(parents=True)
        source = args.out / 'tum_26044_all_runs.tar'
        run_digest = sha256(source)
        assert run_digest == 'f81975aa6adeab8f43502f63b788cbd8d4733fa4152b8305c38df8b1d86b00bf'
        with tarfile.open(source) as bundle:
            for clip in clips:
                for arm in ('native', 'all'):
                    for name in ('traj.npy', 'kf_idx.npy', 'kf_poses.npy'):
                        member = bundle.getmember(f'runs/{clip}/{arm}/{name}')
                        assert member.isfile()
                        bundle.extract(member, reviewed_runs)
        write_json(args.work / 'context/reviewed_sources.json', dict(job=26044,
            context_sha256=context_digest, all_runs_sha256=run_digest,
            control_use='accuracy only; timing not paired'))
    if args.stage == 'graph-track':
        for spec in clips.values():
            spec['frames'] = 256
        # Read-only inventory; no training pixels or labels are processed here.
        write_json(args.work / 'training_dataset_inventory.json', dict(
            root=str(TUM_ZIPS), zips=[dict(name=p.name, bytes=p.stat().st_size)
                for p in sorted(TUM_ZIPS.glob('*.zip'))],
            excluded_evaluation_scenes=[s['scene'] for s in clips.values()],
            decision='inventory only; training split/labels require reviewed gates'))
    inputs = {}
    if any(s['task'] == 'object' for s in clips.values()):
        from kvt_arctic_run import DATASET_DIR, OUT as ARCTIC_OUT, stage_dataset
        manifest = json.loads((ARCTIC_OUT / 'initial_frames' / 'manifest.json').read_text())
        stage_dataset(manifest)
        inputs['arctic_prepared_sha256'] = sha256(ARCTIC_OUT / 'prepared.tar')
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
            if reviewed is None:
                archive_inputs(staged, args.out / f'{args.tag}_atd_inputs_{clip}.tar')
            else:
                assert digest == reviewed['inputs'][clip]['zip_sha256']
                pins = dict(office='88c00b3229ce6e454e5f7ec69129c126a283e6e9db5093bb5c6f86a43694ff8d',
                    with_loop='fd731f32e4316bd3b810f113d1997357f8f08a37ebaaf0d064fca2fcaa480432',
                    no_loop='e47b84da8cf1883a2dce9f0819a89cb2ba1b83030f0f23a739288a0b0e40367f')
                original = args.out / f'tum_26044_atd_inputs_{clip}.tar'
                assert sha256(original) == pins[clip]
                with tarfile.open(original) as bundle:
                    assert json.load(bundle.extractfile(f'{clip}/manifest.json')) == json.loads((staged / 'manifest.json').read_text())
                inputs[clip]['reviewed_model_pixels'] = dict(archive=str(original), sha256=pins[clip])
    checkpoints = Path(os.environ['HF_HOME']) / 'hub' / 'models--yyfz233--Pi3' / 'snapshots'
    weights = list(checkpoints.glob('**/*.safetensors')) + list(checkpoints.glob('**/pytorch_model.bin'))
    assert weights, f'No cached Pi3 checkpoint found at {checkpoints}'
    inputs['pi3_weights'] = {str(w): sha256(w) for w in weights}
    inputs['pi3_refs'] = {str(ref): ref.read_text() for ref in
                          (checkpoints.parent / 'refs').rglob('*') if ref.is_file()}
    sam = CHECKOUT / 'thirdparty/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt'
    inputs['sam_checkpoint'] = dict(path=str(sam), sha256=sha256(sam))
    if reviewed is not None:
        for key in ('pi3_weights', 'pi3_refs', 'sam_checkpoint'):
            assert inputs[key] == reviewed['inputs'][key], key
    write_json(args.work / 'protocol.json', dict(stage=args.stage, clips=clips,
        arms=('native', 'all', 'oracle_spread') if args.stage == 'spread-headroom' else
             ('native', 'all', 'uniform', 'all_graph_checked', 'all_graph',
              'uniform_graph_checked', 'uniform_graph') if args.stage == 'graph-track' else
             ('native', 'oracle', 'oracle_prefix') if args.stage == 'prefix' else
             ('native', 'all_repeat') + ARMS, inputs=inputs,
        routing_arms=('passive', 'passive_prefix') if args.stage == 'scenes' else (),
        prefixes=('oracle_spread',) if args.stage == 'spread-headroom' else
                 ('all_graph', 'uniform_graph') if args.stage == 'graph-track' else
                 ('heuristic', 'oracle', 'random17') if args.stage != 'prefix' else ('oracle',),
        seeds=SEEDS, scene_fraction=.5, object_background=64,
        admission='native schedule replay; arrival choices frozen and reused at rebuild',
        heuristic='scene: half of kept budget highest gradient energy, rest spread; object: bg64',
        oracle='six local body-pose gate derivatives; independent dense teacher cache; not deployable',
        oracle_spread='half of kept budget top scores, rest spread; same learned allocation',
        reused_controls=None if reviewed is None else dict(job=26044, policies=CONTROLS,
            accuracy_only=True, fresh_native_all_arrays_must_match_exactly=True),
        scene_speed='256-frame execution pilot; no training eligibility' if args.stage == 'graph-track' else
                    'not tested; no training eligibility' if args.stage in ('scenes', 'spread-headroom') else
                    'at least 20% faster at all three cache sizes', deadline_hours=args.hours))
    results = {c: {} for c in clips} if reviewed is None else reviewed_results
    schedules = {}

    def condition(clip, name, length=None):
        assert time.monotonic() < deadline, 'ATD deadline; incomplete, not a negative result'
        spec = clips[clip]
        frames = spec['frames'] if length is None else length
        policy = name.removesuffix('_repeat').removesuffix('_prefix')
        graph = policy.endswith(('_graph', '_graph_checked'))
        checked = policy.endswith('_graph_checked')
        policy = policy.removesuffix('_graph_checked').removesuffix('_graph')
        observe = policy == 'passive'
        if observe:
            policy = 'native'
        schedule = None if policy == 'native' else [f for f in schedules[clip] if f < frames]
        target = args.work / 'runs' / clip / name
        target.mkdir(parents=True)
        write_json(target / 'config.json', dict(spec, frames=frames, clip=clip, name=name,
                                              policy=policy, schedule=schedule, observe=observe,
                                              graph=graph, graph_checked=checked))
        print('ATD RUN', clip, name, frames, flush=True)
        with (target / 'run.log').open('w') as log:
            process = subprocess.run([sys.executable, __file__, '--run', str(target / 'config.json')],
                                     stdout=log, stderr=subprocess.STDOUT)
        write_json(target / 'process.json', dict(returncode=process.returncode))
        if process.returncode:
            print((target / 'run.log').read_text()[-10000:], flush=True)
            raise RuntimeError(f'{clip}/{name} failed; partial run preserved by wrapper')
        metric = json.loads((target / 'metrics.json').read_text())
        if graph:
            prefix = name.endswith('_prefix')
            reference = target.parent / (name.removesuffix('_prefix') if prefix else policy)
            np.testing.assert_allclose(np.load(target / 'traj.npy'),
                np.load(reference / 'traj.npy')[:frames], rtol=1e-4, atol=1e-4)
            expected_ids = np.load(reference / 'kf_idx.npy')
            np.testing.assert_array_equal(np.load(target / 'kf_idx.npy'),
                                          expected_ids[expected_ids < frames])
            if not prefix:
                np.testing.assert_allclose(np.load(target / 'kf_poses.npy'),
                    np.load(reference / 'kf_poses.npy'), rtol=1e-4, atol=1e-4)
            full_events = json.loads((reference / 'events.json').read_text())
            short_events = json.loads((target / 'events.json').read_text())
            assert short_events == [e for e in full_events if e['frame'] < frames]
            captures = json.loads((target / 'graph.json').read_text())['captures']
            assert len(captures) == sum(f < frames - 1 for f in metric['keyframes'])
            if not prefix:
                assert len(captures) >= 2, 'Pilot must exercise a real recapture'
            print('ATD TRACKING GRAPH PREFIX OK' if prefix else
                  'ATD TRACKING GRAPH FIDELITY OK', clip, name, flush=True)
        elif observe:
            reference = target.parent / ('passive' if name.endswith('_prefix') else 'native')
            prefix = name.endswith('_prefix')
            # Later rebuilds refit old keyframe poses. Compare contemporaneous
            # cache snapshots below, not the full run's final keyframe poses.
            for file in (('traj.npy', 'kf_idx.npy') if prefix else
                         ('traj.npy', 'kf_idx.npy', 'kf_poses.npy')):
                actual, expected = np.load(target / file), np.load(reference / file)
                if prefix and file == 'traj.npy':
                    np.testing.assert_allclose(actual, expected[:frames], rtol=1e-4, atol=1e-4)
                elif prefix:
                    ids = np.load(reference / 'kf_idx.npy') < frames
                    np.testing.assert_array_equal(actual, expected[ids])
                else:
                    np.testing.assert_array_equal(actual, expected)
            route_rows = json.loads((target / 'routing.json').read_text())
            assert [r['frame'] for r in route_rows] == list(range(1, frames))
            assert all(max(r['frame_ids']) < r['frame'] for r in route_rows)
            assert len(list(target.glob('route_*.npz'))) == frames - 1
            assert len(list(target.glob('cache_*.npz'))) == len(metric['keyframes'])
            if prefix:
                original = json.loads((reference / 'routing.json').read_text())
                for short, full in zip(route_rows, original):
                    for key in ('frame', 'cache', 'frame_ids', 'supported_fraction',
                                'valid_fraction', 'metadata_bytes'):
                        assert short[key] == full[key], key
                for path in (*target.glob('route_*.npz'), *target.glob('cache_*.npz')):
                    with np.load(path) as short, np.load(reference / path.name) as full:
                        assert short.files == full.files
                        for key in short.files:
                            np.testing.assert_array_equal(short[key], full[key])
                print('SCENE ROUTING CAUSAL PREFIX OK', clip, flush=True)
            else:
                print('SCENE ROUTING CONTROL OK', clip, flush=True)
        elif policy == 'native':
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
        if reviewed is not None and name in ('native', 'all'):
            for file in ('traj.npy', 'kf_idx.npy', 'kf_poses.npy'):
                np.testing.assert_array_equal(np.load(target / file),
                    np.load(reviewed_runs / 'runs' / clip / name / file))
            print('ATD REVIEWED REFERENCE EXACT OK', clip, name, flush=True)
        archive = args.out / f'{args.tag}_atd_{clip}_{name}.tar'
        with tarfile.open(archive.with_suffix('.tar.partial'), 'w') as bundle:
            bundle.add(target, arcname=f'{clip}/{name}')
        archive.with_suffix('.tar.partial').replace(archive)
        with tarfile.open(archive) as bundle:
            assert bundle.extractfile(f'{clip}/{name}/traj.npy').read() == (target / 'traj.npy').read_bytes()
        results[clip][name] = metric
        write_json(args.work / 'comparison.json', results)
        print('ATD ARCHIVE OK', clip, name, flush=True)

    if args.stage == 'spread-headroom':
        # Confirm reference fidelity on every scene before spending time on teacher gradients.
        for clip in clips:
            condition(clip, 'native')
            condition(clip, 'all')
        for clip in clips:
            condition(clip, 'oracle_spread')
            condition(clip, 'oracle_spread_prefix', schedules[clip][1] + 2)
        decision = decide({c: dict(r, oracle=r['oracle_spread']) for c, r in results.items()}, None, clips)
        decision['teacher_policy'] = 'oracle_spread'
        decision['reference_job'] = 26044
        decision['original_oracle_replaced'] = False
        write_json(args.work / 'decision.json', decision)
        (args.work / 'JOB_OK').write_text('Coverage-matched teacher retest complete; manual review required.\n')
        print('ATD SPREAD HEADROOM OK', json.dumps(decision), flush=True)
        return

    if args.stage == 'graph-track':
        report = {}
        for clip in clips:
            for name in ('native', 'all', 'uniform', 'all_graph_checked', 'all_graph',
                         'uniform_graph_checked', 'uniform_graph'):
                condition(clip, name)
            length = min(clips[clip]['frames'] - 1, schedules[clip][1] + 2)
            for name in ('all_graph', 'uniform_graph'):
                condition(clip, name + '_prefix', length)
            report[clip] = dict(frames=clips[clip]['frames'], prefix_frames=length, policies={})
            for policy in ('all', 'uniform'):
                eager, replay = results[clip][policy], results[clip][policy + '_graph']
                target = args.work / 'runs' / clip / (policy + '_graph')
                graph_record = json.loads((target / 'graph.json').read_text())
                queries = [r['seconds'] for r in json.loads((target / 'inference.json').read_text())
                           if r['kind'] == 'query']
                native_queries = [r['seconds'] for r in json.loads((target.parent / 'native' /
                                    'inference.json').read_text()) if r['kind'] == 'query']
                report[clip]['policies'][policy] = dict(
                    tracking_seconds=dict(eager=eager['tracking_seconds'], graph=replay['tracking_seconds']),
                    graph_to_eager_tracking_ratio=replay['tracking_seconds'] / eager['tracking_seconds'],
                    graph_to_native_tracking_ratio=replay['tracking_seconds'] /
                                                   results[clip]['native']['tracking_seconds'],
                    graph_to_native_query_sum_ratio=sum(queries) / sum(native_queries),
                    query_seconds=dict(median=float(np.median(queries)), p95=float(np.percentile(queries, 95)),
                                       maximum=max(queries), total=sum(queries)),
                    peak_allocated_bytes=dict(eager=eager['peak_allocated_bytes'], graph=replay['peak_allocated_bytes']),
                    peak_reserved_bytes=dict(eager=eager['peak_reserved_bytes'], graph=replay['peak_reserved_bytes']),
                    setup_seconds=sum(c['setup_seconds'] for c in graph_record['captures']),
                    backend_load_seconds=graph_record['backend_load_seconds'],
                    captures=len(graph_record['captures']))
            write_json(args.work / 'graph_tracking_summary.json', dict(clips=report,
                decision='bounded engineering pilot; full-sequence speed/memory and training gates pending',
                timing='timed runs include selection, recapture and first-cache checks; checked runs excluded'))
        (args.work / 'JOB_OK').write_text('Real-scene graph pilot complete; manual review required.\n')
        print('ATD TRACKING GRAPH PILOT OK', flush=True)
        return

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
    if args.stage == 'scenes':
        (args.work / 'routing_review').mkdir()
        routing = {}
        for clip in clips:
            condition(clip, 'passive')
            assert len(schedules[clip]) >= 2
            length = min(clips[clip]['frames'] - 1, schedules[clip][1] + 2)
            condition(clip, 'passive_prefix', length)
            target = args.work / 'runs' / clip / 'passive'
            route_rows = json.loads((target / 'routing.json').read_text())
            seconds = [r['routing_seconds'] for r in route_rows]
            routing[clip] = dict(frames=clips[clip]['frames'], prefix_frames=length,
                mapping_seconds=dict(median=float(np.median(seconds)),
                    p95=float(np.percentile(seconds, 95)), p99=float(np.percentile(seconds, 99)),
                    maximum=float(max(seconds))),
                tracking_seconds=dict(native=results[clip]['native']['tracking_seconds'],
                                      passive=results[clip]['passive']['tracking_seconds']),
                supported_fraction_range=[min(r['supported_fraction'] for r in route_rows),
                                          max(r['supported_fraction'] for r in route_rows)],
                metadata_bytes_range=[min(r['metadata_bytes'] for r in route_rows),
                                      max(r['metadata_bytes'] for r in route_rows)],
                latency='synchronized mapping only; excludes NPZ/JSON serialization',
                control='exact native/passive arrays; natural admission; no selector',
                decision='manual map review pending; routing benefit not tested')
            write_json(args.work / 'routing_summary.json', routing)
            subprocess.run([sys.executable, str(Path(__file__).with_name('kvt_tum_viz.py')),
                'routing', '--inputs', clips[clip]['scene_dir'], '--result', str(target),
                '--out', str(args.work / 'routing_review' / clip)], check=True)
    else:
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
    for clip in (clips if args.stage == 'scenes' else ('espresso', 'office')):
        # Prefix extends beyond first real admission, exercising rebuilt sparse caches.
        assert len(schedules[clip]) >= 2
        length = min(clips[clip]['frames'] - 1, schedules[clip][1] + 2)
        for policy in ('heuristic', 'oracle', 'random17'):
            condition(clip, policy + '_prefix', length)
    timing = None if args.stage == 'scenes' else json.loads((args.work / 'benchmark.json').read_text())
    decision = decide(results, timing, clips)
    write_json(args.work / 'decision.json', decision)
    (args.work / 'JOB_OK').write_text('ATD diagnostics complete; manual archive review required before training.\n')
    print('ATD SCENE BUNDLE OK' if args.stage == 'scenes' else 'ATD GATE OK',
          json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
