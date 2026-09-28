"""Semantic/geometric quarter-patch KV screen on KV-Tracker; CAMP only.

Protocol: tools/SEMANTIC_KV_OVERNIGHT_PLAN.md. Policies: the fork's
kv_tracker/patch_select.py. Entry: `bash tools/kvt_tum.sbatch patch-select <stage>`.

- preflight: 128-frame espresso and office clips. Native, all-kept, the three
  sparse controls and the ten candidates, plus a causal prefix run of every
  sparse condition. Checks the contracts and projects the screen's cost.
  Its accuracy values do not select candidates.
- screen: the frozen 102-run screen on six clips, then the frozen decision
  report. Requires the reviewed preflight context archive.

Every condition is a fresh process (`--run config.json`) archived on completion.
Arms replay native's admission schedule, so they differ only in which patch rows
the persistent cache keeps after each native dense rebuild.
"""
import argparse
import hashlib
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

from kvt_tum_run import CHECKOUT, evaluate as evaluate_tum, prepare, write_json
from kvt_tum_sweep import archive_inputs

# Dict order is the round-robin order: tasks alternate.
CLIPS = {
    'espresso': dict(task='object', scene='espressomachine_grab_01'),
    'office': dict(task='scene', scene='freiburg3_long_office_household', frames=1150),
    'ketchup': dict(task='object', scene='ketchup_grab_01'),
    'with_loop': dict(task='scene', scene='freiburg2_large_with_loop', frames=1024),
    'box': dict(task='object', scene='box_grab_01'),
    'no_loop': dict(task='scene', scene='freiburg2_large_no_loop', frames=1024),
}
PREFLIGHT_CLIPS = ('espresso', 'office')
PREFLIGHT_FRAMES = 128
# K2/K7/K4/K6 first, as the design prioritises; the rest by ID.
PRIORITY = ('K2', 'K7', 'K4', 'K6', 'K1', 'K3', 'K5', 'K8', 'K9', 'K10')
SPARSE_CONTROLS = ('uniform', 'random17', 'random29')
CONTROLS = ('native', 'all') + SPARSE_CONTROLS
REPEATS = ('native_repeat', 'all_repeat')
RESIZE = dict(object=518, scene=308)
# Established fidelity tolerances: kvt_active_objects.py (ARCTIC) and
# kvt_active_keyframes.py (TUM).
TOLERANCE = dict(object=dict(rtol=1e-4, atol=1e-4), scene=dict(rtol=1e-4, atol=1e-5))
TUM_ZIPS = Path('/mnt/datasets/tum-rgbd')
MAX_GT_DIFFERENCE = .02
SPECIAL = 5
KEYS = ('rpe_translation_m', 'translation_p99_m', 'ate_m', 'rpe_rotation_deg')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def screen_order():
    """All controls first, candidates round-robin across tasks, repeats last."""
    runs = [(clip, name) for clip in CLIPS for name in CONTROLS]
    runs += [(clip, name) for name in PRIORITY for clip in CLIPS]
    runs += [(clip, name) for clip in CLIPS for name in REPEATS]
    assert len(runs) == len(set(runs)) == 102
    return runs


class Instrument:
    """Synchronized timing of every Pi3 call and cache update.

Native runs carry it without a selection policy, so every arm pays the same
instrumentation. Tracking time starts after the model is loaded.
"""

    def __init__(self, selection, log):
        self.selection = selection
        self.log = log
        self.calls = 0
        self.frame = 0

    def attach(self, model):
        self.model = model
        model.register_forward_pre_hook(self.before, with_kwargs=True)
        model.register_forward_hook(self.after, with_kwargs=True)
        if self.selection is not None:
            self.selection.attach(model)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()

    def before(self, module, args, kwargs):
        self.kind = 'query' if kwargs.get('use_cache', False) else (
            'rebuild' if self.calls else 'bootstrap')
        torch.cuda.synchronize()
        self.call_started = time.perf_counter()

    def after(self, module, args, kwargs, output):
        torch.cuda.synchronize()
        self.write(self.kind, time.perf_counter() - self.call_started)
        self.calls += 1

    def begin_query(self, frame_id):
        self.frame = frame_id
        if self.selection is not None:
            self.selection.begin_query(frame_id)

    def end_query(self):
        if self.selection is not None:
            self.selection.end_query()

    def after_rebuild(self, *args):
        torch.cuda.synchronize()
        started = time.perf_counter()
        if self.selection is not None:
            self.selection.after_rebuild(*args)
        torch.cuda.synchronize()
        self.write('select', time.perf_counter() - started)

    def write(self, kind, seconds):
        cache = sum(t.numel() * t.element_size() for layer in self.model.cache.values()
                    for t in layer.values())
        self.log.write(json.dumps(dict(kind=kind, frame=self.frame, seconds=seconds,
            cache_bytes=cache, allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved())) + '\n')


def pair_summary(translation, rotation, times):
    """RMS, p99 and per-half RMS. Halves split the span of the GT-valid pair
(end) times at its middle; an empty half is reported as unevaluable (None)."""
    middle = (times.min() + times.max()) / 2
    halves = [translation[times < middle], translation[times >= middle]]
    return dict(rpe_translation_m=float(np.sqrt(np.mean(translation ** 2))),
                rpe_rotation_deg=float(np.sqrt(np.mean(rotation ** 2))),
                translation_p99_m=float(np.quantile(translation, .99)),
                half_rms_m=[float(np.sqrt(np.mean(h ** 2))) if len(h) else None for h in halves],
                half_pairs=[len(h) for h in halves], half_split_time=float(middle))


def evaluate_object(scene, result, frames):
    import eval as kvt_eval
    from kv_tracker.eval_tools.evo_utils import align_pair
    from kv_tracker.geometry import umeyama_alignment

    gt = kvt_eval.load_gt_arctic(scene)[:frames]
    estimate = np.load(result / 'traj.npy')
    assert gt.shape == estimate.shape == (frames, 4, 4)
    ate, evo_rpe_t, evo_rpe_rot = align_pair(dict(traj_gt=gt, traj_est=estimate, name=scene),
                                             'traj', ret_np=False)
    aligned, gt = align_pair(dict(traj_gt=gt, traj_est=estimate), 'traj')
    error = np.linalg.inv(np.linalg.inv(gt[:-1]) @ gt[1:]) @ (np.linalg.inv(aligned[:-1]) @ aligned[1:])
    translation = np.linalg.norm(error[:, :3, 3], axis=1)
    rotation = np.rad2deg(np.arccos(np.clip((np.trace(error[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    _, _, scale = umeyama_alignment(estimate[:, :3, 3].T, gt[:, :3, 3].T, with_scale=True)
    ends = np.arange(1, frames)
    np.savez(result / 'evaluation.npz', pair_end_frames=ends, translation_m=translation,
             rotation_deg=rotation, aligned=aligned)
    return dict(pair_summary(translation, rotation, ends), ate_m=float(ate),
                evo_rpe_t_m=float(evo_rpe_t), evo_rpe_rot=float(evo_rpe_rot),
                alignment_scale=float(scale), evaluated_frames=frames, rpe_pairs=frames - 1,
                half_time_unit='frame index; ARCTIC frames are consecutive')


def evaluate_scene(scene_dir, result, frames):
    metrics = evaluate_tum(scene_dir, result, MAX_GT_DIFFERENCE)
    manifest = json.loads((scene_dir / 'manifest.json').read_text())
    times = np.array([r['timestamp'] for r in manifest['inputs'][:frames]])
    with np.load(result / 'evaluation.npz') as e:
        ends = e['rpe_pair_start_indices'] + 1
        summary = pair_summary(e['rpe_translation_per_pair_m'], e['rpe_rotation_per_pair_deg'],
                               times[ends] - times[0])
        valid = e['timestamps']
        metrics.update(alignment_scale=float(e['alignment_scale']))
    # Same RMS definitions as evaluate(); recomputed here from the same arrays.
    np.testing.assert_allclose(summary['rpe_translation_m'], metrics['rpe_translation_m'], rtol=1e-9)
    return dict(metrics, **summary, clip_duration_s=float(times[-1] - times[0]),
                gt_valid_time_span_s=[float(valid[0] - times[0]), float(valid[-1] - times[0])],
                half_time_unit='seconds from clip start')


def run(config_path):
    import main as tracker
    from kv_tracker.patch_select import PatchSelectCache

    config = json.loads(config_path.read_text())
    result = config_path.parent
    os.chdir(CHECKOUT)  # SAM checkpoint and ARCTIC GT paths are relative to it.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    task, frames = config['task'], config['frames']
    if task == 'object':
        from kvt_arctic_run import ArcticFrames, DATASET_DIR
        mask_dir = result / 'sam_masks'
        mask_dir.mkdir()
        source = ArcticFrames('cuda:0', mask_dir, scene_dir=DATASET_DIR / config['scene'], offset=2,
                              obj_mode=True, resize_dim=RESIZE[task])
        assert source.length == config['source_frames'], (source.length, config['source_frames'])
        source.length = frames
        tracker_args = ['--obj_mode', '--resize_dim', str(RESIZE[task])]
    else:
        from kv_tracker.dataloaders.tum import TUMLoader
        scene_dir = Path(config['scene_dir'])
        manifest = json.loads((scene_dir / 'manifest.json').read_text())

        class Frames(TUMLoader):
            def __iter__(self):
                for index in range(frames):
                    frame = self.get_frame(index)
                    frame['idx'] = index
                    assert frame['resized_mask_np'].all()
                    assert hashlib.sha256(frame['resized_rgb_masked_np'].tobytes()).hexdigest() == (
                        manifest['inputs'][index]['model_rgb_sha256'])
                    yield frame

        source = Frames('cuda:0', scene_dir=scene_dir, obj_mode=False, resize_dim=RESIZE[task], offset=0)
        tracker_args = ['--cam_only', '--resize_dim', str(RESIZE[task]), '--kf_auto', '50']
    schedule = config['schedule']
    selection = None if config['policy'] == 'native' else PatchSelectCache(
        config['policy'], task, schedule)
    with (result / 'inference.jsonl').open('w') as log:
        instrument = Instrument(selection, log)
        tracker.run_track3r(cfg=dict(results_path=str(result), que_size=1), args=tracker_args,
                            frame_source=source, patch_cache=instrument,
                            keyframe_indices=None if selection is None else schedule[1:])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - instrument.started
    if selection is not None:
        selection.close()
        write_json(result / 'patch_events.json', selection.events)
    trajectory = np.load(result / 'traj.npy')
    assert trajectory.shape == (frames, 4, 4) and np.isfinite(trajectory).all()
    if not (result / 'kf_idx.npy').exists():
        np.save(result / 'kf_idx.npy', np.array([0]))  # Upstream writes it at the first insertion.
    # Upstream point-cloud dump, ~120 MB per object run and unused here.
    (result / 'pcd.npy').unlink(missing_ok=True)
    rows = [json.loads(line) for line in (result / 'inference.jsonl').read_text().splitlines()]
    metrics = dict(task=task, clip=config['clip'], name=config['name'], policy=config['policy'],
                   frames=frames, keyframes=np.load(result / 'kf_idx.npy').tolist(),
                   tracking_seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
                   timing='synchronous instrumented wall time after model load, including selection')
    for kind in ('bootstrap', 'query', 'rebuild', 'select'):
        values = [row['seconds'] for row in rows if row['kind'] == kind]
        metrics[f'{kind}_calls'] = len(values)
        metrics[f'{kind}_seconds'] = float(sum(values))
    queries = [row['cache_bytes'] for row in rows if row['kind'] == 'query']
    metrics['max_query_cache_bytes'] = max(queries)
    if selection is None:
        metrics.update(persistent_bytes=max(queries), state_bytes=0)
    else:
        events = selection.events
        chosen = events[1:]
        metrics.update(state_bytes=events[-1]['state_bytes'],
            final_query_cache_bytes=events[-1]['query_cache_bytes'],
            persistent_bytes=events[-1]['query_cache_bytes'] + events[-1]['state_bytes'],
            rebuild_input_bytes=events[-1]['rebuild_input_bytes'],
            read_demand_seconds=float(sum(e['read_demand_seconds'] for e in events)),
            selector_host_seconds=float(sum(e['selector_host_seconds'] for e in events)),
            selected_patches=sum(len(e['selected']) for e in chosen),
            weak_corner_selected=sum(e.get('weak_corner_selected', 0) for e in chosen),
            target_selected=sum(e.get('target_selected', 0) for e in chosen),
            k2_fill=sum(e.get('k2_fill', 0) for e in chosen))
    metrics.update(evaluate_object(config['scene'], result, frames) if task == 'object'
                   else evaluate_scene(Path(config['scene_dir']), result, frames))
    write_json(result / 'metrics.json', metrics)
    print('RUN METRICS', json.dumps(metrics), flush=True)


def audit(target, config, schedule, native):
    """Contracts every arm must satisfy against its clip's native run."""
    metric = json.loads((target / 'metrics.json').read_text())
    policy = config['policy']
    assert np.load(target / 'kf_idx.npy').tolist() == schedule, (config['name'], 'keyframes')
    rows = [json.loads(line) for line in (target / 'inference.jsonl').read_text().splitlines()]
    assert metric['rebuild_calls'] == len(schedule) - 1
    assert [r['frame'] for r in rows if r['kind'] == 'query'] == list(range(1, config['frames']))
    if config['task'] == 'object':
        for mask in (native / 'sam_masks').glob('*.png'):
            if int(mask.stem) < config['frames']:
                assert mask.read_bytes() == (target / 'sam_masks' / mask.name).read_bytes(), mask.name
    if policy == 'native':
        return metric
    events = json.loads((target / 'patch_events.json').read_text())
    assert [e['frame'] for e in events] == schedule
    total = events[0]['patches']
    budget = math.ceil(total / 4)
    kept = total if policy == 'all' else budget
    assert events[0]['selected'] == list(range(total))
    for i, event in enumerate(events):
        assert event['retained_frame_ids'] == schedule[:i + 1] and event['budget'] == budget
        if i:
            picked = event['selected']
            assert len(picked) == kept and picked == sorted(set(picked))
            assert 0 <= picked[0] and picked[-1] < total
        slots = 2 if i == 0 else i + 1
        expected = (slots if i == 0 else 1) * (SPECIAL + total) + (i * (SPECIAL + kept) if i else 0)
        assert event['retained_tokens'] == expected, (i, event['retained_tokens'], expected)
        dense_tokens = slots * (SPECIAL + total)
        # Every layer holds the same dtype, so bytes scale exactly with kept tokens.
        assert event['query_cache_bytes'] * dense_tokens == event['dense_cache_bytes'] * expected
        if policy != 'all' and i:
            assert event['query_cache_bytes'] < event['dense_cache_bytes']
    # Queries read the gathered cache: no dense shadow survives a rebuild.
    current = None
    by_frame = {e['frame']: e['query_cache_bytes'] for e in events}
    for row in rows:
        if row['kind'] == 'select':
            current = by_frame[row['frame']]
        elif row['kind'] == 'query':
            assert row['cache_bytes'] == current, row
    return metric


def overlap(run_dir, clip_results):
    """Selected-set Jaccard between sparse arms per admission: identical choices
are not independent evidence."""
    names = [n for n in SPARSE_CONTROLS + PRIORITY if n in clip_results]
    chosen = {n: {e['frame']: set(e['selected'])
                  for e in json.loads((run_dir / n / 'patch_events.json').read_text())[1:]}
              for n in names}
    table = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(set(chosen[a]) & set(chosen[b]))
            values = [len(chosen[a][f] & chosen[b][f]) / len(chosen[a][f] | chosen[b][f]) for f in shared]
            table[f'{a}|{b}'] = dict(mean_jaccard=float(np.mean(values)) if values else None,
                                     identical=bool(values) and all(v == 1 for v in values),
                                     admissions=len(values))
    return table


def decide(results):
    """The frozen next-morning decision, per task (design: 'Fixed decisions')."""
    decision = {}
    for task in ('object', 'scene'):
        clips = [c for c, spec in CLIPS.items() if spec['task'] == task]
        state = {}
        for clip in clips:
            r = results.get(clip, {})
            missing = [n for n in CONTROLS + REPEATS if n not in r]
            deviation = {} if missing else {
                f'{b}/{key}': abs(r[b][key] / r[a][key] - 1)
                for a, b in (('native', 'native_repeat'), ('all', 'all_repeat'))
                for key in ('rpe_translation_m', 'translation_p99_m', 'ate_m')}
            stable = not missing and all(v <= .05 for v in deviation.values())
            state[clip] = dict(missing_controls=missing, repeat_deviation=deviation,
                               interpretable=stable,
                               label='incomplete' if missing else ('stable' if stable else 'unstable'))
        interpretable = [c for c in clips if state[c]['interpretable']]
        candidates = {}
        for method in sorted(PRIORITY, key=lambda m: int(m[1:])):
            per_clip = {}
            for clip in interpretable:
                r = results[clip]
                if method not in r:
                    per_clip[clip] = 'incomplete'
                    continue
                m, reference, native = r[method], r['all'], r['native']
                checks = {key: m[key] <= 1.05 * reference[key] for key in KEYS}
                checks['halves'] = all(h is not None and g is not None and h <= 1.05 * g
                                       for h, g in zip(m['half_rms_m'], reference['half_rms_m']))
                checks['time'] = m['tracking_seconds'] <= 1.10 * native['tracking_seconds']
                checks['persistent'] = m['persistent_bytes'] <= .60 * reference['persistent_bytes']
                gain = {s: m['rpe_translation_m'] <= .95 * r[s]['rpe_translation_m']
                        and m['translation_p99_m'] <= r[s]['translation_p99_m'] for s in SPARSE_CONTROLS}
                per_clip[clip] = dict(checks=checks, eligible=all(checks.values()),
                    gain_over_sparse_controls=gain, gain=all(gain.values()),
                    ratio_to_all_kept={key: m[key] / reference[key] for key in KEYS},
                    time_ratio_to_native=m['tracking_seconds'] / native['tracking_seconds'],
                    persistent_ratio_to_all_kept=m['persistent_bytes'] / reference['persistent_bytes'])
            evaluated = [v for v in per_clip.values() if v != 'incomplete']
            complete = len(evaluated) == len(interpretable)
            candidates[method] = dict(per_clip=per_clip, complete=complete, qualifies=(
                len(interpretable) >= 2 and complete and all(v['eligible'] for v in evaluated)
                and sum(v['gain'] for v in evaluated) >= 2))
        qualifying = [m for m, c in candidates.items() if c['qualifies']]
        winner = min(qualifying, key=lambda m: (
            max(v['ratio_to_all_kept']['translation_p99_m'] for v in candidates[m]['per_clip'].values()),
            sum(results[c][m]['tracking_seconds'] for c in interpretable),
            int(m[1:]))) if qualifying else None
        if len(interpretable) < 2:
            status = 'inconclusive: fewer than two interpretable clips'
        elif winner is not None:
            status = 'winner'
        elif not all(c['complete'] for c in candidates.values()):
            status = 'incomplete: missing runs are not negative results'
        else:
            status = 'stop: no candidate qualifies; no retuning'
        decision[task] = dict(status=status, winner=winner, clips=state,
                              interpretable=interpretable, candidates=candidates)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Internal: one condition config.json')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--stage', choices=('preflight', 'screen'))
    parser.add_argument('--reviewed-preflight', type=Path,
                        help='Screen only: the reviewed preflight context tar')
    parser.add_argument('--hours', type=float, default=22.,
                        help='Start no inference after this many hours (Slurm limit is 24 h)')
    args = parser.parse_args()
    if args.run is not None:
        return run(args.run)
    assert sys.platform == 'linux' and args.work.is_relative_to('/tmp')
    assert (args.stage == 'screen') == (args.reviewed_preflight is not None)
    started = time.time()
    deadline = started + args.hours * 3600
    preflight = args.stage == 'preflight'
    clips = {c: dict(CLIPS[c]) for c in (PREFLIGHT_CLIPS if preflight else CLIPS)}
    os.chdir(CHECKOUT)

    inputs = {}
    if any(c['task'] == 'object' for c in clips.values()):
        from kvt_arctic_run import DATASET_DIR, OUT as ARCTIC_OUT, stage_dataset
        manifest = json.loads((ARCTIC_OUT / 'initial_frames' / 'manifest.json').read_text())
        stage_dataset(manifest)
        inputs.update(arctic_prepared=str(ARCTIC_OUT / 'prepared.tar'),
                      arctic_prepared_sha256=sha256(ARCTIC_OUT / 'prepared.tar'), init_masks={})
    frames = {}
    for clip, spec in clips.items():
        if spec['task'] == 'object':
            reviewed = ARCTIC_OUT / 'masks' / spec['scene'] / 'init_mask.png'
            shutil.copyfile(reviewed, DATASET_DIR / spec['scene'] / 'init_mask.png')
            inputs['init_masks'][clip] = sha256(reviewed)
            spec['source_frames'] = manifest['scenes'][spec['scene']]['tracking_frames']
            frames[clip] = PREFLIGHT_FRAMES if preflight else spec['source_frames']
        else:
            frames[clip] = PREFLIGHT_FRAMES if preflight else spec['frames']
            staged = args.work / 'inputs' / clip
            source = TUM_ZIPS / f"rgbd_dataset_{spec['scene']}.zip"
            prepared = prepare(source, staged, RESIZE['scene'], MAX_GT_DIFFERENCE, count=frames[clip])
            (staged / 'archive.sha256').write_text(f'{sha256(source)}  {source}\n')
            archive_inputs(staged, args.out / f'{args.tag}_ps_inputs_{clip}.tar')
            spec['scene_dir'] = str(staged)
            inputs[clip] = dict(zip=str(source), frames=prepared['frames'],
                                gt_valid_frames=prepared['gt_valid_frames'],
                                duration_seconds=prepared['duration_seconds'])
    staging_seconds = time.time() - started
    if preflight:
        order = [(clip, name) for clip in clips for name in CONTROLS + PRIORITY]
    else:
        order = screen_order()
    write_json(args.work / 'protocol.json', dict(
        stage=args.stage, clips=clips, frames=frames, order=order, inputs=inputs,
        random_seeds=[17, 29], patch_fraction='ceil(P/4) ordinary patches per non-anchor keyframe',
        tolerance=TOLERANCE, staging_seconds=staging_seconds, deadline_hours=args.hours,
        reviewed_preflight=None if preflight else dict(
            path=str(args.reviewed_preflight), sha256=sha256(args.reviewed_preflight)),
        admission='native schedule per clip, replayed by every arm',
        scope='screen on previously inspected clips; not a held-out generalization test'))

    results = {clip: {} for clip in clips}
    schedules = {}

    def condition(clip, name, length=None, schedule=None):
        spec = clips[clip]
        task = spec['task']
        policy = name.removesuffix('_repeat').removesuffix('_prefix')
        length = length or frames[clip]
        target = args.work / 'runs' / clip / name
        target.mkdir(parents=True)
        if schedule is None and policy != 'native':
            schedule = schedules[clip]
        config = dict(spec, clip=clip, name=name, policy=policy, frames=length, schedule=schedule)
        write_json(target / 'config.json', config)
        print('RUN', clip, name, length, flush=True)
        run_started = time.time()
        with (target / 'run.log').open('w') as log:
            completed = subprocess.run([sys.executable, __file__, '--run', str(target / 'config.json')],
                                       stdout=log, stderr=subprocess.STDOUT)
        wall = time.time() - run_started
        write_json(target / 'process.json', dict(returncode=completed.returncode, wall_seconds=wall))
        if completed.returncode:
            print((target / 'run.log').read_text()[-8000:], flush=True)
            (args.work / 'FAILED').write_text(f'{clip}/{name}\n')
            raise RuntimeError(f'{clip}/{name} failed; the host stops here, see run.log')
        native = args.work / 'runs' / clip / 'native'
        if name == 'native':
            schedules[clip] = np.load(target / 'kf_idx.npy').tolist()
            assert schedules[clip][0] == 0 and schedules[clip] == sorted(set(schedules[clip]))
            print('SCHEDULE', clip, schedules[clip], flush=True)
        if name == 'native_repeat':
            # A numerical repeat of native admission: a changed schedule is reported, not asserted.
            replay = np.load(target / 'kf_idx.npy').tolist()
        else:
            replay = schedules[clip] if schedule is None else schedule
        metric = audit(target, config, replay, native)
        metric['schedule_matches_native'] = metric['keyframes'] == schedules[clip]
        if name.startswith('all'):
            np.testing.assert_allclose(np.load(native / 'traj.npy'), np.load(target / 'traj.npy'),
                                       **TOLERANCE[task])
            np.testing.assert_array_equal(np.load(native / 'kf_poses.npy'), np.load(target / 'kf_poses.npy'))
            print('FULL RETENTION FIDELITY OK', clip, name, flush=True)
        if task == 'object' and name != 'native':
            shutil.rmtree(target / 'sam_masks')  # Byte-identical to native's, which are kept.
        metric.update(wall_seconds=wall, pose_holds=(target / 'run.log').read_text().count('low conf detected'))
        write_json(target / 'summary.json', metric)
        archive = args.out / f'{args.tag}_ps_{clip}_{name}.tar'
        with tarfile.open(archive.with_suffix('.tar.partial'), 'w') as bundle:
            bundle.add(target, arcname=f'{clip}/{name}')
        archive.with_suffix('.tar.partial').replace(archive)
        with tarfile.open(archive) as bundle:
            assert bundle.extractfile(f'{clip}/{name}/traj.npy').read() == (target / 'traj.npy').read_bytes()
        print('RUN OK', clip, name, ' '.join(f'{k}={metric[k]:.5g}' for k in KEYS),
              f"seconds={metric['tracking_seconds']:.1f} persistent={metric['persistent_bytes']}", flush=True)
        return target, metric

    def prefix(clip):
        """Causal prefix: each sparse arm again, ending just after an admission
that the full run follows with more frames; choices and poses must match."""
        schedule = [s for s in schedules[clip] if s < frames[clip] - 1]
        assert len(schedule) >= 3, f'{clip}: two admissions are needed to exercise history'
        length = schedule[-1] + 1
        keys = ('frame', 'selected', 'retained_frame_ids', 'retained_tokens', 'budget')
        for name in SPARSE_CONTROLS + PRIORITY:
            target, results[clip][f'{name}_prefix'] = condition(clip, f'{name}_prefix', length, schedule)
            full = args.work / 'runs' / clip / name
            np.testing.assert_allclose(np.load(full / 'traj.npy')[:length], np.load(target / 'traj.npy'),
                                       **TOLERANCE[clips[clip]['task']])
            events = [json.loads((run / 'patch_events.json').read_text()) for run in (full, target)]
            assert [{k: e[k] for k in keys} for e in events[0][:len(schedule)]] == [
                {k: e[k] for k in keys} for e in events[1]], (clip, name)
        print('CAUSAL PREFIX OK', clip, length, schedule, flush=True)

    def finish_preflight():
        """Costs and resources per arm, and a conservative projection of the screen."""
        summary = {}
        for clip in clips:
            r = results[clip]
            summary[clip] = {name: dict(
                time_ratio_to_native=m['tracking_seconds'] / r['native']['tracking_seconds'],
                persistent_ratio_to_all_kept=m['persistent_bytes'] / r['all']['persistent_bytes'],
                **{k: m.get(k) for k in ('tracking_seconds', 'select_seconds', 'read_demand_seconds',
                                         'state_bytes', 'persistent_bytes', 'peak_allocated_bytes',
                                         'peak_reserved_bytes', 'weak_corner_selected', 'target_selected',
                                         'selected_patches', 'k2_fill', 'wall_seconds', 'pose_holds')})
                for name, m in r.items() if not name.endswith('_prefix')}
        cost = {}
        for task in ('object', 'scene'):
            measured = [m for c in clips if clips[c]['task'] == task for m in results[c].values()]
            cost[task] = dict(
                overhead_seconds=max(m['wall_seconds'] - m['tracking_seconds'] for m in measured),
                seconds_per_frame=max(m['tracking_seconds'] / m['frames'] for m in measured))
        full = {c: manifest['scenes'][spec['scene']]['tracking_frames'] if spec['task'] == 'object'
                else spec['frames'] for c, spec in CLIPS.items()}
        linear = sum(cost[CLIPS[c]['task']]['overhead_seconds']
                     + cost[CLIPS[c]['task']]['seconds_per_frame'] * full[c] for c, _ in screen_order())
        # Short clips miss the larger late rebuilds and caches: double the linear
        # estimate. Staging scales with the prepared TUM frames.
        estimate = 2 * linear + staging_seconds * sum(full[c] for c in CLIPS if CLIPS[c]['task'] == 'scene') / (
            PREFLIGHT_FRAMES)
        projection = dict(cost=cost, screen_frames=full, linear_run_seconds=linear,
                          conservative_hours=estimate / 3600, deadline_hours=args.hours,
                          fits=estimate / 3600 <= args.hours,
                          note='estimate from 128-frame preflight runs; not a measured runtime')
        write_json(args.work / 'preflight.json', dict(summary=summary, projection=projection))
        print('PROJECTION', json.dumps(projection), flush=True)
        (args.work / 'JOB_OK').write_text('Patch-select preflight complete; review before the screen.\n')
        print('PATCH SELECT PREFLIGHT OK', flush=True)

    incomplete = []
    for index, (clip, name) in enumerate(order):
        if time.time() > deadline:
            incomplete = order[index:]
            print('DEADLINE: no further inference;', len(incomplete), 'runs incomplete', flush=True)
            break
        _, results[clip][name] = condition(clip, name)
        write_json(args.work / 'comparison.json', results)
        if preflight and name == PRIORITY[-1]:
            prefix(clip)

    if preflight:
        return finish_preflight()
    write_json(args.work / 'overlap.json', {clip: overlap(args.work / 'runs' / clip, results[clip])
                                            for clip in clips})
    decision = decide(results)
    write_json(args.work / 'decision.json', decision)
    print('DECISION', json.dumps({task: decision[task]['status'] for task in ('object', 'scene')}), flush=True)
    for task in ('object', 'scene'):
        print('DECISION', task, decision[task]['status'], decision[task]['winner'], flush=True)
    if incomplete:
        write_json(args.work / 'incomplete.json', incomplete)
        print('SCREEN INCOMPLETE: missing runs are incomplete, not negative results', flush=True)
        return
    (args.work / 'JOB_OK').write_text('Patch-select screen complete; artifact review required.\n')
    print('PATCH SELECT SCREEN OK', flush=True)


if __name__ == '__main__':
    main()
