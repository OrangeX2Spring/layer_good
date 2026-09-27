"""Bounded StreamVGGT dynamic-history pilot. Execute only on CAMP Linux."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from stream_cache_adapters import StreamAdapter, tensor_bytes
from stream_cache_sweep import (digest, evaluate, fidelity, load_frame, load_model,
                                pose_arrays, write_json)

REPO = Path(__file__).resolve().parents[1]
LOW, HIGH, PERIOD = 4, 16, 8
CLIPS = [('calibration', 'freiburg3_long_office_household', 256),
         ('with_loop', 'freiburg2_large_with_loop', 256),
         ('no_loop', 'freiburg2_large_no_loop', 256)]
FEATURES = ['rgb_change_mean', 'rgb_change_p90', 'rgb_gradient',
            'previous_log_confidence_mean', 'previous_log_confidence_std',
            'previous_translation_step', 'previous_rotation_step_rad']


def history_ids(frame, budget):
    """Past frames only: anchor zero and the newest budget-1 observations."""
    assert frame >= 0 and budget >= 2
    return list(range(frame)) if frame <= budget else [0, *range(frame - budget + 1, frame)]


def attention_indices(token_frames, selected, current_tokens):
    assert current_tokens > 0
    old = torch.isin(token_frames, torch.tensor(selected, dtype=torch.long)).nonzero().flatten()
    return torch.cat((old, torch.arange(len(token_frames), len(token_frames) + current_tokens)))


class History:
    """Same attention selection, two explicit storage semantics; camera cache untouched."""
    def __init__(self, adapter, backend):
        assert adapter.host == 'streamvggt' and backend in ('active', 'stored')
        self.adapter, self.backend = adapter, backend

    def unselect(self):
        for attention, _ in self.adapter.original_attention:
            if hasattr(attention, 'active_token_indices'):
                del attention.active_token_indices

    def snapshot(self):
        a = self.adapter
        # Upstream mutates cache LISTS, but creates new tensors with cat. Copy lists;
        # never share the mutable containers between counterfactual branches.
        return dict(aggregator=list(a.aggregator_cache), camera=list(a.camera_cache),
                    positions=[attn.cache_positions for attn, _ in a.original_attention],
                    frames=a.token_frames.clone())

    def restore(self, state):
        a = self.adapter
        self.unselect()
        a.aggregator_cache = list(state['aggregator'])
        a.camera_cache = list(state['camera'])
        a.token_frames = state['frames'].clone()
        for (attn, _), pos in zip(a.original_attention, state['positions']):
            attn.cache_positions = pos

    def prune(self, ids, patches):
        self.adapter.prune(torch.arange(patches), ids)

    def forward(self, image, frame, budget, root, frames):
        a = self.adapter
        selected = history_ids(frame, budget)
        available = sorted(set(a.token_frames.tolist()))
        replay_ids, replay_pairs, replay_seconds = [], 0, 0.
        patches = (image.shape[-2] // a.patch_size) * (image.shape[-1] // a.patch_size)
        tokens = patches + a.special
        self.unselect()
        if self.backend == 'stored' and not set(selected).issubset(available):
            # Rebuild aggregator only from causal source pixels. Replaying the camera
            # head would duplicate its history and alter its temporal indexing.
            if image.is_cuda:
                torch.cuda.synchronize()
            replay_start = time.perf_counter()
            camera = list(a.camera_cache)
            a.aggregator_cache = [None] * a.core.aggregator.depth
            a.token_frames = torch.empty(0, dtype=torch.long)
            for attn, _ in a.original_attention:
                attn.cache_positions = None
            for old in selected:
                assert old < frame
                replay_image, _ = load_frame(root, frames[old])
                outputs, special, a.aggregator_cache = a.model.aggregator(
                    replay_image, past_key_values=a.aggregator_cache, use_cache=True,
                    past_frame_idx=old)
                assert special == a.special
                a.token_frames = torch.cat((a.token_frames, torch.full((tokens,), old)))
                replay_ids.append(old)
                replay_pairs += tokens * len(a.token_frames) * len(a.original_attention)
                del outputs, replay_image
            assert all(x is y for x, y in zip(camera, a.camera_cache))
            del camera
            if image.is_cuda:
                torch.cuda.synchronize()
            replay_seconds = time.perf_counter() - replay_start
        elif self.backend == 'stored' and available:
            self.prune(selected, patches)
        assert set(selected).issubset(set(a.token_frames.tolist()))
        indices = attention_indices(a.token_frames, selected, tokens)
        for attn, _ in a.original_attention:
            attn.active_token_indices = indices
        resident_before = tensor_bytes(a.aggregator_cache)
        output = a.forward(image, frame)
        self.unselect()
        # Post-frame storage contains the anchor plus recent entries for the next query.
        retained = history_ids(frame + 1, HIGH if self.backend == 'active' else budget)
        self.prune(retained, patches)
        assert sorted(set(a.token_frames.tolist())) == retained
        event = dict(frame=frame, requested_budget=budget, attended_past_ids=selected,
                     retained_frames=retained, replay_ids=replay_ids,
                     replay_frames=len(replay_ids), replay_seconds=replay_seconds,
                     resident_before_bytes=resident_before,
                     aggregator_attention_pairs=tokens * len(indices) * len(a.original_attention),
                     replay_attention_pairs=replay_pairs, **a.memory())
        return output, event


class Signals:
    """Only current RGB and strictly previous prediction signals; never GT."""
    def __init__(self):
        self.previous_rgb = None
        self.previous_pose = None
        self.confidence = [0., 0.]
        self.step = [0., 0.]

    def before(self, path):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('L').resize((32, 32)), dtype=np.float64) / 255.
        diff = np.zeros_like(rgb) if self.previous_rgb is None else np.abs(rgb - self.previous_rgb)
        features = [float(diff.mean()), float(np.quantile(diff, .9)),
                    float((np.abs(np.diff(rgb, axis=0)).mean() +
                           np.abs(np.diff(rgb, axis=1)).mean()) / 2),
                    *self.confidence, *self.step]
        self.previous_rgb = rgb
        assert np.isfinite(features).all()
        return features

    def after(self, output, pose):
        conf = output['depth_conf'].float()
        self.confidence = [float(torch.log1p(conf.mean())),
                           float(torch.log1p(conf.std(unbiased=False)))]
        if self.previous_pose is not None:
            relative = np.linalg.inv(self.previous_pose) @ pose
            self.step = [float(np.linalg.norm(relative[:3, 3])), rotation_angle(relative)]
        self.previous_pose = pose


def rotation_angle(matrix):
    return float(np.arccos(np.clip((np.trace(matrix[:3, :3]) - 1) / 2, -1, 1)))


def c2w(encoding, hw):
    ex = pose_arrays('streamvggt', {'pose_enc': encoding.detach().float().cpu()}, hw, 8)['extrinsics'][0]
    matrix = np.eye(4)
    matrix[:3] = ex[:3]
    return np.linalg.inv(matrix)


def fit_controller(features, losses):
    """Small ridge regressor; fixed regularization, no hyperparameter search."""
    x, losses = np.asarray(features, dtype=np.float64), np.asarray(losses, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == len(FEATURES) and len(x) == len(losses)
    assert len(x) > len(FEATURES) and np.isfinite(x).all() and np.isfinite(losses).all()
    mean, std = x.mean(0), x.std(0)
    std = np.where(std < 1e-8, 1., std)  # Constant calibration signals have zero weight.
    design = np.column_stack((np.ones(len(x)), (x - mean) / std))
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ np.log1p(losses))
    predicted = design @ weights
    threshold = float(np.median(predicted))
    return dict(features=FEATURES, mean=mean.tolist(), std=std.tolist(), weights=weights.tolist(),
                threshold=threshold, ridge=1., training_samples=len(x),
                training_high_fraction=float(np.mean(predicted > threshold)))


def predict(controller, features):
    assert controller['features'] == FEATURES
    x = (np.asarray(features) - controller['mean']) / controller['std']
    return float(np.dot(np.r_[1., x], controller['weights']))


def budget_for(mode, frame, features, controller, previous):
    if mode.startswith('fixed'):
        return int(mode.removeprefix('fixed'))
    if frame < HIGH:
        return HIGH
    if frame % PERIOD:
        return previous
    if mode == 'alternate':
        return LOW if (frame - HIGH) // PERIOD % 2 == 0 else HIGH
    assert mode == 'learned'
    return HIGH if predict(controller, features) > controller['threshold'] else LOW


def worker(args):
    root = args.work / 'runs' / args.clip
    frames = json.loads((root / 'inputs/manifest.json').read_text())['frames']
    for frame in frames:
        assert digest(root / 'inputs' / frame['rgb']) == frame['rgb_sha256']
    model = load_model(SimpleNamespace(host='streamvggt', checkpoint=args.checkpoint))
    target = root / args.worker
    target.mkdir()
    if args.worker == 'fidelity':
        fidelity(SimpleNamespace(host='streamvggt', keyframe_stride=8, refresh=4,
                 gate_frames=4, gate_atol=1e-5, gate_rtol=1e-4, seed=0, out=target),
                 model, root / 'inputs', frames)
        return
    backend, mode = args.worker.split('_', 1)
    assert backend in ('active', 'stored')
    controller = json.loads((args.work / 'controller.json').read_text()) if mode == 'learned' else None
    adapter = StreamAdapter(model, 'streamvggt')
    history, signals = History(adapter, backend), Signals()
    calibration = mode == 'collect'
    encodings, features_log, low_poses, high_poses, events = [], [], [], [], []
    previous_budget = HIGH
    torch.manual_seed(0)
    with torch.no_grad(), (target / 'events.jsonl').open('w') as log:
        for index, frame in enumerate(frames):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            features = signals.before(root / 'inputs' / frame['rgb'])
            image, _ = load_frame(root / 'inputs', frame)
            budget = HIGH if calibration else budget_for(mode, index, features, controller, previous_budget)
            selection_end = time.perf_counter()
            if calibration and index >= HIGH:
                snapshot = history.snapshot()
                low, _ = history.forward(image, index, LOW, root / 'inputs', frames)
                low_pose = c2w(low['pose_enc'], frame['model_hw'])
                del low
                history.restore(snapshot)
                del snapshot
            output, event = history.forward(image, index, budget, root / 'inputs', frames)
            pose = c2w(output['pose_enc'], frame['model_hw'])
            if calibration:
                high_poses.append(pose)
                if index >= HIGH:
                    low_poses.append(low_pose)
                    features_log.append(features)
            encoding = output['pose_enc'].detach().float().cpu()
            encodings.append(encoding)
            signals.after(output, pose)
            del output, image
            torch.cuda.synchronize()
            event.update(features=features, signal_latest_prediction_frame=index - 1,
                         decision=index >= HIGH and index % PERIOD == 0,
                         controller_score=predict(controller, features) if controller is not None else None,
                         controller_and_input_seconds=selection_end - start,
                         seconds=time.perf_counter() - start,
                         peak_allocated=torch.cuda.max_memory_allocated(),
                         peak_reserved=torch.cuda.max_memory_reserved())
            log.write(json.dumps(event, allow_nan=False) + '\n'); log.flush()
            events.append(event)
            previous_budget = budget
    np.savez_compressed(target / 'camera.npz', **pose_arrays('streamvggt',
        {'pose_enc': torch.cat(encodings, dim=1)}, frames[0]['model_hw'], 8))
    if calibration:
        np.savez(target / 'counterfactuals.npz', features=np.asarray(features_log),
                 low=np.asarray(low_poses), high=np.asarray(high_poses),
                 frames=np.arange(HIGH, len(frames)))
    write_json(target / 'summary.json', dict(backend=backend, mode=mode, frames=len(frames),
        total_seconds=sum(e['seconds'] for e in events),
        replay_frames=sum(e['replay_frames'] for e in events),
        replay_seconds=sum(e['replay_seconds'] for e in events),
        mean_past_frames=float(np.mean([len(e['attended_past_ids']) for e in events])),
        mean_aggregator_bytes=float(np.mean([e['aggregator_bytes'] for e in events])),
        mean_camera_bytes=float(np.mean([e['camera_bytes'] for e in events])),
        peak_allocated=max(e['peak_allocated'] for e in events),
        attention_pairs=sum(e['aggregator_attention_pairs'] + e['replay_attention_pairs'] for e in events),
        high_fraction=float(np.mean([e['requested_budget'] == HIGH for e in events[HIGH:]])),
        switches=sum(a['requested_budget'] != b['requested_budget']
                     for a, b in zip(events[HIGH:], events[HIGH + 1:]))))
    adapter.close()
    print('DYNAMIC RUN OK', args.clip, args.worker, flush=True)


def launch(args, clip, name):
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--work', str(args.work),
                    '--checkpoint', str(args.checkpoint), '--clip', clip, '--worker', name],
                   cwd=REPO, check=True)


def train(args):
    data = np.load(args.work / 'runs/calibration/active_collect/counterfactuals.npz')
    high, low = data['high'], data['low']
    relative = np.linalg.inv(high[:-1]) @ high[1:]
    unit = float(np.sqrt(np.mean(np.sum(relative[:, :3, 3] ** 2, axis=1))))
    assert unit > 1e-8, 'Calibration teacher has insufficient motion'
    error = np.linalg.inv(high[HIGH:]) @ low
    angles = np.asarray([rotation_angle(e) for e in error])
    loss = np.sum(error[:, :3, 3] ** 2, axis=1) / unit ** 2 + (angles / np.deg2rad(1.)) ** 2
    # Labels sum distortion over each next decision block, but branch predictions
    # share the high-history teacher state; this is not a closed-loop oracle.
    starts = np.arange(0, len(loss) - PERIOD + 1, PERIOD)
    block_loss = np.asarray([loss[i:i + PERIOD].mean() for i in starts])
    controller = fit_controller(data['features'][starts], block_loss)
    controller.update(translation_unit=unit, source_clip='calibration',
                      labels='one-step low-vs-high pose distortion, averaged over 8 teacher steps',
                      oracle_warning='teacher-state screening; not a closed-loop performance bound')
    write_json(args.work / 'controller.json', controller)
    # Same high-action count for oracle and learned screening comparisons.
    count = len(block_loss) // 2
    ordered = np.argsort(block_loss, kind='stable')
    selected = ordered[-count:]
    residual = block_loss.copy(); residual[selected] = 0
    learned = np.asarray([predict(controller, row) > controller['threshold']
                          for row in data['features'][starts]])
    write_json(args.work / 'calibration_screen.json', dict(blocks=len(starts),
        block_start_frames=data['frames'][starts].tolist(), block_losses=block_loss.tolist(),
        low_mean_distortion=float(block_loss.mean()),
        oracle_half_high_mean_distortion=float(residual.mean()),
        uniform_half_high_expected_distortion=float(block_loss.mean() * .5),
        learned_training_mean_distortion=float(np.mean(block_loss * ~learned)),
        high_action_count=count, teacher_only=True, training_fit_not_validation=True))
    print('DYNAMIC CONTROLLER FROZEN', digest(args.work / 'controller.json'), flush=True)


def contracts(root, names):
    reports = {}
    camera_bytes = None
    for name in names:
        backend, mode = name.split('_', 1)
        rows = [json.loads(t) for t in (root / name / 'events.jsonl').read_text().splitlines()]
        sizes = [row['camera_bytes'] for row in rows]
        if camera_bytes is None:
            camera_bytes = sizes
        assert sizes == camera_bytes, 'Native camera history changed size across conditions'
        frame_bytes = rows[0]['aggregator_bytes']
        previous = HIGH
        for i, row in enumerate(rows):
            budget = row['requested_budget']
            assert row['frame'] == i and row['signal_latest_prediction_frame'] == i - 1
            assert row['attended_past_ids'] == history_ids(i, budget)
            assert row['retained_frames'] == history_ids(i + 1, HIGH if backend == 'active' else budget)
            assert all(j < i for j in row['replay_ids'])
            if row['replay_ids']:
                assert backend == 'stored' and row['replay_ids'] == history_ids(i, budget)
            elif backend == 'active':
                assert row['replay_attention_pairs'] == 0 and row['replay_seconds'] == 0
            assert row['aggregator_bytes'] > 0 and row['camera_bytes'] > 0
            assert row['aggregator_bytes'] == frame_bytes * len(row['retained_frames'])
            if mode in ('learned', 'alternate') and i >= HIGH and i % PERIOD:
                assert budget == previous
            previous = budget
        reports[name] = json.loads((root / name / 'summary.json').read_text())
    if 'stored_alternate' in names:
        assert reports['stored_alternate']['replay_frames'] > 0
        assert reports['active_alternate']['replay_frames'] == 0
        assert reports['stored_alternate']['mean_aggregator_bytes'] < reports['active_alternate']['mean_aggregator_bytes']
    write_json(root / 'dynamic_contracts.json', reports)
    print('DYNAMIC CONTRACTS OK', root.name, flush=True)


def compare(root, names):
    """Post-hoc reporting only; these evaluation quantities never enter the policy."""
    def poses(name):
        ex = np.load(root / name / 'camera.npz')['extrinsics']
        matrix = np.tile(np.eye(4), (len(ex), 1, 1)); matrix[:, :3] = ex[:, :3]
        return np.linalg.inv(matrix)
    reference = poses('active_fixed16')
    relative = np.linalg.inv(reference[:-1]) @ reference[1:]
    unit = float(np.sqrt(np.mean(np.sum(relative[:, :3, 3] ** 2, axis=1))))
    assert unit > 1e-8
    summaries = json.loads((root / 'dynamic_contracts.json').read_text())
    metrics = json.loads((root / 'camera_metrics.json').read_text())
    result = {}
    for name in names:
        candidate = poses(name)
        error = np.linalg.inv(relative) @ (np.linalg.inv(candidate[:-1]) @ candidate[1:])
        angles = np.asarray([rotation_angle(e) for e in error])
        loss = np.sum(error[:, :3, 3] ** 2, axis=1) / unit ** 2 + (angles / np.deg2rad(1.)) ** 2
        summary = summaries[name]
        result[name] = dict(summary, native_relative_distortion=float(loss.mean()),
                            native_rms_step=unit, gt_metrics=metrics[name],
                            adaptation_active=summary['switches'] > 0,
                            seconds_vs_fixed16=summary['total_seconds'] / summaries['active_fixed16']['total_seconds'],
                            note='Single-run measured time; different average history is not equal cost. '
                                 'Replay work included. Camera history remains native and grows.')
    write_json(root / 'dynamic_comparison.json', result)


def implementation_gate(args, calibration):
    """Independent established-runner controls, before learning or evaluation."""
    root = args.work / 'runs/implementation'
    inputs = root / 'inputs'; inputs.mkdir(parents=True)
    manifest = json.loads((calibration / 'inputs/manifest.json').read_text())
    manifest['frames'] = manifest['frames'][:32]
    for frame in manifest['frames']:
        destination = inputs / frame['rgb']; destination.parent.mkdir(exist_ok=True)
        shutil.copy2(calibration / 'inputs' / frame['rgb'], destination)
    write_json(inputs / 'manifest.json', manifest)
    spec = root / 'reference_sweep.json'
    write_json(spec, dict(max_frames=32, isolate_conditions=True, geometry_export='final',
                         conditions=[dict(name=f'recent{b}', policy=dict(frame_budget=b))
                                     for b in (LOW, HIGH)]))
    subprocess.run([sys.executable, 'tools/stream_cache_sweep.py', 'run', '--host', 'streamvggt',
                    '--checkpoint', str(args.checkpoint), '--inputs', str(inputs), '--sweep', str(spec),
                    '--out', str(root / 'established'), '--git-provenance',
                    str(args.work / 'git_provenance')], cwd=REPO, check=True)
    report = {}
    for budget in (LOW, HIGH):
        reference = np.load(root / 'established' / f'recent{budget}' / 'camera.npz')
        for backend in ('active', 'stored'):
            name = f'{backend}_fixed{budget}'
            launch(args, 'implementation', name)
            candidate = np.load(root / name / 'camera.npz')
            errors = {}
            for key in ('extrinsics', 'intrinsics'):
                np.testing.assert_allclose(candidate[key], reference[key], atol=1e-5, rtol=1e-4)
                errors[key] = float(np.max(np.abs(candidate[key] - reference[key])))
            report[name] = errors
    names = ['active_fixed4', 'stored_fixed4', 'active_fixed16', 'stored_fixed16']
    for name in ('active_alternate', 'stored_alternate'):
        launch(args, 'implementation', name); names.append(name)
    contracts(root, names)
    write_json(root / 'fidelity.json', dict(atol=1e-5, rtol=1e-4, comparisons=report))
    print('DYNAMIC IMPLEMENTATION GATE OK', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--worker')
    parser.add_argument('--clip')
    args = parser.parse_args()
    assert sys.platform == 'linux' and args.work.resolve().is_relative_to('/tmp')
    if args.worker:
        worker(args)
        return
    source = args.work / 'dynamic_source'; source.mkdir()
    for name in ('stream_cache_dynamic.py', 'stream_cache_dynamic.json',
                 'stream_cache_dynamic.sbatch', 'stream_cache_adapters.py',
                 'stream_cache_sweep.py', 'stream_cache_job.py',
                 'test_stream_cache_dynamic.py', 'DYNAMIC_HISTORY.md'):
        shutil.copy2(REPO / 'tools' / name, source)
    write_json(args.work / 'dynamic_protocol.json', dict(host='streamvggt', clips=CLIPS,
        budgets=[LOW, HIGH], decision_period=PERIOD, seed=0, features=FEATURES,
        backbone='frozen native dtype; native camera history preserved',
        stored_recovery='anchor plus recent source RGB; aggregator-only causal replay',
        scope='bounded supervised distillation pilot; no GT policy labels, no full evaluation'))
    controller_hash = None
    for clip, scene, length in CLIPS:
        root = args.work / 'runs' / clip; root.mkdir(parents=True)
        extracted = args.work / ('extracted_' + clip)
        archive = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
        subprocess.run([sys.executable, 'tools/stream_cache_tum.py', '--zip', str(archive),
                        '--out', str(extracted), '--start', '0', '--count', str(length)],
                       cwd=REPO, check=True)
        subprocess.run([sys.executable, 'tools/stream_cache_sweep.py', 'prepare',
                        '--manifest', str(extracted / 'manifest.json'), '--out',
                        str(root / 'inputs'), '--width', '308'], cwd=REPO, check=True)
        write_json(root / 'provenance.json', dict(checkpoint_sha256=digest(args.checkpoint),
                   dataset_zip_sha256=digest(archive), checkpoint=str(args.checkpoint),
                   parent=(args.work / 'git_provenance/superproject_commit.txt').read_text().strip(),
                   model=(args.work / 'git_provenance/model_commit.txt').read_text().strip()))
        launch(args, clip, 'fidelity')
        if clip == 'calibration':
            implementation_gate(args, root)
            launch(args, clip, 'active_collect')
            launch(args, clip, 'active_fixed16')
            reference = np.load(root / 'active_fixed16/camera.npz')
            collected = np.load(root / 'active_collect/camera.npz')
            differences = {}
            for key in ('extrinsics', 'intrinsics'):
                np.testing.assert_allclose(collected[key], reference[key], atol=1e-5, rtol=1e-4)
                differences[key] = float(np.max(np.abs(collected[key] - reference[key])))
            write_json(root / 'counterfactual_identity.json', differences)
            print('DYNAMIC COUNTERFACTUAL IDENTITY OK', flush=True)
            train(args)
            controller_hash = digest(args.work / 'controller.json')
            names = ['active_collect', 'active_fixed16']
        else:
            names = ['active_fixed4', 'active_fixed10', 'active_fixed16',
                     'active_alternate', 'stored_alternate', 'active_learned', 'stored_learned']
            for name in names:
                assert digest(args.work / 'controller.json') == controller_hash
                launch(args, clip, name)
                assert digest(args.work / 'controller.json') == controller_hash
        write_json(root / 'sweep.json', dict(conditions=[dict(name=n) for n in names]))
        evaluate(SimpleNamespace(out=root, max_gt_delta=.02, max_pair_gap=.1))
        contracts(root, names)
        if clip != 'calibration':
            compare(root, names)
    (args.work / 'JOB_OK').write_text('Dynamic history pilot complete; review before expansion.\n')
    print('DYNAMIC HISTORY PILOT OK', flush=True)


if __name__ == '__main__':
    main()
