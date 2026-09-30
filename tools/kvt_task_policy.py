"""One capped CAMP scene pilot: task-trained hard token actions, frozen Pi3."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from kvt_tum_run import CHECKOUT, evaluate, periodic_indices, prepare, write_json
from kvt_tum_sweep import archive_inputs, release_page_cache, LONG_SCENES
from kvt_patch_select import sha256, TUM_ZIPS
from kv_tracker.atd_training import PatchScorer
from kv_tracker.geometry import umeyama_alignment
from kv_tracker.task_token_policy import TaskTokenPolicy

# All Freiburg1 room/desk sources stay in training; the separate Freiburg3
# structure scene family stays in development. Final evaluation clips are absent.
SPLITS = dict(train=('freiburg1_desk', 'freiburg1_desk2', 'freiburg1_room'),
              development=('freiburg3_structure_texture_far', 'freiburg3_structure_notexture_far'))
STARTS = dict(train=(0, 128, 256, 384), development=(0, 128))
FRAMES, UPDATES, SEED = 128, 256, 17


def trajectory_cost(estimate, reference, depth, timestamps):
    """Offline output distillation; no gradient and no GT inputs or labels."""
    assert estimate.shape == reference.shape and estimate.shape[1:] == (4, 4)
    assert len(estimate) == len(timestamps) and depth > 0
    assert np.isfinite(estimate).all() and np.isfinite(reference).all()
    assert np.all(np.diff(timestamps) > 0)
    adjacent = np.diff(timestamps) <= .1
    assert adjacent.any(), 'No adjacent RGB pairs in episode'
    estimate, reference = estimate.astype(np.float64), reference.astype(np.float64)
    x, y = estimate[:, :3, 3].T, reference[:, :3, 3].T
    centered_x, centered_y = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    assert np.linalg.matrix_rank(centered_y) >= 2, 'Degenerate dense-reference episode'
    # A legal sparse action can collapse motion. Define its reward explicitly,
    # rather than letting an undefined Sim(3) interrupt policy optimization.
    if np.linalg.matrix_rank(centered_y @ centered_x.T) < 2:
        return dict(cost=1e6, collapsed_motion=True, mean_pair_cost=None, tail_pair_cost=None)
    rotation, offset, scale = umeyama_alignment(x, y, with_scale=True)
    aligned = estimate.astype(np.float64).copy()
    aligned[:, :3, 3] = scale * (aligned[:, :3, 3] @ rotation.T) + offset
    aligned[:, :3, :3] = rotation @ aligned[:, :3, :3]
    reference_relative = np.linalg.inv(reference[:-1]) @ reference[1:]
    estimate_relative = np.linalg.inv(aligned[:-1]) @ aligned[1:]
    error = (np.linalg.inv(reference_relative) @ estimate_relative)[adjacent]
    translation_squared = np.square(error[:, :3, 3]).sum(1) / depth ** 2
    rotation_squared = np.square(Rotation.from_matrix(error[:, :3, :3]).magnitude())
    pair_cost = translation_squared + rotation_squared
    mean = float(pair_cost.mean())
    tail = float(np.sort(pair_cost)[-math.ceil(.1 * len(pair_cost)):].mean())
    assert np.isfinite(mean + tail)
    return dict(cost=mean + tail, collapsed_motion=False, mean_pair_cost=mean,
        tail_pair_cost=tail, translation_squared_mean=float(translation_squared.mean()),
        rotation_squared_mean=float(rotation_squared.mean()), alignment_scale=float(scale))


def calibration(scene, shape):
    height, width = shape
    if scene.startswith('freiburg1_'):
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
        distortion = torch.tensor([.2624, -.9531, -.0054, .0026, 1.1633], device='cuda')
    else:
        assert scene.startswith('freiburg3_')
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
        distortion = None
    intrinsics = torch.tensor([[fx * width / 640, 0, (cx + .5) * width / 640 - .5],
        [0, fy * height / 480, (cy + .5) * height / 480 - .5], [0, 0, 1]], device='cuda')
    return intrinsics, distortion


def rollout(model, scorer, inputs, target, mode, deadline, sample=False, seed=SEED, frames=FRAMES):
    """The real tracker loop; shared frozen weights, fresh episode cache/state."""
    import main as tracker
    target.mkdir(parents=True)
    manifest = json.loads((inputs / 'manifest.json').read_text())
    images = []
    for row in manifest['inputs'][:frames]:
        bgr = cv2.imread(str(inputs / 'model_rgb' / Path(row['file']).name))
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['model_rgb_sha256']
        images.append(rgb)
    height, width = images[0].shape[:2]
    intrinsics, distortion = calibration(manifest['scene'], (height, width))
    policy = TaskTokenPolicy(mode, intrinsics, distortion,
        scorer if mode == 'actor' else None, sample, seed)
    write_json(target / 'calibration.json', dict(K=intrinsics.cpu().tolist(),
        distortion=None if distortion is None else distortion.cpu().tolist(),
        source='https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats',
        source_size=[480, 640], model_size=[height, width], sampling='7::14',
        gauge='raw Pi3 points and raw last pose; both reset together at rebuild'))
    schedule = periodic_indices(frames, 50, 20)

    def source():
        for frame, rgb in enumerate(images):
            if time.time() >= deadline:
                raise TimeoutError('Task-policy wall deadline reached')
            mask = torch.ones(height, width, device='cuda', dtype=torch.bool)
            yield dict(idx=frame, rgb_np=rgb, resized_rgb_masked_np=rgb,
                resized_rgb_masked=torch.tensor(rgb, device='cuda', dtype=torch.float32)[None, None] / 255.,
                resized_mask_np=np.ones((height, width), dtype=bool), resized_mask=mask)

    model.cache = {}
    model.kept_cache = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with (target / 'run.log').open('w') as log, redirect_stdout(log):
            tracker.run_track3r(cfg=dict(results_path=str(target), que_size=1),
                args=['--cam_only', '--resize_dim', '308', '--kf_auto', '50'],
                frame_source=source(), token_policy=policy, keyframe_indices=schedule[1:],
                pi3_model=model)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        # The wrapper also archives a partial rollout if a deadline or failure
        # interrupts it. No sampled decisions/actor inputs disappear with /tmp.
        write_json(target / 'events.json', policy.events)
        write_json(target / 'history.json', policy.history)
        for number, snapshot in enumerate(policy.snapshots):
            np.savez_compressed(target / f'cache_{number:02d}.npz',
                **{key: value.numpy() if isinstance(value, torch.Tensor) else value
                   for key, value in snapshot.items()})
        if policy.inputs:
            np.savez_compressed(target / 'actor_inputs.npz',
                features=torch.stack(policy.inputs).numpy(), logits=torch.stack(policy.logits).numpy(),
                orders=torch.stack(policy.orders).numpy())
    trajectory = np.load(target / 'traj.npy')
    assert trajectory.shape == (frames, 4, 4) and np.isfinite(trajectory).all()
    assert np.load(target / 'kf_idx.npy').tolist() == schedule
    assert [e['frame'] for e in policy.events] == list(range(frames))
    assert all(e['budget'] == e['target_budget'] for e in policy.events)
    assert len(policy.history) == frames - 1 + len(schedule)
    assert all(parameter.grad is None for parameter in model.parameters())
    metrics = dict(tracking_seconds=seconds, frames=frames, keyframes=schedule,
        bootstrap_depth=policy.bootstrap_depth, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), mode=mode, sampled=sample,
        timing='diagnostic eager tracker wall time including actor/maps and CPU feature exports; no deployment speed gate',
        own_sparse_history=True, first_arrival_decisions=len(policy.saved))
    write_json(target / 'metrics.json', metrics)
    (target / 'pcd.npy').unlink(missing_ok=True)
    print('TASK POLICY ROLLOUT OK', target.name, mode, frames, f'{seconds:.2f}s', flush=True)
    return trajectory, policy, metrics


def development(model, scorer, episodes, work, label, deadline, mode="actor"):
    rows = []
    for name, inputs in episodes:
        target = work / 'runs' / 'development' / name / label
        trajectory, policy, metrics = rollout(model, scorer, inputs, target, mode, deadline)
        reference = work / 'runs' / 'references' / name
        ref_metrics = json.loads((reference / 'metrics.json').read_text())
        manifest = json.loads((inputs / 'manifest.json').read_text())
        times = np.array([row['timestamp'] for row in manifest['inputs']])
        fidelity = trajectory_cost(trajectory, np.load(reference / 'traj.npy'),
                                   ref_metrics['bootstrap_depth'], times)
        quality = evaluate(inputs, target, .02)
        with np.load(target / 'evaluation.npz') as saved:
            errors = saved['rpe_translation_per_pair_m']
            angles = saved['rpe_rotation_per_pair_deg']
            starts = saved['rpe_pair_start_indices']
            rows.append(dict(episode=name, fidelity=fidelity, metrics=metrics,
                quality=dict(quality, translation_p99_m=float(np.quantile(errors, .99))),
                squared_translation_sum=float(np.square(errors).sum()),
                squared_rotation_sum=float(np.square(angles).sum()), pairs=len(errors),
                pair_errors=errors.tolist(),
                post_refresh_rms_m=float(np.sqrt(np.square(errors[starts >= 49]).mean()))))
        write_json(target / 'quality.json', rows[-1])
        if label in ('initial', 'final') and name == episodes[0][0]:
            prefix_target = target.parent / f'{label}_prefix'
            prefix, short, _ = rollout(model, scorer, inputs, prefix_target,
                                       'actor', deadline, frames=51)
            np.testing.assert_array_equal(prefix, trajectory[:51])
            assert short.events == policy.events[:51]
            with np.load(prefix_target / 'actor_inputs.npz') as a, np.load(target / 'actor_inputs.npz') as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key][:51])
            print('TASK POLICY CAUSAL PREFIX OK', label, name, flush=True)
    pairs = sum(row['pairs'] for row in rows)
    summary = dict(label=label, episodes=rows,
        fidelity_cost=float(np.mean([row['fidelity']['cost'] for row in rows])),
        rpe_translation_m=float(np.sqrt(sum(row['squared_translation_sum'] for row in rows) / pairs)),
        rpe_rotation_deg=float(np.sqrt(sum(row['squared_rotation_sum'] for row in rows) / pairs)),
        translation_p99_m=float(np.quantile([e for row in rows for e in row['pair_errors']], .99)),
        post_refresh_rms_m=float(np.sqrt(np.mean([row['post_refresh_rms_m'] ** 2 for row in rows]))))
    write_json(work / f'development_{label}.json', summary)
    print('TASK POLICY DEVELOPMENT OK', label, json.dumps({k: v for k, v in summary.items()
                                                         if k != 'episodes'}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and args.work.is_relative_to('/tmp'), 'CAMP only'
    assert len(set(sum(SPLITS.values(), ()))) == sum(map(len, SPLITS.values()))
    assert not set(sum(SPLITS.values(), ())).intersection(LONG_SCENES)
    assert args.deadline > time.time()
    os.chdir(CHECKOUT)

    def deadline_signal(signum, frame):
        raise TimeoutError('Task-policy allocation deadline reached')

    signal.signal(signal.SIGALRM, deadline_signal)
    signal.signal(signal.SIGTERM, deadline_signal)
    signal.setitimer(signal.ITIMER_REAL, args.deadline - time.time())
    decision = dict(complete=False, pilot_pass=False, next_action='Review pilot archive; no automatic continuation')
    updates = []
    try:
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
        # Import tracker before restoring its module-level TF32 defaults.
        import main as tracker
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained('cuda').eval())
        model.requires_grad_(False)
        scorer = PatchScorer('B').cuda()
        assert sum(p.numel() for p in scorer.parameters()) == 12769
        optimizer = torch.optim.Adam(scorer.parameters(), lr=1e-4)
        (args.work / 'checkpoints').mkdir()

        def checkpoint(label, step):
            destination = args.work / 'checkpoints' / f'{label}.pt'
            partial = destination.with_suffix('.pt.partial')
            torch.save(dict(state_dict=scorer.state_dict(), optimizer=optimizer.state_dict(),
                candidate='B', step=step, seed=SEED, eligible_for_deployment=False,
                protocol_sha256=sha256(args.work / 'task_policy_protocol.json')), partial)
            partial.replace(destination)

        checkpoints = Path(os.environ['HF_HOME']) / 'hub/models--yyfz233--Pi3/snapshots'
        weights = list(checkpoints.glob('**/*.safetensors')) + list(checkpoints.glob('**/pytorch_model.bin'))
        assert weights, 'No offline Pi3 checkpoint'
        protocol = dict(split=SPLITS, starts=STARTS, excluded=LONG_SCENES, frames=FRAMES,
            max_updates=UPDATES, seed=SEED, parameters=12769, learning_rate=1e-4,
            gradient_clip=1., deadline=args.deadline, actor='B; new initialization',
            teacher='frozen dense causal all-kept tracker trajectories; no sensitivity labels; no GT training labels',
            objective='mean + worst10% relative translation squared / bootstrap depth squared + angle squared',
            collapsed_motion_cost=1e6, history='each rollout owns sparse cache and raw-gauge maps',
            admissions='fixed replay of native scene interval50/cap20; no natural-admission or speed gate',
            scene_grouping='Freiburg1 room/desk train versus Freiburg3 structure development; no claim of audited physical-site metadata',
            pi3_weights={str(p): sha256(p) for p in weights},
            inputs={}, eligible_for_deployment=False)
        write_json(args.work / 'task_policy_protocol.json', protocol)
        episodes = dict(train=[], development=[])
        for split, scenes in SPLITS.items():
            for scene in scenes:
                source = TUM_ZIPS / f'rgbd_dataset_{scene}.zip'
                digest = sha256(source)
                for start in STARTS[split]:
                    name = f'{scene}_{start:04d}'
                    staged = args.work / 'inputs' / split / name
                    manifest = prepare(source, staged, 308, .02, start=start, count=FRAMES)
                    (staged / 'archive.sha256').write_text(f'{digest}  {source}\n')
                    archive_inputs(staged, args.out / f'{args.tag}_task_inputs_{name}.tar')
                    release_page_cache(staged, source)
                    times = np.array([row['timestamp'] for row in manifest['inputs']])
                    assert np.all(np.diff(times) > 0) and (np.diff(times) <= .1).any()
                    assert split != 'development' or manifest['gt_valid_fraction'] >= .9
                    protocol['inputs'][name] = dict(split=split, scene=scene, source_start=start,
                        zip_sha256=digest, manifest_sha256=sha256(staged / 'manifest.json'),
                        staged_frames=manifest['frames'], input_archive_sha256=sha256(
                            args.out / f'{args.tag}_task_inputs_{name}.tar'))
                    write_json(args.work / 'task_policy_protocol.json', protocol)
                    episodes[split].append((name, staged))
        checkpoint('initial', 0)
        references = {}
        for name, inputs in episodes['train'] + episodes['development']:
            trajectory, policy, _ = rollout(model, scorer, inputs,
                args.work / 'runs' / 'references' / name, 'all', args.deadline)
            times = np.array([row['timestamp'] for row in json.loads((inputs / 'manifest.json').read_text())['inputs']])
            assert trajectory_cost(trajectory, trajectory, policy.bootstrap_depth, times)['cost'] < 1e-12
            references[name] = (trajectory, policy.bootstrap_depth, times)
        # Uniform is a matched hard-selection control, not another trained arm.
        uniform = development(model, scorer, episodes['development'], args.work,
                              'uniform', args.deadline, mode='uniform')
        scorer.eval()
        development_started = time.time()
        initial = development(model, scorer, episodes['development'], args.work, 'initial', args.deadline)
        # Preserve time for final deployed-policy evidence instead of spending
        # the entire allocation on updates. The reserve uses measured dev cost.
        reserve = max(300., 2 * (time.time() - development_started))
        training_deadline = args.deadline - reserve
        pair_seconds = 0.
        for update in range(UPDATES):
            if time.time() + 1.5 * pair_seconds >= training_deadline:
                print('TASK POLICY TRAINING CAP', len(updates), 'reserving final development time', flush=True)
                break
            update_started = time.time()
            # One fixed shuffled schedule per epoch, no adaptive episode mining.
            epoch, slot = divmod(update, len(episodes['train']))
            order = np.random.default_rng(SEED + epoch).permutation(len(episodes['train']))
            name, inputs = episodes['train'][int(order[slot])]
            target = args.work / 'runs' / 'training' / f'update_{update + 1:04d}'
            optimizer.zero_grad(set_to_none=True)
            scorer.train()
            sampled, actor, sample_metrics = rollout(model, scorer, inputs,
                target / 'sampled', 'actor', args.deadline, sample=True, seed=SEED + update * 1009)
            scorer.eval()
            greedy, _, _ = rollout(model, scorer, inputs, target / 'greedy', 'actor', args.deadline)
            ref, depth, times = references[name]
            sampled_cost, greedy_cost = trajectory_cost(sampled, ref, depth, times), trajectory_cost(greedy, ref, depth, times)
            advantage = sampled_cost['cost'] - greedy_cost['cost']
            assert len(actor.log_probabilities) == FRAMES
            loss = advantage * torch.stack(actor.log_probabilities).mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None for p in model.parameters())
            norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1., error_if_nonfinite=True)
            assert any(p.grad is not None for p in scorer.parameters()), 'Disconnected actor gradient'
            optimizer.step()
            record = dict(update=update + 1, episode=name, sampled=sampled_cost, greedy=greedy_cost,
                advantage=advantage, loss=float(loss.detach()), gradient_norm=float(norm),
                sampled_metrics=sample_metrics)
            updates.append(record)
            write_json(target / 'update.json', record)
            write_json(args.work / 'training_updates.json', updates)
            checkpoint('latest', update + 1)
            print('TASK POLICY UPDATE OK', update + 1, name, json.dumps(record), flush=True)
            del actor, loss
            pair_seconds = time.time() - update_started
            if update + 1 == UPDATES // 2:
                scorer.eval()
                checkpoint('middle', update + 1)
                development(model, scorer, episodes['development'], args.work, 'middle', args.deadline)
        scorer.eval()
        checkpoint('final', len(updates))
        reloaded = PatchScorer('B').cuda().eval()
        reloaded.load_state_dict(torch.load(args.work / 'checkpoints/final.pt', weights_only=True)['state_dict'])
        with torch.no_grad():
            with np.load(args.work / 'runs' / 'development' / episodes['development'][0][0] /
                         'initial' / 'actor_inputs.npz') as saved:
                x = torch.from_numpy(saved['features'][:2]).to('cuda')
            torch.testing.assert_close(scorer(x), reloaded(x), rtol=0, atol=0)
        final = development(model, scorer, episodes['development'], args.work, 'final', args.deadline)
        clauses = {label: dict(fidelity=final['fidelity_cost'] < control['fidelity_cost'],
            translation_rms=final['rpe_translation_m'] < control['rpe_translation_m'],
            translation_p99=final['translation_p99_m'] <= control['translation_p99_m'],
            rotation=final['rpe_rotation_deg'] <= control['rpe_rotation_deg'],
            post_refresh=final['post_refresh_rms_m'] <= control['post_refresh_rms_m'])
            for label, control in (('initial', initial), ('uniform', uniform))}
        passed = all(all(values.values()) for values in clauses.values())
        decision.update(complete=True, pilot_pass=passed, clauses=clauses, updates=len(updates),
            training_update_cap_reached=len(updates) == UPDATES,
            final_development_reserved_seconds=reserve,
            speed_pass=None, memory_pass=None, objects_pass=None, eligible_for_deployment=False,
            next_action='Review pilot archive before any full evaluation' if passed else
                'Stop this configuration; no automatic sweeps or further training')
        (args.work / 'JOB_OK').write_text('Task-policy pilot completed; review decision, not a deployment claim.\n')
    except TimeoutError as error:
        decision.update(updates=len(updates), status='budget_incomplete', reason=str(error),
            next_action='Review partial evidence and compute cost; do not automatically resubmit')
        print('TASK POLICY BUDGET STOP', json.dumps(decision), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        write_json(args.work / 'task_policy_decision.json', decision)
    (args.work / 'PILOT_DONE').write_text('Pilot decision recorded; inspect complete and pilot_pass.\n')
    print('TASK POLICY PILOT DECISION', json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
