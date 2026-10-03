"""One office handoff, three arms, fresh processes. Execute on CAMP only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from kvt_tum_run import prepare, write_json, evaluate
from kvt_tum_sweep import archive_inputs, archive_directory


def arm(config_path):
    import cv2
    import torch
    from kv_tracker.map_handoff import MapHandoff
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
    config = json.loads(config_path.read_text())
    result, inputs = config_path.parent, Path(config['scene_dir'])
    manifest = json.loads((inputs / 'manifest.json').read_text())
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())
    executor = None
    if config.get('depth_contrast') is not None:
        assert not config.get('query_method')
        from kv_tracker.depth_contrast import DepthContrastQueries
        executor = DepthContrastQueries(**config['depth_contrast'])
    if config.get('query_method'):
        import os
        from kv_tracker.query_acceleration import QueryAcceleration
        executor = QueryAcceleration(config['query_method'],
            os.environ.get('KVT_GRAPH_ROPE_BUILD'), config.get('query_checked', False))
    rows = []

    def log(row):
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()

    def save_bridge(event, evidence):
        write_json(result / 'bridge_attempt.json', event)
        np.savez(result / 'bridge_evidence.npz', **evidence)

    with (result / 'inference.jsonl').open('w') as stream, torch.inference_mode():
        tracker = MapHandoff(model, config['mode'], log, save_bridge, query_executor=executor)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        poses = []
        for i, row in enumerate(manifest['inputs'][:config['frames']]):
            pixels = cv2.imread(str(inputs / 'model_rgb' / Path(row['file']).name))
            assert pixels is not None
            pixels = pixels[:, :, ::-1].copy()
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == row['model_rgb_sha256']
            poses.append(tracker.bootstrap(pixels) if i == 0 else tracker.step(pixels, i))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        np.save(result / 'traj.npy', np.asarray(poses))
        np.save(result / 'kf_idx.npy', np.asarray(tracker.ids))
        write_json(result / 'handoff_events.json', tracker.events)
        write_json(result / 'final_transform.json', dict(scale=float(tracker.transform[0]),
            rotation=tracker.transform[1].tolist(), translation=tracker.transform[2].tolist()))
        metrics = dict(frames=len(poses), tracking_seconds=seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            max_cache_bytes=max(r['cache_bytes'] for r in rows),
            final_image_bytes=sum(x.nbytes for x in tracker.images),
            final_geometry_bytes=sum(x.numel() * x.element_size()
                                     for x in (tracker.latest_points, tracker.latest_conf)),
            accepted=bool(tracker.events and tracker.events[0]['accepted']),
            timing='model load/staging excluded; pixel loading, hashing, geometry copies, bridge and logging included')
        for kind in ('bootstrap', 'query', 'rebuild', 'candidate', 'update_total'):
            values = [r['seconds'] for r in rows if r['kind'] == kind]
            metrics[kind + '_calls'] = len(values)
            metrics[kind + '_seconds'] = sum(values)
        metrics['timing_warning'] = 'update_total includes candidate/rebuild; do not add these twice'
        if config['evaluate']:
            metrics.update(evaluate(inputs, result, .02))
            with np.load(result / 'evaluation.npz') as quality:
                metrics['translation_p99_m'] = float(np.quantile(quality['rpe_translation_per_pair_m'], .99))
                transitions = []
                for boundary in ((749, 799, 849, 899, 949)
                                 if config.get('transition_windows', True) else ()):
                    starts = quality['rpe_pair_start_indices']
                    window = (starts >= boundary - 5) & (starts <= boundary + 5)
                    translations = quality['rpe_translation_per_pair_m'][window]
                    rotations = quality['rpe_rotation_per_pair_deg'][window]
                    assert len(translations), 'No valid GT pairs in transition window'
                    transitions.append(dict(boundary=boundary,
                        pair_start_indices=starts[window].tolist(),
                        translation_per_pair_m=translations.tolist(),
                        rotation_per_pair_deg=rotations.tolist(),
                        translation_rmse_m=float(np.sqrt(np.mean(translations ** 2))),
                        translation_max_m=float(translations.max())))
                write_json(result / 'transition_errors.json', dict(
                    alignment='same full-window Sim(3); no transition refit', windows=transitions))
        write_json(result / 'metrics.json', metrics)
        if executor is not None:
            write_json(result / 'query_execution.json', executor.summary())
        write_json(result / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), seed=0, resize_dim=308))


def main(args):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    checkpoint = {}
    for path in sorted(snapshot.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            checkpoint[str(path.relative_to(snapshot))] = digest.hexdigest()
    assert checkpoint
    write_json(args.work / 'checkpoint.json', dict(snapshot=str(snapshot), files=checkpoint))
    scene = 'freiburg3_long_office_household'
    inputs = args.work / 'inputs' / scene
    archive = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
    prepare(archive, inputs, 308, .02, count=975)
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    (inputs / 'archive.sha256').write_text(f'{digest.hexdigest()}  {archive}\n')
    archive_inputs(inputs, args.out / f'{args.tag}_handoff_inputs.tar')
    summary, run_paths = {}, {}

    def run(name, mode, frames, reference=False):
        result = args.work / 'runs' / name
        result.mkdir(parents=True)
        config = dict(name=name, mode=mode, scene_dir=str(inputs), frames=frames,
                      evaluate=frames == 975)
        if reference:
            config.update(scene=scene, policy='original', interval=50, cap=20,
                          resize_dim=308, max_gt_difference=.02, evaluate_trajectory=False,
                          save_final_scene=False)
            if mode == 'fixed':
                config.update(policy='combined', cap=2, patch_fraction=1.)
        write_json(result / 'config.json', config)
        command = ([sys.executable, str(Path(__file__).with_name('kvt_tum_run.py')),
                    str(result / 'config.json')] if reference else
                   [sys.executable, str(Path(__file__)), '--run', str(result / 'config.json')])
        print('RUN', name, flush=True)
        with (result / 'run.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        write_json(result / 'process.json', dict(returncode=completed.returncode))
        archive_directory(result, args.out / f'{args.tag}_handoff_{name}.tar')
        if completed.returncode:
            print((result / 'run.log').read_text()[-8000:], flush=True)
            raise RuntimeError(f'{name} failed with exit {completed.returncode}')
        poses = np.load(result / 'traj.npy')
        assert poses.shape == (frames, 4, 4) and np.isfinite(poses).all()
        if not reference:
            rows = [json.loads(x) for x in (result / 'inference.jsonl').read_text().splitlines()]
            assert len([r for r in rows if r['kind'] == 'query']) == frames - 1
            cap = 20 if mode == 'native' else 2
            assert all(r['input_images'] <= cap for r in rows if 'input_images' in r)
            assert all(len(r['bank_ids']) <= cap for r in rows if 'bank_ids' in r)
            expected = list(range(49, frames, 50))
            assert [r['frame'] for r in rows if r['kind'] == 'update_total'] == expected
        run_paths[name] = result
        return poses

    references = {mode: run('fidelity_reference_' + mode, mode, 128, reference=True)
                  for mode in ('native', 'fixed')}
    for mode in ('native', 'fixed', 'handoff'):
        poses = run('fidelity_' + mode, mode, 128)
        # CombinedCache with cap2/full patches has only one eligible victim:
        # the previous non-anchor frame. It supplies the existing tracker oracle.
        if mode == 'native':
            np.testing.assert_allclose(poses, references['native'], atol=1e-5, rtol=1e-4)
        else:
            np.testing.assert_allclose(poses, references['fixed'], atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(np.load(run_paths['fidelity_fixed'] / 'traj.npy'),
                                  np.load(run_paths['fidelity_handoff'] / 'traj.npy'))
    print('HANDOFF NATIVE FIDELITY OK', flush=True)
    for mode in ('native', 'fixed', 'handoff'):
        run(mode, mode, 975)
        summary[mode] = json.loads((run_paths[mode] / 'metrics.json').read_text())
    with np.load(run_paths['native'] / 'evaluation.npz') as reference:
        for mode in ('fixed', 'handoff'):
            with np.load(run_paths[mode] / 'evaluation.npz') as quality:
                for key in ('rgb_indices', 'gt_indices', 'rpe_pair_start_indices'):
                    np.testing.assert_array_equal(quality[key], reference[key])
    fixed = np.load(run_paths['fixed'] / 'traj.npy')
    handoff = np.load(run_paths['handoff'] / 'traj.npy')
    np.testing.assert_array_equal(handoff[:750], fixed[:750])
    events = json.loads((run_paths['handoff'] / 'handoff_events.json').read_text())
    assert len(events) == 1 and events[0]['frame'] == 749
    event = events[0]
    assert event['old_ids'] == [0, 699] and event['candidate_ids'] == [699, 749]
    if event['accepted']:
        assert event['old_kv_released'] and event['retained_ids'] == [699, 749]
    else:
        assert not event['old_kv_released'] and event['retained_ids'] == [0, 749]
        np.testing.assert_array_equal(handoff, fixed)
    query_rows = [json.loads(x) for x in (run_paths['handoff'] / 'inference.jsonl').read_text().splitlines()]
    assert all(r['bank_ids'][0] == (699 if event['accepted'] else 0)
               for r in query_rows if r['kind'] == 'query' and r['frame'] >= 750)
    assert event['dual_cache_bytes'] == 2 * summary['fixed']['max_cache_bytes']
    for mode in ('fixed', 'handoff'):
        prefix = run(mode + '_prefix', mode, 800)
        np.testing.assert_array_equal(prefix, np.load(run_paths[mode] / 'traj.npy')[:800])
        assert json.loads((run_paths[mode + '_prefix'] / 'handoff_events.json').read_text()) == json.loads(
            (run_paths[mode] / 'handoff_events.json').read_text())
    for mode in ('fixed', 'handoff'):
        summary[mode]['within_native_5pct'] = all(summary[mode][key] <= 1.05 * summary['native'][key]
            for key in ('ate_m', 'rpe_translation_m', 'translation_p99_m'))
    summary['handoff_event'] = event
    summary['interpretation'] = ('Connection committed; review continuous quality and transition costs'
                                 if event['accepted'] else 'Connection rejected; this is not a successful reanchor')
    write_json(args.work / 'handoff_summary.json', summary)
    write_json(args.work / 'handoff_gates.json', dict(native_and_fixed_fidelity=True,
        pre_handoff_identity=True, causal_prefixes=True, bridge_attempts=1,
        accepted=event['accepted'], scientific_acceptance_pending=True))
    hashes = {}
    for path in sorted(args.out.glob(f'{args.tag}_handoff_*.tar')):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    write_json(args.work / 'handoff_archives.json', hashes)
    (args.work / 'JOB_OK').write_text('Execution/contracts passed; bridge acceptance and accuracy separate\n')
    print('HANDOFF PILOT COMPLETE: 3 arms; bridge accepted=' + str(event['accepted']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.run:
        arm(args.run)
    else:
        assert args.work is not None and args.out is not None and args.tag
        main(args)
