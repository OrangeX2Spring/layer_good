"""CAMP-only ARCTIC reanchor transfer and controlled object-coverage comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_tum_run import write_json
from kvt_tum_sweep import archive_directory, release_page_cache

SCENES = ('box_grab_01', 'ketchup_grab_01', 'espressomachine_grab_01')


def replay_native(config_path):
    """Original admission with saved model pixels; object masks exclude live SAM."""
    import cv2
    import torch
    import main as tracker
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
    from kvt_patch_select import evaluate_object
    from kvt_tum_run import CHECKOUT

    config = json.loads(config_path.read_text())
    inputs, result = Path(config['inputs']), config_path.parent
    manifest = json.loads((inputs / 'manifest.json').read_text())
    camera = config.get('task') == 'camera'
    resize = 308 if camera else 518
    if camera:
        assert manifest['resize_dim'] == resize
    os.chdir(CHECKOUT)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained('cuda:0').eval())
    if config.get('profile_forward', False):
        from kvt_inference_timing import InferenceTiming
        forward_timing = InferenceTiming(model)
    timing = {}

    def frames():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for index, row in enumerate(manifest['inputs']):
            bgr = cv2.imread(str(inputs / 'model_rgb' / Path(row['file']).name))
            assert bgr is not None
            rgb = bgr[:, :, ::-1].copy()
            if camera:
                mask = np.ones(rgb.shape[:2], dtype=bool)
            else:
                mask = cv2.imread(str(inputs / 'model_masks' / row['file']), cv2.IMREAD_GRAYSCALE)
                assert mask is not None
                mask = mask > 127
            assert mask.shape == rgb.shape[:2] and mask.any()
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['model_rgb_sha256']
            if not camera:
                assert hashlib.sha256(mask.tobytes()).hexdigest() == row['model_mask_sha256']
            yield dict(idx=index, rgb_np=rgb, resized_rgb_masked_np=rgb,
                       resized_mask_np=mask, resized_mask=torch.tensor(mask, device='cuda:0'),
                       resized_rgb_masked=torch.tensor(rgb, device='cuda:0', dtype=torch.float32)[None, None] / 255.)
        torch.cuda.synchronize()
        timing.update(tracking_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())

    tracker.run_track3r(cfg=dict(results_path=str(result), que_size=1),
        args=['--cam_only' if camera else '--obj_mode', '--resize_dim', str(resize)],
        frame_source=frames(), pi3_model=model)
    trajectory = np.load(result / 'traj.npy')
    assert trajectory.shape == (manifest['frames'], 4, 4) and np.isfinite(trajectory).all()
    # Saved pixels must reproduce the historical native trajectory before comparison.
    np.testing.assert_allclose(trajectory, np.load(config['reference_trajectory']), rtol=1e-4, atol=1e-4)
    timing['includes'] = 'saved pixels/masks, hashing, upload and tracking; excludes SAM/model load/final serialization'
    if camera:
        from kvt_tum_run import evaluate
        metrics = evaluate(inputs, result, .02)
        with np.load(result / 'evaluation.npz') as pairs:
            metrics['translation_p99_m'] = float(np.quantile(pairs['rpe_translation_per_pair_m'], .99))
    else:
        metrics = evaluate_object(config['scene'], result, manifest['frames'])
    write_json(result / 'summary.json', dict(metrics=metrics, timing=timing, native_fidelity=True))
    if config.get('profile_forward', False):
        write_json(result / 'forward_timing.json', forward_timing.finish())
        from kvt_inference_timing import benchmark_cached_heads
        last = cv2.imread(str(inputs / 'model_rgb' / Path(manifest['inputs'][-1]['file']).name))
        write_json(result / 'cached_head_benchmark.json', benchmark_cached_heads(model, last[:, :, ::-1]))


def main(args):
    import cv2
    from kv_tracker.image import pi3_resize_image
    from kvt_arctic_run import DATASET_DIR, OUT as ARCTIC_OUT
    from kvt_patch_select import evaluate_object

    tools = Path(__file__).parent
    summary = args.work / 'runs' / ('segment_arctic_object_comparison' if args.object_coverage
                                   else 'segment_arctic_comparison')
    summary.mkdir(parents=True)
    policy = dict(detector='orb', minimum_inliers=25, maximum_segment_frames=200)
    write_json(summary / 'policy.json', policy)
    records = {}
    for scene in args.scenes:
        name = f'{args.tag}_segment_native_{scene}'
        # Original object tracker, angular admission and live SAM, fresh process.
        completed = subprocess.run([sys.executable, str(tools / 'kvt_arctic_run.py'),
            '--results', name, '--scenes', scene, '--resize-dim', '518', '--no-viz'])
        native = DATASET_DIR / scene / name
        if completed.returncode:
            if native.exists():
                archive_directory(native, args.out / f'{args.tag}_arctic_native_{scene}_partial.tar')
            completed.check_returncode()
        metrics = evaluate_object(scene, native, len(np.load(native / 'traj.npy')))
        write_json(native / 'segment_metrics.json', metrics)
        evidence = summary / scene
        evidence.mkdir()
        write_json(evidence / 'native_metrics.json', metrics)
        shutil.copyfile(native / 'evaluation.npz', evidence / 'native_evaluation.npz')
        # Native archive already preserves RGB source/masks, timing and revisions.
        packed_paths = list(ARCTIC_OUT.glob(f'arctic_{name}_*.tar'))
        assert len(packed_paths) == 1
        with tarfile.open(packed_paths[0]) as packed:
            timing = json.load(packed.extractfile('metrics.json'))['timings'][scene]
        prepared = json.loads((ARCTIC_OUT / 'initial_frames' / 'manifest.json').read_text())
        frames = prepared['scenes'][scene]['tracking_frames']
        assert metrics['evaluated_frames'] == frames
        inputs = args.work / 'inputs' / scene
        (inputs / 'model_rgb').mkdir(parents=True)
        if args.object_coverage:
            (inputs / 'model_masks').mkdir()
        rows = []
        paths = sorted((DATASET_DIR / scene / '0').glob('*.jpg'))[2:]
        assert len(paths) == frames
        for i, path in enumerate(paths):
            image = cv2.imread(str(path))
            mask_path = native / 'sam_masks' / f'{i:05d}.png'
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            assert image is not None and mask is not None
            assert mask.shape == image.shape[:2] and (mask > 127).any()
            pixels = pi3_resize_image(image[:, :, ::-1] * (mask > 127)[..., None], (518, 518))
            filename = f'{i:06d}.png'
            assert cv2.imwrite(str(inputs / 'model_rgb' / filename), pixels[:, :, ::-1])
            rows.append(dict(file=filename, model_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                             source=str(path), source_sha256=sha256(path), mask_sha256=sha256(mask_path)))
            if args.object_coverage:
                model_mask = pi3_resize_image(mask > 127, (518, 518))
                assert model_mask.shape == pixels.shape[:2] and model_mask.any()
                assert cv2.imwrite(str(inputs / 'model_masks' / filename), model_mask.astype(np.uint8) * 255)
                rows[-1]['model_mask_sha256'] = hashlib.sha256(model_mask.tobytes()).hexdigest()
        write_json(inputs / 'manifest.json', dict(inputs=rows, frames=frames, scene=scene,
            resize_dim=518, offset=2, masks='fresh native SAM masks; reused unchanged',
            native_archive=str(packed_paths[0]), native_archive_sha256=sha256(packed_paths[0]),
            prepared_archive_sha256=sha256(ARCTIC_OUT / 'prepared.tar')))
        archive_directory(inputs, args.out / f'{args.tag}_segment_arctic_{scene}_inputs.tar')
        release_page_cache(inputs, args.out / f'{args.tag}_segment_arctic_{scene}_inputs.tar')
        if args.object_coverage:
            replay = args.work / 'runs' / f'arctic_native_replay_{scene}'
            replay.mkdir()
            write_json(replay / 'config.json', dict(inputs=str(inputs), scene=scene,
                reference_trajectory=str((native / 'traj.npy').resolve())))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--native-replay',
                            str(replay / 'config.json')], check=True)
            archive_directory(replay, args.out / f'{args.tag}_arctic_native_replay_{scene}.tar')
            baseline = json.loads((replay / 'summary.json').read_text())
            with np.load(native / 'evaluation.npz') as n, np.load(replay / 'evaluation.npz') as r:
                np.testing.assert_array_equal(n['pair_end_frames'], r['pair_end_frames'])
            object_runs = {}
            for coverage in ('image', 'anchor_support'):
                selected_policy = dict(policy, coverage_mode=coverage, minimum_anchor_coverage=.5)
                policy_path = evidence / f'policy_{coverage}.json'
                write_json(policy_path, selected_policy)
                name = f'arctic_reanchor_{coverage}_{scene}'
                subprocess.run([sys.executable, str(tools / 'kvt_reanchor_maps.py'),
                    '--work', str(args.work), '--out', str(args.out), '--tag', args.tag,
                    '--inputs', str(inputs), '--native-metrics', str(native / 'segment_metrics.json'),
                    '--detector-config', str(policy_path), '--object-scene', scene,
                    '--name', name], check=True)
                result = args.work / 'runs' / name
                run = json.loads((result / 'summary.json').read_text())
                trajectory = np.load(result / 'traj.npy')
                assert trajectory.shape == (frames, 4, 4) and np.isfinite(trajectory).all()
                with np.load(replay / 'evaluation.npz') as n, np.load(result / 'evaluation.npz') as r:
                    np.testing.assert_array_equal(n['pair_end_frames'], r['pair_end_frames'])
                ratios = {key: run['metrics'][key] / baseline['metrics'][key]
                          for key in ('ate_m', 'rpe_translation_m', 'translation_p99_m')}
                decisions = [json.loads(line) for line in
                             (result / 'segmentation_live.jsonl').read_text().splitlines()]
                cuts = [row['frame'] for row in decisions if row['boundary']]
                object_runs[coverage] = dict(run=run, policy=selected_policy,
                    cut_frames=cuts,
                    native_ratios=ratios, quality_pass=all(v <= 1.05 for v in ratios.values()),
                    observed_speedup=baseline['timing']['tracking_seconds'] / run['timing']['tracking_seconds'])
                records[scene] = dict(native_replay=baseline, reference_native=metrics,
                    native_archive=str(packed_paths[0]), arms=object_runs, frames=frames)
                write_json(summary / 'comparison.json', dict(records=records, complete=False))
                archive_directory(summary, args.out / f'{args.tag}_segment_arctic_object_comparison.tar')
                print('OBJECT COVERAGE COMPARE', scene, coverage, ratios, 'cuts', cuts, flush=True)
            continue
        reanchor_name = f'arctic_reanchor_{scene}'
        subprocess.run([sys.executable, str(tools / 'kvt_reanchor_maps.py'),
            '--work', str(args.work), '--out', str(args.out), '--tag', args.tag,
            '--inputs', str(inputs), '--native-metrics', str(native / 'segment_metrics.json'),
            '--detector-config', str(summary / 'policy.json'), '--object-scene', scene,
            '--name', reanchor_name], check=True)
        result = args.work / 'runs' / reanchor_name
        run = json.loads((result / 'summary.json').read_text())
        assert np.load(result / 'traj.npy').shape == (frames, 4, 4)
        with np.load(native / 'evaluation.npz') as n, np.load(result / 'evaluation.npz') as r:
            np.testing.assert_array_equal(n['pair_end_frames'], r['pair_end_frames'])
        record = dict(native=metrics, native_timing=timing, reanchor=run,
            native_archive=str(packed_paths[0]), frames=frames,
            ratios={k: run['metrics'][k] / metrics[k]
                    for k in ('ate_m', 'rpe_translation_m', 'translation_p99_m')})
        records[scene] = record
        write_json(summary / 'comparison.json', dict(records=records, policy=policy, complete=False))
        archive_directory(summary, args.out / f'{args.tag}_segment_arctic_comparison.tar')
        print('SEGMENT ARCTIC COMPARE', scene, 'ratios', record['ratios'],
              'native seconds incl SAM', timing['seconds'],
              'reanchor seconds excl SAM', run['timing']['tracking_seconds'], flush=True)
    if args.object_coverage:
        write_json(summary / 'comparison.json', dict(records=records, complete=True,
            gt_used_by_tracker=False, quality_margin=1.05,
            timing='All compared arms exclude SAM and model load; saved input I/O, hashing, uploads and tracking included. '
                   'Native also reads masks; reanchor includes detector/bridge/logging. One ordered observation; no statistical speed claim.',
            changed='Anchor-relative coverage only; same ORB matching, absolute inlier threshold, age cap and handoff.',
            semantic_kv='Future research; no semantic labels or cache selection in these arms.'))
        archive_directory(summary, args.out / f'{args.tag}_segment_arctic_object_comparison.tar')
        (args.work / 'JOB_OK').write_text('Object coverage comparison completed; inspect quality before any successor\n')
        print('OBJECT COVERAGE COMPLETE', flush=True)
        return
    write_json(summary / 'comparison.json', dict(records=records, policy=policy, complete=True,
        gt_used_by_tracker=False, offset=2, resize_dim=518,
        timing_caveat='native includes live SAM/mask export; reanchor reuses masks and excludes SAM; '
                      'no direct speedup verdict',
        mechanism_caveat='existing scene reanchor operates on masked RGB; shared-point geometry '
                         'uses confidence filtering, not an explicit foreground mask; '
                         'native object angular admission differs from TUM cap20 scheduling',
        accuracy='same upstream object GT and full-run Sim(3); adjacent-pair RPE/p99'))
    archive_directory(summary, args.out / f'{args.tag}_segment_arctic_comparison.tar')
    (args.work / 'JOB_OK').write_text('Three ARCTIC comparisons completed; inspect accuracy and timing scopes\n')
    print('SEGMENT ARCTIC COMPLETE: 3 sequences, native + ORB25/max200 reanchor', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    parser.add_argument('--object-coverage', action='store_true')
    parser.add_argument('--native-replay', type=Path)
    args = parser.parse_args()
    if args.native_replay:
        replay_native(args.native_replay)
    else:
        if args.work is None or args.out is None or args.tag is None:
            parser.error('--work, --out and --tag are required for comparisons')
        main(args)
