"""CAMP causal Pi3/StreamVGGT budget pilot; no offline layer selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'kv_tracker'))
from kvt_tum_run import prepare, write_json
from kvt_patch_select import Instrument, evaluate_object, evaluate_scene
from kv_tracker.online_layer_cache import OnlineLayerCache

SCENES = ('freiburg3_long_office_household', 'freiburg2_large_with_loop', 'freiburg2_large_no_loop')
OBJECTS = ('box_grab_01', 'ketchup_grab_01', 'espressomachine_grab_01')


def pi3_arm(config, result):
    import main as tracker
    from kvt_arctic_run import ArcticFrames, DATASET_DIR
    from kv_tracker.dataloaders.tum import TUMLoader
    os.chdir(ROOT / 'kv_tracker')
    if config['task'] == 'object':
        mask_dir = result / 'sam_masks'
        mask_dir.mkdir()
        class Frames(ArcticFrames):
            def get_segmentation(self, frame):
                import cv2
                original = Path(config['inputs']) / 'masks' / f'{self.index:06d}.png'
                binary = cv2.imread(str(original), cv2.IMREAD_GRAYSCALE)
                assert binary is not None
                shutil.copyfile(original, self.mask_dir / original.name)
                return torch.from_numpy(binary > 127).to('cuda:0')
            def __iter__(self):
                import cv2
                directory = Path(config['inputs']) / 'model_rgb'
                for frame in super().__iter__():
                    pixels = frame['resized_rgb_masked_np']
                    path = directory / f"{frame['idx']:06d}.png"
                    assert np.array_equal(cv2.imread(str(path))[:, :, ::-1], pixels)
                    yield frame
        source = Frames('cuda:0', mask_dir, scene_dir=DATASET_DIR / config['scene'],
                        offset=2, obj_mode=True, resize_dim=518)
        source.length = config['frames']
        args = ['--obj_mode', '--resize_dim', '518']
    else:
        class Frames(TUMLoader):
            def __iter__(self):
                manifest = json.loads((Path(config['inputs']) / 'manifest.json').read_text())
                for i in range(config['frames']):
                    frame = self.get_frame(i)
                    frame['idx'] = i
                    assert hashlib.sha256(frame['resized_rgb_masked_np'].tobytes()).hexdigest() == manifest['inputs'][i]['model_rgb_sha256']
                    yield frame
        source = Frames('cuda:0', scene_dir=Path(config['inputs']), offset=0,
                        obj_mode=False, resize_dim=308)
        args = ['--cam_only', '--resize_dim', '308', '--kf_auto', '50']
    selection = (None if config['policy'] == 'native' else
                 OnlineLayerCache(config['budget'], config['policy']))
    with (result / 'inference.jsonl').open('w') as log:
        instrument = Instrument(selection, log)
        tracker.run_track3r(cfg=dict(results_path=str(result), que_size=1), args=args,
                            frame_source=source, patch_cache=instrument)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - instrument.started
    if selection is not None:
        selection.close()
        write_json(result / 'allocation.json', selection.events)
        write_json(result / 'probes.json', selection.probes)
    (result / 'pcd.npy').unlink(missing_ok=True)
    quality = (evaluate_object(config['scene'], result, config['frames']) if config['task'] == 'object'
               else evaluate_scene(Path(config['inputs']), result, config['frames']))
    rows = [json.loads(line) for line in (result / 'inference.jsonl').read_text().splitlines()]
    metrics = dict(quality, tracking_seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   admission='unchanged native rule in each arm; closed loop, not replayed schedule')
    for kind in ('bootstrap', 'query', 'rebuild', 'select'):
        times = [r['seconds'] for r in rows if r['kind'] == kind]
        metrics[kind + '_calls'] = len(times)
        metrics[kind + '_seconds'] = sum(times)
        metrics[kind + '_p50_seconds'] = float(np.median(times)) if times else None
        metrics[kind + '_p95_seconds'] = float(np.quantile(times, .95)) if times else None
    metrics['max_query_cache_bytes'] = max(r['cache_bytes'] for r in rows if r['kind'] == 'query')
    physical = {(t.device, t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for layer in instrument.model.cache.values() for t in layer.values()}
    metrics['final_physical_kv_bytes'] = sum(physical.values())
    metrics['keyframe_indices'] = (np.load(result / 'kf_idx.npy').tolist()
                                  if (result / 'kf_idx.npy').exists() else [0])
    metrics['cache_accounting'] = 'query cache_bytes are logical K/V; final_physical_kv_bytes deduplicates backing storage'
    if selection is not None:
        metrics.update(budget_bytes=selection.limit, max_persistent_bytes=max(e['persistent_bytes'] for e in selection.events),
                       budget_binding=any(e['persistent_bytes'] == selection.limit for e in selection.events),
                       probe_count=len(selection.probes))
    else:
        metrics['max_persistent_bytes'] = sum(physical.values())
    return metrics


def stream_arm(config, result):
    from stream_cache_sweep import load_model, load_frame
    from stream_depth_ablation import step, quality
    from streamvggt.models.depth_cache import DepthCache, storage_bytes
    model = load_model(SimpleNamespace(host='streamvggt', checkpoint=config['checkpoint']))
    inputs = Path(config['inputs'])
    frames = json.loads((inputs / 'manifest.json').read_text())['frames'][:config['frames']]
    camera = [None] * model.camera_head.trunk_depth
    if config['policy'] == 'native':
        class Native:
            def __init__(self):
                self.bank = [None] * model.aggregator.depth
            def forward(self, image, frame):
                outputs, special, self.bank = model.aggregator(image, past_key_values=self.bank,
                                                              use_cache=True, past_frame_idx=frame)
                return outputs, special
        cache = Native()
    else:
        cache = DepthCache(model.aggregator, mode=('online' if config['policy'] == 'online' else 'budget_uniform'),
                           capacity=128, history_policy='anchor', frame_equivalents=config['budget'])
    torch.cuda.reset_peak_memory_stats()
    poses, events = [], []
    for i, row in enumerate(frames):
        image, _ = load_frame(inputs, row)
        torch.cuda.synchronize()
        start = time.perf_counter()
        pose, camera = step(model, cache, camera, image, i, True)
        torch.cuda.synchronize()
        query = time.perf_counter() - start
        start = time.perf_counter()
        if config['policy'] == 'native':
            event = dict(frame=i, persistent_bytes=storage_bytes(cache.bank))
        else:
            event = cache.retain(True)
        torch.cuda.synchronize()
        event.update(query_seconds=query, write_seconds=time.perf_counter()-start,
                     camera_bytes=storage_bytes(camera))
        poses.append(pose.cpu().numpy())
        events.append(event)
    poses = np.asarray(poses)
    np.save(result / 'traj.npy', poses)
    write_json(result / 'allocation.json', events)
    if config['policy'] != 'native':
        write_json(result / 'probes.json', cache.probes)
        cache.close()
    metrics = dict(quality(poses, frames, result, 64),
        max_persistent_bytes=max(e['persistent_bytes'] for e in events),
        max_camera_bytes=max(e['camera_bytes'] for e in events),
        query_seconds=sum(e['query_seconds'] for e in events),
        write_seconds=sum(e['write_seconds'] for e in events),
        query_plus_write_p50_seconds=float(np.median([e['query_seconds']+e['write_seconds'] for e in events[64:]])),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        admission='every frame; native untouched aggregator/camera cache; all geometry heads active')
    if config['policy'] != 'native':
        metrics.update(budget_bytes=events[-1]['budget_bytes'],
                       budget_binding=any(e['persistent_bytes']==e['budget_bytes'] for e in events),
                       probe_count=len(cache.probes))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', choices=('pi3', 'streamvggt'))
    parser.add_argument('--work', type=Path)
    parser.add_argument('--checkpoint')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    assert sys.platform.startswith('linux') and torch.cuda.is_available()
    torch.manual_seed(0)
    np.random.seed(0)
    if args.run:
        config = json.loads(args.run.read_text())
        with torch.inference_mode():
            metrics = (pi3_arm if args.host == 'pi3' else stream_arm)(config, args.run.parent)
        write_json(args.run.parent / 'metrics.json', metrics)
        print('ONLINE CACHE ARM OK', config['scene'], config['policy'], config['budget'], flush=True)
        return
    work = args.work
    inputs = work / 'inputs'
    inputs.mkdir()
    if args.host == 'pi3':
        from kvt_arctic_run import stage_dataset, OUT, DATASET_DIR
        with tarfile.open(OUT / 'prepared.tar') as packed:
            manifest = json.load(packed.extractfile('manifest.json'))
        stage_dataset(manifest)
        for scene in OBJECTS:
            shutil.copyfile(OUT / 'masks' / scene / 'init_mask.png', DATASET_DIR / scene / 'init_mask.png')
    records = []
    for task, scenes in ([('object', OBJECTS), ('scene', SCENES)] if args.host == 'pi3' else [('scene', SCENES)]):
        for scene in scenes:
            count = (min(1024, manifest['scenes'][scene]['tracking_frames']) if task == 'object' else
                     1024 if args.host == 'pi3' else 128)
            if task == 'object':
                subprocess.run([sys.executable, str(ROOT / 'tools/stream_cache_arctic.py'),
                    '--prepared', str(OUT / 'prepared.tar'),
                    '--kvt', str(OUT / 'arctic_r518_20260915T152911Z.tar'),
                    '--scene', scene, '--count', str(count), '--out', str(inputs / scene), '--object-gt'], check=True)
                import cv2
                from kv_tracker.image import pi3_resize_image
                directory = inputs / scene / 'model_rgb'
                directory.mkdir()
                rows = json.loads((inputs / scene / 'manifest.json').read_text())['frames']
                for i, row in enumerate(rows):
                    original = DATASET_DIR / scene / '0' / Path(row['source_member']).name
                    rgb = cv2.imread(str(original))[:, :, ::-1]
                    mask = cv2.imread(str(inputs / scene / row['mask']), cv2.IMREAD_GRAYSCALE) > 127
                    pixels = pi3_resize_image(rgb * mask[..., None], (518, 518))
                    assert cv2.imwrite(str(directory / f'{i:06d}.png'), pixels[:, :, ::-1])
            else:
                if args.host == 'pi3':
                    prepare(Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip', inputs / scene,
                            308, .02, count=count)
                else:
                    from stream_cache_sweep import prepare as stream_prepare
                    source = work / ('source_' + scene)
                    subprocess.run([sys.executable, str(ROOT / 'tools/stream_cache_tum.py'),
                        '--zip', f'/mnt/datasets/tum-rgbd/rgbd_dataset_{scene}.zip', '--out', str(source),
                        '--count', str(count)], check=True)
                    stream_prepare(SimpleNamespace(manifest=source / 'manifest.json', out=inputs / scene, width=308))
            cases = [('native', None)] + [(p, b) for b in ((3, 5) if task == 'object' else (4, 8, 16))
                                        for p in ('uniform', 'online')]
            for policy, budget in cases:
                name = 'native' if policy == 'native' else f'{policy}_{budget}'
                result = work / 'runs' / scene / name
                result.mkdir(parents=True)
                config = dict(host=args.host, task=task, scene=scene, frames=count, policy=policy,
                              budget=budget, inputs=str(inputs / scene),
                              checkpoint=args.checkpoint)
                write_json(result / 'config.json', config)
                with (result / 'run.log').open('w') as log:
                    subprocess.run([sys.executable, __file__, args.host, '--run', str(result / 'config.json')],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                metrics = json.loads((result / 'metrics.json').read_text())
                records.append(dict(config=config, metrics=metrics))
                write_json(work / 'summary.json', dict(complete=False, records=records))
                if policy == 'online':
                    prefix = result / 'prefix'
                    prefix.mkdir()
                    prefix_frames = 96 if args.host == 'streamvggt' else 128
                    write_json(prefix / 'config.json', dict(config, frames=prefix_frames))
                    with (prefix / 'run.log').open('w') as log:
                        subprocess.run([sys.executable, __file__, args.host, '--run', str(prefix / 'config.json')],
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    np.testing.assert_allclose(np.load(prefix / 'traj.npy'), np.load(result / 'traj.npy')[:prefix_frames],
                                               atol=1e-4, rtol=1e-4)
                    full_events = json.loads((result / 'allocation.json').read_text())
                    prefix_events = json.loads((prefix / 'allocation.json').read_text())
                    state = lambda e: {k: v for k, v in e.items() if k not in ('query_seconds', 'write_seconds', 'scores')}
                    assert [state(e) for e in prefix_events] == [state(e) for e in full_events if e['frame'] < prefix_frames]
                    for a, b in zip(prefix_events, full_events):
                        av = list(a['scores'].values()) if isinstance(a['scores'], dict) else a['scores']
                        bv = list(b['scores'].values()) if isinstance(b['scores'], dict) else b['scores']
                        np.testing.assert_allclose(av, bv, atol=1e-6, rtol=1e-5)
                    records[-1]['causal_prefix_pass'] = True
                print('ONLINE CACHE CASE OK', args.host, scene, name, metrics, flush=True)
    write_json(work / 'summary.json', dict(complete=True, records=records,
        limits='Causal local attention proxy, no offline teacher; native rebuilds unchanged; no automatic successor'))
    print('ONLINE CACHE PILOT COMPLETE', args.host, len(records), flush=True)


if __name__ == '__main__':
    main()
