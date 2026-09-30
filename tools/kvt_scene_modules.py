"""Separate CAMP-only graph and passive-routing pilots; no training or integration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from kvt_patch_select import CHECKOUT, TUM_ZIPS, MAX_GT_DIFFERENCE, archive_inputs, sha256, write_json
from kvt_tum_run import prepare


def graph_case(config, out):
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
    from kv_tracker.token_drop import forward_kept
    from kv_tracker.adaptive_tokens import keep_set
    from kv_tracker.graph_rope import GraphRoPE, load_graph_rope, cuRoPE2D
    bank, fraction = config['bank'], config['fraction']
    torch.manual_seed(17)
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained('cuda').eval())
    model.requires_grad_(False)
    grid, patches = (19, 25), 475
    count = math.ceil(patches * fraction)
    reference = torch.rand(1, bank, 3, 266, 350, device='cuda')
    images = [torch.rand(1, 1, 3, 266, 350, device='cuda') for _ in range(3)]
    keep = keep_set(torch.zeros(patches, dtype=torch.bool, device='cuda'), count, grid)[None]
    indices = [keep[0].nonzero().flatten()]
    indices += [torch.randperm(patches, device='cuda')[:count].sort().values for _ in range(2)]
    images.append(images[0])
    indices.append(indices[1])  # Same RGB, different selected patches at half budget.
    for index in indices:
        assert index.numel() == count and index.min() >= 0 and index.max() < patches
        assert (index[1:] > index[:-1]).all()
    host_images = [image.cpu().pin_memory() for image in images]
    host_indices = [index.cpu().pin_memory() for index in indices]
    image_buffer = images[0].clone()
    index_buffer = indices[0].clone()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        forward_kept(model, reference, keep.expand(bank, -1), store_cache=True, cam_only=True)
        cache = {i: {k: v.clone() for k, v in layer.items()} for i, layer in model.cache.items()}
        pointers = {i: {k: v.data_ptr() for k, v in layer.items()} for i, layer in model.cache.items()}
        labels = model.kept_cache['labels'].clone()
        expected = []
        expected_features = []
        for image, index in zip(images, indices):
            mask = torch.zeros_like(keep)
            mask[0, index] = True
            native = forward_kept(model, image, mask, cam_only=True, use_cache=True)['camera_poses']
            prepared = forward_kept(model, image, keep, cam_only=True, use_cache=True,
                                    query_indices=index)['camera_poses']
            torch.testing.assert_close(prepared, native, rtol=0, atol=0)
            features = forward_kept(model, image, keep, cam_only=True, use_cache=True,
                                    query_indices=index, defer_camera_head=True)['camera_features']
            with torch.amp.autocast(device_type='cuda', enabled=False):
                split = model.camera_head(features, *grid).reshape(1, 1, 4, 4)
            torch.testing.assert_close(split, native, rtol=0, atol=0)
            expected.append(native.clone())
            expected_features.append(features.clone())
        # Native eager references/cache above keep the saved Pi3 backend.
        # Only this fresh graph case opts into the otherwise identical kernel.
        original = model.rope
        assert isinstance(original, cuRoPE2D)
        backend_started = time.perf_counter()
        replacement = GraphRoPE(load_graph_rope(os.environ['KVT_GRAPH_ROPE_BUILD']),
                                freq=original.base, F0=original.F0)
        backend_load_seconds = time.perf_counter() - backend_started
        rope_modules = [module for module in model.modules()
                        if getattr(module, 'rope', None) is original]
        for module in rope_modules:
            module.rope = replacement
        for image, index, target in zip(images, indices, expected):
            adapted = forward_kept(model, image, keep, cam_only=True, use_cache=True,
                                   query_indices=index)['camera_poses']
            torch.testing.assert_close(adapted, target, rtol=0, atol=0)
        # CUDA capture requires warmup on a side stream. Retain all static buffers.
        torch.cuda.synchronize()
        setup_started = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                side_features = forward_kept(model, image_buffer, keep, cam_only=True,
                    use_cache=True, query_indices=index_buffer,
                    defer_camera_head=True)['camera_features']
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - setup_started
        with torch.amp.autocast(device_type='cuda', enabled=False):
            side_pose = model.camera_head(side_features, *grid).reshape(1, 1, 4, 4)
        side_check = dict(feature_max_abs=float((side_features - expected_features[0]).abs().max()),
                          pose_max_abs=float((side_pose - expected[0]).abs().max()))
        write_json(out / 'side_stream.json', side_check)
        print('SCENE GRAPH SIDE STREAM', json.dumps(side_check), flush=True)
        torch.testing.assert_close(side_pose, expected[0], rtol=1e-4, atol=1e-4)
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_features = forward_kept(model, image_buffer, keep, cam_only=True,
                use_cache=True, query_indices=index_buffer,
                defer_camera_head=True)['camera_features']
        torch.cuda.synchronize()
        capture_seconds = time.perf_counter() - started
        setup_seconds = warmup_seconds + capture_seconds + backend_load_seconds
        capture_peak = torch.cuda.max_memory_allocated()
        max_error = 0.
        fidelity = []
        for j, (image, index, target) in enumerate(zip(images, indices, expected)):
            image_buffer.copy_(image)
            index_buffer.copy_(index)
            graph.replay()
            # Preserve Pi3's head/SVD exactly; execute and charge it outside capture.
            with torch.amp.autocast(device_type='cuda', enabled=False):
                graph_output = model.camera_head(graph_features, *grid).reshape(1, 1, 4, 4)
            torch.cuda.synchronize()
            record = dict(input=j,
                feature_max_abs=float((graph_features - expected_features[j]).abs().max()),
                pose_max_abs=float((graph_output - target).abs().max()))
            fidelity.append(record)
            write_json(out / 'fidelity.json', fidelity)
            print('SCENE GRAPH FIDELITY', json.dumps(record), flush=True)
            torch.testing.assert_close(graph_output, target, rtol=1e-4, atol=1e-4)
            max_error = max(max_error, float((graph_output - target).abs().max()))
        assert not torch.equal(expected[0], expected[1]), 'Input variation must exercise outputs'
        if fraction < 1.:
            assert not torch.equal(expected[0], expected[-1]), 'Index variation must exercise outputs'
        times = {'native_eager': [], 'eager': [], 'graph': []}
        # Alternate order; charge pinned-host RGB and index uploads in all paths.
        # Selection/model-input preparation are excluded and explicitly reported.
        for repeat in range(12):
            j = repeat % len(images)
            modes = ('native_eager', 'eager', 'graph')
            for mode in (modes if repeat % 2 == 0 else reversed(modes)):
                for module in rope_modules:
                    module.rope = original if mode == 'native_eager' else replacement
                torch.cuda.synchronize()
                started = time.perf_counter()
                image_buffer.copy_(host_images[j], non_blocking=True)
                index_buffer.copy_(host_indices[j], non_blocking=True)
                if mode == 'graph':
                    graph.replay()
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        query_output = model.camera_head(graph_features, *grid).reshape(1, 1, 4, 4)
                else:
                    query_output = forward_kept(model, image_buffer, keep, cam_only=True,
                        use_cache=True, query_indices=index_buffer)['camera_poses']
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if repeat >= 2:
                    times[mode].append(elapsed)
                torch.testing.assert_close(query_output, expected[j], rtol=1e-4, atol=1e-4)
                if mode == 'graph':
                    max_error = max(max_error, float((query_output - expected[j]).abs().max()))
        for i, layer in cache.items():
            for key, value in layer.items():
                assert torch.equal(model.cache[i][key], value)
                assert model.cache[i][key].data_ptr() == pointers[i][key]
        assert torch.equal(labels, model.kept_cache['labels'])
    medians = {k: float(np.median(v)) for k, v in times.items()}
    savings = medians['native_eager'] - medians['graph']
    result = dict(config, seconds=times, median_seconds=medians,
        capture_seconds=capture_seconds, setup_seconds=setup_seconds,
        backend_load_seconds=backend_load_seconds,
        replay_max_abs_pose_error=max_error, capture_peak_bytes=capture_peak,
        graph_live_extra_bytes=torch.cuda.memory_allocated() - allocated_before,
        setup_break_even_queries=None if savings <= 0 else math.ceil(setup_seconds / savings),
        setup_break_even_reference='native_eager; excludes one-time job-local backend compilation reported in graph_rope_build/backend.json',
        cache_bytes=sum(v.numel()*v.element_size() for layer in model.cache.values() for v in layer.values()),
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        rope_backend='opt-in current-stream Pi3 kernel; exact native eager pose equality required',
        timing='pinned-host RGB/index uploads + query including unchanged eager camera head/SVD; excludes selector, image preparation, capture and rebuild',
        graph_scope='encoder, decoder and camera decoder; fp32 camera features returned to eager head',
        scope='fixed cache per graph; no automatic replay across rebuilds; not end-to-end speed gate')
    write_json(out / 'result.json', result)
    print('SCENE GRAPH CASE OK', json.dumps(result), flush=True)


def routing_case(config, out):
    import random
    import main as tracker
    from kv_tracker.dataloaders.tum import TUMLoader
    from kv_tracker.scene_routing import SceneRoutingObserver
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    staged = Path(config['inputs'])
    manifest = json.loads((staged / 'manifest.json').read_text())

    class Frames(TUMLoader):
        def __iter__(self):
            for index in range(config['frames']):
                frame = self.get_frame(index)
                frame['idx'] = index
                assert hashlib.sha256(frame['resized_rgb_masked_np'].tobytes()).hexdigest() == manifest['inputs'][index]['model_rgb_sha256']
                yield frame

    source = Frames('cuda:0', scene_dir=staged, obj_mode=False, resize_dim=308, offset=0)
    assert (source.height, source.width) == (480, 640)
    h, w, _ = manifest['inputs'][0]['shape']
    # Freiburg3 images are already undistorted. OpenCV linear resize pixel centers.
    k = torch.tensor([[535.4*w/640, 0, (320.1+.5)*w/640-.5],
                      [0, 539.2*h/480, (247.6+.5)*h/480-.5], [0, 0, 1]], device='cuda')
    observer = SceneRoutingObserver(out, k, h, w) if config['observe'] else None
    write_json(out / 'calibration.json', dict(K=k.cpu().tolist(), source_size=[480, 640],
        model_size=[h, w], source='TUM Freiburg3 RGB calibration; cv2 resize; no crop',
        pose='last estimated camera-to-world; canonical tracker gauge',
        sampling='source pixel centers 7::14; patch samples, not dense visibility'))
    tracker.run_track3r(cfg=dict(results_path=str(out), que_size=1),
        args=['--cam_only', '--resize_dim', '308', '--kf_auto', '50'],
        frame_source=source, scene_observer=observer)
    print('SCENE ROUTING RUN OK', config['name'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('graph', 'routing'))
    p.add_argument('--work', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--tag')
    p.add_argument('--case', type=Path)
    args = p.parse_args()
    assert sys.platform == 'linux', 'CAMP only'
    os.chdir(CHECKOUT)
    if args.case:
        config = json.loads(args.case.read_text())
        return (graph_case if config['mode'] == 'graph' else routing_case)(config, args.case.parent)
    snapshots = Path(os.environ['HF_HOME']) / 'hub' / 'models--yyfz233--Pi3' / 'snapshots'
    weights = list(snapshots.glob('**/*.safetensors')) + list(snapshots.glob('**/pytorch_model.bin'))
    assert weights, 'Pi3 weights must already be cached'
    sam = CHECKOUT / 'thirdparty/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt'
    write_json(args.work / 'module_protocol.json', dict(mode=args.mode,
        pi3_weights={str(w): sha256(w) for w in weights}, sam_sha256=sha256(sam),
        training=False, integration=False))
    root = args.work / 'runs' / args.mode
    root.mkdir(parents=True)
    if args.mode == 'graph':
        configs = [dict(mode='graph', name=f'bank{b}_fraction{f}', bank=b, fraction=f)
                   for b in (1, 10, 20) for f in (1., .5)]
    else:
        staged = args.work / 'inputs' / 'office'
        archive = TUM_ZIPS / 'rgbd_dataset_freiburg3_long_office_household.zip'
        prepare(archive, staged, 308, MAX_GT_DIFFERENCE, count=128)
        (staged / 'archive.sha256').write_text(f'{sha256(archive)}  {archive}\n')
        archive_inputs(staged, args.out / f'{args.tag}_routing_inputs.tar')
        configs = [dict(mode='routing', name=n, frames=f, observe=o, inputs=str(staged))
                   for n, f, o in (('native', 128, False), ('passive', 128, True), ('prefix', 64, True))]
    for config in configs:
        out = root / config['name']
        out.mkdir()
        path = out / 'config.json'
        write_json(path, config)
        with (out / 'run.log').open('w') as log:
            proc = subprocess.run([sys.executable, __file__, '--case', str(path)],
                                  stdout=log, stderr=subprocess.STDOUT)
        write_json(out / 'process.json', dict(returncode=proc.returncode))
        print((out / 'run.log').read_text()[-8000:], flush=True)
        assert proc.returncode == 0, f'{config["name"]} failed; wrapper archives partial runs'
    if args.mode == 'routing':
        native, passive, prefix = [root / n for n in ('native', 'passive', 'prefix')]
        for f in ('traj.npy', 'kf_idx.npy', 'kf_poses.npy'):
            np.testing.assert_array_equal(np.load(native / f), np.load(passive / f))
        np.testing.assert_allclose(np.load(passive / 'traj.npy')[:64],
                                   np.load(prefix / 'traj.npy'), rtol=1e-4, atol=1e-4)
        for path in sorted(prefix.glob('*.npz')):
            with np.load(path) as a, np.load(passive / path.name) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
        rows = json.loads((passive / 'routing.json').read_text())
        assert len(rows) == 127 and len(list(prefix.glob('route_*.npz'))) == 63
        write_json(args.work / 'routing_summary.json', dict(
            routing_median_seconds=float(np.median([r['routing_seconds'] for r in rows])),
            supported_fraction_range=[min(r['supported_fraction'] for r in rows),
                                      max(r['supported_fraction'] for r in rows)],
            latency='mapping only, synchronized; excludes artifact serialization; not end-to-end speed',
            decision='manual map inspection required; no selector benefit established'))
    else:
        results = [json.loads((root / c['name'] / 'result.json').read_text()) for c in configs]
        write_json(args.work / 'graph_summary.json', results)
    (args.work / 'JOB_OK').write_text(f'{args.mode} pilot contracts passed; manual review required\n')
    print('SCENE MODULE PILOT OK', args.mode, flush=True)


if __name__ == '__main__':
    main()
