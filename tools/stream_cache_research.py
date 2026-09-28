"""StreamVGGT research conditions; called by the shared archived sweep driver."""

from dataclasses import asdict
from collections import Counter
import json
import time

import numpy as np
import torch

from stream_cache_adapters import StreamAdapter, tensor_bytes


def research_fidelity(args, model, root, frames, target):
    """New ragged path must reproduce the established all-kept adapter first."""
    from streamvggt.models.research_cache import ResearchCache, ResearchConfig
    from stream_cache_sweep import load_frame, cpu_predictions, write_json

    count = min(max(args.gate_frames, 6), len(frames))
    budget = 3  # Exercise eviction as well as the initial append-only prefix.
    adapter = StreamAdapter(model, 'streamvggt')
    reference = []
    with torch.no_grad():
        for index, frame in enumerate(frames[:count]):
            image, _ = load_frame(root, frame)
            reference.append(cpu_predictions(adapter.forward(image, index)))
            kept = list(range(index + 1)) if index < budget else [0, *range(index - budget + 2, index + 1)]
            adapter.prune(torch.arange(adapter.features.shape[0]), kept)
    adapter.close()
    del adapter, image
    session = ResearchCache(model.aggregator, ResearchConfig(frame_budget=budget,
                                                           refresh_frames=1))
    camera = [None] * model.camera_head.trunk_depth
    worst = {}
    with torch.no_grad():
        for index, frame in enumerate(frames[:count]):
            image, _ = load_frame(root, frame)
            tokens = session.forward(image, index)
            output, camera = heads(model, tokens, image, camera)
            for key, value in cpu_predictions(output).items():
                error = float((value - reference[index][key]).abs().max())
                worst[key] = max(worst.get(key, 0.), error)
                torch.testing.assert_close(value, reference[index][key],
                                           atol=args.gate_atol, rtol=args.gate_rtol)
            session.maintain(index)
            del tokens, image, output
    write_json(target / 'research_fidelity.json', dict(frames=count, frame_budget=budget, max_abs=worst,
               atol=args.gate_atol, rtol=args.gate_rtol,
               scope='Disabled intervention only; does not validate merged or refreshed representations'))
    print('RESEARCH FIDELITY OK', flush=True)


def evaluate_object(args, manifest):
    """Masked-input camera-in-object readout; never label this world-camera GT."""
    from stream3r_arctic_eval import evaluate
    from stream_cache_sweep import write_json, digest

    assert manifest['mask_input'] and manifest['pose_reference'] == 'camera-in-object'
    frames = manifest['frames']
    assert all(row['gt_row'] == row['kvt_index'] + 2 for row in frames)
    assert all(b['kvt_index'] == a['kvt_index'] + 1 for a, b in zip(frames, frames[1:]))
    for name, checksum in manifest['object_gt_files'].items():
        assert digest(args.out / 'inputs' / name) == checksum
    gt = np.asarray([row['gt_object_c2w'] for row in frames], dtype=np.float64)
    reports = {}
    for condition in json.loads((args.out / 'sweep.json').read_text())['conditions']:
        name = condition['name']
        extrinsics = np.load(args.out / name / 'camera.npz')['extrinsics']
        poses = np.tile(np.eye(4), (len(frames), 1, 1))
        assert extrinsics.shape == (len(frames), 3, 4)
        poses[:, :3] = extrinsics
        reports[name] = evaluate(np.linalg.inv(poses), gt)
    write_json(args.out / 'object_metrics.json', dict(
        pose_reference='camera-in-object', rows=reports,
        caveat='Masked pose-head readout; no world-camera or reconstruction-accuracy claim. '
               'One full-prefix Sim(3); adjacent frames; rotation Frobenius norm, not degrees.'))
    print('OBJECT EVALUATION OK', flush=True)


def verify_research_run(args, configs, frame_count):
    """Cross-condition checks; passing contracts does not pass the scientific gate."""
    from stream_cache_sweep import write_json

    events = {}
    reports = {}
    for name, config in configs:
        if config.research is None:
            continue
        rows = [json.loads(line) for line in (args.out / name / 'events.jsonl').read_text().splitlines()]
        assert len(rows) == frame_count
        for index, row in enumerate(rows):
            assert row['frame'] == index
            assert row['retained_frames'] == sorted(set(row['retained_frames']))
            assert row['retained_frames'][0] == 0 and row['retained_frames'][-1] == index
            assert row['total_persistent_bytes'] == (row['state_bytes'] + row['camera_bytes']
                                                   + row.get('diagnostic_state_bytes', 0))
            if row['state_budget_bytes'] is not None:
                assert row['state_bytes'] <= row['state_budget_bytes']
        events[name] = rows
        refreshed_counts = Counter(key for row in rows for key in row['refreshed_frames'])
        reports[name] = dict(mean_state_bytes=float(np.mean([row['state_bytes'] for row in rows])),
                             max_state_bytes=max(row['state_bytes'] for row in rows),
                             refreshed_groups=sum(refreshed_counts.values()),
                             refreshed_frame_counts=dict(sorted(refreshed_counts.items())),
                             actual_aggregator_dtypes=sorted({row['aggregator_dtype'] for row in rows}))
    if 'spread64' in events:
        for field in ('processed_patches', 'aggregator_bytes', 'state_bytes'):
            expected = [row[field] for row in events['spread64']]
            for name in ('spatial64', 'appearance64'):
                assert expected == [row[field] for row in events[name]], (name, field)
        assert any(row['processed_patches'] < row['dense_patches'] for row in events['spread64'])
    if 'spatial64_mass' in events:
        for plain, corrected in (('dense8', 'dense8_mass'), ('spatial64', 'spatial64_mass')):
            for a, b in zip(events[plain], events[corrected]):
                for field in ('dense_to_sparse', 'representative_positions', 'token_mass',
                              'processed_patches', 'retained_frames', 'patches_per_frame',
                              'aggregator_bytes', 'state_bytes', 'camera_bytes'):
                    assert a[field] == b[field], (plain, field)
                special = len(a['token_mass']) - a['processed_patches']
                assert a['token_mass'][:special] == [1.] * special
                assert sum(a['token_mass']) == a['dense_patches'] + special
                assert min(a['token_mass']) >= 1
        assert any(max(row['token_mass']) > 1 for row in events['spatial64_mass'])
        with np.load(args.out / 'dense8/pose_encodings.npz') as plain, \
                np.load(args.out / 'dense8_mass/pose_encodings.npz') as corrected:
            np.testing.assert_allclose(plain['pose_enc'], corrected['pose_enc'],
                                       atol=args.gate_atol, rtol=args.gate_rtol)
        with np.load(args.out / 'dense8' / f'frame_{frame_count - 1:06d}.npz') as plain, \
                np.load(args.out / 'dense8_mass' / f'frame_{frame_count - 1:06d}.npz') as corrected:
            for key in plain.files:
                np.testing.assert_allclose(plain[key], corrected[key],
                                           atol=args.gate_atol, rtol=args.gate_rtol)
        print('CONTEXT MASS CONTRACTS OK', flush=True)
    if 'temporal_coverage' in events:
        for name in ('temporal_fifo', 'temporal_uniform', 'temporal_coverage'):
            assert any(row['allocation_actions'] for row in events[name]), name
        ceilings = {rows[-1]['state_budget_bytes'] for name, rows in events.items()
                    if name.startswith('temporal_')}
        assert len(ceilings) == 1
        assert events['recent8'][-1]['state_bytes'] == ceilings.pop()
    if 'refresh_none' in events:
        with np.load(args.out / 'recent4' / 'pose_encodings.npz') as first, \
                np.load(args.out / 'refresh_none' / 'pose_encodings.npz') as second:
            np.testing.assert_allclose(first['pose_enc'], second['pose_enc'],
                                       atol=args.gate_atol, rtol=args.gate_rtol)
        for name in ('refresh_random', 'refresh_selective', 'refresh_oracle', 'refresh_full'):
            assert any(row['refreshed_frames'] for row in events[name]), name
            assert [row['state_bytes'] for row in events[name]] == [
                row['state_bytes'] for row in events['refresh_none']]
    write_json(args.out / 'research_gate.json', dict(
        contracts='passed', conditions=reports, scientific_gate='requires archive review',
        caveat='Temporal policies share a byte ceiling, not exact occupancy; '
               'native camera storage and measured CUDA peaks are reported separately.'))
    print('RESEARCH CROSS-CONDITION GATE OK', flush=True)


def heads(model, tokens, image, camera):
    with torch.autocast(device_type=image.device.type, enabled=False):
        pose, camera = model.camera_head(tokens, past_key_values_camera=camera, use_cache=True)
        depth, confidence = model.depth_head(tokens, images=image,
                                              patch_start_idx=model.aggregator.patch_start_idx)
        points, points_conf = model.point_head(tokens, images=image,
                                               patch_start_idx=model.aggregator.patch_start_idx)
    return dict(pose_enc=pose[-1], depth=depth, depth_conf=confidence,
                world_points=points, world_points_conf=points_conf), camera


def run_research_condition(args, model, root, frames, name, config):
    from streamvggt.models.research_cache import ResearchCache, ResearchConfig
    from stream_cache_sweep import load_frame, cpu_predictions, pose_arrays, write_json

    assert args.host == 'streamvggt'
    target = args.out / name
    target.mkdir()
    settings = ResearchConfig(**config.research)
    write_json(target / 'config.json', asdict(config))
    write_json(target / 'research_config.json', asdict(settings))
    research_fidelity(args, model, root, frames, target)
    torch.cuda.empty_cache()
    torch.manual_seed(settings.seed)
    observer = None
    if settings.context_probe:
        from streamvggt.models.context_probe import ContextProbe
        observer = ContextProbe(model.aggregator.patch_start_idx, settings.context_tokens)
    session = ResearchCache(model.aggregator, settings, observer=observer)
    camera = [None] * model.camera_head.trunk_depth
    poses, rows, probe_rows = [], [], []
    with torch.no_grad(), (target / 'events.jsonl').open('w') as log:
        for index, frame in enumerate(frames):
            image, mask = load_frame(root, frame)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            tokens = session.forward(image, index, mask)
            torch.cuda.synchronize()
            aggregated = time.perf_counter()
            output, camera = heads(model, tokens, image, camera)
            torch.cuda.synchronize()
            predicted = time.perf_counter()
            before = session.memory()
            del tokens
            event = session.maintain(index)
            torch.cuda.synchronize()
            maintained = time.perf_counter()
            session.refresh(index)
            torch.cuda.synchronize()
            end = time.perf_counter()
            memory = session.memory()
            assert memory['state_bytes'] == memory['aggregator_bytes'] + memory['auxiliary_bytes']
            if settings.method == 'temporal':
                assert memory['state_bytes'] <= session.byte_budget
            assert len(session.records) <= index + 1
            assert 0 in session.records and index in session.records
            prediction = cpu_predictions(output)
            diagnostic_bytes = 0 if observer is None else observer.memory()
            if observer is not None:
                probe = observer.report
                probe['dense_background_confidence_mean'] = float(
                    prediction['depth_conf'].reshape(-1)[~mask.reshape(-1)].mean())
                probe_rows.append(probe)
            poses.append(prediction['pose_enc'])
            if args.geometry_export == 'all' or index == len(frames) - 1:
                np.savez_compressed(target / f'frame_{index:06d}.npz',
                                    **{key: value.numpy() for key, value in prediction.items()})
            event.update(memory, pre_maintenance_state_bytes=before['state_bytes'],
                         camera_bytes=tensor_bytes(camera),
                         diagnostic_state_bytes=diagnostic_bytes,
                         total_persistent_bytes=memory['state_bytes'] + tensor_bytes(camera) + diagnostic_bytes,
                         aggregator_dtype=str(session.records[index]['kv'][0][0].dtype),
                         aggregate_seconds=aggregated - start,
                         head_seconds=predicted - aggregated,
                         allocation_seconds=maintained - predicted,
                         refresh_seconds=end - maintained, seconds=end - start,
                         peak_allocated=torch.cuda.max_memory_allocated(),
                         peak_reserved=torch.cuda.max_memory_reserved())
            log.write(json.dumps(event, allow_nan=False) + '\n')
            log.flush()
            rows.append(dict(event))
            del output, prediction, image
    poses = dict(pose_enc=torch.cat(poses, 1))
    np.savez_compressed(target / 'pose_encodings.npz', **{key: value.numpy() for key, value in poses.items()})
    np.savez_compressed(target / 'camera.npz',
                        **pose_arrays(args.host, poses, frames[0]['model_hw'], args.keyframe_stride))
    times = [row['seconds'] for row in rows]
    if observer is not None:
        with (target / 'context_probes.jsonl').open('w') as output:
            for row in probe_rows:
                output.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(target / 'summary.json', dict(frames=len(rows), total_seconds=sum(times),
               p50_seconds=float(np.median(times)), p95_seconds=float(np.quantile(times, .95)),
               max_state_bytes=max(row['state_bytes'] for row in rows),
               max_total_persistent_bytes=max(row['total_persistent_bytes'] for row in rows),
               max_diagnostic_state_bytes=max(row['diagnostic_state_bytes'] for row in rows),
               diagnostic_only=settings.context_probe,
               max_peak_allocated=max(row['peak_allocated'] for row in rows),
               max_peak_reserved=max(row['peak_reserved'] for row in rows),
               refreshed_groups=sum(len(row['refreshed_frames']) for row in rows),
               inference='causal observation stream; native camera history; post-output maintenance',
               geometry_export=args.geometry_export,
               caveat='Single-run timing includes selection/refresh; excludes input loading and output export. '
                      'Oracle includes diagnostic candidate recomputation. State bytes include auxiliary tensors; '
                      'camera history is separate and grows. Dense heads use reconstructed context tokens.'))
    write_json(target / 'research_contracts.json', dict(
        frames=len(rows), finite_predictions=True, anchor_and_current_preserved=True,
        byte_budget_checked=settings.method == 'temporal',
        method=settings.method, runtime_verified=True))
    print(f'RESEARCH CONTRACTS OK {name}', flush=True)
    print(f'RUN OK {name}', flush=True)
