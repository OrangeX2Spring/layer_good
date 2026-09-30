"""CAMP-only calibration and bounded live historical-KV depth ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'kv_tracker'))
from kv_tracker.layer_cache import LayerCache
from kvt_tum_run import prepare, write_json


def calibrate(work, dataset):
    import cv2
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
    inputs = work / 'inputs' / 'calibration'
    manifest = prepare(dataset / 'rgbd_dataset_freiburg1_desk.zip', inputs, 308, .02, count=128)
    images = []
    for record in manifest['inputs'][:16]:
        image = cv2.cvtColor(cv2.imread(str(inputs / 'model_rgb' / Path(record['file']).name)), cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(image.tobytes()).hexdigest() == record['model_rgb_sha256']
        images.append(torch.from_numpy(image.copy()).permute(2, 0, 1).float().cuda() / 255.)
    torch.manual_seed(17)
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained()).eval()
    identity = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        identity.update(name.encode())
        identity.update(str((value.dtype, tuple(value.shape))).encode())
        identity.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    write_json(work / 'model_identity.json', dict(loaded_state_sha256=identity.hexdigest()))
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        bank_poses = model(torch.stack(images[:8])[None], cam_only=True, store_cache=True)['camera_poses'][0].float()
        dense = model.cache
        layers = sorted(dense)
        assert layers == list(range(1, len(model.decoder), 2))
        assert len(layers) % 3 == 0
        radius = float((bank_poses[:, :3, 3] - bank_poses[0, :3, 3]).square().sum(-1).mean().sqrt())
        assert radius > 0
        calls = []
        hooks = [block.register_forward_hook(lambda module, args, out, i=i: calls.append(i))
                 for i, block in enumerate(model.decoder)]
        try:
            def query():
                poses = []
                for image in images[8:16]:
                    calls.clear()
                    poses.append(model(image[None, None], cam_only=True, use_cache=True)['camera_poses'][0, 0].float())
                    assert calls == list(range(len(model.decoder))), 'Every decoder block must execute'
                return torch.stack(poses)
            native = query()
            scores = []
            saved = dict(native=native.cpu().numpy(), bank=bank_poses.cpu().numpy())
            for layer in layers:
                # Dense teacher is kept only during calibration, never in deployed runs.
                model.cache = {i: dict(values) for i, values in dense.items()}
                transform = LayerCache('omit', [layer])
                transform(model.cache, list(range(8)))
                poses = query()
                assert torch.isfinite(poses).all()
                translation = (poses[:, :3, 3] - native[:, :3, 3]).square().sum(-1) / radius ** 2
                rotation = (poses[:, :3, :3] - native[:, :3, :3]).square().sum((-1, -2)) / 6
                score = float((translation + rotation).mean())
                scores.append(dict(layer=layer, score=score,
                    normalized_translation_mse=float(translation.mean()), rotation_matrix_mse=float(rotation.mean())))
                saved[f'omit_{layer}'] = poses.cpu().numpy()
                print('LAYER CACHE CALIBRATION', layer, score, flush=True)
            chosen = sorted(row['layer'] for row in sorted(scores, key=lambda row: (row['score'], row['layer']))[:len(layers) // 3])
            model.cache = dense
            torch.testing.assert_close(query(), native, rtol=1e-5, atol=1e-5)
        finally:
            for hook in hooks:
                hook.remove()
    np.savez(work / 'calibration_poses.npz', **saved)
    selection = dict(omitted_layers=chosen, scores=scores, bank_frames=list(range(8)),
        query_frames=list(range(8, 16)), bank_translation_radius=radius,
        criterion='mean squared native-pose translation/radius plus rotation Frobenius squared/6',
        scope='single-layer ablations; combined omission unverified until evaluation',
        mapping='zero-based pinned Pi3 decoder; odd blocks alone own historical KV',
        budget='omit one third of cache-bearing blocks; no budget sweep')
    write_json(work / 'selection.json', selection)
    return chosen


def pilot(work, dataset, chosen):
    records = []
    for scene, start in (('freiburg1_desk', 128), ('freiburg3_structure_notexture_far', 0)):
        inputs = work / 'inputs' / scene
        prepare(dataset / f'rgbd_dataset_{scene}.zip', inputs, 308, .02, start=start, count=128)
        for name in ('native', 'omit', 'uniform'):
            result = work / 'runs' / scene / name
            result.mkdir(parents=True)
            config = dict(name=name, scene_dir=str(inputs), frames=128, resize_dim=308,
                policy='fixed', insertion_indices=[31, 63, 95], interval=32, cap=4,
                layer='decoder0', score='cosine', threshold=.05, similarity_floor=.95,
                max_gt_difference=.02, evaluate_trajectory=True, save_final_scene=False)
            config['historical_kv'] = dict(mode=name, omitted_layers=chosen)
            path = result / 'config.json'
            write_json(path, config)
            print('LAYER CACHE RUN', scene, name, 'log', result / 'run.log', flush=True)
            with (result / 'run.log').open('w') as log:
                process = subprocess.run([sys.executable, str(REPO / 'tools/kvt_tum_run.py'), str(path)],
                                         stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                print('\n'.join((result / 'run.log').read_text().splitlines()[-80:]), flush=True)
            process.check_returncode()
            metrics = json.loads((result / 'metrics.json').read_text())
            inference = [json.loads(line) for line in (result / 'inference.jsonl').read_text().splitlines()]
            metrics['max_query_cache_bytes'] = max(row['cache_bytes'] for row in inference if row['kind'] == 'query')
            with np.load(result / 'evaluation.npz') as evaluation:
                metrics['translation_p99_m'] = float(np.quantile(evaluation['ate_per_frame_m'], .99))
            events = json.loads((result / 'historical_kv.json').read_text())
            metrics['max_persistent_cache_bytes'] = max(e['persistent_bytes'] for e in events)
            if name == 'omit':
                omission_events = events
            elif name == 'uniform':
                assert [(e['frame_ids'], e['persistent_bytes']) for e in events] == [
                    (e['frame_ids'], e['persistent_bytes']) for e in omission_events]
            records.append(dict(scene=scene, name=name, **metrics))
            write_json(work / 'summary.json', dict(complete=False, records=records))
            print('LAYER CACHE RUN OK', scene, name, metrics, flush=True)
        trajectories = {name: np.load(work / 'runs' / scene / name / 'traj.npy')
                        for name in ('native', 'omit', 'uniform')}
        for name in ('omit', 'uniform'):
            assert np.isfinite(trajectories[name]).all()
            # Stored keyframe poses come from the dense rebuild; arrival-frame
            # trajectory poses were queried before that rebuild and can differ.
            np.testing.assert_allclose(np.load(work / 'runs' / scene / name / 'kf_poses.npy'),
                np.load(work / 'runs' / scene / 'native' / 'kf_poses.npy'), rtol=1e-5, atol=1e-5)
    write_json(work / 'summary.json', dict(complete=True, records=records,
        next='manual per-scene quality/tail/bytes/query-time review; no automatic full evaluation',
        evidence='short fixed-admission live tracker; dense rebuilds unchanged; no dense shadow during runs'))
    print('LAYER CACHE PILOT COMPLETE: inspect summary; no automatic next job', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('/mnt/datasets/tum-rgbd'))
    args = parser.parse_args()
    assert sys.platform.startswith('linux') and torch.cuda.is_available(), 'CAMP GPU only'
    (args.work / 'inputs').mkdir()
    write_json(args.work / 'protocol.json', dict(calibration='desk0:16; prepared0:128',
        evaluation=['desk128:256', 'textureless0:128'], cases=['native', 'omit', 'uniform'],
        fixed_insertions=[31, 63, 95], architecture='all original blocks/weights active',
        control='uniform flattened history tokens; integer exact matching of omission payload'))
    chosen = calibrate(args.work, args.dataset)
    torch.cuda.empty_cache()
    pilot(args.work, args.dataset, chosen)
