"""CAMP shared A/B label and training contract pilot; never grants selection eligibility."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import cv2
import numpy as np
import torch

from kvt_patch_select import TUM_ZIPS, prepare, archive_inputs, sha256, write_json
from kv_tracker.adaptive_tokens import sensitivity
from kv_tracker.atd_training import PatchScorer, patch_features, ranking_loss, scorer_keep
from kv_tracker.scene_routing import project_cache
from kv_tracker.token_drop import forward_kept

SPLIT = dict(train='freiburg1_desk', validation='freiburg3_structure_texture_far')
RESERVED = ('freiburg3_long_office_household', 'freiburg2_large_with_loop',
            'freiburg2_large_no_loop')
LABEL_FRAMES = (1, 8, 16, 32, 48, 50, 56, 63)


def labels(inputs, output, count):
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
    manifest = json.loads((inputs / 'manifest.json').read_text())
    assert manifest['scene'] in SPLIT.values() and manifest['scene'] not in RESERVED
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained('cuda').eval())
    model.requires_grad_(False)
    model.cache = {}
    images = []
    for row in manifest['inputs'][:count]:
        bgr = cv2.imread(str(inputs / 'model_rgb' / Path(row['file']).name))
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['model_rgb_sha256']
        images.append(torch.from_numpy(rgb.copy()).to('cuda').permute(2, 0, 1).float() / 255.)
    height, width = images[0].shape[-2:]
    cells = height // 14 * (width // 14)
    if manifest['scene'].startswith('freiburg1_'):
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
        distortion = torch.tensor([.2624, -.9531, -.0054, .0026, 1.1633], device='cuda')
    else:
        assert manifest['scene'].startswith('freiburg3_')
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
        distortion = None
    intrinsics = torch.tensor([[fx * width / 640, 0, (cx + .5) * width / 640 - .5],
        [0, fy * height / 480, (cy + .5) * height / 480 - .5], [0, 0, 1]], device='cuda')
    write_json(output / 'calibration.json', dict(K=intrinsics.cpu().tolist(),
        distortion=None if distortion is None else distortion.cpu().tolist(),
        source='https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats',
        source_size=[480, 640], model_size=[height, width], sampling='7::14',
        gauge='unrebased Pi3 dense history, reset together at each rebuild'))
    records = []
    admitted = [0]
    features, scores, keeps, frame_numbers = [], [], [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for frame in range(count):
            if frame == 0:
                physical = [0, 0]
            else:
                assert max(physical) < frame
                pointers = {i: {k: v.data_ptr() for k, v in layer.items()}
                            for i, layer in model.cache.items()}
                one = images[frame][None, None]
                if frame in LABEL_FRAMES:
                    cache_before = {i: {key: value.clone() for key, value in layer.items()}
                                    for i, layer in model.cache.items()}
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    # Do not let autocast lower projection arithmetic to bf16.
                    with torch.autocast('cuda', enabled=False):
                        maps = project_cache(points, confidence, previous_pose, intrinsics,
                                             height, width, distortion)
                        feature = patch_features(images[frame], maps)
                    torch.cuda.synchronize()
                    map_feature_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    score = sensitivity(model, one, use_cache=True)[0]
                    for i, layer in cache_before.items():
                        for key, value in layer.items():
                            assert torch.equal(value, model.cache[i][key])
                    del cache_before
                    assert all(parameter.grad is None for parameter in model.parameters())
                    torch.cuda.synchronize()
                    teacher_seconds = time.perf_counter() - started
                    keep = scorer_keep(score, (height // 14, width // 14), frame)
                    features.append(feature.cpu().numpy()); scores.append(score.cpu().numpy())
                    keeps.append(keep.cpu().numpy()); frame_numbers.append(frame)
                    records.append(dict(frame=frame, physical_ids=list(physical),
                        cache_frame=0 if len(admitted) == 1 else admitted[-1],
                        input_sha256=manifest['inputs'][frame]['model_rgb_sha256'],
                        previous_pose=previous_pose.cpu().tolist(),
                        supported_fraction=float((maps['support'] > 0).float().mean()),
                        map_feature_seconds=map_feature_seconds, teacher_seconds=teacher_seconds))
                    np.savez(output / f'route_{frame:06d}.npz',
                        **{key: value.cpu().numpy() for key, value in maps.items()})
                result = forward_kept(model, one, torch.ones(1, cells, device='cuda', dtype=torch.bool),
                                      cam_only=True, use_cache=True)
                previous_pose = result['camera_poses'][0, 0].float()
                assert pointers == {i: {k: v.data_ptr() for k, v in layer.items()}
                                    for i, layer in model.cache.items()}
                if frame != 49:
                    continue
                admitted.append(frame)
                physical = list(admitted)
            result = forward_kept(model, torch.stack([images[i] for i in physical])[None],
                torch.ones(len(physical), cells, device='cuda', dtype=torch.bool), store_cache=True)
            points = result['points'][0, :, 7::14, 7::14].reshape(-1, 3).float().clone()
            confidence = result['conf'][0, :, 7::14, 7::14, 0].sigmoid().reshape(-1).float().clone()
            previous_pose = result['camera_poses'][0, -1].float().clone()
            np.savez(output / f'cache_{frame:06d}.npz', points=points.cpu().numpy(),
                confidence=confidence.cpu().numpy(), frame_ids=np.array(physical),
                pose=previous_pose.cpu().numpy())
    np.savez(output / 'labels.npz', features=np.stack(features), scores=np.stack(scores),
             teacher_keep=np.stack(keeps), frames=np.array(frame_numbers))
    write_json(output / 'records.json', records)
    print('ATD LABEL CONTRACT OK', manifest['scene'], count, len(records), flush=True)


def train(work, candidate):
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    data = {}
    for split in SPLIT:
        with np.load(work / 'labels' / split / 'labels.npz') as saved:
            channels = 7 if candidate == 'A' else 12
            data[split] = (torch.from_numpy(saved['features'][:, :channels]).to('cuda'),
                           torch.from_numpy(saved['scores']).to('cuda'))
    model = PatchScorer(candidate).cuda()
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x, y = data['train']
    with torch.no_grad():
        initial = float(ranking_loss(model(x), y))
        initial_validation = float(ranking_loss(model(data['validation'][0]), data['validation'][1]))
    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = ranking_loss(model(x), y)
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        final = float(ranking_loss(model(x), y))
        validation = float(ranking_loss(model(data['validation'][0]), data['validation'][1]))
        assert final < initial, 'Training smoke failed to decrease training ranking loss'
        expected = model(data['validation'][0])
    target = work / 'training' / candidate
    target.mkdir(parents=True)
    checkpoint = target / 'scorer.pt'
    torch.save(dict(candidate=candidate, state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(), seed=17, learning_rate=.001,
        label_sha256={split: sha256(work / 'labels' / split / 'labels.npz') for split in SPLIT},
        protocol_sha256=sha256(work / 'training_protocol.json'),
        split=SPLIT, feature_channels=x.shape[1], selection='half top-score, half spread',
        teacher='six-component first-order sensitivity; diagnostic only', steps=20,
        eligible_for_tracking=False), checkpoint)
    reloaded = PatchScorer(candidate).cuda()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict']); reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(data['validation'][0]), expected, rtol=0, atol=0)
        grid = tuple(x.shape[-2:])
        validation_keeps = []
        for frame, score in enumerate(expected):
            keep = scorer_keep(score, grid, frame)
            assert int(keep.sum()) == (score.numel() + 1) // 2
            validation_keeps.append(keep.cpu().numpy())
        with np.load(work / 'labels/validation/labels.npz') as saved:
            np.savez(target / 'validation_predictions.npz', frames=saved['frames'],
                     scores=expected.cpu().numpy(), keep=np.stack(validation_keeps),
                     teacher_keep=saved['teacher_keep'])
        rgb_features = data['validation'][0][:1]
        for _ in range(5):
            reloaded(rgb_features)
        torch.cuda.synchronize(); started = time.perf_counter()
        for _ in range(20):
            scorer_keep(reloaded(rgb_features)[0], grid, 0)
        torch.cuda.synchronize()
        score_select_seconds = (time.perf_counter() - started) / 20
    write_json(target / 'summary.json', dict(candidate=candidate, parameters=parameter_count,
        initial_train_loss=initial, final_train_loss=final,
        initial_validation_loss=initial_validation, validation_loss=validation,
        train_losses=losses, checkpoint_sha256=sha256(checkpoint), reload_exact=True,
        score_select_seconds=score_select_seconds,
        timing='cached features + scorer + spread selection; excludes live mapping/features/Pi3',
        eligible_for_tracking=False, eligible_for_bulk_training=False))
    print('ATD TRAINING SMOKE OK', candidate, initial, final, validation, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--label-inputs', type=Path)
    parser.add_argument('--label-output', type=Path)
    parser.add_argument('--frames', type=int, default=64)
    args = parser.parse_args()
    assert sys.platform == 'linux' and args.work.is_relative_to('/tmp'), 'CAMP only'
    if args.label_inputs:
        return labels(args.label_inputs, args.label_output, args.frames)
    assert args.out is not None and args.tag is not None
    assert not set(SPLIT.values()).intersection(RESERVED)
    assert len(set(SPLIT.values())) == len(SPLIT)
    checkpoints = Path(os.environ['HF_HOME']) / 'hub/models--yyfz233--Pi3/snapshots'
    weights = list(checkpoints.glob('**/*.safetensors')) + list(checkpoints.glob('**/pytorch_model.bin'))
    assert weights, 'No offline Pi3 checkpoint'
    protocol = dict(scope='engineering contract pilot, not selection/headroom or bulk training',
        split=SPLIT, excluded=RESERVED, frames=64, staged_frames=128, label_frames=LABEL_FRAMES,
        teacher='six-component first-order sensitivity; original all-top policy failed scene gate',
        selection='ceil(P/2) kept: floor(kept/2) top scores, remainder spatial spread',
        history='dense causal teacher; bootstrap duplicate [0,0]; rebuild after49',
        geometry='dense-history patch projection; unsupported explicit; no GT features or labels',
        sparse_history_validation='pending; dense-history labels do not establish B deployment',
        pi3_weights={str(p): sha256(p) for p in weights}, inputs={},
        pi3_refs={str(p): p.read_text() for p in (checkpoints.parent / 'refs').rglob('*') if p.is_file()},
        seed=17, learning_rate=.001, training_steps=20, projection_tf32=False,
        eligible_for_bulk_training=False)
    write_json(args.work / 'training_protocol.json', protocol)
    for split, scene in SPLIT.items():
        source = TUM_ZIPS / f'rgbd_dataset_{scene}.zip'
        staged = args.work / 'inputs' / split
        # Reuse the existing loader's minimum staging contract; process only64.
        prepare(source, staged, 308, .02, count=128)
        digest = sha256(source)
        (staged / 'archive.sha256').write_text(f'{digest}  {source}\n')
        archive_inputs(staged, args.out / f'{args.tag}_training_inputs_{split}.tar')
        protocol['inputs'][split] = dict(zip_sha256=digest,
            manifest_sha256=sha256(staged / 'manifest.json'))
        write_json(args.work / 'training_protocol.json', protocol)
        for suffix, count in ((split, 64), ('train_prefix', 51)) if split == 'train' else ((split, 64),):
            target = args.work / 'labels' / suffix
            target.mkdir(parents=True)
            with (target / 'run.log').open('w') as log:
                process = subprocess.run([sys.executable, __file__, '--work', str(args.work),
                    '--label-inputs', str(staged), '--label-output', str(target),
                    '--frames', str(count)], stdout=log, stderr=subprocess.STDOUT)
            print((target / 'run.log').read_text()[-10000:], flush=True)
            process.check_returncode()
    with np.load(args.work / 'labels/train/labels.npz') as full, np.load(args.work / 'labels/train_prefix/labels.npz') as short:
        rows = full['frames'] < 51
        for key in full.files:
            np.testing.assert_allclose(short[key], full[key][rows], rtol=1e-4, atol=1e-5)
    full = json.loads((args.work / 'labels/train/records.json').read_text())
    short = json.loads((args.work / 'labels/train_prefix/records.json').read_text())
    for a, b in zip(short, full):
        for key in ('frame', 'physical_ids', 'cache_frame', 'input_sha256', 'previous_pose', 'supported_fraction'):
            assert a[key] == b[key], key
    for path in list((args.work / 'labels/train_prefix').glob('route_*.npz')) + list((args.work / 'labels/train_prefix').glob('cache_*.npz')):
        with np.load(path) as a, np.load(args.work / 'labels/train' / path.name) as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    print('ATD LABEL CAUSAL PREFIX OK', flush=True)
    for candidate in ('A', 'B'):
        train(args.work, candidate)
        gc.collect(); torch.cuda.empty_cache()
    archive = args.out / f'{args.tag}_training_pilot.tar'
    with tarfile.open(archive.with_suffix('.tar.partial'), 'w') as tar:
        for name in ('training_protocol.json', 'labels', 'training'):
            tar.add(args.work / name, arcname=name)
    archive.with_suffix('.tar.partial').replace(archive)
    (args.work / 'JOB_OK').write_text('A/B engineering contracts passed; selection and bulk-training gates pending.\n')
    print('ATD SHARED TRAINING PILOT OK', flush=True)


if __name__ == '__main__':
    main()
