"""Rebuild job 26127's frozen local maps and record the cameras its bridges did not save.

Diagnostic only: no tracking pass, no GT, no acceptance change. CAMP only.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from kvt_office_segments import BOUNDARIES
from kvt_tum_run import write_json
from kvt_tum_sweep import archive_directory

# Archives verified on the Mac (cluster_results/kvt_oracle_maps_26127/archive_review.json).
SOURCES = {
    'tum_26127_oracle_maps_inputs.tar': '9ecd7546a56f2135bfbfc2d33a1be4de45c1670cb8cf56e1bedf2639a2933b33',
    'tum_26127_oracle_maps.tar': '7580f611922d9cbb0ce5b2ec542e1f3026f93e2b496363533794e4983aad2094',
    'tum_26127_context.tar': 'e7c21e5b9505e1d96945fb6243dc3703d8a2c1c0a5975f2593c13ef72d9db39d',
}
# Maximum absolute differences from 26127 (map units); 26127 itself matched 26115 to 1.3e-7.
POSE_LIMIT = 1e-4
POINT_LIMIT = 1e-3


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8388608), b''):
            digest.update(chunk)
    return digest.hexdigest()


def angle(a, b):
    """Degrees between two rotation matrices."""
    cosine = (np.trace(np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def main(args):
    import cv2
    import torch
    from huggingface_hub import snapshot_download
    from kv_tracker.map_handoff import MapHandoff, bridge, transform_pose
    from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32

    for name, expected in SOURCES.items():
        assert sha256(args.out / name) == expected, f'{name} differs from the reviewed archive'
    with tarfile.open(args.out / 'tum_26127_context.tar') as packed:
        expected_checkpoint = json.load(packed.extractfile('./oracle_sources.json'))['checkpoint']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    checkpoint = {name: sha256(snapshot / name) for name in expected_checkpoint}
    assert checkpoint == expected_checkpoint, 'Checkpoint differs from job 26127'

    saved = {}
    with tarfile.open(args.out / 'tum_26127_oracle_maps.tar') as packed:
        local_traj = np.load(io.BytesIO(packed.extractfile('oracle_maps/local_traj.npy').read()))
        for b in BOUNDARIES[1:-1]:
            saved[b] = dict(event=json.load(packed.extractfile(f'oracle_maps/bridge_{b:04d}.json')),
                            evidence=dict(np.load(io.BytesIO(
                                packed.extractfile(f'oracle_maps/bridge_{b:04d}.npz').read()))))
    assert local_traj.shape == (2585, 4, 4)

    inputs = tarfile.open(args.out / 'tum_26127_oracle_maps_inputs.tar')
    manifest = json.load(inputs.extractfile('office/manifest.json'))['inputs']
    assert len(manifest) == 2585

    def image(frame):
        row = manifest[frame]
        data = inputs.extractfile(f'office/model_rgb/{Path(row["file"]).name}').read()
        rgb = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['model_rgb_sha256'], frame
        return rgb

    result = args.work / 'runs' / 'bridge_cameras'
    result.mkdir(parents=True)
    write_json(result / 'config.json', dict(source_job=26127, sources=SOURCES, checkpoint=checkpoint,
        boundaries=BOUNDARIES, pose_limit=POSE_LIMIT, point_limit=POINT_LIMIT,
        tracking_pass=False, gt_used=False, acceptance_changed=False))
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())

    fidelity, old_side, rows = [], {}, []
    with (result / 'inference.jsonl').open('w') as stream, torch.inference_mode():
        def log(row):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()

        for start, end in zip(BOUNDARIES, BOUNDARIES[1:]):
            # A map's frozen state depends only on its bootstrap image and its
            # [0, 49] rebuild; cached queries never write the bank.
            model.cache = {}
            tracker = MapHandoff(model, 'oracle',
                                 lambda row, s=start: log(dict(row, segment_start=s)), None)
            frames = [start, start + 49, start + 50, *range(end - 10, end)]
            poses = [tracker.bootstrap(image(start)), tracker.step(image(start + 49), 49)]
            poses += [tracker.step(image(f), f - start) for f in frames[2:]]
            fidelity.append(dict(kind='local_traj', start=start, frames=frames,
                                 max_difference=float(np.abs(np.stack(poses) - local_traj[frames]).max())))
            if start > 0:
                # New side of boundary `start`: post-rebuild, same-cache cameras.
                # Local frame 49 would trigger a second rebuild through step(),
                # so read it with the dense query, which asserts the cache is unchanged.
                scale, rotation, translation = tracker.transform
                curve = [tracker.step(image(start + k), k) for k in range(49)]
                curve.append(tracker.query_geometry(image(start + 49), 49)[1].float().numpy())
                old = old_side.pop(start)
                fitted, event, evidence = bridge(
                    old['points'], scale * (tracker.anchor_points.double() @ rotation.T) + translation,
                    old['conf'], tracker.anchor_conf, torch.from_numpy(old['curve'][49]),
                    transform_pose(tracker.bank_last_pose, tracker.transform))
                reference = saved[start]
                for key in ('old_points', 'new_points'):
                    fidelity.append(dict(kind=key, boundary=start, max_difference=float(
                        np.abs(evidence[key] - reference['evidence'][key]).max())))
                for key in ('old_pose', 'new_pose', 'mapped_pose'):
                    fidelity.append(dict(kind=key, boundary=start, max_difference=float(
                        np.abs(evidence[key] - reference['evidence'][key]).max())))
                fidelity.append(dict(kind='bridge_checks', boundary=start,
                                     identical=event['checks'] == reference['event']['checks'],
                                     camera_translation_difference=abs(
                                         event['camera_translation_normalized'] -
                                         reference['event']['camera_translation_normalized'])))
                sim3 = tuple(torch.tensor(event[k], dtype=torch.float64)
                             for k in ('scale', 'rotation', 'translation'))
                extent = event['extent']
                new_rebuild_b = transform_pose(tracker.anchor_pose, tracker.transform)
                new_rebuild_b49 = transform_pose(tracker.bank_last_pose, tracker.transform)
                new_curve = np.stack(curve)
                old_b, old_b49 = old['query_b'], old['curve'][49]

                def mapped(pose):
                    return transform_pose(torch.as_tensor(pose), sim3).numpy()

                def disagreement(new, old_pose):
                    m = mapped(new)
                    return dict(translation_normalized=float(
                                    np.linalg.norm(m[:3, 3] - old_pose[:3, 3]) / extent),
                                rotation_deg=angle(m, old_pose))

                row = dict(boundary=start, extent=extent, scale=event['scale'],
                    gate_b49_rebuild_vs_old_query=disagreement(new_rebuild_b49, old_b49),
                    d1_b_rebuild_vs_old_query=disagreement(new_rebuild_b, old_b),
                    d1_b_new_query_vs_old_query=disagreement(new_curve[0], old_b),
                    d2_b49_new_query_vs_rebuild=dict(translation_normalized=float(
                        event['scale'] * np.linalg.norm(new_curve[49][:3, 3] - new_rebuild_b49[:3, 3].numpy())
                        / extent), rotation_deg=angle(new_curve[49], new_rebuild_b49)),
                    d2_b49_new_query_vs_old_query=disagreement(new_curve[49], old_b49))
                rows.append(row)
                np.savez(result / f'cameras_{start:04d}.npz', old_query_b=old_b,
                         old_curve=old['curve'], old_dense_b=old['dense_b'],
                         new_rebuild_b=new_rebuild_b.numpy(), new_rebuild_b49=new_rebuild_b49.numpy(),
                         new_curve=new_curve, new_emitted=local_traj[start:start + 50],
                         sim3_scale=event['scale'], sim3_rotation=np.asarray(event['rotation']),
                         sim3_translation=np.asarray(event['translation']), extent=extent)
                write_json(result / f'cameras_{start:04d}.json', dict(row, recomputed_event=event))
                print('BRIDGE CAMERAS', start, json.dumps({k: v for k, v in row.items()
                                                            if k.startswith(('gate', 'd1', 'd2'))}),
                      flush=True)
                del fitted, evidence
            if end < BOUNDARIES[-1]:
                # Old side of boundary `end`: the frozen bank extrapolates over end..end+49.
                points, dense_b, conf = tracker.query_geometry(image(end), end - start)
                curve = np.stack([tracker.step(image(end + k), end + k - start) for k in range(50)])
                old_side[end] = dict(points=points, conf=conf, dense_b=dense_b.numpy(),
                                     query_b=curve[0], curve=curve)
            del tracker
    inputs.close()
    assert not old_side and len(rows) == 5
    write_json(result / 'fidelity.json', fidelity)
    write_json(result / 'summary.json', dict(boundaries=rows, fidelity=fidelity,
        note='Diagnostic only. Distances are fractions of the old shared pointmap RMS extent; '
             'GT scoring of the curves happens offline.'))
    archive_directory(result, args.out / f'{args.tag}_bridge_cameras.tar')
    write_json(args.work / 'bridge_cameras_archive.json',
               {f'{args.tag}_bridge_cameras.tar': sha256(args.out / f'{args.tag}_bridge_cameras.tar')})

    # Gates are checked after archiving, so a fidelity failure keeps its evidence.
    for item in fidelity:
        if item['kind'] == 'bridge_checks':
            assert item['identical'] and item['camera_translation_difference'] <= 1e-3, item
        else:
            limit = POINT_LIMIT if item['kind'].endswith('points') else POSE_LIMIT
            assert item['max_difference'] <= limit, item
    (args.work / 'JOB_OK').write_text('Bridge cameras recorded; 26127 maps reproduced\n')
    print('BRIDGE CAMERAS COMPLETE: 5 boundaries, 26127 maps reproduced', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    main(parser.parse_args())
