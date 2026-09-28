"""Stage consecutive ARCTIC object frames and saved KV-Tracker SAM masks on CAMP."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--kvt', type=Path, required=True)
    parser.add_argument('--scene', default='box_grab_01')
    parser.add_argument('--count', type=int, default=121)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--object-gt', action='store_true',
                        help='Archive and verify camera-in-object GT for research pilots')
    args = parser.parse_args()
    assert args.count >= 2
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'rgb').mkdir()
    (args.out / 'masks').mkdir()
    gt_files = {}
    with tarfile.open(args.prepared) as source, tarfile.open(args.kvt) as baseline:
        manifest = json.load(source.extractfile('manifest.json'))
        reference = json.load(baseline.extractfile('manifest.json'))
        assert manifest == reference['prepare_manifest']
        assert manifest['loader_offset'] == reference['offset'] == 2
        frames = manifest['scenes'][args.scene]['frames'][2:]
        assert len(frames) >= args.count
        mask_names = {member.name for member in baseline.getmembers() if member.isfile()
                      and member.name.startswith(f'{args.scene}/results/sam_masks/')}
        assert mask_names == {f'{args.scene}/results/sam_masks/{i:05d}.png'
                              for i in range(len(frames))}
        rows = []
        gt = None
        if args.object_gt:
            from scipy.spatial.transform import Rotation
            from stream3r_arctic_eval import evaluate, read_array
            prefix = f'data/raw_seqs/s01/{args.scene}'
            obj = read_array(source, prefix + '.object.npy', True)[:, 1:]
            cam = read_array(source, prefix + '.egocam.dist.npy', True).item()
            object_pose = np.tile(np.eye(4), (len(obj), 1, 1))
            object_pose[:, :3, :3] = Rotation.from_rotvec(obj[:, :3]).as_matrix()
            object_pose[:, :3, 3] = obj[:, 3:] * 1e-3
            camera = np.tile(np.eye(4), (len(obj), 1, 1))
            camera[:, :3, :3] = cam['R_k_cam_np']
            camera[:, :3, 3] = cam['T_k_cam_np'].squeeze()
            gt = np.linalg.inv(np.linalg.inv(camera) @ object_pose)[2:]
            trajectory = read_array(baseline, f'{args.scene}/results/traj.npy').astype(np.float64)
            check = evaluate(trajectory, gt)
            recorded = next(row for row in json.load(baseline.extractfile('metrics.json'))['rows']
                            if row['scene'] == args.scene)
            assert check['status'] == 'ok'
            for metric in ('ate_m', 'rpe_t_m', 'rpe_rot'):
                assert abs(check[metric] - recorded[metric]) < 1e-5, metric
            for suffix in ('.object.npy', '.egocam.dist.npy'):
                raw = source.extractfile(prefix + suffix).read()
                name = 'gt' + suffix
                (args.out / name).write_bytes(raw)
                gt_files[name] = hashlib.sha256(raw).hexdigest()
            (args.out / 'object_gt_check.json').write_text(json.dumps(dict(
                scene=args.scene, metrics=check, source_sha256=gt_files,
                convention='camera-in-object; exact KV-Tracker algebra; offset 2'), indent=2) + '\n')
            gt_files['object_gt_check.json'] = hashlib.sha256(
                (args.out / 'object_gt_check.json').read_bytes()).hexdigest()
        for index, frame in enumerate(frames[:args.count]):
            rgb_bytes = source.extractfile(frame['path']).read()
            mask_member = f'{args.scene}/results/sam_masks/{index:05d}.png'
            mask_bytes = baseline.extractfile(mask_member).read()
            rgb = np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert('RGB'))
            mask = np.asarray(Image.open(io.BytesIO(mask_bytes)).convert('L')) > 127
            assert rgb.shape[:2] == mask.shape and mask.any() and (~mask).any()
            rgb_path = f'rgb/{index:06d}.png'
            mask_path = f'masks/{index:06d}.png'
            Image.fromarray(rgb).save(args.out / rgb_path)
            (args.out / mask_path).write_bytes(mask_bytes)
            rows.append(dict(rgb=rgb_path, mask=mask_path, source_member=frame['path'],
                             source_sha256=hashlib.sha256(rgb_bytes).hexdigest(),
                             mask_member=mask_member,
                             source_mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
                             image_id=frame.get('image_id'), kvt_index=index))
            if gt is not None:
                rows[-1]['gt_object_c2w'] = gt[index].tolist()
                rows[-1]['gt_row'] = index + 2
    result = dict(
        sequence=f'ARCTIC s01 {args.scene}', frames=rows, mask_input=True,
        selection='first consecutive post-offset frames',
        prepared_archive=str(args.prepared), kvt_archive=str(args.kvt))
    if args.object_gt:
        result.update(pose_reference='camera-in-object', object_gt_files=gt_files)
    (args.out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'ARCTIC STAGED {args.scene} {args.count}', flush=True)


if __name__ == '__main__':
    main()
