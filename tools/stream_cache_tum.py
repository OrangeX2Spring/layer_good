"""Write a stream_cache source manifest for one TUM RGB-D sequence.

Reads the TUM ZIP directly from `/mnt/datasets/tum-rgbd` and writes the selected
frames plus a `manifest.json` in the schema `stream_cache_sweep.py prepare` reads:
per-frame `rgb`, `timestamp`, `intrinsics` (original resolution -- prepare derives
`model_intrinsics` itself) and, where a GT pose falls within tolerance, `gt_c2w`
and `gt_timestamp`.

GT association mirrors `kvt_tum_run.associate_gt`: nearest timestamp, never an
extrapolated pose, `null` beyond tolerance. The two sweeps therefore agree on which
frames are scorable. Standard library only, so it runs on a bare allocated node
without entering the container.
"""
import argparse
import bisect
import json
import zipfile
from pathlib import Path

# TUM RGB camera intrinsics, per the dataset's own calibration page. Chosen by
# sequence name because the three cameras differ and mixing them is silent.
INTRINSICS = {
    'freiburg1': (517.306408, 516.469215, 318.643040, 255.313989),
    'freiburg2': (520.908620, 521.007327, 325.141442, 249.701764),
    'freiburg3': (535.4, 539.2, 320.1, 247.6),
}


def quaternion_to_c2w(tx, ty, tz, qx, qy, qz, qw):
    """TUM groundtruth.txt is the camera pose in the world frame, so this is c2w."""
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** .5
    assert norm > 1e-6, 'degenerate quaternion'
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw), tx],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw), ty],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy), tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def nearest_gt(time, gt_times):
    """Nearest GT row to one RGB timestamp. gt_times must be non-decreasing."""
    right = min(bisect.bisect_left(gt_times, time), len(gt_times) - 1)
    left = max(right - 1, 0)
    if abs(gt_times[left] - time) < abs(gt_times[right] - time):
        return left
    return right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0, help='first source frame index')
    parser.add_argument('--count', type=int, required=True, help='frames to emit')
    parser.add_argument('--stride', type=int, default=1,
                        help='1 keeps the clip consecutive, which is what a streaming '
                             'model is actually being asked to do')
    parser.add_argument('--max-gt-delta', type=float, default=.02)
    args = parser.parse_args()
    assert args.count >= 2 and args.stride >= 1 and args.start >= 0

    scene = args.zip.stem.removeprefix('rgbd_dataset_')
    camera = scene.split('_')[0]
    assert camera in INTRINSICS, f'unknown TUM camera in {scene}'
    fx, fy, cx, cy = INTRINSICS[camera]
    intrinsics = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    prefix = f'rgbd_dataset_{scene}/'
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'rgb').mkdir()

    with zipfile.ZipFile(args.zip) as packed:
        names = sorted(n for n in packed.namelist()
                       if n.startswith(prefix + 'rgb/') and n.endswith('.png'))
        assert names, f'no rgb/*.png under {prefix}'
        selected = names[args.start::args.stride][:args.count]
        assert len(selected) == args.count, (
            f'only {len(selected)} frames available from start={args.start} '
            f'stride={args.stride}; sequence holds {len(names)}')

        gt_rows = []
        with packed.open(prefix + 'groundtruth.txt') as handle:
            for line in handle.read().decode().splitlines():
                if line.startswith('#') or not line.strip():
                    continue
                gt_rows.append([float(v) for v in line.split()])
    assert len(gt_rows) >= 3, 'groundtruth.txt is too short to align against'
    gt_times = [row[0] for row in gt_rows]
    # freiburg2_large_no_loop repeats one timestamp; non-decreasing, not strict.
    assert all(b >= a for a, b in zip(gt_times, gt_times[1:])), 'GT timestamps go backwards'

    frames, valid = [], 0
    with zipfile.ZipFile(args.zip) as packed:
        for index, name in enumerate(selected):
            timestamp = float(Path(name).stem)
            relative = f'rgb/{index:06d}.png'
            (args.out / relative).write_bytes(packed.read(name))
            row = {'rgb': relative, 'timestamp': timestamp,
                   'intrinsics': intrinsics, 'source_frame': Path(name).name}
            match = nearest_gt(timestamp, gt_times)
            if abs(gt_times[match] - timestamp) <= args.max_gt_delta:
                row['gt_c2w'] = quaternion_to_c2w(*gt_rows[match][1:8])
                row['gt_timestamp'] = gt_times[match]
                valid += 1
            else:
                row['gt_c2w'] = None
                row['gt_timestamp'] = None
            frames.append(row)

    assert valid >= 3, f'only {valid} timestamp-valid GT poses; evaluation needs 3'
    manifest = {
        'sequence': scene,
        'mask_source': 'none',
        'source_archive': str(args.zip),
        'selection': {'start': args.start, 'count': args.count, 'stride': args.stride},
        'span_seconds': frames[-1]['timestamp'] - frames[0]['timestamp'],
        'max_gt_delta_seconds': args.max_gt_delta,
        'gt_valid_frames': valid,
        'frames': frames,
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'TUM MANIFEST OK {scene} frames={len(frames)} '
          f'span={manifest["span_seconds"]:.2f}s gt_valid={valid}', flush=True)


if __name__ == '__main__':
    main()
