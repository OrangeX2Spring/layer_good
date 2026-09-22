"""Small object-first correspondence pilot, executed only in the CAMP container."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

from kvt_arctic_run import CHECKOUT, DATASET_DIR, OUT, ROOT, SCENES, kvt_eval, align_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--scene', choices=SCENES, default=SCENES[0])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.tag.replace('_', '').isalnum()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    os.chdir(CHECKOUT)
    specification = dict(scene=args.scene, interval=30, cap=32, resolution=518,
        prefix_frames=128, pilot_frames=160, ordinary_patch_fraction=.5,
        spatial_bins=[4, 4], match_cosine_floor=.9, geometry_patch_spacing_multiplier=2.,
        variants=['dense', 'uniform', 'correspondence', 'semantic_correspondence'],
        scope='target-masked ARCTIC object tracking; diagnostic prefix, not held-out accuracy',
        novelty='unverified; no merging or newly inferred object identities')
    (args.out / 'protocol.json').write_text(json.dumps(specification, indent=2) + '\n')
    reports = {}
    events_by_mode = {}
    reference_name = f'{args.tag}_native'
    for mode in ['native'] + specification['variants']:
        lengths = [160] if mode in ('native', 'dense') else [128, 160]
        for length in lengths:
            name = f'{args.tag}_{mode}' + ('_prefix' if length == 128 else '')
            command = [sys.executable, str(ROOT / 'tools/kvt_arctic_run.py'),
                '--results', name, '--scenes', args.scene, '--no-viz', '--resize-dim', '518',
                '--online-policy', 'interval', '--interval', '30', '--max-keyframes', '32',
                '--prefix-frames', str(length)]
            if mode != 'native':
                command += ['--cache-policy', mode, '--check-masks-from', reference_name]
            if mode not in ('native', 'dense') and length == 160:
                command += ['--compare-prefix', f'{args.tag}_{mode}_prefix']
            print(f'PILOT {mode} frames={length}', flush=True)
            subprocess.run(command, check=True)
            result = DATASET_DIR / args.scene / name
            expected_ids = list(range(0, length, 30))
            assert np.load(result / 'kf_idx.npy').tolist() == expected_ids
            trajectory = np.load(result / 'traj.npy')
            assert trajectory.shape == (length, 4, 4) and np.isfinite(trajectory).all()
            if mode == 'native':
                continue
            events = [json.loads(line) for line in (result / 'cache_events.jsonl').read_text().splitlines()]
            assert [row['frame'] for row in events] == expected_ids
            patch_count = len(events[0]['patch_indices']['0'])
            for row in events:
                assert not row['evicted'] and len(set(row['retained_frame_ids'])) <= 32
                assert len(row['patch_indices']['0']) == patch_count
                for frame, picked in row['patch_indices'].items():
                    expected = patch_count if mode == 'dense' or frame == '0' else (patch_count + 1) // 2
                    assert len(picked) == len(set(picked)) == expected
                    assert min(picked) >= 0 and max(picked) < patch_count
                assert row['query_cache_bytes'] <= row['dense_cache_bytes']
                if mode != 'dense' and row['frame']:
                    assert row['query_cache_bytes'] < row['dense_cache_bytes']
                # Every recorded match references a patch that actually survived.
                for link in row['links']:
                    assert link['previous_frame'] < row['frame']
                    assert link['previous_patch'] in row['patch_indices'][str(link['previous_frame'])]
            if mode == 'dense':
                reference = np.load(DATASET_DIR / args.scene / reference_name / 'traj.npy')
                np.testing.assert_array_equal(trajectory, reference)
                print('FULL RETENTION GATE OK', flush=True)
            if length == 128:
                continue
            if mode not in ('native', 'dense'):
                prefix = DATASET_DIR / args.scene / f'{args.tag}_{mode}_prefix'
                before = [json.loads(line) for line in (prefix / 'cache_events.jsonl').read_text().splitlines()]
                for a, b in zip(before, events):
                    for key in ('frame', 'retained_frame_ids', 'patch_indices', 'links'):
                        assert a[key] == b[key], (mode, key)
            events_by_mode[mode] = events
            gt = kvt_eval.load_gt_arctic(args.scene)[:length]
            aligned, gt = align_pair(dict(traj_gt=gt, traj_est=trajectory), 'traj')
            gt_steps = np.linalg.inv(gt[:-1]) @ gt[1:]
            predicted_steps = np.linalg.inv(aligned[:-1]) @ aligned[1:]
            error = np.linalg.inv(gt_steps) @ predicted_steps
            translation = np.linalg.norm(error[:, :3, 3], axis=1)
            rotation = np.rad2deg(np.arccos(np.clip(
                (np.trace(error[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
            np.savez(args.out / f'{mode}_pose_errors.npz', query_frames=np.arange(1, length),
                     translation_m=translation, rotation_deg=rotation)
            archives = list(OUT.glob(f'arctic_{name}_[0-9]*.tar'))
            assert len(archives) == 1, archives
            with tarfile.open(archives[0]) as bundle:
                saved = json.load(bundle.extractfile('metrics.json'))
            timings = [json.loads(line) for line in (result / 'inference.jsonl').read_text().splitlines()]
            queries = [row for row in timings if row['kind'] == 'query']
            assert [row['frame'] for row in queries] == list(range(1, length))
            reports[mode] = {'archive': str(archives[0]), **saved['rows'][0],
                **saved['timings'][args.scene], 'rpe_rotation_deg': float(np.sqrt(np.mean(rotation ** 2))),
                'translation_p99_m': float(np.quantile(translation, .99)),
                'rotation_p99_deg': float(np.quantile(rotation, .99)),
                'mean_query_cache_bytes': float(np.mean([row['cache_bytes'] for row in queries])),
                'max_selector_metadata_bytes': max(row['feature_bytes'] for row in events),
                'query_p50_seconds': float(np.median([row['seconds'] for row in queries])),
                'inference_seconds': sum(row['seconds'] for row in timings),
                'selector_host_seconds': sum(row['selector_host_seconds'] for row in events),
                'matched_kept': sum(row['matched_kept'] for row in events),
                'object_matched_kept': sum(row['object_matched_kept'] for row in events),
                'object_patches_kept': [row['object_patches_kept'] for row in events]}
            (args.out / 'review.json').write_text(json.dumps(reports, indent=2, allow_nan=False) + '\n')
    checks = dict(full_retention_exact=True, masks_identical=True, fixed_schedules=True,
                  prefixes_consistent=True, half_patch_counts=True,
                  generic_matches_active=reports['correspondence']['matched_kept'] > 0,
                  semantic_object_matches_active=reports['semantic_correspondence']['object_matched_kept'] > 0,
                  semantic_choices_differ=any(
                      a['patch_indices'] != b['patch_indices'] for a, b in zip(
                          events_by_mode['correspondence'], events_by_mode['semantic_correspondence'])))
    (args.out / 'gates.json').write_text(json.dumps(checks, indent=2) + '\n')
    print('CORRESPONDENCE PILOT GATES OK', json.dumps(checks), flush=True)
    print('Review mechanism activity, errors and resources before selecting any full run.', flush=True)


if __name__ == '__main__':
    main()
