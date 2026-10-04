"""CAMP-only fixed ORB25/max200 reanchor transfer on the three ARCTIC objects."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_tum_run import write_json
from kvt_tum_sweep import archive_directory, release_page_cache

SCENES = ('box_grab_01', 'ketchup_grab_01', 'espressomachine_grab_01')


def main(args):
    import cv2
    from kv_tracker.image import pi3_resize_image
    from kvt_arctic_run import DATASET_DIR, OUT as ARCTIC_OUT
    from kvt_patch_select import evaluate_object

    tools = Path(__file__).parent
    summary = args.work / 'runs' / 'segment_arctic_comparison'
    summary.mkdir(parents=True)
    policy = dict(detector='orb', minimum_inliers=25, maximum_segment_frames=200)
    write_json(summary / 'policy.json', policy)
    records = {}
    for scene in SCENES:
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
        write_json(inputs / 'manifest.json', dict(inputs=rows, frames=frames, scene=scene,
            resize_dim=518, offset=2, masks='fresh native SAM masks; reused unchanged',
            native_archive=str(packed_paths[0]), native_archive_sha256=sha256(packed_paths[0]),
            prepared_archive_sha256=sha256(ARCTIC_OUT / 'prepared.tar')))
        archive_directory(inputs, args.out / f'{args.tag}_segment_arctic_{scene}_inputs.tar')
        release_page_cache(inputs, args.out / f'{args.tag}_segment_arctic_{scene}_inputs.tar')
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
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    main(parser.parse_args())
