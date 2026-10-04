"""CAMP-only fixed office-policy transfer: loop/no-loop native versus live reanchor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

from kvt_bridge_cameras import sha256
from kvt_reanchor_maps import SOURCES
from kvt_tum_run import prepare, write_json
from kvt_tum_sweep import archive_directory, archive_inputs, release_page_cache


def main(args):
    from huggingface_hub import snapshot_download
    for name, digest in SOURCES.items():
        assert sha256(args.out / name) == digest
    with tarfile.open(args.out / 'tum_26127_context.tar') as packed:
        checkpoint = json.load(packed.extractfile('./oracle_sources.json'))['checkpoint']
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    assert {name: sha256(snapshot / name) for name in checkpoint} == checkpoint
    scene = args.scene
    label = 'with_loop' if scene == 'freiburg2_large_with_loop' else 'no_loop'
    archive = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
    inputs = args.work / 'inputs' / scene
    prepare(archive, inputs, 308, .02)
    digest = sha256(archive)
    (inputs / 'archive.sha256').write_text(f'{digest}  {archive}\n')
    archive_inputs(inputs, args.out / f'{args.tag}_segment_{label}_inputs.tar')
    release_page_cache(archive, inputs, args.out / f'{args.tag}_segment_{label}_inputs.tar')
    manifest = json.loads((inputs / 'manifest.json').read_text())
    frames = len(manifest['inputs'])
    assert frames == {'freiburg2_large_no_loop': 3359, 'freiburg2_large_with_loop': 5182}[scene]
    tools = Path(__file__).parent
    native = args.work / 'runs' / f'{label}_native'
    native.mkdir(parents=True)
    write_json(native / 'config.json', dict(mode='native', scene_dir=str(inputs),
        frames=frames, evaluate=True, sources={str(archive): digest}, checkpoint=checkpoint))
    subprocess.run([sys.executable, str(tools / 'kvt_handoff_pilot.py'),
                    '--run', str(native / 'config.json')], check=True)
    archive_directory(native, args.out / f'{args.tag}_{label}_native.tar')
    baseline = json.loads((native / 'metrics.json').read_text())
    assert baseline['evaluated_frames'] == manifest['gt_valid_frames']
    comparison = args.work / 'runs' / f'segment_{label}_comparison'
    comparison.mkdir()
    policy = dict(detector='orb', minimum_inliers=25, maximum_segment_frames=200)
    write_json(comparison / 'policy.json', policy)
    subprocess.run([sys.executable, str(tools / 'kvt_reanchor_maps.py'),
        '--work', str(args.work), '--out', str(args.out), '--tag', args.tag,
        '--inputs', str(inputs), '--native-metrics', str(native / 'metrics.json'),
        '--detector-config', str(comparison / 'policy.json'), '--name', f'{label}_reanchor'], check=True)
    result = args.work / 'runs' / f'{label}_reanchor'
    run = json.loads((result / 'summary.json').read_text())
    assert json.loads((result / 'environment.json').read_text()) == json.loads(
        (native / 'environment.json').read_text())
    windows = {}
    for name, path in (('native', native), ('reanchor', result)):
        assert np.load(path / 'traj.npy').shape == (frames, 4, 4)
        with np.load(path / 'evaluation.npz') as quality, np.load(native / 'evaluation.npz') as reference:
            for key in ('rgb_indices', 'gt_indices', 'timestamps', 'rpe_pair_start_indices', 'reference'):
                np.testing.assert_array_equal(quality[key], reference[key])
            indices = quality['rgb_indices']
            split = np.flatnonzero((np.diff(indices) != 1) | (np.diff(quality['timestamps']) > .1)) + 1
            spans = [dict(start=int(span[0]), end=int(span[-1]), frames=len(span))
                     for span in np.split(indices, split)]
            windows[name] = {}
            for start, end in ((0, 1000), (1000, 2500), (1025, 2026), (2000, frames)):
                ate = quality['ate_per_frame_m'][(indices >= start) & (indices < end)]
                pairs = quality['rpe_pair_start_indices']
                rpe = quality['rpe_translation_per_pair_m'][(pairs >= start) & (pairs + 1 < end)]
                windows[name][f'{start}_{end - 1}'] = dict(evaluated_frames=len(ate), rpe_pairs=len(rpe),
                    ate_m=float(np.sqrt(np.mean(ate ** 2))) if len(ate) else None,
                    rpe_translation_m=float(np.sqrt(np.mean(rpe ** 2))) if len(rpe) else None,
                    translation_p99_m=float(np.quantile(rpe, .99)) if len(rpe) else None)
    ratios = {key: run['metrics'][key] / baseline[key]
              for key in ('ate_m', 'rpe_translation_m', 'translation_p99_m')}
    write_json(comparison / 'comparison.json', dict(complete=True, scene=scene, frames=frames,
        policy=policy, native=baseline, reanchor=run, native_ratios=ratios, windows=windows,
        gt_spans=spans, gt_valid_frames=manifest['gt_valid_frames'],
        speedup=baseline['tracking_seconds'] / run['timing']['tracking_seconds'],
        gt_used_by_tracker=False, alignment='one full-valid Sim(3) per arm; windows retain it',
        caveat='office-calibrated policy; one timing per arm; sparse GT cannot score the whole route; '
               'historical native provenance remains pending; fresh native is the matched control'))
    archive_directory(comparison, args.out / f'{args.tag}_segment_{label}_comparison.tar')
    (args.work / 'JOB_OK').write_text(f'{scene} comparison complete; inspect accuracy and sparse GT\n')
    print('SEGMENT', label.replace('_', ' ').upper(),
          'COMPLETE: native + ORB25/max200 reanchor;', ratios, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--scene', default='freiburg2_large_no_loop',
                        choices=('freiburg2_large_no_loop', 'freiburg2_large_with_loop'))
    main(parser.parse_args())
