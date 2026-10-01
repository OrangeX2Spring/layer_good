"""One full-office experiment: six independent two-keyframe segments, CAMP only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

from kvt_tum_run import prepare, write_json
from kvt_tum_sweep import archive_inputs, archive_directory

BOUNDARIES = (0, 275, 575, 700, 975, 1400, 2585)


def main(args):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download('yyfz233/Pi3', local_files_only=True))
    checkpoint = {}
    for path in sorted(snapshot.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            checkpoint[str(path.relative_to(snapshot))] = digest.hexdigest()
    assert checkpoint
    write_json(args.work / 'checkpoint.json', dict(snapshot=str(snapshot), files=checkpoint))
    scene = 'freiburg3_long_office_household'
    archive = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
    with zipfile.ZipFile(archive) as packed:
        images = sorted(n for n in packed.namelist()
                        if n.startswith(f'rgbd_dataset_{scene}/rgb/') and n.endswith('.png'))
        assert len(images) == BOUNDARIES[-1]
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    archive_sha = digest.hexdigest()
    summary = []
    archives = []
    write_json(args.work / 'segment_protocol.json', dict(
        boundaries=BOUNDARIES, keyframe_local_indices=[0, 49], seed=0,
        history='fresh process and model for every segment; no previous state',
        alignment='independent Sim(3) per segment; no concatenated trajectory score',
        source_archive=str(archive), source_sha256=archive_sha))
    for start, end in zip(BOUNDARIES, BOUNDARIES[1:]):
        name = f'office_{start:04d}_{end - 1:04d}'
        length = end - start
        inputs = args.work / 'inputs' / name
        manifest = prepare(archive, inputs, 308, .02, start=start, count=length)
        assert [Path(r['file']).name for r in manifest['inputs']] == [
            Path(n).name for n in images[start:end]]
        (inputs / 'archive.sha256').write_text(f'{archive_sha}  {archive}\n')
        input_archive = args.out / f'{args.tag}_{name}_inputs.tar'
        archive_inputs(inputs, input_archive)
        archives.append(input_archive)
        result = args.work / 'runs' / name
        result.mkdir(parents=True)
        config = dict(name=name, scene=scene, scene_dir=str(inputs), frames=length,
                      resize_dim=308, interval=50, cap=2, layer='encoder',
                      policy='fixed', insertion_indices=[49], record_input_frames=True,
                      max_gt_difference=.02, evaluate_trajectory=True, save_final_scene=True)
        write_json(result / 'config.json', config)
        write_json(result / 'frame_mapping.json', dict(
            global_start=start, global_end_exclusive=end, local_to_global_offset=start,
            global_keyframes=[start, start + 49], local_keyframes=[0, 49]))
        print('RUN', name, 'global keyframes', [start, start + 49], flush=True)
        with (result / 'run.log').open('w') as log:
            completed = subprocess.run([sys.executable,
                str(Path(__file__).with_name('kvt_tum_run.py')), str(result / 'config.json')],
                stdout=log, stderr=subprocess.STDOUT)
        write_json(result / 'process.json', dict(returncode=completed.returncode))
        output_archive = args.out / f'{args.tag}_{name}.tar'
        archive_directory(result, output_archive)
        archives.append(output_archive)
        if completed.returncode:
            print((result / 'run.log').read_text()[-8000:], flush=True)
            raise RuntimeError(f'{name} failed with exit {completed.returncode}')
        poses = np.load(result / 'traj.npy')
        assert poses.shape == (length, 4, 4) and np.isfinite(poses).all()
        assert np.load(result / 'kf_idx.npy').tolist() == [0, 49]
        with np.load(result / 'final_scene.npz') as final:
            assert final['frame_ids'].tolist() == [0, 49]
            assert np.isfinite(final['poses']).all()
        decisions = [json.loads(line) for line in (result / 'decisions.jsonl').read_text().splitlines()]
        assert len(decisions) == length - 1
        assert [d['frame'] for d in decisions if d['selected']] == [49]
        for d in decisions:
            assert d['cache_frame_ids'] == ([0] if d['frame'] <= 49 else [0, 49])
        inference = [json.loads(line) for line in (result / 'inference.jsonl').read_text().splitlines()]
        for kind, count, input_count in (('bootstrap', 1, 2), ('rebuild', 1, 2), ('query', length - 1, 1)):
            rows = [r for r in inference if r['kind'] == kind]
            assert len(rows) == count and all(r['input_frame_count'] == input_count for r in rows)
        rebuild = next(r for r in inference if r['kind'] == 'rebuild')
        assert rebuild['frame'] == 49
        with np.load(result / 'evaluation.npz') as evaluation:
            p99 = float(np.quantile(evaluation['rpe_translation_per_pair_m'], .99))
            scale = float(evaluation['alignment_scale'])
        metrics = json.loads((result / 'metrics.json').read_text())
        row = dict(segment=name, global_start=start, global_end_exclusive=end,
                   global_keyframes=[start, start + 49], translation_p99_m=p99,
                   alignment_scale=scale, metrics=metrics)
        write_json(result / 'segment_checks.json', dict(passed=True,
            verified_keyframes=[start, start + 49], bootstrap_images=2,
            bootstrap_unique_frames=1, rebuild_images=2, rebuilds=1,
            no_cross_segment_state='fresh subprocess; segment-only input directory'))
        archive_directory(result, output_archive)
        summary.append(row)
        write_json(args.work / 'segment_summary.json', summary)
        print('SEGMENT OK', name, flush=True)
    assert sum(r['metrics']['frames'] for r in summary) == 2585
    hashes = {}
    for path in archives:
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    write_json(args.work / 'segment_archives.json', hashes)
    (args.work / 'JOB_OK').write_text('Six independent office segments complete; metric review pending\n')
    print('OFFICE SEGMENTS COMPLETE: 6 independent segments, 2585 frames', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    main(parser.parse_args())
