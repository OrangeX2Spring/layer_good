"""Reusable in-container setup and execution for one streaming host batch job."""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('stream3r', 'streamvggt', 'longstream'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--width', type=int, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and args.work.resolve().is_relative_to('/tmp')
    repo = Path(__file__).resolve().parents[1]
    python = sys.executable
    before = subprocess.check_output([python, '-m', 'pip', 'list', '--format=json'], text=True)
    (args.work / 'packages_before.json').write_text(before)
    installed = json.loads(before)
    constraints = args.work / 'installed_constraints.txt'
    constraints.write_text(''.join(f"{row['name']}=={row['version']}\n" for row in installed))
    # Only add absent inference dependencies, without changing existing packages.
    requirements = {'einops': 'einops'}
    if args.host == 'streamvggt':
        requirements['transformers'] = 'transformers'
    if args.host == 'longstream':
        requirements.update(cv2='opencv-python-headless', yaml='PyYAML')
    missing = [package for module, package in requirements.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.run([python, '-m', 'pip', 'install', '--no-cache-dir', '--only-binary=:all:',
                        '--constraint', str(constraints), '--report', str(args.work / 'install.json'),
                        *missing], check=True)
    after = subprocess.check_output([python, '-m', 'pip', 'list', '--format=json'], text=True)
    (args.work / 'packages_after.json').write_text(after)
    versions = {row['name']: row['version'] for row in json.loads(after)}
    assert all(versions[row['name']] == row['version'] for row in installed), 'Existing package changed'
    sys.path.insert(0, str(repo / ('streamvggt/src' if args.host == 'streamvggt' else args.host)))
    modules = {'stream3r': 'stream3r.models.stream3r',
               'streamvggt': 'streamvggt.models.streamvggt',
               'longstream': 'longstream.core.model'}
    importlib.import_module(modules[args.host])
    print(f'MODEL IMPORT OK {args.host}', flush=True)
    subprocess.run([python, '-m', 'unittest', 'discover', '-s', 'tools',
                    '-p', 'test_*cache*.py'], cwd=repo, check=True)
    manifest = args.manifest
    if manifest is None:
        if args.host != 'stream3r':
            raise ValueError('Supply CACHE_INPUT_TAR for this host; no dataset is guessed')
        images = sorted((repo / 'stream3r/examples/static_room').glob('*.png'))
        assert len(images) >= 2, 'STream3R example images missing'
        manifest = args.work / 'example_manifest.json'
        manifest.write_text(json.dumps(dict(sequence='STream3R-static_room-smoke',
            purpose='integration only, no GT or long-term-memory claim',
            frames=[dict(rgb=str(path)) for path in images]), indent=2) + '\n')
    prepared = args.work / 'prepared'
    subprocess.run([python, 'tools/stream_cache_sweep.py', 'prepare',
                    '--manifest', str(manifest), '--out', str(prepared),
                    '--width', str(args.width)], cwd=repo, check=True)
    extra = (['--model-config', str(repo / 'longstream/configs/longstream_infer.yaml')]
             if args.host == 'longstream' else [])
    subprocess.run([python, 'tools/stream_cache_sweep.py', 'run', '--host', args.host,
                    '--checkpoint', str(args.checkpoint), '--inputs', str(prepared),
                    '--sweep', str(args.sweep), '--out', str(args.work / 'run'),
                    '--git-provenance', str(args.work / 'git_provenance'), *extra],
                   cwd=repo, check=True)


if __name__ == '__main__':
    main()
