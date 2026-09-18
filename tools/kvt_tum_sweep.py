"""Fail-fast TUM experiment controller, executed inside the existing CAMP image."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

from kvt_tum_run import prepare, write_json

LONG_SCENES = ('freiburg3_long_office_household', 'freiburg2_large_with_loop',
               'freiburg2_large_no_loop')
BASELINE_ATE = {'freiburg1_xyz': .021, 'freiburg1_rpy': .045,
                'freiburg1_desk2': .083, 'freiburg1_desk': .059, 'freiburg1_room': .361}
COSINE_THRESHOLDS = (.01, .025, .05, .10)
COVERAGE_THRESHOLDS = (.05, .10, .20, .40)
CHAMFER_THRESHOLDS = (.025, .05, .10, .20)


def semantic_configs():
    configs = []
    for layer, score, thresholds in (
            ('decoder0', 'cosine', COSINE_THRESHOLDS),
            ('encoder', 'cosine', COSINE_THRESHOLDS),
            ('decoder0', 'coverage', COVERAGE_THRESHOLDS),
            ('decoder0', 'chamfer', CHAMFER_THRESHOLDS)):
        for threshold in thresholds:
            configs.append(dict(name=f'{layer}_{score}_{threshold:g}', policy='semantic',
                                layer=layer, score=score, threshold=threshold))
    return configs


def archive_directory(source, destination):
    temporary = destination.with_suffix('.tar.partial')
    with tarfile.open(temporary, 'w') as archive:
        archive.add(source, arcname=source.name)
    temporary.replace(destination)


def compare_prefix(prefix, full):
    short = np.load(prefix / 'traj.npy')
    long = np.load(full / 'traj.npy')
    np.testing.assert_allclose(short, long[:len(short)], atol=1e-4, rtol=1e-4)
    a = [json.loads(line) for line in (prefix / 'decisions.jsonl').read_text().splitlines()]
    b = [json.loads(line) for line in (full / 'decisions.jsonl').read_text().splitlines()][:len(a)]
    assert len(a) == len(short) - 1 == len(b)
    for left, right in zip(a, b):
        for key in ('frame', 'candidate', 'selected', 'capped', 'cap_blocked', 'cache_frame_ids'):
            assert left[key] == right[key], (key, left, right)
        if left['score'] is not None:
            np.testing.assert_allclose(left['score'], right['score'], atol=1e-5, rtol=1e-4)
    write_json(full / 'prefix_gate.json', dict(passed=True, frames=len(short),
               max_pose_difference=float(np.max(np.abs(short - long[:len(short)])))))
    print('PREFIX GATE OK', full.name, flush=True)


class Sweep:
    def __init__(self, args):
        self.args = args
        self.work = args.work
        self.records = []
        self.manifests = {}
        (self.work / 'runs').mkdir(parents=True)
        (self.work / 'inputs').mkdir()
        self.base = dict(resize_dim=308, interval=50, cap=20, layer='decoder0',
                         score='cosine', threshold=.05, similarity_floor=.95,
                         max_gt_difference=.02)
        write_json(self.work / 'protocol.json', dict(base=self.base,
            semantic=semantic_configs(), long_scenes=LONG_SCENES,
            memory_probe_caps=[64, 48, 32], memory_headroom_gib=2,
            ate_match_relative_tolerance=.05, historical_ate_tolerance_m=.002,
            periodic_search='start at semantic count, double to K, one midpoint refinement',
            selection='per-sequence best of 16, exploratory same-sequence selection',
            prefix_frames=128))

    def checkpoint(self):
        write_json(self.work / 'summary.json', self.records)
        with (self.work / 'summary.csv').open('w', newline='') as stream:
            fields = ['scene', 'name', 'policy', 'layer', 'score', 'threshold', 'interval',
                      'cap', 'frames', 'keyframes', 'last_keyframe', 'tail_frames', 'tail_seconds',
                      'cap_reached', 'ate_m', 'rpe_translation_m', 'rpe_rotation_deg',
                      'evaluated_fraction', 'seconds', 'peak_allocated_bytes',
                      'peak_reserved_bytes', 'estimated_peak_device_bytes']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.records)

    def stage(self):
        for scene in (*BASELINE_ATE, *LONG_SCENES):
            archive = self.args.datasets / f'rgbd_dataset_{scene}.zip'
            assert archive.is_file(), archive
            destination = self.work / 'inputs' / scene
            self.manifests[scene] = prepare(archive, destination, 308, .02)
            # Source ZIP checksum is streamed, never loaded into memory.
            digest = hashlib.sha256()
            with archive.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            (destination / 'archive.sha256').write_text(f'{digest.hexdigest()}  {archive}\n')
            archive_directory(destination, self.args.out / f'{self.args.tag}_inputs_{scene}.tar')
        write_json(self.work / 'inventory.json', {scene: {k: v for k, v in m.items() if k != 'inputs'}
                    for scene, m in self.manifests.items()})

    def run_one(self, scene, settings, prefix=False, preflight=False):
        config = dict(self.base, **settings)
        name = settings['name'] + ('_prefix' if prefix else '')
        result = self.work / 'runs' / scene / name
        result.mkdir(parents=True)
        frames = 128 if prefix else settings.get('frames', self.manifests[scene]['frames'])
        config.update(name=name, scene=scene, scene_dir=str(self.work / 'inputs' / scene), frames=frames)
        write_json(result / 'config.json', config)
        started = time.time()
        print('RUN', scene, name, frames, 'frames', flush=True)
        with (result / 'run.log').open('w') as log:
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name('kvt_tum_run.py')),
                                        str(result / 'config.json')], stdout=log, stderr=subprocess.STDOUT)
        write_json(result / 'process.json', dict(start_unix=started, end_unix=time.time(),
                                               returncode=completed.returncode))
        destination = self.args.out / f'{self.args.tag}_{scene}_{name}.tar'
        if completed.returncode:
            archive_directory(result, destination)
            print((result / 'run.log').read_text()[-8000:], flush=True)
            if preflight and completed.returncode == 42 and (result / 'CUDA_OOM').is_file():
                print('PREFLIGHT CUDA OOM', scene, config['cap'], flush=True)
                return result, None
            raise RuntimeError(f'{scene}/{name} failed, exit {completed.returncode}; see archived run.log')
        metrics = json.loads((result / 'metrics.json').read_text())
        if not prefix and not preflight:
            self.records.append(dict(config, **metrics))
            self.checkpoint()
        archive_directory(result, destination)
        return result, metrics

    def paired(self, scene, settings):
        short, _ = self.run_one(scene, settings, prefix=True)
        full, metrics = self.run_one(scene, settings)
        compare_prefix(short, full)
        archive_directory(full, self.args.out / f'{self.args.tag}_{scene}_{settings["name"]}.tar')
        return full, metrics

    def fidelity(self):
        from kv_tracker.dataloaders.tum import TUMLoader
        from kv_tracker.eval_tools.evo_utils import align_pair

        for scene, expected in BASELINE_ATE.items():
            result, _ = self.paired(scene, dict(name='original', policy='original'))
            # Historical reproduction deliberately retains upstream nearest-GT
            # semantics; the experimental headline uses the bounded evaluator.
            reference = TUMLoader.load_gt(str(self.work / 'inputs' / scene))
            estimate = np.load(result / 'traj.npy')
            actual, _, _ = align_pair(dict(traj_gt=reference, traj_est=estimate), ret_np=False)
            actual = float(actual)
            write_json(result / 'historical_gate.json', dict(ate_m=actual, expected_m=expected,
                       tolerance_m=.002, passed=abs(actual - expected) <= .002))
            archive_directory(result, self.args.out / f'{self.args.tag}_{scene}_original.tar')
            assert abs(actual - expected) <= .002, (scene, actual, expected)
            print('BASELINE GATE OK', scene, actual, flush=True)
        # Observational hook fidelity against a genuinely unmodified decision path.
        scene = 'freiburg1_xyz'
        bare, _ = self.run_one(scene, dict(name='bare', policy='bare'), prefix=True)
        observed = self.work / 'runs' / scene / 'original_prefix'
        np.testing.assert_allclose(np.load(bare / 'traj.npy'), np.load(observed / 'traj.npy'),
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_array_equal(np.load(bare / 'kf_idx.npy'), np.load(observed / 'kf_idx.npy'))
        print('HOOK FIDELITY GATE OK', flush=True)

    def choose_cap(self):
        probes = []
        for cap in (64, 48, 32):
            accepted = True
            for scene in LONG_SCENES:
                _, metrics = self.run_one(scene, dict(name=f'probe_{cap}', policy='probe',
                    cap=cap, score='coverage', frames=cap + 8), preflight=True)
                safe = metrics is not None and (
                    metrics['estimated_peak_device_bytes'] <= metrics['device_total_bytes'] - 2 * 1024**3)
                if metrics is not None:
                    assert metrics['keyframes'] == cap
                probes.append(dict(scene=scene, cap=cap, accepted=safe, metrics=metrics))
                write_json(self.work / 'memory_probes.json', probes)
                if not safe:
                    accepted = False
                    break
            if accepted:
                write_json(self.work / 'chosen_cap.json', dict(cap=cap, probes=probes))
                print('MEMORY GATE OK', cap, 'keyframes; 2 GiB estimated headroom', flush=True)
                return cap
        raise RuntimeError('All three higher-cap preflights failed; no long sweep was started')

    def periodic_search(self, scene, semantics, cap):
        # Same-sequence best is an exploratory target, never a held-out optimum.
        best = min(semantics, key=lambda entry: entry['ate_m'])
        limit = best['ate_m'] * 1.05
        tested = []
        budget = min(cap, max(2, best['keyframes']))
        previous = None
        while True:
            interval = max(2, self.manifests[scene]['frames'] // (budget - 1))
            name = f'spread_b{budget}_i{interval}'
            _, metrics = self.paired(scene, dict(name=name, policy='periodic', cap=budget, interval=interval))
            tested.append(dict(name=name, target_budget=budget, interval=interval, **metrics))
            if metrics['ate_m'] <= limit or budget == cap:
                break
            previous = budget
            budget = min(cap, budget * 2)
        # One lower/midpoint probe makes the diagnostic bounded; monotonicity is
        # NOT assumed, and the report says smallest TESTED matching count.
        if tested[-1]['ate_m'] <= limit:
            lower = 2 if previous is None else previous
            middle = (lower + budget) // 2
            if lower < middle < budget:
                interval = max(2, self.manifests[scene]['frames'] // (middle - 1))
                name = f'spread_b{middle}_i{interval}'
                _, metrics = self.paired(scene, dict(name=name, policy='periodic', cap=middle, interval=interval))
                tested.append(dict(name=name, target_budget=middle, interval=interval, **metrics))
        # Include stock and extended originals: otherwise a successful 20-frame
        # original could be hidden behind a worse sequence-spread control.
        original_runs = [entry for entry in self.records if entry['scene'] == scene
                         and entry['policy'] in ('original', 'periodic')]
        matches = [entry for entry in original_runs if entry['ate_m'] <= limit]
        smallest = min(matches, key=lambda entry: entry['keyframes']) if matches else None
        report = dict(scene=scene, target_semantic=best, ate_limit_m=limit, relative_tolerance=.05,
                      sequence_spread_runs=tested, all_original_runs=original_runs,
                      smallest_tested_match=smallest,
                      keyframe_ratio=None if smallest is None else smallest['keyframes'] / best['keyframes'],
                      interpretation='exploratory per-sequence oracle; no global optimum or monotonicity claim')
        write_json(self.work / f'match_{scene}.json', report)

    def execute(self):
        self.stage()
        self.fidelity()
        cap = self.choose_cap()
        for scene in LONG_SCENES:
            self.paired(scene, dict(name='original', policy='original'))
            self.paired(scene, dict(name='original_extended', policy='periodic', cap=cap))
            semantics = []
            for settings in semantic_configs():
                _, metrics = self.paired(scene, dict(settings, cap=cap))
                semantics.append(dict(settings, **metrics))
            self.periodic_search(scene, semantics, cap)
        (self.work / 'JOB_OK').write_text('All gates and configurations completed\n')
        print('JOB OK', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--datasets', type=Path, default=Path('/mnt/datasets/tum-rgbd'))
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    Sweep(options).execute()
