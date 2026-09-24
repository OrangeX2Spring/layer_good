"""LLM / uniform / random keyframe comparison on full-office TUM. CAMP only.

Runs six conditions in isolated processes against the same staged inputs:
    stock       — instrumented stock interval-50/cap-20 (saves actual IDs)
    llm         — frozen LLM selection replayed via fixed policy
    uniform     — full-clip uniform spacing, 20 frames including bootstrap
    random_s0   — random 20 frames, seed 0
    random_s1   — random 20 frames, seed 1
    random_s2   — random 20 frames, seed 2

Every condition uses 20 unique keyframes (including bootstrap frame 0),
identical staged inputs with verified pixel hashes, the same model revision
and container. The stock run is instrumented (not bare) so its decisions log
is available for diagnostic comparison.

The LLM selection is frozen in this file (LLM_SELECTION below). It was
produced in a score-blind subscription session from the job 25851 contact
sheet packet. Provenance: model label "GPT-6 (Codex)", date 2026-09-24,
19 insertion indices (bootstrap frame 0 implicit). The original
selection.json and inspection_log.md are preserved locally at
cluster_results/kvt_tum/tum_25851_llm_packet/llm_packet/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random as random_module
import subprocess
import sys

import numpy as np

from kvt_tum_run import periodic_indices, prepare, write_json
from kvt_tum_sweep import archive_directory, archive_inputs, release_page_cache


SCENE = 'freiburg3_long_office_household'
BUDGET = 20  # Including bootstrap frame 0.

# Frozen LLM selection: 19 insertion indices (bootstrap 0 is implicit).
# Source: cluster_results/kvt_tum/tum_25851_llm_packet/llm_packet/selection.json
# Packet job: 25851. Model label: "GPT-6 (Codex)". Date: 2026-09-24.
# Fidelity gate: job 25855 (stock-ID replay passed, max_pose_difference=0.0).
LLM_SELECTION = [186, 294, 414, 546, 606, 726, 834, 918,
                 1194, 1266, 1412, 1506, 1712, 1914, 1999,
                 2118, 2262, 2394, 2582]
LLM_PROVENANCE = dict(
    model_label='GPT-6 (Codex)',
    user_reported_model='gpt-6-astra low',
    date='2026-09-24',
    packet_job=25851,
    fidelity_job=25855,
    local_path='cluster_results/kvt_tum/tum_25851_llm_packet/llm_packet/selection.json',
    selection_json_sha256='0b27ac5d97738745eda22761208e176d06c730453ed8ab997c1da049bd6fecde',
    inspection_log_sha256='ef399e0d28bccb0e96607cb8a348fc8515ca93957c32d7d1e2a7019b22682ba3',
    note='Score-blind subscription session; indices frozen before scoring')

# Inputs the selector saw, from the job 25851 packet manifest. The RGB digest is
# sha256 over the concatenated per-frame rgb_sha256 hex strings in index order;
# rgb_sha256 there is prepare()'s model_rgb_sha256.
PACKET_SOURCE_ZIP_SHA256 = 'fc7a09a2748f03c8329aee4b6694ba599129d85eb656738ce490e569bf56dc01'
PACKET_RGB_DIGEST = 'e2f8703186c8d3385dedfbceaee9357992e4e8e1bd93206c0aa79f1847a68b1e'
PACKET_FRAMES = 2585

assert LLM_SELECTION == sorted(set(LLM_SELECTION))
assert all(0 < i for i in LLM_SELECTION)
assert len(LLM_SELECTION) == BUDGET - 1


def uniform_indices(length, budget):
    """Full-clip uniform spacing: budget frames including bootstrap at 0.

    Places budget-1 additional frames at evenly-spaced positions across
    [1, length-1]. This uses the clip length, so it is an offline control.
    """
    assert budget >= 2 and length >= budget
    step = (length - 1) / (budget - 1)
    return [int(round(i * step)) for i in range(1, budget)]


def random_indices(length, budget, seed):
    """Random budget-1 unique frames from [1, length-1], sorted."""
    assert budget >= 2 and length > budget
    rng = random_module.Random(seed)
    chosen = sorted(rng.sample(range(1, length), budget - 1))
    return chosen


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()

    llm_indices = list(LLM_SELECTION)
    print('LLM SELECTION', len(llm_indices), 'frozen indices', flush=True)

    # --- Stage the scene ---
    source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{SCENE}.zip'
    staged = args.work / 'inputs' / SCENE
    manifest = prepare(source, staged, 308, .02)
    length = manifest['frames']

    # Validate LLM indices against actual frame count.
    assert all(i < length for i in llm_indices), \
        f'LLM index {max(llm_indices)} >= frame count {length}'

    # Record source ZIP hash for provenance.
    source_sha = sha256_file(source)
    (staged / 'archive.sha256').write_text(f'{source_sha}  {source}\n')

    # Staged model inputs must be the pixels the LLM selected from.
    assert [r['index'] for r in manifest['inputs']] == list(range(length))
    rgb_digest = hashlib.sha256(''.join(
        r['model_rgb_sha256'] for r in manifest['inputs']).encode()).hexdigest()
    assert length == PACKET_FRAMES, (length, PACKET_FRAMES)
    assert source_sha == PACKET_SOURCE_ZIP_SHA256, source_sha
    assert rgb_digest == PACKET_RGB_DIGEST, rgb_digest
    print('PACKET INPUT HASHES OK', length, flush=True)

    # Archive the inputs (metadata + model_rgb, not source PNGs).
    inputs_archive = args.out / f'{args.tag}_inputs_{SCENE}.tar'
    archive_inputs(staged, inputs_archive)
    release_page_cache(staged, inputs_archive)

    # --- Build the six conditions ---
    base = dict(scene=SCENE, scene_dir=str(staged), frames=length,
                resize_dim=308, interval=50, cap=BUDGET,
                max_gt_difference=.02, evaluate_trajectory=True)

    conditions = []

    # 1. Stock: instrumented original decisions
    conditions.append(('stock', dict(base, name='stock', policy='original')))

    # 2. LLM: frozen selection
    conditions.append(('llm', dict(base, name='llm', policy='fixed',
                                   insertion_indices=llm_indices)))

    # 3. Uniform: full-clip uniform spacing
    u_indices = uniform_indices(length, BUDGET)
    conditions.append(('uniform', dict(base, name='uniform', policy='fixed',
                                       insertion_indices=u_indices)))

    # 4-6. Random seeds 0, 1, 2
    for seed in (0, 1, 2):
        r_indices = random_indices(length, BUDGET, seed)
        conditions.append((f'random_s{seed}',
                           dict(base, name=f'random_s{seed}', policy='fixed',
                                insertion_indices=r_indices)))

    # --- Write the protocol ---
    write_json(args.work / 'protocol.json', dict(
        base=base, stage='llm-compare',
        budget=BUDGET, scene=SCENE, frames=length,
        conditions=[name for name, _ in conditions],
        llm_selection=dict(indices=llm_indices, **LLM_PROVENANCE),
        uniform_indices=u_indices,
        random_seeds=[0, 1, 2],
        stock_expected_ids=periodic_indices(length, 50, BUDGET),
        source_zip_sha256=source_sha,
        comparison='LLM vs stock vs full-clip uniform vs random seeds 0/1/2',
        note='All conditions use 20 unique keyframes including bootstrap frame 0'))

    # --- Run each condition ---
    results = {}
    all_metrics = {}
    for name, config in conditions:
        result = args.work / 'runs' / name
        result.mkdir(parents=True)
        write_json(result / 'config.json', config)
        print('RUN', name, length, 'frames', flush=True)
        with (result / 'run.log').open('w') as log:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('kvt_tum_run.py')),
                 str(result / 'config.json')],
                stdout=log, stderr=subprocess.STDOUT)
        write_json(result / 'process.json', dict(returncode=completed.returncode))

        # Archive the result.
        archive = args.out / f'{args.tag}_{SCENE}_{name}.tar'
        archive_directory(result, archive)
        release_page_cache(result, archive)

        if completed.returncode:
            print((result / 'run.log').read_text()[-8000:], flush=True)
            raise RuntimeError(f'{name} failed: exit {completed.returncode}; see archived run.log')

        results[name] = result
        metrics = json.loads((result / 'metrics.json').read_text())
        all_metrics[name] = metrics
        print('RUN OK', name, metrics.get('ate_m'), metrics.get('keyframes'), flush=True)

    # --- Verify the stock run matches expected periodic indices ---
    stock_ids = np.load(results['stock'] / 'kf_idx.npy').tolist()
    expected_stock = periodic_indices(length, 50, BUDGET)
    assert stock_ids == expected_stock, f'Stock IDs mismatch: {stock_ids} != {expected_stock}'

    # --- Write provenance ---
    write_json(args.work / 'input_provenance.json', dict(
        source_zip=str(source), source_zip_sha256=source_sha,
        staged_frames=length, resize_dim=308,
        packet_job=25851, packet_rgb_digest=rgb_digest, packet_inputs_match=True,
        note='Staged model-input hashes match the LLM packet; per-frame pixel '
             'hashes verified inside each run by kvt_tum_run.py'))

    # --- Write comparison summary ---
    comparison = {}
    for name in [n for n, _ in conditions]:
        m = all_metrics[name]
        comparison[name] = dict(
            ate_m=m['ate_m'],
            rpe_translation_m=m['rpe_translation_m'],
            rpe_rotation_deg=m['rpe_rotation_deg'],
            keyframes=m['keyframes'],
            last_keyframe=m['last_keyframe'],
            tail_frames=m['tail_frames'],
            seconds=m['seconds'],
            peak_allocated_bytes=m['peak_allocated_bytes'],
            evaluated_frames=m['evaluated_frames'])

    write_json(args.work / 'comparison.json', dict(
        scene=SCENE, frames=length, budget=BUDGET,
        conditions=comparison,
        provenance=dict(llm=LLM_PROVENANCE,
                        source_zip_sha256=source_sha, packet_rgb_digest=rgb_digest,
                        note='Stock-ID fidelity passed in job 25855. Revisions and '
                             'container SHA of this job are in the context archive; '
                             'compare them with 25855 at review')))

    print('COMPARISON', json.dumps(
        {name: f"ATE={v['ate_m']:.4f}" for name, v in comparison.items()}), flush=True)
    (args.work / 'JOB_OK').write_text(
        'LLM/stock/uniform/random comparison completed.\n')
    print('LLM COMPARE JOB OK', flush=True)


if __name__ == '__main__':
    main()
