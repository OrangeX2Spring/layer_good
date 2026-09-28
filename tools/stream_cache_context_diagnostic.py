"""Evaluate read-only causal context probes against same-job object-pose errors."""

import json

import numpy as np
from scipy.stats import spearmanr


def correlation(score, target):
    assert len(score) == len(target) and len(score) >= 3
    assert np.isfinite(score).all() and np.isfinite(target).all()
    if np.ptp(score) == 0 or np.ptp(target) == 0:
        return dict(status='constant signal or target; correlation undefined', rho=None)
    rho = float(spearmanr(score, target).statistic)
    assert np.isfinite(rho)
    return dict(status='ok', rho=rho)


def evaluate_context_diagnostic(root, atol, rtol):
    from stream_cache_sweep import write_json

    rows = [json.loads(line) for line in (root / 'dense8_probe/context_probes.jsonl').read_text().splitlines()]
    assert len(rows) == 128 and [row['frame'] for row in rows] == list(range(128))
    with np.load(root / 'dense8/pose_encodings.npz') as dense, \
            np.load(root / 'dense8_probe/pose_encodings.npz') as observed:
        np.testing.assert_allclose(observed['pose_enc'], dense['pose_enc'], atol=atol, rtol=rtol)
        pose_error = float(np.max(np.abs(observed['pose_enc'] - dense['pose_enc'])))
    # Check every exported dense field, not only the camera readout.
    with np.load(root / 'dense8/frame_000127.npz') as dense, \
            np.load(root / 'dense8_probe/frame_000127.npz') as observed:
        for key in dense.files:
            np.testing.assert_allclose(observed[key], dense[key], atol=atol, rtol=rtol)
    events = [json.loads(line) for line in (root / 'dense8/events.jsonl').read_text().splitlines()]
    probe_events = [json.loads(line) for line in (root / 'dense8_probe/events.jsonl').read_text().splitlines()]
    assert len(events) == len(probe_events) == 128
    for reference, observed in zip(events, probe_events):
        for key in ('retained_frames', 'patches_per_frame', 'state_bytes', 'camera_bytes'):
            assert reference[key] == observed[key], key
    metrics = json.loads((root / 'object_metrics.json').read_text())['rows']
    dense_errors = np.asarray(metrics['dense8']['rpe_translation_m'])
    assert dense_errors.shape == (127,)
    controls = dict(frame_index=np.arange(1, 128),
        dense_local_error=dense_errors,
        negative_dense_confidence=-np.asarray([row['dense_background_confidence_mean'] for row in rows]))
    controls['negative_dense_confidence'] = np.maximum(
        controls['negative_dense_confidence'][:-1], controls['negative_dense_confidence'][1:])
    report = {}
    for name in ('spread64', 'spatial64', 'appearance64'):
        candidate_events = [json.loads(line) for line in (root / name / 'events.jsonl').read_text().splitlines()]
        assert len(candidate_events) == 128
        for index, row in enumerate(rows):
            candidate = row['conditions'][name]
            assert candidate['processed_patches'] == candidate_events[index]['processed_patches']
            assert [layer['layer'] for layer in candidate['layers']] == [0, 4, 11, 17, 23]
            expected_history = events[index - 1]['retained_frames'] if index else []
            assert row['past_frames'] == expected_history
        score = np.asarray([np.mean([
            (layer['history_and_current']['camera_rse'] + layer['history_and_current']['register_rse']) / 2
            for layer in row['conditions'][name]['layers']]) for row in rows])
        variance = np.asarray([row['conditions'][name]['feature_variance'] for row in rows])
        predictors = dict(probe=np.maximum(score[:-1], score[1:]),
                          feature_variance=np.maximum(variance[:-1], variance[1:]), **controls)
        # Positive means compression hurt this adjacent-pair error versus dense8.
        sparse_errors = np.asarray(metrics[name]['rpe_translation_m'])
        assert sparse_errors.shape == dense_errors.shape
        target = sparse_errors - dense_errors
        results = {}
        for label, selection in (('all', slice(None)), ('first_half', slice(0, 63)),
                                 ('second_half', slice(63, None))):
            results[label] = {key: correlation(value[selection], target[selection])
                              for key, value in predictors.items()}
        supported = all(
            results[half]['probe']['rho'] is not None
            and results[half]['probe']['rho'] > 0
            and all(results[half][control]['rho'] is not None
                    and results[half]['probe']['rho'] > results[half][control]['rho']
                    for control in ('feature_variance', 'frame_index'))
            for half in ('first_half', 'second_half'))
        report[name] = dict(correlations=results, exploratory_signal_supported=supported,
                           pair_end_frames=list(range(1, 128)),
                           excess_translation_rpe_m=target.tolist(),
                           predictors={key: value.tolist() for key, value in predictors.items()})
    write_json(root / 'context_diagnostic.json', dict(
        contracts='passed', dense_probe_pose_max_abs=pose_error, conditions=report,
        exploratory_signal_supported=sum(row['exploratory_signal_supported'] for row in report.values()) >= 2,
        next_action='Review archive before selecting or submitting the main experiment',
        caveat='Local counterfactual on dense layer inputs/history, not full compressed-trajectory fidelity. '
               'Scores use causal observations; GT and full-model confidence are offline evaluation only. '
               'Correlations are descriptive, temporally dependent and from one previously inspected clip. '
               'No deployable timing, pose-error bound or novelty claim.'))
    print('CONTEXT DIAGNOSTIC CONTRACTS OK', flush=True)
