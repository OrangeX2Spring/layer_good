"""CAMP-only contracts: dynamic attention, causal policy, and replay state isolation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from stream_cache_adapters import sparse_vggt_attention
from stream_cache_dynamic import (FEATURES, HIGH, LOW, History, Signals,
                                  attention_indices, budget_for, fit_controller,
                                  history_ids, predict, running_threshold, SCORE_WINDOW, QUANTILES)


class DynamicTests(unittest.TestCase):
    def test_anchor_recent_indices_and_no_future(self):
        self.assertEqual(history_ids(0, LOW), [])
        self.assertEqual(history_ids(3, LOW), [0, 1, 2])
        self.assertEqual(history_ids(20, LOW), [0, 17, 18, 19])
        frames = torch.tensor(history_ids(20, HIGH)).repeat_interleave(3)
        selected = attention_indices(frames, history_ids(20, LOW), 3)
        ids = torch.cat((frames, torch.full((3,), 20)))[selected]
        self.assertEqual(ids.tolist(), [0] * 3 + [17] * 3 + [18] * 3 + [19] * 3 + [20] * 3)
        self.assertEqual(len(selected.unique()), len(selected))

    def test_gather_matches_physical_prune_and_preserves_bank(self):
        torch.manual_seed(8)
        attn = SimpleNamespace(training=False, fused_attn=True, num_heads=2, head_dim=4,
            qkv=torch.nn.Linear(8, 24), q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity(),
            rope=lambda value, positions: value + positions[:, None, :, :1],
            proj_drop=torch.nn.Identity(), proj=torch.nn.Identity())
        old = (torch.randn(1, 2, 1, 12, 4), torch.randn(1, 2, 1, 12, 4))
        before = [t.clone() for t in old]
        x = torch.randn(1, 3, 8)
        pos = torch.arange(6).reshape(1, 3, 2)
        old_pos = torch.arange(24).reshape(1, 12, 2)
        selected = torch.tensor([0, 1, 2, 9, 10, 11, 12, 13, 14])
        attn.cache_positions = old_pos
        attn.active_token_indices = selected
        active, bank = sparse_vggt_attention(attn, x, pos, past_key_values=old, use_cache=True)
        self.assertEqual(bank[0].shape[3], 15)
        torch.testing.assert_close(attn.cache_positions, torch.cat((old_pos, pos), 1))
        del attn.active_token_indices
        attn.cache_positions = old_pos[:, [0, 1, 2, 9, 10, 11]]
        smaller = tuple(t[:, :, :, [0, 1, 2, 9, 10, 11]] for t in old)
        stored, _ = sparse_vggt_attention(attn, x, pos, past_key_values=smaller, use_cache=True)
        torch.testing.assert_close(active, stored, rtol=0, atol=0)
        for value, expected in zip(old, before):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        # All-selected gather is identical to the pre-existing no-gather path.
        attn.cache_positions = old_pos
        native, _ = sparse_vggt_attention(attn, x, pos, past_key_values=old, use_cache=True)
        attn.cache_positions = old_pos
        attn.active_token_indices = torch.arange(15)
        full, _ = sparse_vggt_attention(attn, x, pos, past_key_values=old, use_cache=True)
        torch.testing.assert_close(full, native, rtol=0, atol=0)

    def test_snapshot_copies_mutated_lists(self):
        tensors = [(torch.randn(1, 1, 1, 2, 2), torch.randn(1, 1, 1, 2, 2))]
        attention = SimpleNamespace(cache_positions=torch.zeros(1, 2, 2))
        adapter = SimpleNamespace(host='streamvggt', aggregator_cache=list(tensors),
            camera_cache=list(tensors), token_frames=torch.tensor([0, 0]),
            original_attention=[(attention, None)])
        h = History(adapter, 'active'); state = h.snapshot()
        adapter.aggregator_cache[0] = None
        adapter.camera_cache[0] = None
        attention.cache_positions = None
        adapter.token_frames = torch.tensor([99])
        h.restore(state)
        self.assertIs(adapter.aggregator_cache[0], tensors[0])
        self.assertIs(adapter.camera_cache[0], tensors[0])
        self.assertEqual(adapter.token_frames.tolist(), [0, 0])
        adapter.camera_cache[0] = None
        self.assertIsNotNone(state['camera'][0])

    def test_stored_growth_replays_only_causal_rgb_without_camera(self):
        # Tiny fake aggregator captures the exact rebuild call sequence. The model
        # equivalence gate on CAMP additionally exercises the real frozen backbone.
        calls = []
        attention = SimpleNamespace(cache_positions=torch.zeros(1, 4, 2))
        camera = [object()]
        adapter = SimpleNamespace(host='streamvggt', patch_size=1, special=0,
            camera_cache=camera, token_frames=torch.tensor([0, 21, 22, 23]),
            aggregator_cache=[(torch.zeros(1, 1, 1, 4, 1),) * 2],
            core=SimpleNamespace(aggregator=SimpleNamespace(depth=1)),
            original_attention=[(attention, None)])
        def aggregate(image, past_key_values, use_cache, past_frame_idx):
            calls.append(past_frame_idx)
            old = past_key_values[0]
            count = 1 if old is None else old[0].shape[3] + 1
            tensor = torch.full((1, 1, 1, count, 1), float(past_frame_idx))
            return [], 0, [(tensor, tensor.clone())]
        adapter.model = SimpleNamespace(aggregator=aggregate)
        def forward(image, frame):
            adapter.token_frames = torch.cat((adapter.token_frames, torch.tensor([frame])))
            return {'current': frame}
        adapter.forward = forward
        adapter.memory = lambda: {'aggregator_bytes': 0}
        h = History(adapter, 'stored')
        def prune(ids, patches):
            adapter.token_frames = adapter.token_frames[torch.isin(adapter.token_frames, torch.tensor(ids))]
        h.prune = prune
        frames = [{'source_index': i} for i in range(25)]
        def load(root, row):
            self.assertLess(row['source_index'], 24)
            return torch.zeros(1, 1, 3, 1, 1), None
        with patch('stream_cache_dynamic.load_frame', side_effect=load) as loader:
            output, event = h.forward(torch.zeros(1, 1, 3, 1, 1), 24, HIGH, Path('.'), frames)
        self.assertEqual(calls, history_ids(24, HIGH))
        self.assertEqual(loader.call_count, HIGH)
        self.assertEqual(event['replay_ids'], calls)
        self.assertIs(adapter.camera_cache, camera)
        self.assertEqual(output, {'current': 24})
        self.assertEqual(adapter.token_frames.tolist(), history_ids(25, HIGH))

    def test_controller_fit_serialization_and_fixed_dwell(self):
        x = np.zeros((40, len(FEATURES)))
        x[:, 0] = np.linspace(0, 2, 40)
        model = fit_controller(x, np.expm1(x[:, 0]))
        model = json.loads(json.dumps(model))
        self.assertLess(predict(model, x[0]), model['threshold'])
        self.assertGreater(predict(model, x[-1]), model['threshold'])
        self.assertEqual(budget_for('learned_q50', 16, x[0], model, HIGH, model['threshold']), LOW)
        self.assertEqual(budget_for('learned_q50', 17, x[-1], model, LOW), LOW)
        self.assertEqual(budget_for('learned_q50', 24, x[-1], model, LOW, model['threshold']), HIGH)
        schedule = [budget_for('alternate', i, None, None, LOW) for i in (16, 24, 32)]
        self.assertEqual(schedule, [LOW, HIGH, LOW])
        self.assertTrue(np.isfinite(model['weights']).all())

    def test_running_threshold_shift_scale_window_and_ties(self):
        scores = np.linspace(5., 10., SCORE_WINDOW)
        threshold = running_threshold(scores, .5)
        self.assertEqual(threshold, 7.5)
        self.assertEqual(list(QUANTILES.values()), [.25, .5, .6, .75])
        thresholds = [running_threshold(scores, q) for q in QUANTILES.values()]
        np.testing.assert_allclose(thresholds, [6.25, 7.5, 8., 8.75])
        for q, cutoff in zip(QUANTILES.values(), thresholds):
            self.assertEqual(running_threshold([-999.] + scores.tolist(), q), cutoff)
            self.assertAlmostEqual(running_threshold(3 * scores + 20, q), 3 * cutoff + 20)
        self.assertEqual(running_threshold([-999.] + scores.tolist(), .5), threshold)
        # Positive affine changes preserve relative decisions, unlike a fixed cutoff.
        for score in (6., 9.):
            self.assertEqual(score > threshold,
                             3 * score + 20 > running_threshold(3 * scores + 20, .5))
        self.assertEqual(running_threshold([6.] * 14, .5), 6.)
        model = dict(features=FEATURES, mean=[0.] * 7, std=[1.] * 7,
                     weights=[0., 1., 0., 0., 0., 0., 0., 0.])
        features = [8.25] + [0.] * 6
        self.assertEqual([budget_for(mode, 16, features, model, HIGH, cutoff)
                          for mode, cutoff in zip(QUANTILES, thresholds)],
                         [HIGH, HIGH, HIGH, LOW])
        features = [6.] + [0.] * 6
        self.assertEqual(budget_for('learned_q50', 16, features, model, HIGH, 6.), LOW)
        self.assertEqual(budget_for('learned_q50', 8, features, model, LOW), HIGH)
        self.assertEqual(budget_for('learned_q50', 17, features, model, HIGH, 9.), HIGH)
        # A huge current score cannot raise its own threshold: history is supplied separately.
        features[0] = 1000.
        self.assertEqual(budget_for('learned_q50', 24, features, model, LOW, threshold), HIGH)

    def test_signal_interface_has_no_gt_or_future_predictions(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'current.png'
            Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(path)
            signals = Signals()
            before = signals.before(path)
            self.assertEqual(before[3:], [0., 0., 0., 0.])
            signals.after({'depth_conf': torch.ones(1, 1, 2, 2)}, np.eye(4))
            after = signals.before(path)
            self.assertEqual(before[:3], after[:3])
            self.assertGreater(after[3], 0)
            # Updating prediction signals does not retroactively modify the sample.
            self.assertEqual(before[3:], [0., 0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
