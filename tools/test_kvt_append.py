"""A1 contracts against the actual Pi3 cache block. Execute on CAMP only."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from kv_tracker.append_cache import AppendOnlyCache
from pi3.models.layers.attention import FlashAttentionRope
from pi3.models.layers.block import BlockRope
from kvt_tum_append import insertion_metrics


class AppendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        decoder = torch.nn.ModuleList([
            BlockRope(dim=16, num_heads=2, qk_norm=True,
                      attn_class=FlashAttentionRope).eval() for _ in range(4)])
        self.model = SimpleNamespace(decoder=decoder, cache={
            i: {name: torch.randn(1, 2, 18, 8) for name in ('k', 'v')} for i in (1, 3)})
        self.policy = AppendOnlyCache([49, 99], verify=True)
        self.policy.attach(self.model)
        self.addCleanup(self.policy.close)

    def query(self, hidden):
        for i, block in enumerate(self.model.decoder):
            hidden = (block(hidden, kv_cache=self.model.cache[i], ret_kv=False)
                      if i % 2 else block(hidden))
        return hidden

    def test_real_block_capture_matches_native_query_and_preserves_history(self):
        for frame in (49, 99):
            hidden = torch.randn(1, 9, 16)
            old = {i: dict(layer) for i, layer in self.model.cache.items()}
            expected = self.query(hidden)
            self.policy.begin_query(frame)
            actual = self.query(hidden)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for i in (1, 3):
                for name in ('k', 'v'):
                    self.assertIs(self.model.cache[i][name], old[i][name])
                    tail = self.policy.pending[i][name]
                    self.assertEqual(tail.shape, (1, 2, 9, 8))
                    self.assertEqual(tail.untyped_storage().nbytes(), tail.numel() * tail.element_size())
            pending = self.policy.pending.copy()
            self.policy.commit(frame)
            for i in (1, 3):
                for name in ('k', 'v'):
                    expected_cache = torch.cat((old[i][name], pending[i][name]), dim=2)
                    torch.testing.assert_close(self.model.cache[i][name], expected_cache, atol=0, rtol=0)
            self.assertFalse(self.policy.pending)
        self.assertEqual(self.policy.frame_ids, [0, 0, 49, 99])

    def test_tail_matches_native_normalized_keys_and_values(self):
        # The native returned K/V is the oracle; this includes normalization and
        # any RoPE configured on the real block (none on this small CPU fixture).
        block = self.model.decoder[1]
        hidden = torch.randn(1, 9, 16)
        _, keys, values = block(hidden, kv_cache=self.model.cache[1], ret_kv=True)
        self.policy.begin_query(49)
        block(hidden, kv_cache=self.model.cache[1], ret_kv=False)
        torch.testing.assert_close(self.policy.pending[1]['k'], keys[:, :, -9:], atol=0, rtol=0)
        torch.testing.assert_close(self.policy.pending[1]['v'], values[:, :, -9:], atol=0, rtol=0)

    def test_unselected_query_does_not_mutate_or_capture(self):
        before = {i: dict(layer) for i, layer in self.model.cache.items()}
        self.policy.begin_query(48)
        self.query(torch.randn(1, 9, 16))
        self.assertFalse(self.policy.pending)
        for i in (1, 3):
            for name in ('k', 'v'):
                self.assertIs(self.model.cache[i][name], before[i][name])

    def test_incomplete_capture_cannot_commit(self):
        self.policy.begin_query(49)
        with self.assertRaises(AssertionError):
            self.policy.commit(49)

    def test_insertion_diagnostic_uses_first_query_after_insertion(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary)
            np.save(result / 'kf_idx.npy', np.array([0, 2]))
            positions = np.zeros((5, 3))
            positions[:, 0] = [0, 1, 2, 12, 13]
            np.savez(result / 'evaluation.npz', rpe_pair_start_indices=[0, 2, 3],
                     rpe_translation_per_pair_m=[.1, 9., .3],
                     full_aligned_positions=positions, alignment_scale=2.)
            values = insertion_metrics(result)
            self.assertEqual(values['insertion_pair_start_indices'], [2])
            self.assertEqual(values['insertion_step_rms_m'], 10.)
            self.assertEqual(values['insertion_rpe_t_rms_m'], 9.)
            self.assertAlmostEqual(values['noninsertion_rpe_t_rms_m'], np.sqrt(.05))


if __name__ == '__main__':
    unittest.main()
