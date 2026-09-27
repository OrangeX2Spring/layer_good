"""CAMP-only tensor and actual Pi3-block contracts for active keyframe access."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from kv_tracker.active_keyframes import ActiveKeyframes
from pi3.models.layers.attention import FlashAttentionRope
from pi3.models.layers.block import BlockRope


class ActiveTests(unittest.TestCase):
    def test_actual_block_subset_and_full_bank_restoration(self):
        torch.manual_seed(4)
        block = BlockRope(dim=16, num_heads=2, qk_norm=True, attn_class=FlashAttentionRope).eval()
        bank = {1: {k: torch.randn(1, 2, 27, 8) for k in ('k', 'v')}}
        model = SimpleNamespace(cache=bank)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        x = torch.randn(1, 9, 16)
        expected = block(x, kv_cache=bank[1], ret_kv=False)
        policy = ActiveKeyframes('all'); policy.bootstrap(model, rgb)
        policy.begin(101, [0, 49, 99], rgb)
        self.assertIs(model.cache, bank)
        torch.testing.assert_close(block(x, kv_cache=model.cache[1], ret_kv=False), expected, rtol=0, atol=0)
        policy.end()
        one = ActiveKeyframes('one'); one.bootstrap(model, rgb)
        old = {k: t.clone() for k, t in bank[1].items()}
        one.begin(101, [0, 49, 99], rgb)
        manual = {k: t[:, :, :9] for k, t in bank[1].items()}
        torch.testing.assert_close(block(x, kv_cache=model.cache[1], ret_kv=False),
                                   block(x, kv_cache=manual, ret_kv=False), rtol=0, atol=0)
        one.end()
        self.assertIs(model.cache, bank)
        for k in old:
            torch.testing.assert_close(bank[1][k], old[k], rtol=0, atol=0)
        # A later full query can access history without replay or cache reconstruction.
        policy.begin(102, [0, 49, 99], rgb)
        torch.testing.assert_close(block(x, kv_cache=model.cache[1], ret_kv=False), expected, rtol=0, atol=0)
        policy.end()

    def test_incremental_count_causal_quantiles_and_newest_selection(self):
        for q in (.25, .5, .6, .75):
            model = SimpleNamespace(cache={1: {k: torch.zeros(1, 2, 40, 8) for k in ('k', 'v')}})
            rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            policy = ActiveKeyframes('motion', q); policy.bootstrap(model, rgb)
            policy.scores = [.1] * 64
            policy.begin(104, [0, 49, 99, 103], np.full_like(rgb, 255))
            self.assertEqual(policy.count, 2)
            self.assertEqual(policy.events[-1]['selected_ids'], [0, 103])
            self.assertAlmostEqual(policy.events[-1]['threshold'], .1)
            policy.end()
            policy.begin(105, [0, 49, 99, 103], np.full_like(rgb, 255))
            self.assertEqual(policy.count, 2)  # Dwell even with low motion.
            policy.end()
            policy.begin(112, [0, 49, 99, 103], np.full_like(rgb, 255))
            self.assertEqual(policy.count, 1)
            policy.end()

    def test_native_bootstrap_duplicate_counts_as_one(self):
        bank = {1: {k: torch.zeros(1, 2, 18, 8) for k in ('k', 'v')}}
        model = SimpleNamespace(cache=bank)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        policy = ActiveKeyframes('one'); policy.bootstrap(model, rgb)
        policy.begin(1, [0, 0], rgb)
        self.assertIs(model.cache, bank)
        self.assertEqual(policy.events[-1]['selected_ids'], [0])
        self.assertEqual(policy.events[-1]['selected_physical_ids'], [0, 0])
        policy.end()

    def test_count_can_traverse_one_to_k_and_back(self):
        model = SimpleNamespace(cache={1: {k: torch.zeros(1, 2, 40, 8) for k in ('k', 'v')}})
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        policy = ActiveKeyframes('motion'); policy.bootstrap(model, rgb)
        for frame, expected in zip((104, 112, 120, 128), (2, 3, 4, 4)):
            policy.scores = [0.] * 64
            rgb = 255 - rgb
            policy.begin(frame, [0, 49, 99, 103], rgb)
            self.assertEqual(policy.count, expected)
            self.assertEqual(len(policy.events[-1]['selected_ids']), expected)
            policy.end()
        for frame, expected in zip((136, 144, 152, 160), (3, 2, 1, 1)):
            policy.begin(frame, [0, 49, 99, 103], rgb)
            self.assertEqual(policy.count, expected)
            policy.end()


if __name__ == '__main__':
    unittest.main()
