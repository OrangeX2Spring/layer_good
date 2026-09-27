"""CAMP-only tensor and actual Pi3-block contracts for active keyframe access."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from kv_tracker.active_keyframes import ActiveKeyframes
from pi3.models.layers.attention import FlashAttentionRope
from pi3.models.layers.block import BlockRope

# Synthetic banks skip the queries that would record these keyframes' visibility.
SEEN = dict.fromkeys((49, 99, 103), 1.)


class ActiveTests(unittest.TestCase):
    def test_actual_block_subset_and_full_bank_restoration(self):
        torch.manual_seed(4)
        block = BlockRope(dim=16, num_heads=2, qk_norm=True, attn_class=FlashAttentionRope).eval()
        bank = {1: {k: torch.randn(1, 2, 27, 8) for k in ('k', 'v')}}
        model = SimpleNamespace(cache=bank)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        x = torch.randn(1, 9, 16)
        expected = block(x, kv_cache=bank[1], ret_kv=False)
        policy = ActiveKeyframes('all'); policy.bootstrap(model, rgb, 1.); policy.visible.update(SEEN)
        policy.begin(101, [0, 49, 99], rgb, 1.)
        self.assertIs(model.cache, bank)
        torch.testing.assert_close(block(x, kv_cache=model.cache[1], ret_kv=False), expected, rtol=0, atol=0)
        policy.end()
        one = ActiveKeyframes('one'); one.bootstrap(model, rgb, 1.); one.visible.update(SEEN)
        old = {k: t.clone() for k, t in bank[1].items()}
        one.begin(101, [0, 49, 99], rgb, 1.)
        manual = {k: t[:, :, :9] for k, t in bank[1].items()}
        torch.testing.assert_close(block(x, kv_cache=model.cache[1], ret_kv=False),
                                   block(x, kv_cache=manual, ret_kv=False), rtol=0, atol=0)
        one.end()
        self.assertIs(model.cache, bank)
        for k in old:
            torch.testing.assert_close(bank[1][k], old[k], rtol=0, atol=0)
        # A later full query can access history without replay or cache reconstruction.
        policy.begin(102, [0, 49, 99], rgb, 1.)
        torch.testing.assert_close(block(x, kv_cache=model.cache[1], ret_kv=False), expected, rtol=0, atol=0)
        policy.end()

    def test_incremental_count_causal_quantiles_and_newest_selection(self):
        for q in (.25, .5, .6, .75):
            model = SimpleNamespace(cache={1: {k: torch.zeros(1, 2, 40, 8) for k in ('k', 'v')}})
            rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            policy = ActiveKeyframes('motion', q); policy.bootstrap(model, rgb, 1.); policy.visible.update(SEEN)
            policy.scores = [.1] * 64
            policy.begin(104, [0, 49, 99, 103], np.full_like(rgb, 255), 1.)
            self.assertEqual(policy.count, 2)
            self.assertEqual(policy.events[-1]['selected_ids'], [0, 103])
            self.assertAlmostEqual(policy.events[-1]['threshold'], .1)
            policy.end()
            policy.begin(105, [0, 49, 99, 103], np.full_like(rgb, 255), 1.)
            self.assertEqual(policy.count, 2)  # Dwell even with low motion.
            policy.end()
            policy.begin(112, [0, 49, 99, 103], np.full_like(rgb, 255), 1.)
            self.assertEqual(policy.count, 1)
            policy.end()

    def test_leave_one_out_drops_exactly_one_keyframe(self):
        torch.manual_seed(5)
        block = BlockRope(dim=16, num_heads=2, qk_norm=True, attn_class=FlashAttentionRope).eval()
        bank = {1: {k: torch.randn(1, 2, 27, 8) for k in ('k', 'v')}}
        model = SimpleNamespace(cache=bank)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        x = torch.randn(1, 9, 16)
        old = {k: t.clone() for k, t in bank[1].items()}
        for drop, keep in ((49, [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
                           (0, list(range(9, 27)))):
            policy = ActiveKeyframes('drop', drop=drop); policy.bootstrap(model, rgb, 1.); policy.visible.update(SEEN)
            policy.begin(101, [0, 49, 99], rgb, 1.)
            manual = {k: t[:, :, keep] for k, t in bank[1].items()}
            torch.testing.assert_close(block(x, kv_cache=model.cache[1], ret_kv=False),
                                       block(x, kv_cache=manual, ret_kv=False), rtol=0, atol=0)
            self.assertEqual(policy.events[-1]['selected_ids'], [f for f in (0, 49, 99) if f != drop])
            policy.end()
            self.assertIs(model.cache, bank)
        for k in old:
            torch.testing.assert_close(bank[1][k], old[k], rtol=0, atol=0)
        # Not yet admitted: full bank. Only keyframe: kept, including both bootstrap slots.
        pair = SimpleNamespace(cache={1: {k: torch.zeros(1, 2, 18, 8) for k in ('k', 'v')}})
        late = ActiveKeyframes('drop', drop=99); late.bootstrap(pair, rgb, 1.); late.visible.update(SEEN)
        late.begin(60, [0, 49], rgb, 1.)
        self.assertEqual(late.events[-1]['selected_ids'], [0, 49]); late.end()
        anchor = ActiveKeyframes('drop', drop=0); anchor.bootstrap(pair, rgb, 1.); anchor.visible.update(SEEN)
        anchor.begin(1, [0, 0], rgb, 1.)
        self.assertEqual(anchor.events[-1]['selected_physical_ids'], [0, 0]); anchor.end()

    def test_visibility_ranked_subsets_prefer_visible_then_newer(self):
        torch.manual_seed(6)
        block = BlockRope(dim=16, num_heads=2, qk_norm=True, attn_class=FlashAttentionRope).eval()
        bank = {1: {k: torch.randn(1, 2, 45, 8) for k in ('k', 'v')}}
        model = SimpleNamespace(cache=bank)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        x = torch.randn(1, 9, 16)
        ids = [0, 49, 99, 149, 199]
        seen = {49: 1., 99: 1., 149: .6, 199: 1.}
        for mode, rank, expected in (('two', 'newest', [0, 199]), ('two', 'visible', [0, 199]),
                                     ('half', 'newest', [0, 149, 199]), ('half', 'visible', [0, 99, 199])):
            policy = ActiveKeyframes(mode, rank=rank); policy.bootstrap(model, rgb, 1.)
            policy.visible.update(seen)
            policy.begin(200, ids, rgb, 0.)
            self.assertEqual(policy.events[-1]['selected_ids'], expected)
            self.assertEqual(policy.events[-1]['available_visible'], [1., 1., 1., .6, 1.])
            keep = [9 * ids.index(f) + t for f in expected for t in range(9)]
            manual = {k: t[:, :, keep] for k, t in bank[1].items()}
            torch.testing.assert_close(block(x, kv_cache=model.cache[1], ret_kv=False),
                                       block(x, kv_cache=manual, ret_kv=False), rtol=0, atol=0)
            policy.end()
            self.assertIs(model.cache, bank)
            # The query's own visibility is recorded only after its selection.
            self.assertEqual(policy.visible[200], 0.)
        with self.assertRaises(AssertionError):
            ActiveKeyframes('drop', drop=49, rank='visible')

    def test_native_bootstrap_duplicate_counts_as_one(self):
        bank = {1: {k: torch.zeros(1, 2, 18, 8) for k in ('k', 'v')}}
        model = SimpleNamespace(cache=bank)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        policy = ActiveKeyframes('one'); policy.bootstrap(model, rgb, 1.); policy.visible.update(SEEN)
        policy.begin(1, [0, 0], rgb, 1.)
        self.assertIs(model.cache, bank)
        self.assertEqual(policy.events[-1]['selected_ids'], [0])
        self.assertEqual(policy.events[-1]['selected_physical_ids'], [0, 0])
        policy.end()

    def test_count_can_traverse_one_to_k_and_back(self):
        model = SimpleNamespace(cache={1: {k: torch.zeros(1, 2, 40, 8) for k in ('k', 'v')}})
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        policy = ActiveKeyframes('motion'); policy.bootstrap(model, rgb, 1.); policy.visible.update(SEEN)
        for frame, expected in zip((104, 112, 120, 128), (2, 3, 4, 4)):
            policy.scores = [0.] * 64
            rgb = 255 - rgb
            policy.begin(frame, [0, 49, 99, 103], rgb, 1.)
            self.assertEqual(policy.count, expected)
            self.assertEqual(len(policy.events[-1]['selected_ids']), expected)
            policy.end()
        for frame, expected in zip((136, 144, 152, 160), (3, 2, 1, 1)):
            policy.begin(frame, [0, 49, 99, 103], rgb, 1.)
            self.assertEqual(policy.count, expected)
            policy.end()


if __name__ == '__main__':
    unittest.main()
