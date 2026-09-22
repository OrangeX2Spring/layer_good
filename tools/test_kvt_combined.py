"""Remote CPU contract tests; do not execute project code on the editing Mac."""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from kv_tracker.combined_cache import CombinedCache


class CombinedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.model = SimpleNamespace(patch_size=14, patch_start_idx=5,
            encoder=torch.nn.Identity(), decoder=[None] * 4, cache={})

    def dense_cache(self, frame_ids):
        # Four spatial tokens and five register tokens per frame.
        self.model.cache = {i: {name: torch.randn(1, 2, len(frame_ids) * 9, 8)
            for name in ('k', 'v')} for i in (1, 3)}
        return {i: {name: t.clone() for name, t in layer.items()}
                for i, layer in self.model.cache.items()}

    def bootstrap(self, fraction=.5, budget=3):
        policy = CombinedCache(budget=budget, interval=2, patch_fraction=fraction)
        policy.attach(self.model)
        policy.tokens = torch.randn(4, 1024)
        self.dense_cache([0, 0])
        policy.after_rebuild([0, 0], torch.ones(1, 2, 28, 28, 1))
        return policy

    def insert(self, policy, frame_id, cached_ids, patches=None):
        policy.tokens = torch.randn(4, 1024) if patches is None else patches
        self.assertTrue(policy.select(frame_id, cached_ids))
        ids = policy.keep_frame_ids + [frame_id]
        dense = self.dense_cache(ids)
        confidence = torch.arange(28 * 28).reshape(1, 1, 28, 28, 1).float()
        policy.after_rebuild(ids, confidence.expand(1, len(ids), -1, -1, -1))
        return ids, dense

    def test_sparse_cache_matches_dense_mask_attention(self):
        policy = self.bootstrap()
        ids, dense = self.insert(policy, 1, [0, 0])
        event = policy.events[-1]
        indices = torch.cat([torch.cat((torch.arange(5),
            policy.records[i]['indices'] + 5)) + slot * 9 for slot, i in enumerate(ids)])
        self.assertEqual(len(indices), 16)  # full anchor 9 + special 5 + half 2
        self.assertLess(event['query_cache_bytes'], event['dense_cache_bytes'])
        query = torch.randn(1, 2, 3, 8)
        mask = torch.zeros(18, dtype=torch.bool)
        mask[indices] = True
        for layer in (1, 3):
            sparse = self.model.cache[layer]
            for name in ('k', 'v'):
                torch.testing.assert_close(sparse[name], dense[layer][name][:, :, indices],
                                           atol=0, rtol=0)
            expected = F.scaled_dot_product_attention(query, dense[layer]['k'],
                dense[layer]['v'], attn_mask=mask[None, None, None, :])
            actual = F.scaled_dot_product_attention(query, sparse['k'], sparse['v'])
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_full_retention_is_bit_identical(self):
        policy = self.bootstrap(fraction=1.)
        _, dense = self.insert(policy, 1, [0, 0])
        for layer in (1, 3):
            for name in ('k', 'v'):
                torch.testing.assert_close(self.model.cache[layer][name], dense[layer][name],
                                           atol=0, rtol=0)

    def test_equal_weight_rank_fusion_and_special_tokens(self):
        policy = self.bootstrap()
        policy.tokens = torch.randn(4, 1024)
        self.assertTrue(policy.select(1, [0, 0]))
        policy.pending['novelty'] = torch.tensor([.9, .1, .8, .2])
        self.dense_cache([0, 1])
        confidence = torch.tensor([[0., 3.], [2., 1.]])
        confidence = confidence.repeat_interleave(14, 0).repeat_interleave(14, 1)
        policy.after_rebuild([0, 1], confidence[None, None, ..., None].expand(1, 2, -1, -1, -1))
        self.assertEqual(policy.records[1]['indices'].tolist(), [0, 2])
        self.assertEqual(policy.records[0]['indices'].tolist(), [0, 1, 2, 3])

    def test_repeated_evictions_preserve_anchor_latest_and_patch_ids(self):
        policy = self.bootstrap()
        ids = [0, 0]
        previous = {0: policy.records[0]['indices'].clone()}
        for frame_id in (1, 3, 5, 7, 9):
            ids, _ = self.insert(policy, frame_id, ids)
            self.assertLessEqual(len(ids), 3)
            self.assertEqual(ids[0], 0)
            self.assertEqual(ids[-1], frame_id)
            self.assertEqual(ids, list(policy.records))
            for retained in set(ids) & set(previous):
                torch.testing.assert_close(policy.records[retained]['indices'], previous[retained])
            previous = {i: r['indices'].clone() for i, r in policy.records.items()}
        self.assertEqual(sum(len(e['evicted']) for e in policy.events), 3)

    def test_redundancy_evicts_duplicate_not_distinct_view(self):
        policy = self.bootstrap()
        distinct = torch.zeros(4, 1024)
        distinct[:, 0] = 1
        duplicate = torch.zeros(4, 1024)
        duplicate[:, 1] = 1
        ids, _ = self.insert(policy, 1, [0, 0], distinct)
        ids, _ = self.insert(policy, 3, ids, duplicate)
        ids, _ = self.insert(policy, 5, ids, duplicate)
        self.assertEqual(ids, [0, 1, 5])
        self.assertEqual(policy.events[-1]['evicted'], [3])

    def test_interval_phase_and_non_candidates_leave_cache_unchanged(self):
        policy = self.bootstrap()
        policy.interval = 50
        self.assertFalse(policy.select(48, [0, 0]))
        policy.begin_query(48)
        self.assertFalse(policy.capture_enabled)
        policy.begin_query(49)
        self.assertTrue(policy.capture_enabled)
        policy.tokens = torch.randn(4, 1024)
        self.assertTrue(policy.select(49, [0, 0]))
        self.assertEqual(policy.keep_frame_ids, [0])


if __name__ == '__main__':
    unittest.main()
