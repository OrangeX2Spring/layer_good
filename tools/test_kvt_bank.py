"""Remote contracts for whole-image eviction; do not execute on the Mac."""
import unittest
from types import SimpleNamespace

import torch

from kv_tracker.bank_cache import BankCache


class BankTests(unittest.TestCase):
    def test_stale_frame_removed_and_dense_cache_untouched(self):
        for mode in ('recent', 'relevance'):
            policy = BankCache(4, 2, mode)
            policy.model = SimpleNamespace(cache={1: {'k': torch.randn(1, 1, 36, 4)}})
            tensor = policy.model.cache[1]['k']
            before = tensor.clone()
            # Old frame 1 is relevant, frame 3 stale; frame 5 is the predecessor.
            relevant = torch.zeros(1024)
            relevant[0] = 1
            stale = torch.zeros(1024)
            stale[1] = 1
            policy.records = {i: {'descriptor': d} for i, d in
                              ((0, stale), (1, relevant), (3, stale), (5, relevant))}
            policy.tokens = relevant.repeat(4, 1)
            self.assertTrue(policy.select(7, [0, 1, 3, 5]))
            ids = policy.keep_frame_ids + [7]
            self.assertEqual(ids, [0, 3, 5, 7] if mode == 'recent' else [0, 1, 5, 7])
            policy.after_rebuild(ids, torch.ones(1, 4, 28, 28, 1))
            self.assertIs(policy.model.cache[1]['k'], tensor)
            torch.testing.assert_close(tensor, before, rtol=0, atol=0)
            self.assertIsNone(policy.tokens)
            self.assertEqual(list(policy.records), ids)
            self.assertFalse(policy.select(8, ids))

    def test_ties_prefer_recent_and_keep_predecessor(self):
        policy = BankCache(4, 2, 'relevance')
        vector = torch.ones(1024) / 32
        policy.records = {i: {'descriptor': vector} for i in (0, 1, 3, 5)}
        policy.tokens = vector.repeat(4, 1)
        policy.select(7, [0, 1, 3, 5])
        self.assertEqual(policy.keep_frame_ids, [0, 3, 5])
        self.assertEqual(policy.pending['evicted'], [1])


if __name__ == '__main__':
    unittest.main()
