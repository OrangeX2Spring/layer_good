"""Run on remote Linux: python -m unittest discover -s tools -p 'test_stream_cache.py'."""

import importlib.util
from pathlib import Path
from types import MethodType
import unittest

import torch
import torch.nn.functional as F

from stream_cache_adapters import sparse_vggt_attention, tensor_bytes
from stream_cache_policy import CachePolicy, PolicyConfig, reciprocal_links


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.patches = torch.eye(4)
        self.confidence = torch.arange(4).float()

    def test_anchor_recent_budget_and_rejected_expiration(self):
        policy = CachePolicy(PolicyConfig(admission='novelty', threshold=0.1, frame_budget=3))
        for frame in range(10):
            _, kept, event = policy.update(frame, self.patches, (2, 2), self.confidence)
            self.assertIn(0, kept)
            self.assertIn(frame, kept)
            self.assertLessEqual(len(kept), 3)
            if frame > 1:
                self.assertEqual(kept, [0, frame])
                self.assertEqual(event['expired_recent'], [frame - 1])

    def test_eviction_protects_anchor_and_recent_for_every_strategy(self):
        for eviction in ('fifo', 'random', 'redundancy'):
            policy = CachePolicy(PolicyConfig(frame_budget=3, eviction=eviction))
            for frame in range(12):
                _, kept, event = policy.update(frame, self.patches, (2, 2), self.confidence)
                self.assertEqual(len(kept), min(frame + 1, 3))
                self.assertIn(0, kept)
                self.assertIn(frame, kept)
                if frame >= 3:
                    self.assertEqual(len(event['evicted']), 1)

    def test_patch_budget_and_confidence_priority(self):
        for mode in ('uniform', 'random', 'confidence', 'novelty', 'confidence_novelty', 'mask'):
            policy = CachePolicy(PolicyConfig(patch_policy=mode, patch_fraction=.5))
            anchor, _, _ = policy.update(0, self.patches, (2, 2), self.confidence)
            self.assertEqual(len(anchor), 4)
            picked, _, _ = policy.update(1, self.patches, (2, 2), self.confidence,
                                         torch.tensor([False, False, True, True]))
            self.assertEqual(len(picked), 2)
            self.assertTrue(bool((picked.diff() > 0).all()))
            if mode in ('confidence', 'mask'):
                torch.testing.assert_close(picked, torch.tensor([2, 3]))

    def test_novelty_detects_unseen_orthogonal_patch(self):
        policy = CachePolicy(PolicyConfig(score='coverage', admission='novelty', threshold=.2))
        old = torch.tensor([[1., 0.]]).repeat(4, 1)
        new = torch.tensor([[0., 1.]]).repeat(4, 1)
        policy.update(0, old, (2, 2), self.confidence)
        _, _, event = policy.update(1, new, (2, 2), self.confidence)
        self.assertEqual(event['score'], 1.)
        self.assertTrue(event['accepted'])

    def test_all_scores_finite_on_centered_anchor_and_refresh(self):
        for score in ('pooled', 'centered', 'coverage', 'chamfer', 'q90'):
            policy = CachePolicy(PolicyConfig(score=score, spatial_grid=2))
            policy.update(0, self.patches, (2, 2), self.confidence)
            _, _, event = policy.update(1, self.patches, (2, 2), self.confidence)
            self.assertTrue(torch.isfinite(torch.tensor(event['score'])))
            policy.clear()
            _, kept, event = policy.update(24, self.patches, (2, 2), self.confidence)
            self.assertEqual(kept, [24])
            self.assertIsNone(event['score'])

    def test_random_is_reproducible(self):
        histories = []
        for _ in range(2):
            policy = CachePolicy(PolicyConfig(frame_budget=3, eviction='random',
                                              patch_policy='random', patch_fraction=.5, seed=12))
            histories.append([policy.update(i, self.patches, (2, 2), self.confidence)[2]
                              for i in range(8)])
        self.assertEqual(*histories)

    def test_correspondence_is_reciprocal_geometric_and_label_consistent(self):
        features = torch.eye(4)
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                               [0., 1., 0.], [1., 1., 0.]])
        links, _ = reciprocal_links(features, points, features, points, .1)
        torch.testing.assert_close(links, torch.arange(4))
        displaced = points + 10
        links, _ = reciprocal_links(features, displaced, features, points, .1)
        self.assertTrue(bool((links == -1).all()))
        labels = torch.tensor([True, False, False, False])
        other = labels.clone()
        other[0] = False
        links, _ = reciprocal_links(features, points, features, points, .1,
                                    labels, other)
        self.assertEqual(int(links[0]), -1)
        torch.testing.assert_close(links[1:], torch.arange(1, 4))

    def test_correspondence_keeps_half_and_surviving_fifo_tracks(self):
        policy = CachePolicy(PolicyConfig(patch_policy='correspondence',
                                          patch_fraction=.5, frame_budget=3))
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                               [0., 1., 0.], [1., 1., 0.]])
        for frame in range(5):
            picked, kept, event = policy.update(frame, self.patches, (2, 2),
                                                 self.confidence, points=points)
            self.assertEqual(len(picked), 4 if frame == 0 else 2)
            self.assertLessEqual(len(kept), 3)
            if frame:
                self.assertGreater(event['matched_kept'], 0)
        self.assertEqual(kept, [0, 3, 4])

    def test_semantic_policy_requires_target_labels(self):
        policy = CachePolicy(PolicyConfig(patch_policy='semantic_correspondence',
                                          patch_fraction=.5))
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                               [0., 1., 0.], [1., 1., 0.]])
        labels = torch.tensor([True, False, False, False])
        policy.update(0, self.patches, (2, 2), self.confidence, labels, points)
        picked, _, event = policy.update(1, self.patches, (2, 2),
                                          self.confidence, labels, points)
        self.assertEqual(len(picked), 2)
        self.assertGreater(event['object_matched_kept'], 0)


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'streamvggt/src/streamvggt/layers/attention.py'
        spec = importlib.util.spec_from_file_location('sweep_test_attention', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.attention = module.Attention

    def test_all_retained_matches_upstream_across_frames(self):
        torch.manual_seed(3)
        original = self.attention(16, num_heads=2, qk_norm=True).eval()
        adapted = self.attention(16, num_heads=2, qk_norm=True).eval()
        adapted.load_state_dict(original.state_dict())
        adapted.forward = MethodType(sparse_vggt_attention, adapted)
        adapted.cache_positions = None
        original_cache = adapted_cache = None
        with torch.no_grad():
            for _ in range(4):
                x = torch.randn(1, 5, 16)
                pos = torch.zeros(1, 5, 2)
                expected, original_cache = original(x, pos=pos, past_key_values=original_cache, use_cache=True)
                actual, adapted_cache = adapted(x, pos=pos, past_key_values=adapted_cache, use_cache=True)
                torch.testing.assert_close(actual, expected)

    def test_sparse_keys_use_matching_positions(self):
        class PositionOffset(torch.nn.Module):
            def forward(self, x, pos):
                return x + pos.sum(-1)[:, None, :, None]

        torch.manual_seed(5)
        attention = self.attention(16, num_heads=2, qk_norm=True, rope=PositionOffset()).eval()
        attention.forward = MethodType(sparse_vggt_attention, attention)
        attention.cache_positions = None
        x = torch.randn(1, 5, 16)
        pos = torch.arange(10).reshape(1, 5, 2).float()
        with torch.no_grad():
            _, cache = attention(x, pos=pos, use_cache=True)
            indices = torch.tensor([0, 3])
            cache = tuple(t.index_select(3, indices) for t in cache)
            attention.cache_positions = attention.cache_positions[:, indices]
            y = torch.randn(1, 5, 16)
            q, k, v = attention.qkv(y).reshape(1, 5, 3, 2, 8).permute(2, 0, 3, 1, 4).unbind(0)
            keys = torch.cat((cache[0].flatten(2, 3), k), dim=2)
            values = torch.cat((cache[1].flatten(2, 3), v), dim=2)
            positions = torch.cat((pos[:, indices], pos), dim=1)
            expected = F.scaled_dot_product_attention(attention.rope(attention.q_norm(q), pos),
                         attention.rope(attention.k_norm(keys), positions), values)
            expected = attention.proj(expected.transpose(1, 2).reshape(1, 5, 16))
            actual, cache = attention(y, pos=pos, past_key_values=cache, use_cache=True)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(cache[0].shape[3], 7)
            self.assertEqual(tensor_bytes(cache), 2 * 1 * 2 * 1 * 7 * 8 * 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
