"""Run in the existing CAMP image; numerical project code is not run on the Mac."""
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from kv_tracker.correspondence_cache import CorrespondenceCache, match_patches, spatial_pick


class CorrespondenceTests(unittest.TestCase):
    def test_matching_follows_physical_points_and_rejects_inconsistent_geometry(self):
        features = torch.eye(4)
        points = torch.tensor([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.], [3., 0., 1.]])
        permutation = torch.tensor([2, 0, 3, 1])
        links, _ = match_patches(features[permutation], points[permutation], features, points, .1)
        self.assertEqual(links.tolist(), permutation.tolist())
        links, _ = match_patches(features[permutation], points[permutation] + 10, features, points, .1)
        self.assertEqual(links.tolist(), [-1] * 4)
        links, _ = match_patches(features, points, features, points, 0.)
        self.assertEqual(links.tolist(), [-1] * 4)

    def test_identity_changes_an_ambiguous_match(self):
        current = torch.tensor([[1., 0.]])
        history = current.repeat(2, 1)
        point = torch.tensor([[0., 0., 1.]])
        generic, _ = match_patches(current, point, history, point.repeat(2, 1), 1.)
        semantic, _ = match_patches(current, point, history, point.repeat(2, 1), 1.,
                                   torch.tensor([True]), torch.tensor([False, True]))
        self.assertEqual(generic.tolist(), [0])
        self.assertEqual(semantic.tolist(), [1])

    def test_reciprocal_match_does_not_duplicate_one_history_token(self):
        current = torch.tensor([[1., 0.], [1., 0.]])
        links, _ = match_patches(current, torch.zeros(2, 3), current[:1], torch.zeros(1, 3), 1.)
        self.assertEqual(links.tolist(), [0, -1])

    def test_spatial_quota_and_exact_odd_budget(self):
        links = torch.full((64,), -1, dtype=torch.long)
        scores = torch.full((64,), -1.)
        tracks = torch.arange(64)
        # Each 2x2 cell keeps two tokens. Its right column continues tracks.
        links[torch.arange(64) % 2 == 1] = torch.arange(32)
        scores[links >= 0] = 1.
        picked = spatial_pick((8, 8), 32, links, scores, tracks)
        self.assertEqual(picked.tolist(), list(range(1, 64, 2)))
        odd = spatial_pick((8, 8), 31, links, scores, tracks)
        self.assertEqual(len(odd.unique()), 31)
        full = spatial_pick((8, 8), 64, links, scores, tracks)
        self.assertEqual(full.tolist(), list(range(64)))

    def test_semantic_quotas_keep_target_support_at_identical_total_count(self):
        labels = torch.arange(64) % 2 == 0
        links = torch.full((64,), -1, dtype=torch.long)
        links[~labels] = torch.arange(32)
        scores = (links >= 0).float()
        generic = spatial_pick((8, 8), 32, links, scores, torch.arange(64))
        semantic = spatial_pick((8, 8), 32, links, scores, torch.arange(64), labels)
        self.assertEqual(len(generic), len(semantic))
        self.assertEqual(int(labels[generic].sum()), 0)
        self.assertEqual(int(labels[semantic].sum()), 16)

    def test_cache_gather_full_retention_cap_and_no_resurrection(self):
        for mode in ('dense', 'uniform', 'correspondence', 'semantic_correspondence'):
            policy = CorrespondenceCache(mode, budget=3, interval=2)
            model = SimpleNamespace(patch_size=14, patch_start_idx=5,
                encoder=torch.nn.Identity(), decoder=[None] * 4, cache={})
            policy.attach(model)
            yy, xx = torch.meshgrid(torch.arange(112), torch.arange(112), indexing='ij')
            points = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).float()
            ids = [0, 0]
            previous = None
            for index in (0, 2, 4):
                policy.tokens = F.pad(torch.eye(64), (0, 960))
                if index:
                    self.assertTrue(policy.select(index, ids))
                    ids = policy.keep_frame_ids + [index]
                count = len(ids) * 69
                model.cache = {layer: {name: torch.randn(1, 2, count, 8)
                                      for name in ('k', 'v')} for layer in (1, 3)}
                dense = {i: {key: value.clone() for key, value in layer.items()}
                         for i, layer in model.cache.items()}
                policy.after_rebuild(ids, torch.ones(1, len(ids), 112, 112, 1),
                    points=points[None, None].expand(1, len(ids), -1, -1, -1),
                    masks=(xx < 56).numpy()[None].repeat(len(ids), axis=0))
                picked = policy.records[index]['indices']
                self.assertEqual(len(picked), 64 if index == 0 or mode == 'dense' else 32)
                if previous is not None:
                    torch.testing.assert_close(policy.records[2]['indices'], previous)
                if index == 2:
                    previous = picked.clone()
                indices = torch.cat([torch.cat((torch.arange(5), policy.records[i]['indices'] + 5))
                                     + slot * 69 for slot, i in enumerate(ids)])
                query = torch.randn(1, 2, 2, 8)
                mask = torch.zeros(count, dtype=torch.bool)
                mask[indices] = True
                for layer in (1, 3):
                    for name in ('k', 'v'):
                        torch.testing.assert_close(model.cache[layer][name], dense[layer][name][:, :, indices],
                                                   atol=0, rtol=0)
                    expected = F.scaled_dot_product_attention(query, dense[layer]['k'], dense[layer]['v'],
                                                              attn_mask=mask[None, None, None])
                    actual = F.scaled_dot_product_attention(query, model.cache[layer]['k'], model.cache[layer]['v'])
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                if mode == 'dense':
                    self.assertEqual(policy.cache_bytes(), policy.events[-1]['dense_cache_bytes'])
            self.assertFalse(policy.select(6, ids))
            policy.begin_query(6)
            self.assertFalse(policy.capture_enabled)
            policy.close()


if __name__ == '__main__':
    unittest.main()
