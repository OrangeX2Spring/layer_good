"""Patch-select contract tests. Run in the CAMP image before any patch-select run;
numerical project code is never run on the editing Mac.

CPU: quota rounding, image scores and the mask-cut exclusion, exact K and
determinism for every policy, K7/K8 history, K10 non-finite handling, and the
cache gather (persistence, anchor, bytes, attention equal to masked dense keys).
GPU: the K9 register-attention hook reproduces the real final global block, and
gathered post-RoPE keys equal masked dense keys through that real block.
"""
import math
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from kv_tracker.patch_select import (CANDIDATES, CLUSTERS, POLICIES, PatchSelectCache, allocate,
                                     cells, image_scores, register_attention, stratified)

H, W = 224, 308  # 16 x 22 = 352 patches, the TUM protocol grid
GRID = (H // 14, W // 14)
P = GRID[0] * GRID[1]
K = math.ceil(P / 4)


def textured(seed):
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def object_frame(seed):
    rgb = textured(seed)
    mask = np.zeros((H, W), dtype=bool)
    mask[40:150, 60:200] = True
    return rgb * mask[..., None], mask


def fake_model():
    return SimpleNamespace(patch_size=14, patch_start_idx=5, encoder=torch.nn.Identity(),
                           decoder=[torch.nn.Identity() for _ in range(36)], cache={})


def policy_cache(policy, task, seed=0):
    """A cache that has seen its bootstrap, with synthetic arrival signals."""
    cache = PatchSelectCache(policy, task, [0, 5, 9])
    cache.attach(fake_model())
    generator = torch.Generator().manual_seed(seed)
    features = F.normalize(torch.randn(P, 1024, generator=generator), dim=-1)
    if policy in ('K7', 'K8'):
        cache.previous = dict(features=features, age=torch.zeros(P, dtype=torch.long))
    if policy == 'K8':
        cache.retained = [features]
    cache.tokens = F.normalize(features + .01 * torch.randn(P, 1024, generator=generator), dim=-1)
    cache.demand = torch.rand(P, generator=generator)
    return cache


def choose(cache, rgb, mask, frame=5, confidence=None, points=None):
    if confidence is None:
        confidence = torch.rand(H, W, generator=torch.Generator().manual_seed(frame))
    if points is None:
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        points = torch.stack((x, y, torch.ones_like(x)), -1).float()
    return cache.choose(frame, GRID, K, rgb, mask, confidence, points)


class AllocationTests(unittest.TestCase):
    def test_largest_remainder_prefers_lower_group(self):
        self.assertEqual(allocate(10, [5, 5, 5], [1, 1, 1]).tolist(), [4, 3, 3])

    def test_clipping_redistributes_by_weight(self):
        quota = allocate(10, [1, 20, 20], [1, 1, 1])
        self.assertEqual(quota.tolist(), [1, 5, 4])

    def test_zero_weight_groups_absorb_the_rest(self):
        self.assertEqual(allocate(5, [2, 10], [1, 0]).tolist(), [2, 3])

    def test_bounds(self):
        self.assertEqual(allocate(0, [3, 3], [1, 1]).tolist(), [0, 0])
        with self.assertRaises(AssertionError):
            allocate(7, [3, 3], [1, 1])

    def test_stratified_quotas_and_ranking(self):
        score = torch.arange(P, dtype=torch.float64)
        groups = cells(GRID)
        picked = stratified(score, groups, K, torch.ones(P, dtype=torch.bool))
        sizes = torch.bincount(groups)
        expected = allocate(K, sizes, sizes)
        self.assertEqual(torch.bincount(groups[picked], minlength=16).tolist(), expected.tolist())
        for g in range(16):
            members = (groups == g).nonzero().flatten()
            # Highest scores (here: highest IDs) inside each cell.
            self.assertEqual(sorted(picked[groups[picked] == g].tolist()),
                             members[-int(expected[g]):].tolist() if expected[g] else [])


class ImageScoreTests(unittest.TestCase):
    def test_corner_is_patch_max_of_opencv_response(self):
        rgb = textured(1)
        corner, gradient, fraction = image_scores(rgb, np.ones((H, W), dtype=bool))
        response = cv2.cornerMinEigenVal(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), 3, 3)
        blocks = response.reshape(GRID[0], 14, GRID[1], 14).max(axis=(1, 3)).flatten()
        np.testing.assert_allclose(corner.numpy(), blocks, rtol=1e-6)
        self.assertTrue(torch.isfinite(gradient).all() and (fraction == 1).all())

    def test_mask_cut_edge_is_not_texture(self):
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=bool)
        mask[40:150, 60:200] = True
        rgb[mask] = 180  # a flat object: its only edge is the artificial cut
        corner, gradient, fraction = image_scores(rgb, mask)
        finite = torch.isfinite(corner)
        self.assertTrue(finite.any())
        self.assertLessEqual(float(corner[finite].max()), 1e-6)
        self.assertLessEqual(float(gradient[torch.isfinite(gradient)].max()), 1e-6)
        # Without the exclusion the same pixels do respond.
        unmasked, _, _ = image_scores(rgb, np.ones((H, W), dtype=bool))
        self.assertGreater(float(unmasked.max()), 1.)
        self.assertEqual(float(fraction.max()), 1.)
        self.assertEqual(float(fraction.min()), 0.)


class PolicyTests(unittest.TestCase):
    def test_every_policy_is_exact_sorted_and_deterministic(self):
        for task in ('object', 'scene'):
            for flat in (False, True):
                rgb, mask = object_frame(2) if task == 'object' else (textured(2), np.ones((H, W), bool))
                if flat:
                    rgb = np.zeros_like(rgb)
                for policy in POLICIES:
                    with self.subTest(task=task, flat=flat, policy=policy):
                        first, details = choose(policy_cache(policy, task), rgb, mask)
                        second, _ = choose(policy_cache(policy, task), rgb, mask)
                        self.assertEqual(first.tolist(), second.tolist())
                        expected = P if policy == 'all' else K
                        self.assertEqual(len(first), expected)
                        self.assertEqual(first.tolist(), sorted(set(first.tolist())))
                        self.assertTrue(0 <= int(first.min()) and int(first.max()) < P)
                        if policy != 'all':
                            self.assertIn('weak_corner_selected', details)

    def test_candidates_differ_from_uniform_on_texture(self):
        rgb, mask = textured(3), np.ones((H, W), bool)
        uniform, _ = choose(policy_cache('uniform', 'scene'), rgb, mask)
        for policy in CANDIDATES:
            picked, _ = choose(policy_cache(policy, 'scene'), rgb, mask)
            self.assertNotEqual(picked.tolist(), uniform.tolist(), policy)

    def test_random_controls_depend_on_seed_and_frame_only(self):
        rgb, mask = textured(4), np.ones((H, W), bool)
        a, _ = choose(policy_cache('random17', 'scene'), rgb, mask, frame=5)
        b, _ = choose(policy_cache('random17', 'scene', seed=9), textured(5), mask, frame=5)
        c, _ = choose(policy_cache('random17', 'scene'), rgb, mask, frame=9)
        d, _ = choose(policy_cache('random29', 'scene'), rgb, mask, frame=5)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertNotEqual(a.tolist(), c.tolist())
        self.assertNotEqual(a.tolist(), d.tolist())

    def test_object_region_split(self):
        rgb, mask = object_frame(6)
        picked, details = choose(policy_cache('K4', 'object'), rgb, mask)
        # Majority-covered patches: rows 3-10 x columns 4-13. 80 exceeds 75% of
        # K = 88, so the split is exact.
        self.assertEqual(details['target_patches'], 80)
        self.assertEqual(details['target_selected'], 66)
        small = np.zeros((H, W), dtype=bool)
        small[42:70, 62:90] = True
        picked, details = choose(policy_cache('K4', 'object'), rgb * small[..., None], small)
        self.assertEqual(details['target_selected'], details['target_patches'])
        self.assertEqual(len(picked), K)

    def test_scene_clusters_and_boundaries(self):
        rgb, mask = textured(7), np.ones((H, W), bool)
        _, details = choose(policy_cache('K4', 'scene'), rgb, mask)
        self.assertEqual(len(details['cluster_sizes']), CLUSTERS)
        self.assertEqual(sum(details['cluster_sizes']), P)
        _, details = choose(policy_cache('K5', 'scene'), rgb, mask)
        self.assertGreater(details['boundary_patches'], 0)

    def test_persistent_landmarks_follow_tracks(self):
        rgb, mask = textured(8), np.ones((H, W), bool)
        cache = policy_cache('K7', 'scene')
        previous = cache.previous['features']
        permutation = torch.randperm(P, generator=torch.Generator().manual_seed(0))
        cache.previous['age'] = torch.arange(P) % 5
        cache.tokens = previous[permutation]  # every patch reappears elsewhere
        picked, details = choose(cache, rgb, mask)
        self.assertEqual(details['matched'], P)
        self.assertEqual(details.get('k2_fill', 0), 0)
        ages = cache.previous['age']
        torch.testing.assert_close(ages, (torch.arange(P) % 5)[permutation] + 1)
        self.assertGreaterEqual(int(ages[picked].min()), int(ages.sort(descending=True).values[K - 1]))
        # Only the latest admission is kept: bounded history.
        self.assertIs(cache.previous['features'], cache.tokens)
        cache.tokens = -previous  # nothing reaches the cosine floor
        _, details = choose(cache, rgb, mask, frame=9)
        self.assertEqual(details['matched'], 0)
        self.assertEqual(details['k2_fill'], K)

    def test_stable_plus_novel_splits_and_grows_retained(self):
        rgb, mask = textured(9), np.ones((H, W), bool)
        cache = policy_cache('K8', 'scene')
        before = sum(len(r) for r in cache.retained)
        picked, details = choose(cache, rgb, mask)
        self.assertEqual(len(picked), K)
        self.assertEqual(sum(len(r) for r in cache.retained) - before, K)

    def test_geometry_excludes_nonfinite_points(self):
        rgb, mask = textured(10), np.ones((H, W), bool)
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        points = torch.stack((x, y, torch.ones_like(x)), -1).float()
        points[:112] = float('nan')  # the top half of the patch rows
        picked, details = choose(policy_cache('K10', 'scene'), rgb, mask, points=points)
        self.assertEqual(details['nonfinite_points'], P // 2)
        self.assertEqual(details.get('k2_fill', 0), 0)
        self.assertTrue((picked >= P // 2).all())
        points[:] = float('nan')
        picked, details = choose(policy_cache('K10', 'scene'), rgb, mask, points=points)
        self.assertEqual(details['k2_fill'], K)


class CacheTests(unittest.TestCase):
    def run_sequence(self, policy):
        cache = PatchSelectCache(policy, 'scene', [0, 5, 9])
        model = fake_model()
        cache.attach(model)
        generator = torch.Generator().manual_seed(1)
        history = {}
        for ids in ([0, 0], [0, 5], [0, 5, 9]):
            if ids[-1]:
                cache.begin_query(ids[-1])
            if cache.capturing:
                cache.capture(None, None, dict(x_norm_patchtokens=torch.randn(len(ids), P, 1024)))
            if ids[-1]:
                cache.end_query()
            count = len(ids) * (5 + P)
            model.cache = {layer: {name: torch.randn(1, 2, count, 8, generator=generator)
                                   for name in ('k', 'v')} for layer in range(1, 36, 2)}
            dense = {i: dict(layer) for i, layer in model.cache.items()}
            rgb = np.stack([textured(i) for i in range(len(ids))])
            cache.after_rebuild(ids, torch.rand(1, len(ids), H, W, 1), torch.rand(1, len(ids), H, W, 3),
                                np.ones((len(ids), H, W), dtype=bool), rgb)
            event = cache.events[-1]
            for frame, picked in history.items():
                self.assertEqual(cache.selected[frame].tolist(), picked)  # choices persist
            history[ids[-1]] = cache.selected[ids[-1]].tolist()
            rows = [torch.cat((torch.arange(5), cache.selected[f] + 5)) + slot * (5 + P)
                    for slot, f in enumerate(ids)]
            indices = torch.cat(rows)
            self.assertEqual(event['retained_tokens'], len(indices))
            mask = torch.zeros(count, dtype=torch.bool)
            mask[indices] = True
            query = torch.randn(1, 2, 3, 8, generator=generator)
            for layer in model.cache:
                if policy == 'all':
                    self.assertIs(model.cache[layer]['k'], dense[layer]['k'])
                torch.testing.assert_close(model.cache[layer]['k'], dense[layer]['k'][:, :, indices],
                                           atol=0, rtol=0)
                expected = F.scaled_dot_product_attention(query, dense[layer]['k'], dense[layer]['v'],
                                                          attn_mask=mask[None, None, None])
                actual = F.scaled_dot_product_attention(query, model.cache[layer]['k'], model.cache[layer]['v'])
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            self.assertEqual(event['query_cache_bytes'] * count, event['dense_cache_bytes'] * len(indices))
        return cache

    def test_gather_anchor_persistence_and_bytes(self):
        for policy in ('all', 'uniform', 'K2', 'K7', 'K8'):
            with self.subTest(policy=policy):
                cache = self.run_sequence(policy)
                self.assertEqual(cache.selected[0].tolist(), list(range(P)))
                self.assertEqual([e['frame'] for e in cache.events], [0, 5, 9])
                if policy == 'all':
                    self.assertEqual(cache.events[-1]['state_bytes'], 0)
                    self.assertEqual(cache.events[-1]['query_cache_bytes'],
                                     cache.events[-1]['dense_cache_bytes'])
                else:
                    self.assertLess(cache.events[-1]['query_cache_bytes'],
                                    cache.events[-1]['dense_cache_bytes'])
                if policy == 'K7':
                    # One P x 1024 float32 descriptor set, ages, and the kept IDs.
                    ids = 8 * (P + 2 * K)
                    self.assertEqual(cache.events[-1]['state_bytes'], P * 1024 * 4 + P * 8 + ids)
                cache.close()

    def test_admission_must_follow_the_replayed_schedule(self):
        cache = PatchSelectCache('uniform', 'scene', [0, 5])
        model = fake_model()
        cache.attach(model)
        model.cache = {layer: {n: torch.zeros(1, 2, 2 * (5 + P), 8) for n in 'kv'} for layer in range(1, 36, 2)}
        cache.after_rebuild([0, 0], torch.rand(1, 2, H, W, 1), torch.rand(1, 2, H, W, 3),
                            np.ones((2, H, W), bool), np.zeros((2, H, W, 3), np.uint8))
        cache.begin_query(6)
        cache.end_query()
        with self.assertRaises(AssertionError):
            cache.after_rebuild([0, 6], torch.rand(1, 2, H, W, 1), torch.rand(1, 2, H, W, 3),
                                np.ones((2, H, W), bool), np.zeros((2, H, W, 3), np.uint8))


@unittest.skipUnless(torch.cuda.is_available(), 'needs the CAMP GPU')
class RealBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32, pi3_inference
        cls.model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained('cuda:0').eval())
        cls.model.cache = {}
        images = torch.rand(1, 3, H, W, 3, generator=torch.Generator().manual_seed(0)).cuda()
        pi3_inference(cls.model, images[:, :2], 'cuda:0', store_cache=True)
        block = cls.model.decoder[-1]
        captured = {}

        def grab(module, args, kwargs):
            captured.update(x=args[0].detach().clone(), xpos=kwargs['xpos'].clone(), cache=kwargs['kv_cache'])

        hook = block.register_forward_pre_hook(grab, with_kwargs=True)
        pi3_inference(cls.model, images[:, 2:], 'cuda:0', use_cache=True)
        hook.remove()
        cls.captured = captured

    def test_register_attention_reproduces_the_block(self):
        block, c = self.model.decoder[-1], self.captured
        attn = block.attn
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            probability = register_attention(block, c['x'], c['xpos'], c['cache'], 5)
            y = block.norm1(c['x'])
            B, L, C = y.shape
            qkv = attn.qkv(y).reshape(B, L, 3, attn.num_heads, C // attn.num_heads).transpose(1, 3)
            values = torch.cat([c['cache']['v'], qkv[:, :, 2]], dim=2)
            mixed = (probability[None] @ values.float()).transpose(1, 2).reshape(1, 5, C)
            expected = attn.forward_w_cache(y, xpos=c['xpos'], kv_cache=c['cache'])[0][:, :5]
            actual = attn.proj(mixed.to(expected.dtype))
        self.assertEqual(probability.shape, (attn.num_heads, 5, c['cache']['k'].shape[2] + L))
        torch.testing.assert_close(probability.sum(-1), torch.ones_like(probability[..., 0]))
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        print('REGISTER ATTENTION relative error', float(error), flush=True)
        self.assertLess(float(error), 2e-2)

    def test_hook_records_read_demand_on_the_admitted_query(self):
        from kv_tracker.pi3_utilts import pi3_inference
        cache = PatchSelectCache('K9', 'scene', [0, 1])
        cache.attach(self.model)
        saved = {i: dict(layer) for i, layer in self.model.cache.items()}
        cache.begin_query(1)
        images = torch.rand(1, 1, H, W, 3, generator=torch.Generator().manual_seed(1)).cuda()
        pi3_inference(self.model, images, 'cuda:0', use_cache=True)
        cache.end_query()
        cache.close()
        self.assertEqual(cache.demand.shape, (P,))
        self.assertGreater(cache.demand_seconds, 0)
        self.assertTrue(0 < float(cache.demand.sum()) < 1)
        for i, layer in self.model.cache.items():
            self.assertIs(layer['k'], saved[i]['k'])
            self.assertIs(layer['v'], saved[i]['v'])

    def test_gathered_keys_equal_masked_dense_keys(self):
        block, c = self.model.decoder[-1], self.captured
        cached = c['cache']['k'].shape[2]
        L = c['x'].shape[1]
        keep = torch.ones(cached, dtype=torch.bool)
        # Second slot: registers kept, three of every four patches dropped.
        keep[5 + P + 5:] = torch.arange(cached - (5 + P + 5)) % 4 == 0
        index = keep.nonzero().flatten().cuda()
        gathered = {n: c['cache'][n].index_select(2, index) for n in ('k', 'v')}
        mask = torch.cat((keep, torch.ones(L, dtype=torch.bool))).cuda()[None, None, None]
        attention = block.attn.forward_w_cache
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            y = block.norm1(c['x'])
            sparse = attention(y, xpos=c['xpos'], kv_cache=gathered)[0]
            dense = attention(y, xpos=c['xpos'], kv_cache=c['cache'], attn_mask=mask)[0]
            unmasked = attention(y, xpos=c['xpos'], kv_cache=c['cache'])[0]
        error = (sparse.float() - dense.float()).norm() / dense.float().norm()
        effect = (unmasked.float() - dense.float()).norm() / dense.float().norm()
        print('GATHER relative error', float(error), 'dropping effect', float(effect), flush=True)
        self.assertLess(float(error), 1e-2)
        self.assertGreater(float(effect), float(error))


if __name__ == '__main__':
    unittest.main()
