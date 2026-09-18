"""Contract tests for the token-selection policies. CPU only; runs on the Mac.

    ~/anaconda3/bin/python tools/test_kvcache_policy.py
"""
import unittest

import numpy as np
import torch

from kvcache_policy import (POLICIES, frames_in_cache, pool_confidence, pool_mask,
                            retained_bytes, select, semantic_budget)

FRAMES, ROWS, COLS, SPECIAL = 4, 3, 5, 2
TOKENS = ROWS * COLS + SPECIAL


def object_mask(frames=FRAMES, per_frame=(3, 5, 2, 4)):
    mask = torch.zeros(frames, TOKENS, dtype=torch.bool)
    for frame, count in enumerate(per_frame):
        mask[frame, SPECIAL:SPECIAL + count] = True
    return mask


class PolicyTests(unittest.TestCase):
    def test_frames_in_cache_requires_exact_division(self):
        self.assertEqual(frames_in_cache(4 * TOKENS, TOKENS), 4)
        with self.assertRaises(AssertionError):
            frames_in_cache(4 * TOKENS + 1, TOKENS)

    def test_causal_keeps_everything(self):
        kept = select('causal', frames=FRAMES, tokens_per_frame=TOKENS,
                      patch_start_idx=SPECIAL)
        np.testing.assert_array_equal(kept.numpy(), np.arange(FRAMES * TOKENS))

    def test_window_reproduces_anchor_plus_recent(self):
        # window_size 2 over 4 frames: frame 0 anchor plus the most recent frame.
        kept = select('window', frames=FRAMES, tokens_per_frame=TOKENS,
                      patch_start_idx=SPECIAL, window_size=2, anchor=True)
        expected = np.concatenate([np.arange(TOKENS), np.arange(3 * TOKENS, 4 * TOKENS)])
        np.testing.assert_array_equal(kept.numpy(), expected)
        # Without the anchor it is a pure suffix.
        kept = select('window', frames=FRAMES, tokens_per_frame=TOKENS,
                      patch_start_idx=SPECIAL, window_size=2, anchor=False)
        np.testing.assert_array_equal(kept.numpy(), np.arange(2 * TOKENS, 4 * TOKENS))

    def test_semantic_budget_rejects_mask_on_special_tokens(self):
        mask = object_mask()
        np.testing.assert_array_equal(semantic_budget(mask, SPECIAL).numpy(), [3, 5, 2, 4])
        mask[1, 0] = True
        with self.assertRaises(AssertionError):
            semantic_budget(mask, SPECIAL)

    def test_semantic_keeps_object_specials_and_anchor(self):
        mask = object_mask()
        kept = select('semantic', frames=FRAMES, tokens_per_frame=TOKENS,
                      patch_start_idx=SPECIAL, mask=mask, anchor=True, keep_special=True)
        flat = torch.zeros(FRAMES * TOKENS, dtype=torch.bool)
        flat[kept] = True
        grid = flat.reshape(FRAMES, TOKENS)
        self.assertTrue(grid[0].all(), 'anchor frame must be kept entire')
        self.assertTrue(grid[1:, :SPECIAL].all(), 'special tokens must survive')
        np.testing.assert_array_equal(grid[1:, SPECIAL:].numpy(), mask[1:, SPECIAL:].numpy())

    def test_nulls_are_budget_matched_to_semantic_frame_by_frame(self):
        mask = object_mask()
        budget = semantic_budget(mask, SPECIAL)
        score = torch.rand(FRAMES, TOKENS)
        reference = select('semantic', frames=FRAMES, tokens_per_frame=TOKENS,
                           patch_start_idx=SPECIAL, mask=mask, anchor=False,
                           keep_special=False)
        for policy in ('confidence', 'random', 'uniform'):
            kept = select(policy, frames=FRAMES, tokens_per_frame=TOKENS,
                          patch_start_idx=SPECIAL, score=score, budget=budget,
                          anchor=False, keep_special=False,
                          generator=torch.Generator().manual_seed(0))
            self.assertEqual(len(kept), len(reference), policy)
            grid = torch.zeros(FRAMES * TOKENS, dtype=torch.bool)
            grid[kept] = True
            grid = grid.reshape(FRAMES, TOKENS)
            np.testing.assert_array_equal(grid.sum(dim=1).numpy(), budget.numpy())
            self.assertFalse(grid[:, :SPECIAL].any(), f'{policy} spent budget on specials')

    def test_confidence_takes_the_highest_scoring_patches(self):
        score = torch.zeros(1, TOKENS)
        score[0, SPECIAL:] = torch.arange(ROWS * COLS, dtype=torch.float32)
        kept = select('confidence', frames=1, tokens_per_frame=TOKENS,
                      patch_start_idx=SPECIAL, score=score,
                      budget=torch.tensor([3]), anchor=False, keep_special=False)
        # The three largest patch scores are the last three patch positions.
        np.testing.assert_array_equal(kept.numpy(), [TOKENS - 3, TOKENS - 2, TOKENS - 1])

    def test_uniform_is_deterministic_and_spread(self):
        budget = torch.tensor([5])
        first = select('uniform', frames=1, tokens_per_frame=TOKENS, patch_start_idx=SPECIAL,
                       budget=budget, anchor=False, keep_special=False)
        second = select('uniform', frames=1, tokens_per_frame=TOKENS, patch_start_idx=SPECIAL,
                        budget=budget, anchor=False, keep_special=False)
        np.testing.assert_array_equal(first.numpy(), second.numpy())
        self.assertEqual(first[0].item(), SPECIAL)
        self.assertEqual(first[-1].item(), TOKENS - 1)

    def test_random_differs_by_seed_but_repeats_within_one(self):
        budget = torch.tensor([6])
        kwargs = dict(frames=1, tokens_per_frame=TOKENS, patch_start_idx=SPECIAL,
                      budget=budget, anchor=False, keep_special=False)
        a = select('random', generator=torch.Generator().manual_seed(0), **kwargs)
        b = select('random', generator=torch.Generator().manual_seed(0), **kwargs)
        c = select('random', generator=torch.Generator().manual_seed(1), **kwargs)
        np.testing.assert_array_equal(a.numpy(), b.numpy())
        self.assertFalse(np.array_equal(a.numpy(), c.numpy()))

    def test_every_policy_returns_sorted_unique_in_range(self):
        mask = object_mask()
        budget = semantic_budget(mask, SPECIAL)
        score = torch.rand(FRAMES, TOKENS)
        for policy in POLICIES:
            kept = select(policy, frames=FRAMES, tokens_per_frame=TOKENS,
                          patch_start_idx=SPECIAL, mask=mask, score=score, budget=budget,
                          window_size=2, generator=torch.Generator().manual_seed(0))
            self.assertEqual(kept.dtype, torch.int64, policy)
            self.assertTrue(bool((kept.diff() > 0).all()), f'{policy} not strictly sorted')
            self.assertGreaterEqual(int(kept.min()), 0)
            self.assertLess(int(kept.max()), FRAMES * TOKENS)

    def test_retained_bytes_counts_keys_and_values_in_every_layer(self):
        kept = torch.arange(10)
        self.assertEqual(retained_bytes(kept, heads=8, head_dim=64, depth=24,
                                        element_size=4), 10 * 8 * 64 * 24 * 2 * 4)

    def test_pool_mask_keeps_any_covered_cell_and_never_a_special(self):
        pixels = torch.zeros(1, ROWS * 2, COLS * 2, dtype=torch.bool)
        pixels[0, 0, 0] = True          # one pixel of the top-left cell
        pixels[0, 2:4, 2:4] = True      # all of the cell at row 1, col 1
        tokens = pool_mask(pixels, (ROWS, COLS), SPECIAL)
        self.assertEqual(tokens.shape, (1, TOKENS))
        self.assertFalse(tokens[0, :SPECIAL].any())
        self.assertTrue(tokens[0, SPECIAL + 0])
        self.assertTrue(tokens[0, SPECIAL + COLS + 1])
        self.assertEqual(int(tokens[0, SPECIAL:].sum()), 2)
        # A coverage threshold drops the single-pixel cell and keeps the full one.
        strict = pool_mask(pixels, (ROWS, COLS), SPECIAL, threshold=0.5)
        self.assertFalse(strict[0, SPECIAL + 0])
        self.assertTrue(strict[0, SPECIAL + COLS + 1])

    def test_pool_confidence_averages_cells_and_protects_specials(self):
        confidence = torch.arange(ROWS * 2 * COLS * 2, dtype=torch.float32)
        confidence = confidence.reshape(1, ROWS * 2, COLS * 2)
        tokens = pool_confidence(confidence, (ROWS, COLS), SPECIAL)
        self.assertEqual(tokens.shape, (1, TOKENS))
        self.assertTrue(torch.isinf(tokens[0, :SPECIAL]).all())
        cell = confidence[0, :2, :2].mean()
        self.assertAlmostEqual(float(tokens[0, SPECIAL]), float(cell), places=5)

    def test_pool_rejects_a_grid_that_does_not_divide(self):
        with self.assertRaises(AssertionError):
            pool_mask(torch.zeros(1, 7, 10, dtype=torch.bool), (ROWS, COLS), SPECIAL)


if __name__ == '__main__':
    unittest.main(verbosity=2)
