"""CAMP GPU contracts for physical historical cache omission."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kv_tracker'))
from kv_tracker.layer_cache import LayerCache


class LayerCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available(), 'Run on CAMP GPU'

    def cache(self):
        return {i: {n: torch.randn(1, 2, 13, 8, device='cuda', dtype=torch.bfloat16)
                    for n in ('k', 'v')} for i in (1, 3, 5)}

    def test_omission_releases_storage_and_preserves_other_layers(self):
        cache = self.cache()
        original = cache[3]['k']
        transform = LayerCache('omit', [1])
        transform(cache, [0, 1])
        torch.testing.assert_close(cache[3]['k'], original, rtol=0, atol=0)
        self.assertEqual(cache[1]['k'].shape[2], 0)
        self.assertEqual(cache[1]['k'].untyped_storage().nbytes(), 0)

    def test_uniform_matches_omission_bytes_including_rounding(self):
        omitted, uniform = self.cache(), self.cache()
        a, b = LayerCache('omit', [1]), LayerCache('uniform', [1])
        a(omitted, [0, 1])
        b(uniform, [0, 1])
        self.assertEqual(a.events[0]['persistent_bytes'], b.events[0]['persistent_bytes'])
        self.assertEqual(sum(x['k'].shape[2] for x in uniform.values()), 26)

    def test_native_attention_empty_history_equals_current_only(self):
        from pi3.models.layers.attention import FlashAttentionRope
        torch.manual_seed(17)
        attention = FlashAttentionRope(16, 2, qk_norm=True).cuda().to(torch.bfloat16).eval()
        current = torch.randn(1, 9, 16, device='cuda', dtype=torch.bfloat16)
        empty = {n: torch.empty(1, 2, 0, 8, device='cuda', dtype=torch.bfloat16) for n in ('k', 'v')}
        with torch.inference_mode():
            expected = attention(current)
            observed, keys, values = attention.forward_w_cache(current, kv_cache=empty)
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        self.assertEqual(keys.shape[2], current.shape[1])
        self.assertEqual(values.shape, keys.shape)

    def test_native_compacts_views_without_changing_values(self):
        cache = {i: {n: torch.randn(1, 2, 13, 24, device='cuda', dtype=torch.bfloat16)[..., :8]
                     for n in ('k', 'v')} for i in (1, 3, 5)}
        original = {i: {n: t.clone() for n, t in values.items()} for i, values in cache.items()}
        transform = LayerCache('native', [1])
        transform(cache, [0, 1])
        for i, values in cache.items():
            for n, t in values.items():
                torch.testing.assert_close(t, original[i][n], rtol=0, atol=0)
                self.assertEqual(t.untyped_storage().nbytes(), t.numel() * t.element_size())

    def test_special_control_preserves_every_frame_special_and_exact_bytes(self):
        # Token-coded values identify the actual gathered entries, not just counts.
        for per_frame in (11, 12):
            count = 2 * per_frame
            cache = {i: {n: torch.arange(count, device='cuda', dtype=torch.bfloat16)
                         .reshape(1, 1, count, 1).clone() for n in ('k', 'v')}
                     for i in (1, 3, 5)}
            omitted = {i: {n: t.clone() for n, t in values.items()} for i, values in cache.items()}
            reference = LayerCache('omit', [1])
            control = LayerCache('uniform_special', [1])
            reference(omitted, [0, 31])
            control(cache, [0, 31])
            self.assertEqual(reference.events[0]['persistent_bytes'], control.events[0]['persistent_bytes'])
            for values in cache.values():
                ids = values['k'].flatten().long().tolist()
                self.assertEqual(ids, sorted(set(ids)))
                self.assertTrue(set(range(5)).issubset(ids))
                self.assertTrue(set(range(per_frame, per_frame + 5)).issubset(ids))
                self.assertEqual(values['k'].untyped_storage().nbytes(), values['k'].numel() * 2)
                torch.testing.assert_close(values['k'], values['v'], rtol=0, atol=0)
