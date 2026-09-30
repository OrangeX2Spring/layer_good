"""Mathematical cache contracts. Run inside a CAMP GPU allocation, not on Mac."""
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kv_tracker'))
from kv_tracker.math_cache import (MathCache, PackedMatrix, allocate_bits,
    dense_response, pack_codes, unpack_codes)
from math_cache_ablation import TraceCapture


class MathCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available(), 'Run on CAMP with GPU'

    def setUp(self):
        torch.manual_seed(17)
        self.k = torch.randn(65, 8, device='cuda') * .2
        self.v = torch.randn(65, 6, device='cuda')
        self.q = torch.randn(12, 8, device='cuda') * .2
        self.ck = torch.randn(9, 8, device='cuda') * .2
        self.cv = torch.randn(9, 6, device='cuda')

    def test_exact_history_current_mass_merge(self):
        cache = MathCache(dict(method='exact')).fit(self.k, self.v, self.q[:4])
        expected, _ = dense_response(self.q, torch.cat((self.k, self.ck)), torch.cat((self.v, self.cv)))
        torch.testing.assert_close(cache.read(self.q, self.ck, self.cv), expected)
        self.assertEqual(cache.persistent_bytes(), (self.k.numel() + self.v.numel()) * 4)

    def test_packing_all_widths_and_partial_bytes(self):
        for bits in range(1, 9):
            codes = torch.arange(19, device='cuda') % (2 ** bits)
            packed = pack_codes(codes, bits)
            self.assertEqual(packed.dtype, torch.uint8)
            self.assertEqual(packed.numel(), (len(codes) * bits + 7) // 8)
            torch.testing.assert_close(unpack_codes(packed, bits, len(codes)), codes)

    def test_packed_constant_zero_bit_and_quantization_error_bound(self):
        x = torch.stack((torch.ones(19, device='cuda') * 3,
                         torch.linspace(-1, 1, 19, device='cuda')), -1)
        packed = PackedMatrix(x, torch.tensor([0, 3], device='cuda', dtype=torch.int32))
        y = packed.decode()
        torch.testing.assert_close(y[:, 0], x[:, 0])
        self.assertLessEqual(float((y[:, 1] - x[:, 1]).abs().max()), 1 / 7 + 1e-6)
        self.assertEqual(packed.state['codes'][0].numel(), 0)
        constant = PackedMatrix(x[:, :1], torch.tensor([4], device='cuda', dtype=torch.int32))
        torch.testing.assert_close(constant.decode(), x[:, :1])

    def test_integer_waterfill_matches_exhaustive_distortion_optimum(self):
        alpha = torch.tensor([1., 4., 9.], device='cuda')
        bits = allocate_bits(alpha, 5, maximum=3)
        self.assertEqual(int(bits.sum()), 5)
        observed = float((alpha * 4. ** (-bits.float())).sum())
        optimum = min(sum(a * 4. ** (-b) for a, b in zip([1., 4., 9.], bs))
                      for bs in product(range(4), repeat=3) if sum(bs) == 5)
        self.assertAlmostEqual(observed, optimum)

    def test_full_rank_readers_preserve_original_scale(self):
        k, v, q = self.k[:, :6], self.v, self.q[:, :6]
        expected, _ = dense_response(q, k, v)
        for method in ('lowrank', 'query_lowrank'):
            cache = MathCache(dict(method=method, rank=6)).fit(k, v, q[:4])
            torch.testing.assert_close(cache.read(q), expected, rtol=2e-5, atol=2e-6)
        cache = MathCache(dict(method='lowrank', rank=3)).fit(k, v, q[:4])
        s = cache.state
        logits = (q @ s['query_map']) @ s['k'].T / 6 ** .5
        expected = logits.softmax(-1) @ s['v'] @ s['value_map'].T
        torch.testing.assert_close(cache.read(q), expected)

    def test_balanced_weights_and_all_kept_identity(self):
        for tokens in (1, 17, 65):
            cache = MathCache(dict(method='probe_balance', tokens=tokens)).fit(self.k, self.v, self.q[:4])
            self.assertEqual(len(cache.state['k']), tokens)
            self.assertAlmostEqual(float(cache.state['log_weights'].exp().sum()), 65, places=4)
            self.assertTrue(torch.isfinite(cache.read(self.q)).all())
            if tokens == 65:
                expected, _ = dense_response(self.q, self.k, self.v)
                torch.testing.assert_close(cache.read(self.q), expected)

    def test_kernel_split_append_matches_one_shot_and_drops_raw_history(self):
        cfg = dict(method='kernel', features=24, seed=17)
        one = MathCache(cfg).fit(self.k, self.v, self.q[:4])
        split = MathCache(cfg).fit(self.k[:31], self.v[:31], self.q[:4])
        split.append(self.k[31:], self.v[31:])
        self.assertEqual(split.count, len(self.k))
        torch.testing.assert_close(split.read(self.q), one.read(self.q))
        self.assertEqual(set(one.state), {'omega', 'key_shift', 'S', 'z'})
        expected_bytes = (8 * 24 + 1 + 24 * 6 + 24) * 4
        self.assertEqual(one.persistent_bytes(), expected_bytes)

    def test_zero_key_kernel_exact_uniform_and_current_mass(self):
        k, q = torch.zeros_like(self.k), torch.zeros_like(self.q)
        ck = torch.zeros_like(self.ck)
        cache = MathCache(dict(method='kernel', features=16)).fit(k, self.v, q[:4])
        expected = torch.cat((self.v, self.cv)).mean(0).expand(len(q), -1)
        torch.testing.assert_close(cache.read(q, ck, self.cv), expected)

    def test_moments_recover_cross_covariance_lost_by_centroid(self):
        k = torch.tensor([[1.], [-1.]], device='cuda')
        v = torch.tensor([[2.], [-2.]], device='cuda')
        q = torch.tensor([[.01]], device='cuda')
        mean = MathCache(dict(method='centroid', groups=1)).fit(k, v, q)
        moments = MathCache(dict(method='moments', groups=1, rank=1)).fit(k, v, q)
        expected, _ = dense_response(q, k, v)
        self.assertEqual(float(mean.read(q)), 0.)
        self.assertLess(float((moments.read(q) - expected).abs()), 1e-5)
        self.assertGreater(moments.persistent_bytes(), mean.persistent_bytes())

    def test_identical_keys_empty_clusters_are_removed(self):
        k = torch.ones_like(self.k)
        for method in ('centroid', 'moments'):
            cache = MathCache(dict(method=method, groups=8, rank=3)).fit(k, self.v, self.q[:4])
            self.assertEqual(len(cache.state['mean_k']), 1)
            expected, _ = dense_response(self.q, k, self.v)
            torch.testing.assert_close(cache.read(self.q), expected)

    def test_quantized_correction_changes_history_mass_not_history_weights(self):
        plain = MathCache(dict(method='quant', bits=2)).fit(self.k, self.v, self.q[:4])
        fixed = MathCache(dict(method='quant', bits=2, bias_correction=True)).fit(self.k, self.v, self.q[:4])
        a, ma = plain.history_response(self.q)
        b, mb = fixed.history_response(self.q)
        torch.testing.assert_close(a, b)
        self.assertGreater(float((ma - mb).abs().max()), 0.)
        self.assertGreater(fixed.persistent_bytes(), plain.persistent_bytes())
        self.assertGreater(float((plain.read(self.q, self.ck, self.cv)
            - fixed.read(self.q, self.ck, self.cv)).abs().max()), 0.)

    def test_all_methods_leave_inputs_unchanged_and_return_finite(self):
        originals = [x.clone() for x in (self.k, self.v, self.q)]
        for method in MathCache.METHODS:
            cfg = dict(method=method, tokens=17, features=32, groups=4, rank=3, bits=4)
            cache = MathCache(cfg).fit(self.k, self.v, self.q[:4])
            result = cache.read(self.q[4:], self.ck, self.cv)
            self.assertEqual(result.shape, (8, 6))
            self.assertTrue(torch.isfinite(result).all(), method)
            for value, original in zip((self.k, self.v, self.q), originals):
                torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_capture_direct_cached_attention_call_and_restore_sdpa(self):
        class Attention(torch.nn.Module):
            def forward_w_cache(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

            def forward(self, q, k, v):
                return self.attn.forward_w_cache(q, k, v)

        block = Block()
        original = F.scaled_dot_product_attention
        q = self.q[None, None].expand(1, 2, -1, -1).contiguous()
        k = self.k[None, None].expand(1, 2, -1, -1).contiguous()
        v = self.v[None, None].expand(1, 2, -1, -1).contiguous()
        expected = block(q, k, v)
        with tempfile.TemporaryDirectory() as temp:
            recorder = TraceCapture([(1, block.attn, block)], Path(temp),
                dict(bank_frames=8, query_rows=4), 'stub')
            recorder.frame = 8
            with recorder.installed():
                actual = block(q, k, v)
            self.assertEqual(len(recorder.records), 2)
            self.assertIs(F.scaled_dot_product_attention, original)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
