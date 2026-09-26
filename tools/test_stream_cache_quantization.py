"""Remote tensor contracts for A3; no model weights required."""

from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from stream_cache_adapters import StreamAdapter
from stream_cache_quantization import fake_quantize


class QuantizationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'GPU dtype contract runs on CAMP')
    def test_cuda_dtypes_and_finite_reconstruction(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            value = torch.randn(1, 2, 35, 64, device='cuda', dtype=dtype)
            for bits in (4, 8):
                for axis in (2, 3):
                    result, size = fake_quantize(value, bits, axis)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.shape, value.shape)
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertGreater(size, 0)

    def test_affine_grid_constants_and_short_groups(self):
        value = torch.tensor([0., 1., 4., 15., 7.]).reshape(1, 1, 5, 1)
        result, size = fake_quantize(value, 4, 2, group_size=4)
        torch.testing.assert_close(result, value, rtol=0, atol=0)
        self.assertEqual(size, (2 + 8) + (1 + 8))
        constant = torch.full((1, 2, 35, 7), -3.5)
        for axis in (2, 3):
            result, _ = fake_quantize(constant, 8, axis)
            torch.testing.assert_close(result, constant, rtol=0, atol=0)

    def test_group_axes_and_error_against_scalar_reference(self):
        torch.manual_seed(42)
        value = torch.randn(1, 2, 35, 37)
        for bits in (4, 8):
            for axis in (2, 3):
                actual, size = fake_quantize(value, bits, axis)
                rows = value.movedim(axis, -1).reshape(-1, value.shape[axis])
                expected = rows.clone()
                byte_count = 0
                for row in range(len(rows)):
                    for start in range(0, rows.shape[1], 32):
                        group = rows[row, start:start + 32]
                        lo, hi = float(group.min()), float(group.max())
                        step = (hi - lo) / (2 ** bits - 1)
                        expected[row, start:start + 32] = torch.tensor([
                            lo if step == 0 else round((float(x) - lo) / step) * step + lo
                            for x in group])
                        byte_count += (len(group) * bits + 7) // 8 + 8
                torch.testing.assert_close(actual.movedim(axis, -1).reshape_as(rows), expected)
                self.assertEqual(size, byte_count)

    def test_cache_entry_eviction_refresh_and_attention_consumption(self):
        torch.manual_seed(7)
        for host in ('streamvggt', 'longstream'):
            adapter = StreamAdapter.__new__(StreamAdapter)
            adapter.host = host
            adapter.special = 1
            adapter.sparse = True
            adapter.original_attention = []
            adapter.quantized_frame_bytes = {}
            adapter.session = SimpleNamespace(clear_cache_only=lambda: None)
            key, value = torch.randn(1, 2, 70, 8), torch.randn(1, 2, 70, 8)
            adapter.token_frames = torch.tensor([0] * 35 + [1] * 35)
            pair = [key.unsqueeze(2), value.unsqueeze(2)] if host == 'streamvggt' else [key, value]
            if host == 'streamvggt':
                adapter.aggregator_cache = [pair]
            else:
                adapter.session.aggregator_kv_cache_list = [pair]
            old = key[:, :, :35].clone()
            query = torch.randn(1, 2, 1, 8)
            before = F.scaled_dot_product_attention(query, key, value)
            event = adapter.quantize_current(4)
            self.assertTrue(event['quantized_current_changed'])
            torch.testing.assert_close(key[:, :, :35], old, rtol=0, atol=0)
            self.assertFalse(torch.equal(before, F.scaled_dot_product_attention(query, key, value)))
            # Simulate another arrival, dropping the first frame via the real prune path.
            snapshot = key[:, :, 35:].clone()
            appended = [torch.cat((t, torch.randn_like(t[:, :, :35])), 2) for t in (key, value)]
            if host == 'streamvggt':
                adapter.aggregator_cache = [[t.unsqueeze(2) for t in appended]]
            else:
                adapter.session.aggregator_kv_cache_list = [appended]
            adapter.token_frames = torch.tensor([0] * 35 + [1] * 35 + [2] * 35)
            adapter.prune(torch.arange(34), [1, 2])
            adapter.quantize_current(4)
            cache = adapter.aggregator_cache if host == 'streamvggt' else adapter.session.aggregator_kv_cache_list
            current_key = cache[0][0].reshape(1, 2, 70, 8)
            torch.testing.assert_close(current_key[:, :, :35], snapshot, rtol=0, atol=0)
            self.assertEqual(set(adapter.quantized_frame_bytes), {1, 2})
            if host == 'longstream':
                adapter.reset_segment(24)
                self.assertEqual(adapter.quantized_frame_bytes, {})
                self.assertEqual(len(adapter.token_frames), 0)


if __name__ == '__main__':
    unittest.main()
