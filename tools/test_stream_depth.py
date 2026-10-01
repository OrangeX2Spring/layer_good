"""GPU contracts; run only inside the CAMP allocation/container."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'streamvggt/src'))
from streamvggt.models.aggregator import Aggregator
from streamvggt.models.depth_cache import DepthCache, storage_bytes


class StreamDepthTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(17)
        return Aggregator(img_size=42, patch_size=14, embed_dim=32, depth=3,
                          num_heads=2, patch_embed='conv').cuda().eval()

    @torch.inference_mode()
    def test_native_intermediates_and_omitted_first_layer(self):
        model = self.model()
        images = torch.rand(1, 3, 3, 42, 42, device='cuda')
        bank, references = [None] * 3, []
        for frame in range(3):
            outputs, _, bank = model(images[:, frame:frame+1], past_key_values=bank, use_cache=True)
            references.append([value.clone() for value in outputs])
        cache = DepthCache(model, capacity=3)
        for frame in range(3):
            outputs, _ = cache.forward(images[:, frame:frame+1], frame)
            for actual, reference in zip(outputs, references[frame]):
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
            cache.retain(True)
        cache.mode, cache.omitted = 'omit', (0,)
        cache.retain(False)
        assert cache.cache[0][0].shape == (1, 2, 1, 0, 16)
        assert storage_bytes(cache.cache[0]) == 0
        attention = model.global_blocks[0].attn
        x = torch.rand(1, 14, 32, device='cuda')
        pos = torch.zeros(1, 14, 2, dtype=torch.long, device='cuda')
        expected, _ = cache.originals[0](x, pos=pos, past_key_values=None, use_cache=True)
        cache.frame = 3
        actual, _ = attention(x, pos=pos, past_key_values=cache.cache[0], use_cache=True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        # Restore metadata changed by that standalone attention read.
        cache.positions[0] = pos[:, :0].clone()
        cache.ids[0] = torch.empty(0, device='cuda', dtype=torch.long)
        specials = []
        hook = model.frame_blocks[0].register_forward_pre_hook(
            lambda module, inputs: specials.append(inputs[0][:, :5].clone()))
        outputs, _ = cache.forward(images[:, :1], 3)
        hook.remove()
        torch.testing.assert_close(specials[0][:, :1], model.camera_token[:, 1])
        torch.testing.assert_close(specials[0][:, 1:], model.register_token[:, 1])
        assert all(torch.isfinite(value).all() for value in outputs)
        cache.retain(False)
        assert cache.ids[0].numel() == 0 and cache.calls == [0, 1, 2]
        cache.close()

    @torch.inference_mode()
    def test_equal_storage_positions_specials_and_fifo(self):
        for size in (42, 56):
            model = self.model()
            image = torch.rand(1, 1, 3, size, size, device='cuda')
            events = {}
            for mode in ('native', 'omit', 'uniform_special'):
                cache = DepthCache(model, mode, (0,), capacity=2)
                rows = []
                for frame in range(4):
                    cache.forward(image, frame)
                    before_ids = [ids.clone() for ids in cache.ids]
                    before_pos = [pos.clone() for pos in cache.positions]
                    before_kv = [tuple(value.clone() for value in pair) for pair in cache.cache]
                    rows.append(cache.retain(frame != 1))
                    for layer, ids in enumerate(cache.ids):
                        selected = torch.searchsorted(before_ids[layer], ids)
                        torch.testing.assert_close(cache.positions[layer], before_pos[layer][:, selected])
                        for actual, original in zip(cache.cache[layer], before_kv[layer]):
                            torch.testing.assert_close(actual, original.index_select(3, selected))
                        assert ids.unique().numel() == ids.numel()
                        if mode == 'uniform_special':
                            for retained in cache.frames:
                                specials = retained * cache.tokens + torch.arange(5, device='cuda')
                                assert torch.isin(specials, ids).all()
                assert cache.frames == [2, 3]
                events[mode] = rows
                cache.close()
            for dense, omitted, uniform in zip(*events.values()):
                assert omitted['persistent_bytes'] == uniform['persistent_bytes']
                assert 3 * omitted['persistent_bytes'] == 2 * dense['persistent_bytes']


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'GPU contract must run on CAMP'
    unittest.main()
