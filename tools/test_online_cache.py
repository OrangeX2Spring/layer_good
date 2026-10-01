"""CAMP-only budget, protection and local-sensitivity contracts."""
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from kv_tracker.online_layer_cache import OnlineLayerCache, sensitivity


class OnlineContracts(unittest.TestCase):
    def test_sensitivity_matches_explicit_removal(self):
        torch.manual_seed(7)
        q = torch.randn(1, 2, 3, 4, device='cuda')
        k = torch.randn(1, 2, 7, 4, device='cuda')
        v = torch.randn_like(k)
        floor = torch.tensor([True, True, False, False, False, True, True], device='cuda')
        a = F.scaled_dot_product_attention(q, k, v)
        b = F.scaled_dot_product_attention(q, k[:, :, floor], v[:, :, floor])
        expected = float((a-b).square().mean()/a.square().mean())
        self.assertAlmostEqual(sensitivity(q, k, v, floor), expected, places=6)
        self.assertEqual(sensitivity(q, k, v, torch.ones(7, device='cuda', dtype=torch.bool)), 0.)

    def test_pi3_fixed_cap_and_protected_values(self):
        layers = (1, 3, 5)
        for policy in ('online', 'uniform'):
            cache = OnlineLayerCache(4, policy)
            cache.layers = list(layers)
            cache.scores = {1: .01, 3: 1., 5: 10.}
            cache.model = SimpleNamespace(cache={})
            cache.frame = 0
            for count in (2, 3, 7, 12):
                ids = [0, 0] if count == 2 else list(range(count))
                dense = {i: {n: torch.arange(count*10, device='cuda').float().reshape(1, 1, count*10, 1)
                             for n in ('k', 'v')} for i in layers}
                cache.model.cache = dense
                original = dense[1]['k'].clone()
                cache.after_rebuild(ids, None, None, None, None)
                event = cache.events[-1]
                self.assertLessEqual(event['persistent_bytes'], event['budget_bytes'])
                self.assertEqual(event['persistent_bytes'], min(count, 4)*10*len(layers)*16)
                for i in layers:
                    rows = cache.rows[i]
                    self.assertEqual(int((rows // 10 == 0).sum()), 20 if count == 2 else 10)
                    self.assertEqual(int((rows // 10 == ids[-1]).sum()), 20 if count == 2 else 10)
                    self.assertTrue(torch.equal(cache.model.cache[i]['k'][:, :, :10], original[:, :, :10]))
                    if count > 2:
                        self.assertGreaterEqual(event['tokens_by_layer'][i] - 20, 8)
                if count == 12 and policy == 'online':
                    self.assertGreater(event['tokens_by_layer'][5], event['tokens_by_layer'][1])

    def test_stream_caps_and_original_positions(self):
        from streamvggt.models.depth_cache import DepthCache
        from test_stream_depth import StreamDepthTests
        model = StreamDepthTests().model()
        for mode in ('online', 'budget_uniform'):
            cache = DepthCache(model, mode=mode, capacity=8, history_policy='anchor', frame_equivalents=4)
            first = None
            for frame in range(12):
                image = torch.randn(1, 1, 3, 28, 28, device='cuda')
                cache.forward(image, frame)
                before_ids = [ids.clone() for ids in cache.ids]
                before_positions = [pos.clone() for pos in cache.positions]
                if first is None:
                    first = [pair[0][:, :, :, :cache.tokens].clone() for pair in cache.cache]
                event = cache.retain(True)
                self.assertLessEqual(event['persistent_bytes'], event['budget_bytes'])
                for layer in range(model.depth):
                    selected = torch.searchsorted(before_ids[layer], cache.ids[layer])
                    torch.testing.assert_close(cache.positions[layer], before_positions[layer][:, selected], atol=0, rtol=0)
                    self.assertTrue(torch.equal(cache.cache[layer][0][:, :, :, :cache.tokens], first[layer]))
                    self.assertEqual(int((cache.ids[layer] // cache.tokens == frame).sum()), cache.tokens)
            cache.close()

    def test_pi3_probe_does_not_change_query(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = torch.nn.Module()
                self.attn.qkv = torch.nn.Linear(4, 12)
                self.attn.q_norm = self.attn.k_norm = torch.nn.Identity()
                self.attn.num_heads = 1
                self.attn.rope = None
            def forward(self, x, xpos=None, kv_cache=None):
                q, k, v = self.attn.qkv(x).reshape(1, 10, 3, 1, 4).transpose(1, 3).unbind(2)
                return F.scaled_dot_product_attention(q, torch.cat((kv_cache['k'], k), 2),
                                                      torch.cat((kv_cache['v'], v), 2))
        blocks = torch.nn.ModuleList([Block() for _ in range(6)]).cuda().eval()
        model = SimpleNamespace(decoder=blocks, cache={i: {n: torch.randn(1, 1, 50, 4, device='cuda')
                    for n in ('k', 'v')} for i in (1, 3, 5)})
        cache = OnlineLayerCache(4)
        cache.attach(model)
        cache.after_rebuild(list(range(5)), None, None, None, None)
        cache.begin_query(16)
        for i in (1, 3, 5):
            x = torch.randn(1, 10, 4, device='cuda')
            cache.active = False
            expected = blocks[i].forward(x, xpos=None, kv_cache=model.cache[i])
            actual = blocks[i](x, xpos=None, kv_cache=model.cache[i])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(len(cache.probes), 3)
        cache.end_query()
        cache.close()


if __name__ == '__main__':
    unittest.main()
