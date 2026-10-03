"""CAMP native-cache query fidelity, read-only cache and recapture contracts."""
import os
import unittest

import numpy as np
import torch

from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32, pi3_inference
from kv_tracker.query_acceleration import QueryAcceleration


class QueryContracts(unittest.TestCase):
    def test_native_cache_and_query_only_selection(self):
        torch.manual_seed(17)
        model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())
        images = np.random.default_rng(17).integers(0, 256, (3, 266, 350, 3), dtype=np.uint8)
        with torch.inference_mode():
            dense = QueryAcceleration('graph', os.environ['KVT_GRAPH_ROPE_BUILD'], checked=True)
            half = QueryAcceleration('half')
            half_graph = QueryAcceleration('half_graph', os.environ['KVT_GRAPH_ROPE_BUILD'], checked=True)
            for bank in (1, 2):
                for executor in (dense, half, half_graph):
                    executor.reset()
                pi3_inference(model, [images[:bank]], 'cuda', store_cache=True)
                cache = {i: {k: v.clone() for k, v in row.items()} for i, row in model.cache.items()}
                for image in images:
                    dense.forward(model, image, 'cuda')  # Includes native pose comparison.
                    expected = half.forward(model, image, 'cuda')
                    actual = half_graph.forward(model, image, 'cuda')
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                    for i, row in cache.items():
                        for key, tensor in row.items():
                            self.assertTrue(torch.equal(model.cache[i][key], tensor))
                    self.assertEqual(int(half.keep.sum()), (475 + 1) // 2)
            self.assertEqual(len(dense.executor.rows), 2)
            self.assertEqual(len(half_graph.executor.rows), 2)
        print('QUERY ACCELERATION CONTRACT OK', flush=True)


if __name__ == '__main__':
    unittest.main()
