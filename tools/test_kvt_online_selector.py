"""CPU policy contract tests; run in the CAMP container before the GPU sweep."""
import io
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from kvt_online_selector import OnlineSelector


class OnlineSelectorTests(unittest.TestCase):
    def selector(self, policy, **overrides):
        config = dict(policy=policy, seed=0, max_keyframes=3, interval=2,
                      angle_degrees=25, novelty_threshold=0.1)
        config.update(overrides)
        selector = OnlineSelector(config, io.StringIO())
        selector.model = SimpleNamespace(cache={})
        return selector

    def step(self, selector, index, original=False):
        pose = torch.eye(4)
        pose[2, 3] = -1
        return selector.select(dict(idx=index, resized_mask_np=np.ones((14, 14), dtype=bool)),
                               torch.zeros(3), pose, pose[None].repeat(len(selector.inserted), 1, 1),
                               selector.inserted.copy(), original)

    def test_interval_and_cap(self):
        selector = self.selector('interval')
        self.assertEqual([self.step(selector, i) for i in range(1, 9)],
                         [False, True, False, True, False, False, False, False])
        self.assertEqual(selector.inserted, [0, 2, 4])

    def test_original_uncapped(self):
        selector = self.selector('original', max_keyframes=0)
        for i in range(1, 7):
            self.assertEqual(self.step(selector, i, i % 2 == 1), i % 2 == 1)

    def test_random_prefix_and_seed(self):
        histories = []
        for seed, length in [(0, 8), (0, 20), (1, 20)]:
            selector = self.selector('random', seed=seed, max_keyframes=32)
            histories.append([self.step(selector, i) for i in range(1, length + 1)])
        self.assertEqual(histories[0], histories[1][:8])
        self.assertNotEqual(histories[1], histories[2])

    def test_angular_geometry(self):
        selector = self.selector('angular')
        pose = torch.eye(4)
        pose[0, 3] = -1
        cached = torch.eye(4)[None]
        cached[0, 2, 3] = -1
        self.assertTrue(selector.select(dict(idx=1), torch.zeros(3), pose, cached, [0], False))
        self.assertAlmostEqual(json.loads(selector.log.getvalue())['angle_degrees'], 90)

    def test_semantic_novelty_and_empty_object(self):
        selector = self.selector('semantic')
        selector.embeddings = [torch.tensor([1., 0.])]
        selector.tokens = torch.tensor([[[1., 0.]]])
        self.assertFalse(self.step(selector, 1))
        selector.tokens = torch.tensor([[[0., 1.]]])
        self.assertTrue(self.step(selector, 2))
        selector.tokens = torch.tensor([[[0., 1.]]])
        self.assertFalse(self.step(selector, 3))
        selector.tokens = torch.tensor([[[1., 0.]]])
        vector, count = selector.embedding(dict(resized_mask_np=np.zeros((14, 14), dtype=bool)))
        self.assertIsNone(vector)
        self.assertEqual(count, 0)

    def test_background_patch_excluded(self):
        selector = self.selector('semantic')
        selector.tokens = torch.tensor([[[1., 0.], [0., 100.]]])
        mask = np.zeros((14, 28), dtype=bool)
        mask[:, :14] = True
        vector, count = selector.embedding(dict(resized_mask_np=mask))
        self.assertEqual(count, 1)
        torch.testing.assert_close(vector, torch.tensor([1., 0.]))

    def test_future_cache_rejected(self):
        selector = self.selector('original')
        with self.assertRaises(AssertionError):
            selector.select(dict(idx=1), torch.zeros(3), torch.eye(4),
                            torch.eye(4)[None], [1], False)


if __name__ == '__main__':
    unittest.main()
