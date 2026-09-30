"""Routing contracts for CAMP; graph contracts run in each fresh model process."""
import unittest
import torch
from kv_tracker.scene_routing import project_cache


class RoutingContracts(unittest.TestCase):
    def test_projection_visibility_and_gauge(self):
        assert torch.cuda.is_available(), 'CAMP GPU required'
        points = torch.tensor([[7., 7., 1.], [14., 14., 2.], [-7., 0., 1.],
                               [0., 0., -1.]], device='cuda')
        conf = torch.tensor([.9, .7, .5, .4], device='cuda')
        pose = torch.eye(4, device='cuda')
        k = torch.eye(3, device='cuda')
        maps = project_cache(points, conf, pose, k, 28, 28)
        self.assertEqual(maps['support'].tolist(), [2, 0, 0, 0])
        self.assertEqual(maps['front_depth'].tolist(), [1., 0., 0., 0.])
        self.assertEqual(maps['secondary_gap'].tolist(), [1., 0., 0., 0.])
        self.assertEqual(maps['source_index'].tolist(), [0, -1, -1, -1])
        self.assertEqual(maps['valid_samples'].tolist(), [True, True, False, False])
        self.assertEqual(maps['confidence'][0], conf[0])
        pose[:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device='cuda')
        pose[:3, 3] = torch.tensor([2., 3., 4.], device='cuda')
        moved = points @ pose[:3, :3].T + pose[:3, 3]
        for _ in range(3):
            repeat = project_cache(moved, conf, pose, k, 28, 28)
            for key in maps:
                self.assertTrue(torch.equal(maps[key], repeat[key]), key)

    def test_empty_support_and_tie(self):
        points = torch.tensor([[7., 7., 1.], [7., 7., 1.]], device='cuda')
        confidence = torch.tensor([.6, .9], device='cuda')
        pose, k = torch.eye(4, device='cuda'), torch.eye(3, device='cuda')
        maps = project_cache(points, confidence, pose, k, 28, 28)
        self.assertEqual(maps['source_index'][0], 0)
        self.assertEqual(maps['confidence'][0], confidence[0])
        pose[2, 3] = 3.
        maps = project_cache(points, confidence, pose, k, 28, 28)
        self.assertEqual(maps['support'].sum(), 0)
        for key in ('front_depth', 'confidence', 'secondary_gap'):
            self.assertEqual(maps[key].sum(), 0)


if __name__ == '__main__':
    unittest.main()
