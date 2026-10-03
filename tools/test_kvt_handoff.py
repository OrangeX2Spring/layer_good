"""CAMP-only geometry contracts for the single map handoff."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from kv_tracker.map_handoff import MapHandoff, bridge, compose, fit_similarity, transform_pose


class HandoffTests(unittest.TestCase):
    def test_native_prefix_freeze_and_extended_bank(self):
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        for stop, cap, count, last in ((200, 20, 5, 199), (500, 20, 11, 499),
                                     (750, 20, 16, 749), (2585, 20, 20, 949),
                                     (1200, 25, 25, 1199), (1600, 33, 33, 1599),
                                     (2000, 41, 41, 1999)):
            with self.subTest(stop=stop, cap=cap):
                model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), cache={})
                tracker = MapHandoff(model, 'native', lambda row: None, lambda event, evidence: None,
                                     native_keyframe_cap=cap)
                tracker.origin = torch.eye(4)
                tracker.ids, tracker.images = [0], [image]
                with patch('kv_tracker.map_handoff.pi3_inference',
                           return_value=torch.eye(4)[None, None]), \
                     patch.object(tracker, 'reconstruct', return_value=(
                         torch.zeros(41, 2, 2, 3), torch.eye(4)[None],
                         torch.ones(41, 2, 2), torch.eye(4))) as rebuild, \
                     patch('torch.cuda.synchronize'):
                    for frame in range(1, 2010):
                        tracker.step(image, frame, update=frame < stop)
                self.assertEqual(len(tracker.ids), count)
                self.assertEqual(tracker.ids[-1], last)
                self.assertEqual(rebuild.call_count, count - 1)
                self.assertTrue(all(call.args[2] < stop for call in rebuild.call_args_list))

    def setUp(self):
        torch.manual_seed(4)
        self.r = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        self.t = torch.tensor([.3, -.7, 1.2], dtype=torch.float64)
        self.s = torch.tensor(1.7, dtype=torch.float64)

    def test_similarity_and_camera_orientation(self):
        x = torch.randn(500, 3, dtype=torch.float64)
        target = self.s * (x @ self.r.T) + self.t
        s, r, t = fit_similarity(x, target)
        torch.testing.assert_close(s, self.s)
        torch.testing.assert_close(r, self.r)
        torch.testing.assert_close(t, self.t)
        pose = torch.eye(4, dtype=torch.float64)
        pose[:3, 3] = torch.tensor([1., 2., 3.])
        mapped = transform_pose(pose, (s, r, t))
        torch.testing.assert_close(mapped[:3, :3].T @ mapped[:3, :3], torch.eye(3, dtype=torch.float64))
        torch.testing.assert_close(mapped[:3, 3], self.s * (self.r @ pose[:3, 3]) + self.t)
        outer = (torch.tensor(.6), self.r.T, -self.t)
        torch.testing.assert_close(transform_pose(mapped, outer),
                                   transform_pose(pose, compose(outer, (s, r, t))))

    def test_shared_geometry_accepts_and_camera_disagreement_rejects(self):
        points = torch.randn(128, 128, 3, dtype=torch.float64)
        target = self.s * (points @ self.r.T) + self.t
        conf = torch.ones(128, 128)
        new_pose = torch.eye(4, dtype=torch.float64)
        old_pose = transform_pose(new_pose, (self.s, self.r, self.t))
        fitted, event, evidence = bridge(target, points, conf, conf, old_pose, new_pose)
        self.assertIsNotNone(fitted)
        self.assertTrue(event['accepted'])
        self.assertEqual(event['spatial_bins'], 16)
        self.assertFalse((evidence['fit_mask'] & evidence['validation_mask']).any())
        wrong = old_pose.clone()
        wrong[:3, 3] += 10
        fitted, event, _ = bridge(target, points, conf, conf, wrong, new_pose)
        self.assertIsNone(fitted)
        self.assertFalse(event['checks']['camera_translation'])

    def test_collinear_geometry_rejected(self):
        x = torch.arange(500, dtype=torch.float64)[:, None].expand(-1, 3)
        self.assertIsNone(fit_similarity(x, x))

    def test_validation_geometry_cannot_hide_behind_fit(self):
        points = torch.randn(128, 128, 3, dtype=torch.float64)
        target = self.s * (points @ self.r.T) + self.t
        y, x = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
        target[((x // 4 + y // 4) % 2) == 1] += 2
        conf = torch.ones(128, 128)
        new_pose = torch.eye(4, dtype=torch.float64)
        old_pose = transform_pose(new_pose, (self.s, self.r, self.t))
        fitted, event, _ = bridge(target, points, conf, conf, old_pose, new_pose)
        self.assertIsNone(fitted)
        self.assertFalse(event['checks']['validation_p90'])

    def test_transaction_commit_and_rejection(self):
        for accept in (True, False):
            model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]),
                                    cache={1: {'k': torch.ones(1, 1, 18, 2)}})
            rows = []
            tracker = MapHandoff(model, 'handoff', rows.append, lambda event, evidence: None)
            tracker.ids = [0, 699]
            tracker.images = [np.zeros((28, 28, 3), dtype=np.uint8),
                              np.ones((28, 28, 3), dtype=np.uint8)]
            tracker.origin = torch.eye(4)
            tracker.latest_points = torch.zeros(28, 28, 3)
            tracker.latest_conf = torch.ones(28, 28)
            identity = (torch.tensor(1., dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                        torch.zeros(3, dtype=torch.float64))
            built = []

            def reconstruct(images, ids, frame, kind):
                built.append((kind, ids))
                if kind == 'rebuild':
                    # Rejected candidate restored the old bank before fallback.
                    self.assertTrue(torch.equal(model.cache[1]['k'], torch.ones(1, 1, 18, 2)))
                model.cache = {1: {'k': torch.zeros(1, 1, 18, 2)}}
                return (torch.zeros(2, 28, 28, 3), torch.eye(4).repeat(2, 1, 1),
                        torch.ones(2, 28, 28), torch.eye(4))

            tracker.reconstruct = reconstruct
            with patch('torch.cuda.synchronize'), \
                    patch('kv_tracker.map_handoff.pi3_inference', return_value=torch.eye(4)[None, None]), \
                    patch('kv_tracker.map_handoff.bridge', return_value=(
                        identity if accept else None, dict(accepted=accept), {})):
                emitted = tracker.step(np.full((28, 28, 3), 2, dtype=np.uint8), 749)
            np.testing.assert_array_equal(emitted, np.eye(4))
            self.assertEqual(tracker.ids, [699, 749] if accept else [0, 749])
            self.assertEqual(tracker.events[0]['old_kv_released'], accept)
            self.assertEqual(len(built), 1 if accept else 2)
            self.assertEqual(rows[-1]['kind'], 'update_total')


if __name__ == '__main__':
    unittest.main()
