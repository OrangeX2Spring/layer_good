"""CAMP-only schedule, coordinate and history-retirement contracts."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from kv_tracker.map_handoff import MapHandoff
from kv_tracker.oracle_maps import OracleMaps


class OracleTests(unittest.TestCase):
    def test_frozen_oracle_schedule(self):
        model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), cache={})
        tracker = MapHandoff(model, 'oracle', lambda row: None, None)
        calls = []

        def reconstruct(images, ids, frame, kind):
            calls.append((frame, kind, ids))
            return (torch.zeros(2, 28, 28, 3), torch.eye(4).repeat(2, 1, 1),
                    torch.ones(2, 28, 28), torch.eye(4))

        tracker.reconstruct = reconstruct
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        with patch('torch.cuda.synchronize'), patch('kv_tracker.map_handoff.pi3_inference',
                return_value=torch.eye(4)[None, None]):
            tracker.bootstrap(image)
            for frame in range(1, 1185):
                tracker.step(image, frame)
        self.assertEqual(calls, [(0, 'bootstrap', [0, 0]), (49, 'rebuild', [0, 49])])
        self.assertEqual(tracker.ids, [0, 49])

    def test_shared_geometry_uses_both_coordinate_offsets(self):
        model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]),
                                cache={1: {'k': torch.zeros(1)}})
        tracker = MapHandoff(model, 'oracle', lambda row: None, None)
        tracker.origin = torch.eye(4)
        tracker.origin[:3, 3] = torch.tensor([1., 2., 3.])
        tracker.transform = (torch.tensor(2.), torch.eye(3, dtype=torch.float64),
                             torch.tensor([4., 5., 6.], dtype=torch.float64))
        outputs = (torch.ones(1, 1, 8, 8, 3), torch.eye(4)[None, None],
                   torch.ones(1, 1, 8, 8, 1), None, None, None)
        with patch('torch.cuda.synchronize'), patch('kv_tracker.map_handoff.pi3_inference',
                                                   return_value=outputs):
            points, pose, _ = tracker.query_geometry(np.zeros((8, 8, 3)), 125)
        torch.testing.assert_close(points[0, 0], torch.tensor([8., 11., 14.], dtype=torch.float64))
        torch.testing.assert_close(pose[:3, 3], torch.tensor([6., 9., 12.], dtype=torch.float64))

    def test_delayed_connection_and_retirement(self):
        class LocalMap:
            def __init__(self, model, mode, log, save):
                self.model = model
                self.transform = (torch.tensor(1., dtype=torch.float64),
                                  torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))

            def bootstrap(self, image):
                self.images, self.ids = [image], [0]
                self.model.cache = {1: {'k': torch.ones(4)}}
                return np.eye(4, dtype=np.float32)

            def step(self, image, frame):
                if frame == 49:
                    self.images, self.ids = [self.images[0], image], [0, 49]
                    self.model.cache = {1: {'k': torch.ones(8)}}
                    self.anchor_points, self.anchor_conf = torch.ones(8, 8, 3), torch.ones(8, 8)
                    self.bank_last_pose = torch.eye(4)
                return np.eye(4, dtype=np.float32)

            def cache_bytes(self):
                return sum(t.numel() * t.element_size()
                           for layer in self.model.cache.values() for t in layer.values())

            def query_geometry(self, image, frame):
                return torch.ones(8, 8, 3), torch.eye(4), torch.ones(8, 8)

        for accepted in (True, False):
            with patch('kv_tracker.oracle_maps.MapHandoff', LocalMap):
                model = SimpleNamespace(cache={})
                tracker = OracleMaps(model, (0, 125, 250), lambda row: None, lambda e, d: None)
                fitted = tracker.world if accepted else None
                with patch('kv_tracker.oracle_maps.bridge', return_value=(
                        fitted, dict(accepted=accepted), {})):
                    for frame in range(250):
                        tracker.step(np.full((8, 8, 3), frame, dtype=np.uint8), frame)
                        if 125 <= frame < 174:
                            self.assertIsNotNone(tracker.pending)
                            self.assertEqual(tracker.events, [])
            event = tracker.events[0]
            self.assertEqual(event['old_ids'], [0, 49])
            self.assertEqual(event['new_ids'], [125, 174])
            self.assertEqual(event['decision_frame'], 174)
            self.assertTrue(event['old_images_released'] and event['old_kv_released'])
            self.assertEqual(tracker.connected, accepted)
            self.assertEqual(len(tracker.transforms), 2)


if __name__ == '__main__':
    unittest.main()
