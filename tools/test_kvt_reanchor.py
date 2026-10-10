"""CAMP-only schedule, retirement and connection contracts for reanchored maps."""
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np
import torch

from kv_tracker.map_handoff import MapHandoff, transform_pose
from kv_tracker.reanchor_maps import ReanchorMaps


def pose(angle, position):
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = position
    return result


class ReanchorTests(unittest.TestCase):
    def test_three_image_map_keeps_previous_view_and_restores_scale(self):
        model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), cache={})
        tracker = MapHandoff(model, 'reanchor', lambda row: None, None, local_keyframe_cap=3)
        calls = []

        def reconstruct(images, ids, frame, kind):
            calls.append(list(ids))
            self.assertEqual([int(image[0, 0, 0]) for image in images], ids)
            points = torch.ones(len(ids), 28, 28, 3) * (2. if frame > 49 else 1.)
            if frame == 49:
                tracker.anchor_points, tracker.anchor_conf = points[0].clone(), torch.ones(28, 28)
            return points, torch.eye(4).repeat(len(ids), 1, 1), torch.ones(len(ids), 28, 28), torch.eye(4)

        tracker.reconstruct = reconstruct
        with patch('torch.cuda.synchronize'), patch('kv_tracker.map_handoff.pi3_inference',
                return_value=torch.eye(4)[None, None]):
            tracker.bootstrap(np.zeros((28, 28, 3), dtype=np.uint8))
            for frame in range(1, 200):
                tracker.step(np.full((28, 28, 3), frame, dtype=np.uint8), frame)
        self.assertEqual(calls, [[0, 0], [0, 49], [0, 49, 99], [0, 99, 149], [0, 149, 199]])
        self.assertEqual(tracker.ids, [0, 149, 199])
        self.assertEqual(tracker.rebuild_scale, .5)

    def test_pinned_rebuild_keeps_refresh_frame_pose(self):
        # The 99 rebuild re-solves the refresh frame at a different pose; later
        # queries live in that new solution, as on the far side of the office.
        agreed = torch.from_numpy(pose(.2, [1., .5, 0.]))
        moved = torch.from_numpy(pose(-.3, [-2., 1., .4]))
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        outputs, rows = {}, {}
        for pinned in (False, True):
            model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), cache={})
            logged = []
            tracker = MapHandoff(model, 'reanchor', logged.append, None, local_keyframe_cap=3,
                                 pin_rebuilds=pinned)
            raw = {'pose': agreed}

            def reconstruct(images, ids, frame, kind):
                if frame == 49:
                    tracker.anchor_points, tracker.anchor_conf = torch.ones(28, 28, 3), torch.ones(28, 28)
                if frame == 99:
                    raw['pose'] = moved
                poses = torch.eye(4).repeat(len(ids), 1, 1)
                poses[-1] = moved if frame == 99 else agreed
                return torch.ones(len(ids), 28, 28, 3), poses, torch.ones(len(ids), 28, 28), torch.eye(4)

            tracker.reconstruct = reconstruct
            with patch('torch.cuda.synchronize'), patch('kv_tracker.map_handoff.pi3_inference',
                    side_effect=lambda *a, **k: raw['pose'][None, None]):
                tracker.bootstrap(image)
                outputs[pinned] = {frame: tracker.step(image, frame) for frame in range(1, 101)}
            rows[pinned] = [row for row in logged if row['kind'] == 'update_total']
        np.testing.assert_allclose(outputs[True][100], outputs[True][99], atol=1e-5)
        self.assertGreater(np.abs(outputs[False][100] - outputs[False][99]).max(), .5)
        np.testing.assert_allclose(outputs[True][99], outputs[False][99], atol=1e-6)
        self.assertNotIn('pin_rotation_deg', rows[False][0])
        self.assertAlmostEqual(rows[True][0]['pin_rotation_deg'], 0., places=3)
        self.assertAlmostEqual(rows[True][1]['pin_rotation_deg'], np.degrees(.5), places=3)
        self.assertAlmostEqual(rows[True][1]['pin_position_step'],
                               float(np.linalg.norm([3., -.5, -.4])), places=5)

    def test_pinned_scale_matches_refresh_frame_depths(self):
        model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), cache={})
        tracker = MapHandoff(model, 'reanchor', lambda row: None, None, local_keyframe_cap=3,
                             pin_rebuilds=True, pin_scale=True)
        reads = []
        camera = torch.from_numpy(pose(.3, [1., -2., .5])).double()
        depth = torch.zeros(28, 28, 3, dtype=torch.float64)
        depth[..., 2] = 4.

        def query_geometry(image, frame):
            # Outgoing bank: refresh-frame points 4 units in front of its camera.
            reads.append(frame)
            return depth @ camera[:3, :3].T + camera[:3, 3], camera.clone(), torch.ones(28, 28)

        def reconstruct(images, ids, frame, kind):
            # Each rebuild sees the same frame 1 unit deep; the anchor points disagree.
            poses = torch.eye(4).repeat(len(ids), 1, 1)
            points = torch.full((len(ids), 28, 28, 3), 7.)
            points[-1] = depth.float() / 4
            if frame == 49:
                tracker.anchor_points, tracker.anchor_conf = points[0].clone(), torch.ones(28, 28)
            return points, poses, torch.ones(len(ids), 28, 28), torch.eye(4)

        tracker.query_geometry, tracker.reconstruct = query_geometry, reconstruct
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        with patch('torch.cuda.synchronize'), patch('kv_tracker.map_handoff.pi3_inference',
                return_value=torch.eye(4)[None, None]):
            tracker.bootstrap(image)
            for frame in range(1, 100):
                tracker.step(image, frame)
                if frame == 49:
                    self.assertAlmostEqual(tracker.rebuild_scale, 4.)
        self.assertEqual(reads, [49, 99])
        self.assertAlmostEqual(tracker.rebuild_scale, 4.)

    def test_scale_pin_requires_rigid_pin(self):
        model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), cache={})
        with self.assertRaises(AssertionError):
            MapHandoff(model, 'reanchor', lambda row: None, None, pin_scale=True)

    def test_pinning_requires_a_single_map(self):
        with self.assertRaises(AssertionError):
            ReanchorMaps(SimpleNamespace(cache={}), (0, 60, 120), Mock(), Mock(), pin_rebuilds=True)

    def test_short_final_segment_connects_at_eof(self):
        class LocalMap:
            def __init__(self, model, mode, log, save, query_executor=None, local_keyframe_cap=2,
                         pin_rebuilds=False, pin_scale=False):
                self.model = model
                self.transform = (torch.tensor(1., dtype=torch.float64),
                                  torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))

            def bootstrap(self, image):
                self.images, self.ids = [image], [0]
                self.model.cache = {1: {'k': torch.ones(4)}}
                return np.eye(4, dtype=np.float32)

            def step(self, image, frame, update=True, dense_query=False):
                return np.eye(4, dtype=np.float32)

            def query_geometry(self, image, frame):
                return torch.ones(8, 8, 3), torch.eye(4, dtype=torch.float64), torch.ones(8, 8)

            def reconstruct(self, images, ids, frame, kind):
                self.model.cache = {1: {'k': torch.ones(8)}}
                return (torch.ones(2, 8, 8, 3), torch.eye(4).repeat(2, 1, 1),
                        torch.ones(2, 8, 8), torch.eye(4))

        # Include a one-frame tail: duplicated anchor/last is still a valid pair.
        for tail in (1, 30, 49):
            model = SimpleNamespace(cache={})
            end = 60 + tail
            with patch('kv_tracker.reanchor_maps.MapHandoff', LocalMap), patch(
                    'kv_tracker.reanchor_maps.bridge',
                    return_value=(None, dict(accepted=False, scale=2.), {})):
                tracker = ReanchorMaps(model, (0, 60, end), lambda row: None, lambda e, d: None)
                for frame in range(end):
                    image = np.full((8, 8, 3), frame, dtype=np.uint8)
                    tracker.step(image, frame)
                self.assertIsNotNone(tracker.pending)
                tracker.finish(image, frame)
                self.assertIsNone(tracker.pending)
                self.assertEqual(len(tracker.transforms), 2)
                self.assertEqual(tracker.tracker.ids, [0, tail - 1])
                self.assertEqual(tracker.events[0]['decision_frame'], end - 1)
                self.assertEqual(tracker.events[0]['delay_frames'], tail - 1)
                self.assertTrue(tracker.events[0]['end_of_video'])
                self.assertEqual(float(tracker.transforms[1][0]), 2.)

    def test_anchor_plus_latest_schedule(self):
        model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), cache={})
        tracker = MapHandoff(model, 'reanchor', lambda row: None, None)
        calls = []

        def reconstruct(images, ids, frame, kind):
            calls.append((frame, kind, ids))
            if frame == 49:
                tracker.anchor_points, tracker.anchor_conf = torch.ones(28, 28, 3), torch.ones(28, 28)
            return (torch.ones(2, 28, 28, 3), torch.eye(4).repeat(2, 1, 1),
                    torch.ones(2, 28, 28), torch.eye(4))

        tracker.reconstruct = reconstruct
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        with patch('torch.cuda.synchronize'), patch('kv_tracker.map_handoff.pi3_inference',
                return_value=torch.eye(4)[None, None]):
            tracker.bootstrap(image)
            for frame in range(1, 200):
                tracker.step(image, frame)
        self.assertEqual(calls, [(0, 'bootstrap', [0, 0]), (49, 'rebuild', [0, 49]),
                                 (99, 'rebuild', [0, 99]), (149, 'rebuild', [0, 149]),
                                 (199, 'rebuild', [0, 199])])

    def test_retiring_bank_skips_scheduled_rebuild(self):
        model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), cache={})
        tracker = MapHandoff(model, 'reanchor', lambda row: None, None)
        calls = []

        def reconstruct(images, ids, frame, kind):
            calls.append(kind)
            return (torch.ones(2, 28, 28, 3), torch.eye(4).repeat(2, 1, 1),
                    torch.ones(2, 28, 28), torch.eye(4))

        tracker.reconstruct = reconstruct
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        tracker.query_executor = Mock()
        with patch('torch.cuda.synchronize'), patch('kv_tracker.map_handoff.pi3_inference',
                return_value=torch.eye(4)[None, None]):
            tracker.bootstrap(image)
            tracker.step(image, 49, update=False, dense_query=True)
        tracker.query_executor.forward.assert_not_called()
        self.assertEqual(calls, ['bootstrap'])
        self.assertEqual(tracker.ids, [0])

    def test_refresh_rebuild_restores_first_rebuild_scale(self):
        model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), cache={})
        tracker = MapHandoff(model, 'reanchor', lambda row: None, None)
        base = torch.ones(2, 28, 28, 3)

        def reconstruct(images, ids, frame, kind):
            # The second rebuild returns the anchor's pointmap at twice the scale.
            points = base * (2. if frame > 49 else 1.)
            if frame == 49:
                tracker.anchor_points, tracker.anchor_conf = points[0].clone(), torch.ones(28, 28)
            return points, torch.eye(4).repeat(2, 1, 1), torch.ones(2, 28, 28), torch.eye(4)

        tracker.reconstruct = reconstruct
        raw = torch.eye(4)
        raw[0, 3] = 1.
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        with patch('torch.cuda.synchronize'), patch('kv_tracker.map_handoff.pi3_inference',
                return_value=raw[None, None]):
            tracker.bootstrap(image)  # centering translation is -1 on every axis
            before = [tracker.step(image, frame) for frame in range(1, 100)][-1]
            after = tracker.step(image, 100)
        self.assertEqual(tracker.rebuild_scale, .5)
        np.testing.assert_allclose(before[:3, 3], [0., -1., -1.])
        np.testing.assert_allclose(after[:3, 3], [-.5, -1., -1.])

    def test_immediate_retirement_and_pose_anchored_connection(self):
        old_b, new_b = pose(.4, [1., 2., 3.]), pose(-.3, [.2, -.1, .5])

        class LocalMap:
            def __init__(self, model, mode, log, save, query_executor=None, local_keyframe_cap=2,
                         pin_rebuilds=False, pin_scale=False):
                self.model = model
                self.transform = (torch.tensor(1., dtype=torch.float64),
                                  torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))

            def bootstrap(self, image):
                self.images, self.ids = [image], [0]
                self.model.cache = {1: {'k': torch.ones(4)}}
                return new_b if int(image[0, 0, 0]) == 125 else np.eye(4, dtype=np.float32)

            def step(self, image, frame, update=True, dense_query=False):
                if frame == 49:
                    self.images, self.ids = [self.images[0], image], [0, 49]
                    self.model.cache = {1: {'k': torch.ones(8)}}
                    self.anchor_points, self.anchor_conf = torch.ones(8, 8, 3), torch.ones(8, 8)
                return old_b if frame == 125 else np.eye(4, dtype=np.float32)

            def query_geometry(self, image, frame):
                return torch.ones(8, 8, 3), torch.from_numpy(old_b).double(), torch.ones(8, 8)

        model = SimpleNamespace(cache={})
        with patch('kv_tracker.reanchor_maps.MapHandoff', LocalMap), patch(
                'kv_tracker.reanchor_maps.bridge',
                return_value=(None, dict(accepted=False, scale=2.), {})):
            tracker = ReanchorMaps(model, (0, 125, 250), lambda row: None, lambda e, d: None)
            for frame in range(250):
                image = np.full((8, 8, 3), frame, dtype=np.uint8)
                if frame == 125:
                    refs = [weakref.ref(t) for layer in model.cache.values() for t in layer.values()]
                    image_refs = [weakref.ref(x) for x in tracker.tracker.images]
                tracker.step(image, frame)
                if frame == 125:
                    self.assertTrue(all(ref() is None for ref in refs + image_refs))
                    self.assertEqual(tracker.events, [])
        self.assertEqual(len(tracker.transforms), 2)
        event = tracker.events[0]
        self.assertEqual((event['old_ids'], event['new_ids'], event['decision_frame']),
                         ([0, 49], [125, 174], 174))
        # A failed point-fit check does not reject; scale still comes from the fit.
        self.assertEqual(float(tracker.transforms[1][0]), 2.)
        torch.testing.assert_close(transform_pose(torch.from_numpy(new_b), tracker.transforms[1]),
                                   torch.from_numpy(old_b).double())


if __name__ == '__main__':
    unittest.main()
