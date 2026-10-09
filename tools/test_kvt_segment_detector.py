"""CAMP-only causality, persistence and image-matching contracts."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from kvt_segment_detector import INLIER_THRESHOLDS, SegmentDetector, FlowSegmentDetector


class DetectorContracts(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        self.image = np.random.default_rng(0).integers(0, 256, (224, 308, 3), dtype=np.uint8)

    def test_same_view_supported_and_textureless_unknown(self):
        detector = SegmentDetector()
        detector.step(self.image, 0)
        row, _ = detector.step(self.image, 1)
        self.assertEqual(row['status'], 'supported')
        row, _ = detector.step(np.zeros_like(self.image), 2)
        self.assertEqual(row['status'], 'unknown_texture')
        self.assertFalse(row['boundary'])

    def test_persistence_interrupt_and_current_frame_reset(self):
        detector = SegmentDetector()
        detector.step(self.image, 0)
        for frame in range(1, 50):
            detector.step(self.image, frame)
        # No matches simulates loss of anchor support with textured frames.
        with patch.object(detector, 'matcher') as matcher:
            matcher.knnMatch.return_value = []
            for frame in range(50, 54):
                row, _ = detector.step(self.image, frame)
                self.assertFalse(row['boundary'])
            row, _ = detector.step(np.zeros_like(self.image), 54)
            self.assertEqual(row['low_run'], 0)
            for frame in range(55, 59):
                row, _ = detector.step(self.image, frame)
                self.assertFalse(row['boundary'])
            row, _ = detector.step(self.image, 59)
            self.assertTrue(row['boundary'])
            self.assertEqual(row['anchor_frame'], 0)
            self.assertEqual(detector.anchor[0], 59)
            row, _ = detector.step(self.image, 60)
            self.assertEqual(row['anchor_frame'], 59)
            self.assertFalse(row['boundary'])
            self.assertEqual(row['low_run'], 0)
            for frame in range(61, 109):
                row, _ = detector.step(self.image, frame)
                self.assertFalse(row['boundary'])
                self.assertEqual(row['low_run'], 0)

    def test_future_suffix_does_not_change_prefix(self):
        def run(suffix):
            cv2.setRNGSeed(0)
            detector = SegmentDetector()
            rows = []
            with patch.object(detector, 'matcher') as matcher:
                matcher.knnMatch.return_value = []
                for frame, image in enumerate([self.image] * 55 + suffix):
                    row, _ = detector.step(image, frame)
                    del row['seconds']
                    rows.append(row)
            return rows
        prefix = run([])
        extended = run([np.zeros_like(self.image)] * 5)
        self.assertEqual(prefix, extended[:len(prefix)])
        self.assertEqual([r['frame'] for r in prefix if r['boundary']], [54])

    def test_threshold_is_strict_and_arms_have_independent_anchors(self):
        keypoints = [cv2.KeyPoint(float(x), float(y), 3.)
                     for y in (28, 84, 140, 196) for x in (38, 115, 192, 269)
                     for _ in range(4)]
        descriptors = np.zeros((64, 32), np.uint8)
        pairs = [[cv2.DMatch(i, i, 10.), cv2.DMatch(i, (i + 1) % 64, 100.)]
                 for i in range(64)]
        mask = np.zeros((64, 1), np.uint8)
        # 20 verified matches across five distinct cells: coverage cannot trigger.
        mask[:20] = 1
        detectors = [SegmentDetector(threshold) for threshold in INLIER_THRESHOLDS]
        for detector in detectors:
            with patch.object(detector, 'orb') as orb, patch.object(
                    detector, 'matcher') as matcher, patch(
                    'kvt_segment_detector.cv2.findHomography', return_value=(None, mask)), patch(
                    'kvt_segment_detector.cv2.findFundamentalMat', return_value=(None, mask)):
                orb.detectAndCompute.return_value = (keypoints, descriptors)
                matcher.knnMatch.return_value = pairs
                for frame in range(55):
                    row, _ = detector.step(self.image, frame)
                self.assertEqual(row['inliers'], 20)
                self.assertEqual(row['cells'], 5)
                self.assertEqual(row['boundary'], detector.minimum_inliers > 20)
        self.assertEqual([d.anchor[0] for d in detectors], [0, 0, 0, 54, 54])

    def test_maximum_age_cuts_current_frame_even_without_texture(self):
        image = np.zeros_like(self.image)
        for detector in (FlowSegmentDetector(maximum_segment_frames=75),
                         SegmentDetector(35, maximum_segment_frames=75)):
            for frame in range(76):
                row, _ = detector.step(image, frame)
                self.assertEqual(row['boundary'], frame == 75)
            self.assertEqual(row['status'], 'maximum_age')
            row, _ = detector.step(image, 76)
            self.assertEqual(row['anchor_frame'], 75)
            self.assertFalse(row['boundary'])

    def test_object_coverage_uses_anchor_support_not_image_area(self):
        # All 64 features occupy one image cell, but span 16 object-relative cells.
        keypoints = [cv2.KeyPoint(float(x), float(y), 3.)
                     for y in (10, 20, 30, 40) for x in (10, 20, 30, 40)
                     for _ in range(4)]
        descriptors = np.zeros((64, 32), np.uint8)
        pairs = [[cv2.DMatch(i, i, 10.), cv2.DMatch(i, (i + 1) % 64, 100.)]
                 for i in range(64)]
        full = np.ones((64, 1), np.uint8)
        for mode in ('image', 'anchor_support'):
            detector = SegmentDetector(25, coverage_mode=mode)
            with patch.object(detector, 'orb') as orb, patch.object(
                    detector, 'matcher') as matcher, patch(
                    'kvt_segment_detector.cv2.findHomography', return_value=(None, full)), patch(
                    'kvt_segment_detector.cv2.findFundamentalMat', return_value=(None, full)):
                orb.detectAndCompute.return_value = (keypoints, descriptors)
                matcher.knnMatch.return_value = pairs
                detector.step(self.image, 0)
                # Current object translated/scaled: anchor support is unchanged.
                orb.detectAndCompute.return_value = (
                    [cv2.KeyPoint(p.pt[0] * .5 + 10, p.pt[1] * .5 + 10, 3.)
                     for p in keypoints], descriptors)
                row, _ = detector.step(self.image, 1)
                self.assertEqual(row['cells'], 1)
                self.assertEqual(row['status'], 'supported' if mode == 'anchor_support' else 'low_overlap')
                if mode == 'anchor_support':
                    self.assertEqual(row['anchor_coverage'], 1.)
                    self.assertEqual(row['anchor_cells'], 16)

    def test_object_support_persistence_unknown_and_prefix(self):
        def run(count):
            cv2.setRNGSeed(0)
            detector = SegmentDetector(25, coverage_mode='anchor_support')
            rows = []
            for frame in range(count):
                if frame < 50:
                    row, _ = detector.step(self.image, frame)
                else:
                    with patch.object(detector, 'matcher') as matcher:
                        matcher.knnMatch.return_value = []
                        row, _ = detector.step(
                            np.zeros_like(self.image) if frame == 53 else self.image, frame)
                row.pop('seconds')
                rows.append(row)
            return rows
        prefix = run(59)
        self.assertEqual(prefix, run(62)[:59])
        self.assertEqual(prefix[53]['status'], 'unknown_texture')
        self.assertEqual([r['frame'] for r in prefix if r['boundary']], [58])

    def test_object_coverage_can_fail_with_enough_inliers(self):
        points = [cv2.KeyPoint(float(x), float(y), 3.)
                  for y in (10, 60, 110, 160) for x in (10, 60, 110, 160)
                  for _ in range(8)]
        descriptors = np.zeros((128, 32), np.uint8)
        pairs = [[cv2.DMatch(i, i, 10.), cv2.DMatch(i, (i + 1) % 128, 100.)]
                 for i in range(128)]
        inliers = np.zeros((128, 1), np.uint8)
        inliers[:32] = 1  # Four of sixteen anchor cells, despite enough matches.
        detector = SegmentDetector(25, coverage_mode='anchor_support')
        with patch.object(detector, 'orb') as orb, patch.object(
                detector, 'matcher') as matcher, patch(
                'kvt_segment_detector.cv2.findHomography', return_value=(None, inliers)), patch(
                'kvt_segment_detector.cv2.findFundamentalMat', return_value=(None, inliers)):
            orb.detectAndCompute.return_value = (points, descriptors)
            matcher.knnMatch.return_value = pairs
            detector.step(self.image, 0)
            row, _ = detector.step(self.image, 1)
        self.assertEqual(row['anchor_coverage'], .25)
        self.assertFalse(row['low_inliers'])
        self.assertTrue(row['low_coverage'])

    def test_flow_tracks_gradual_motion_and_checks_prefix_causality(self):
        images = [np.roll(self.image, shift, axis=1) for shift in range(12)]
        def run(frames):
            detector = FlowSegmentDetector()
            rows = []
            for frame, image in enumerate(frames):
                row, _ = detector.step(image, frame)
                del row['seconds']
                rows.append(row)
            return rows
        prefix = run(images[:8])
        self.assertEqual(prefix, run(images)[:8])
        self.assertGreater(prefix[-1]['surviving_fraction'], .5)
        self.assertEqual(prefix[-1]['status'], 'supported')

    def test_flow_does_not_replenish_anchor_support_and_resets_at_current_frame(self):
        detector = FlowSegmentDetector()
        detector.step(self.image, 0)
        original = detector.initial_features
        with patch('kvt_segment_detector.cv2.calcOpticalFlowPyrLK') as flow:
            # Lose every original feature, despite the current image being textured.
            points = detector.points.copy()
            flow.return_value = (points, np.zeros((len(points), 1), np.uint8), None)
            row, _ = detector.step(self.image, 1)
            self.assertEqual(row['features'], 0)
            self.assertEqual(row['anchor_features'], original)
            for frame in range(2, 54):
                row, _ = detector.step(self.image, frame)
                self.assertFalse(row['boundary'])
            row, _ = detector.step(self.image, 54)
            self.assertTrue(row['boundary'])
            self.assertEqual(row['anchor_frame'], 0)
            self.assertEqual(detector.anchor_frame, 54)
            self.assertGreater(detector.initial_features, 40)


if __name__ == '__main__':
    unittest.main()
