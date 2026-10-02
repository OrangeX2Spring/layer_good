"""CAMP-only causality, persistence and image-matching contracts."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from kvt_segment_detector import SegmentDetector


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


if __name__ == '__main__':
    unittest.main()
