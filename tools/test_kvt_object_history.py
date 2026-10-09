"""CAMP-only saved-error grouping and no-retirement contracts."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from kvt_object_history import error_groups
from kv_tracker.reanchor_maps import ReanchorMaps


class ObjectHistoryTests(unittest.TestCase):
    def test_no_boundaries_never_retire_or_connect(self):
        local = Mock()
        local.bootstrap.return_value = np.eye(4)
        local.step.return_value = np.eye(4)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with patch('kv_tracker.reanchor_maps.MapHandoff', return_value=local) as factory:
            tracker = ReanchorMaps(SimpleNamespace(cache={}), [0, 652], Mock(), Mock())
            for frame in range(652):
                tracker.step(image, frame)
            tracker.finish(image, frame)
        factory.assert_called_once()
        local.bootstrap.assert_called_once()
        self.assertEqual(local.step.call_count, 651)
        local.query_geometry.assert_not_called()
        self.assertEqual(tracker.events, [])
        self.assertEqual(len(tracker.transforms), 1)
        self.assertIsNone(tracker.pending)

    def test_error_groups_priority_and_native_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, reference = root / 'result', root / 'reference'
            result.mkdir()
            reference.mkdir()
            np.savez(result / 'evaluation.npz', pair_end_frames=np.arange(1, 6),
                     translation_m=np.array([1., 2., 3., 4., 5.]))
            np.savez(reference / 'evaluation.npz', pair_end_frames=np.arange(1, 6),
                     translation_m=np.ones(5))
            (result / 'events.json').write_text(json.dumps([{'decision_frame': 3}]))
            (result / 'inference.jsonl').write_text('\n'.join(json.dumps(
                dict(kind='update_total', global_frame=i)) for i in (2, 3, 4)))
            groups = error_groups(result, reference, [0, 2, 6])
            self.assertEqual([groups[k]['pairs'] for k in ('cut', 'connection', 'refresh', 'other')],
                             [1, 1, 1, 2])
            self.assertEqual(groups['other']['excess_squared_error_sum'], 24.)
            self.assertEqual(groups['segment_2_6']['pairs'], 3)
            self.assertEqual(groups['cut']['squared_error_sum'], 4.)
            np.savez(reference / 'evaluation.npz', pair_end_frames=np.arange(2, 7),
                     translation_m=np.ones(5))
            with self.assertRaises(AssertionError):
                error_groups(result, reference, [0, 2, 6])


if __name__ == '__main__':
    unittest.main()
