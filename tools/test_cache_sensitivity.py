"""CAMP tensor and selection contracts for the bounded sensitivity diagnostic."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from cache_sensitivity_sweep import calibration_conditions, pose_score
from stream_cache_quantization import fake_quantize
from stream_cache_sensitivity import CacheCompression


class SensitivityTests(unittest.TestCase):
    def pairs(self):
        torch.manual_seed(17)
        return [(torch.randn(1, 4, 35, 8), torch.randn(1, 4, 35, 8))
                for _ in range(6)]

    def test_int2_scalar_grid_and_payload(self):
        value = torch.tensor([0., 1., 2., 3., 7.]).reshape(1, 1, 5, 1)
        restored, size = fake_quantize(value, 2, 2, group_size=4)
        torch.testing.assert_close(restored, value, rtol=0, atol=0)
        self.assertEqual(size, 18)  # two one-byte code groups + two FP32 min/scale pairs

    def test_native_identity_profile_and_exact_bytes(self):
        pairs = self.pairs()
        before = [(k.clone(), v.clone()) for k, v in pairs]
        compression = CacheCompression(dict(bits=0, profile=True))
        report = compression.apply(pairs)
        self.assertFalse(report['changed'])
        self.assertEqual(report['actual_bytes'], report['analytical_payload_bytes'])
        self.assertEqual(compression.profile_calls, 1)
        for pair, original in zip(pairs, before):
            for value, expected in zip(pair, original):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
        totals = [sum(values) for values in compression.variance.values()]
        np.testing.assert_allclose(totals, [totals[0]] * 3)

    def test_protection_exact_values_and_equal_family_budgets(self):
        for family, count in [('layer', 6), ('head', 4), ('channel', 4)]:
            budgets = []
            for group in range(count):
                pairs = self.pairs()
                before = [(k.clone(), v.clone()) for k, v in pairs]
                compression = CacheCompression(dict(bits=2, family=family, group=group))
                report = compression.apply(pairs)
                budgets.append(report['analytical_payload_bytes'])
                expected_bytes = 0
                for layer, (pair, original) in enumerate(zip(pairs, before)):
                    for which, (value, native) in enumerate(zip(pair, original)):
                        expected, size = fake_quantize(native, 2, 2 if which == 0 else 3)
                        expected_bytes += size
                        mask = torch.zeros_like(native, dtype=torch.bool)
                        if family == 'layer' and layer == group:
                            mask[:] = True
                        elif family == 'head':
                            mask[:, group:group + 1] = True
                        elif family == 'channel':
                            mask[:, :, :, group * 2:(group + 1) * 2] = True
                        expected[mask] = native[mask]
                        expected_bytes += int(mask.sum()) * native.element_size()
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                self.assertEqual(report['analytical_payload_bytes'], expected_bytes)
                self.assertLess(expected_bytes, report['actual_bytes'])
                self.assertTrue(report['changed'])
            self.assertEqual(budgets, [budgets[0]] * count)

    def test_stream_old_entries_eviction_and_refresh(self):
        for host in ('streamvggt', 'longstream'):
            compression = CacheCompression(dict(bits=2, family='head', group=1))
            adapter = SimpleNamespace(host=host, session=SimpleNamespace())
            previous = None
            for frame, retained, refresh in [(0, [0], False), (1, [0, 1], False),
                                              (2, [1, 2], False), (2, [2], True)]:
                pairs = self.pairs()
                if len(retained) == 2:
                    pairs = [(torch.cat((old[0], new[0]), 2),
                              torch.cat((old[1], new[1]), 2))
                             for old, new in zip(previous, pairs)]
                adapter.token_frames = torch.tensor(retained).repeat_interleave(35)
                if host == 'streamvggt':
                    adapter.aggregator_cache = [(k.unsqueeze(2), v.unsqueeze(2)) for k, v in pairs]
                else:
                    adapter.session.aggregator_kv_cache_list = pairs
                event = compression.stream(adapter, frame, refresh=refresh)
                self.assertEqual(set(compression.frame_bytes), set(retained))
                self.assertEqual(event['analytical_compression_bytes'],
                                 sum(compression.frame_bytes.values()) + 8)
                if len(retained) == 2:
                    for pair, old in zip(pairs, previous):
                        for value, expected in zip(pair, old):
                            torch.testing.assert_close(value[:, :, :35], expected, rtol=0, atol=0)
                previous = [(k[:, :, -35:].clone(), v[:, :, -35:].clone()) for k, v in pairs]

    def test_kvt_rebuild_records_and_descriptor(self):
        compression = CacheCompression(dict(bits=2, family='layer', group=0))
        for ids in ([0, 0], [0, 49]):
            cache = {2 * i + 1: dict(k=k, v=v) for i, (k, v) in enumerate(self.pairs())}
            compression.kvt(cache, ids)
            report = compression.events[-1]
            self.assertEqual(report['frame_ids'], list(ids))
            self.assertEqual(report['layer_ids'], list(range(1, 12, 2)))
            self.assertTrue(report['changed'])
        self.assertEqual(len(compression.events), 2)

    def test_pose_score_and_fixed_calibration_grid(self):
        native = np.tile(np.eye(4), (5, 1, 1))
        native[:, 0, 3] = np.arange(5)
        self.assertEqual(pose_score(native, native)['score'], 0.)
        candidate = native.copy()
        candidate[:, 0, 3] *= 2
        self.assertAlmostEqual(pose_score(candidate, native)['score'], 1.)
        rows = calibration_conditions()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({r['name'] for r in rows}), 18)


if __name__ == '__main__':
    unittest.main()
