"""Remote contracts for passive context probes; do not execute on the editing Mac."""

import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

import test_stream_cache_research as fixtures
from stream_cache_context_diagnostic import correlation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'streamvggt/src'))
from streamvggt.models.context_probe import ContextProbe
from streamvggt.models.research_cache import ResearchCache, ResearchConfig, nbytes
from streamvggt.layers.rope import RotaryPositionEmbedding2D


class ContextProbeTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ResearchTests()
        fixture.setUp()
        self.model, self.images = fixture.model, fixture.images
        for block in self.model.global_blocks:
            original = block.attn
            attention = nn.Module()
            attention.num_heads, attention.head_dim = 2, 4
            attention.fused_attn = True
            attention.qkv, attention.proj = original.qkv, original.proj
            attention.proj_drop = nn.Identity()
            attention.q_norm, attention.k_norm = nn.LayerNorm(4), nn.LayerNorm(4)
            attention.rope = RotaryPositionEmbedding2D()
            block.attn = attention.eval()
        self.mask = torch.zeros(8, 8, dtype=torch.bool)
        self.mask[:2, :2] = True
        self.config = ResearchConfig(method='context', context='dense', frame_budget=3)

    def test_identity_probes_and_dense_trajectory_preservation_with_eviction(self):
        observer = ContextProbe(2, count=16, layers=(0, 1, 2))
        reference = ResearchCache(self.model, self.config)
        observed = ResearchCache(self.model, self.config, observer)
        with torch.no_grad():
            for index, image in enumerate(self.images[:6]):
                expected = reference.forward(image, index, self.mask)
                actual = observed.forward(image, index, self.mask)
                for x, y in zip(expected, actual):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
                self.assertEqual(observer.report['past_frames'], list(reference.records)[:-1])
                for row in observer.report['conditions'].values():
                    for layer in row['layers']:
                        for key in ('history_and_current', 'current_only'):
                            self.assertLess(layer[key]['camera_rse'], 1e-10)
                            self.assertLess(layer[key]['register_rse'], 1e-10)
                reference.maintain(index)
                observed.maintain(index)
                self.assertEqual(reference.memory(), observed.memory())
                for bank in observer.banks.values():
                    self.assertEqual(list(bank), list(observed.records))

    def test_compressed_shadow_banks_do_not_mutate_reference_cache(self):
        observer = ContextProbe(2, count=2, layers=(0, 2))
        reference = ResearchCache(self.model, self.config)
        observed = ResearchCache(self.model, self.config, observer)
        with torch.no_grad():
            for index, image in enumerate(self.images[:5]):
                expected = reference.forward(image, index, self.mask)
                actual = observed.forward(image, index, self.mask)
                for x, y in zip(expected, actual):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
                reference.maintain(index)
                observed.maintain(index)
                for frame, record in reference.records.items():
                    for a, b in zip(record['kv'], observed.records[frame]['kv']):
                        for x, y in zip(a, b):
                            torch.testing.assert_close(x, y, atol=0, rtol=0)
        self.assertGreater(observer.memory(), 0)
        for bank in observer.banks.values():
            for layers in bank.values():
                for pair in layers.values():
                    for value in pair:
                        self.assertEqual(value.untyped_storage().nbytes(), nbytes(value))
        scores = observer.report['conditions']['appearance64']['layers']
        self.assertTrue(any(row['history_and_current']['camera_rse'] > 0 for row in scores))
        self.assertTrue(all(row['compact_keys'] < row['dense_keys'] for row in scores))

    def test_probe_prefix_is_independent_of_future_frames(self):
        reports = []
        for count in (4, 7):
            observer = ContextProbe(2, count=2, layers=(0, 2))
            session = ResearchCache(self.model, self.config, observer)
            rows = []
            with torch.no_grad():
                for index, image in enumerate(self.images[:count]):
                    session.forward(image, index, self.mask)
                    session.maintain(index)
                    rows.append(copy.deepcopy(observer.report))
            reports.append(rows)
        self.assertEqual(reports[0], reports[1][:4])

    def test_correlations_handle_ranks_and_undefined_constants(self):
        self.assertAlmostEqual(correlation([1, 2, 3, 4], [4, 3, 2, 1])['rho'], -1.)
        self.assertAlmostEqual(correlation([1, 1, 3, 4], [1, 1, 3, 4])['rho'], 1.)
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3])['rho'])


if __name__ == '__main__':
    unittest.main()
