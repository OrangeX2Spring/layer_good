"""B-level tensor contracts, executed by the existing CAMP test discovery."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from stream_cache_structure import attention_bins, retention_mask, channel_basis, StructureExperiment


class StructureTests(unittest.TestCase):
    def test_profile_uniform_attention_matches_frame_mass(self):
        attention = SimpleNamespace()
        adapter = SimpleNamespace(host='streamvggt', special=1,
                                  original_attention=[(attention, None)],
                                  token_frames=torch.arange(9).repeat_interleave(5))
        experiment = StructureExperiment(adapter, 'profile', Path('.'))
        experiment.current = 9
        mask = experiment.attend(0, torch.zeros(1, 2, 5, 4),
                                 torch.zeros(1, 2, 50, 4), None)
        self.assertIsNone(mask)
        torch.testing.assert_close(experiment.samples[0][0],
                                   torch.tensor([[.1, .1, .4, .4]]).expand(2, 4))

    def test_disjoint_bins_and_attention_mask_consumption(self):
        frames = torch.arange(10).repeat_interleave(2)
        bins = attention_bins(frames, 9)
        self.assertTrue(torch.equal(bins.sum(0), torch.ones(20, dtype=torch.long)))
        self.assertEqual(bins.sum(1).tolist(), [2, 2, 8, 8])
        mask = retention_mask(frames, 9, torch.tensor([True, False]))
        q, k = torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 20, 4)
        v = torch.arange(20.).reshape(1, 1, 20, 1).expand(1, 2, 20, 4)
        actual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(actual[0, 0], v[0, 0, mask[0, 0, 0]].mean(0)[None])
        torch.testing.assert_close(actual[0, 1], v[0, 1].mean(0)[None])
        self.assertEqual(int(retention_mask(frames, 9, torch.tensor([True, True]), True).sum()), 2)
        self.assertTrue(retention_mask(frames, 9, torch.tensor([True, False]), True).all())

    def test_basis_preserves_rank_one_and_has_sorted_energy(self):
        value = torch.arange(1., 21.).reshape(1, 1, 20, 1) * torch.tensor([1., 2., 3., 4.])
        spectrum, basis = channel_basis(value)
        self.assertTrue((spectrum.diff(dim=-1) <= 0).all())
        reduced = basis[..., :1]
        torch.testing.assert_close(value @ reduced @ reduced.transpose(-1, -2), value)

    def test_register_eviction_preserves_specials_positions_and_heads(self):
        adapter = SimpleNamespace(host='streamvggt', special=2, original_attention=[])
        experiment = StructureExperiment(adapter, 'registers', Path('.'))
        frames = torch.arange(7).repeat_interleave(5)
        values = torch.arange(35.).reshape(1, 1, 1, 35, 1)
        attention = SimpleNamespace(cache_positions=torch.arange(35).reshape(1, 35, 1))
        adapter.original_attention = [(attention, None)]
        adapter.token_frames = frames
        adapter.aggregator_cache = [(values.clone(), values.clone())]
        camera = torch.randn(3)
        adapter.camera_cache = camera
        experiment.current = 6
        experiment.after_prune()
        expected = torch.tensor([0, 1, 5, 6, 10, 11, *range(15, 35)])
        torch.testing.assert_close(adapter.aggregator_cache[0][0].flatten(), expected.float())
        torch.testing.assert_close(attention.cache_positions.flatten(), expected)
        self.assertIs(adapter.camera_cache, camera)

    def test_fake_projection_changes_only_new_entry(self):
        adapter = SimpleNamespace(token_frames=torch.tensor([0, 0, 1, 1]))
        tensor = torch.arange(16.).reshape(1, 1, 1, 4, 4)
        adapter.aggregator_cache = [(tensor.clone(), tensor.clone())]
        experiment = StructureExperiment.__new__(StructureExperiment)
        experiment.adapter, experiment.mode, experiment.current = adapter, 'rank', 1
        experiment.profile = {'basis': torch.eye(4).reshape(1, 1, 1, 4, 4).expand(1, 2, 1, 4, 4)}
        experiment.after_prune()
        for result in adapter.aggregator_cache[0]:
            torch.testing.assert_close(result[..., :2, :], tensor[..., :2, :])
            torch.testing.assert_close(result[..., 2:, :2], tensor[..., 2:, :2])
            self.assertEqual(int(torch.count_nonzero(result[..., 2:, 2:])), 0)


if __name__ == '__main__':
    unittest.main()
