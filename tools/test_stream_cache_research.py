"""Remote CPU contracts for the research algorithms; never run on the editing Mac."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'streamvggt/src/streamvggt/models/research_cache.py'
SPEC = importlib.util.spec_from_file_location('research_cache_contract', MODULE)
research = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = research
SPEC.loader.exec_module(research)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.head_dim, self.fused_attn = 2, 4, True
        self.qkv, self.proj = nn.Linear(8, 24), nn.Linear(8, 8)
        self.q_norm, self.k_norm, self.proj_drop = nn.Identity(), nn.Identity(), nn.Identity()

    def rope(self, value, positions):
        # Position-sensitive fixture: catches row/position misalignment.
        return value + positions.sum(-1)[:, None, :, None].to(value.dtype) * .01

    def forward(self, value, pos):
        q, k, v = self.qkv(value).reshape(1, -1, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
        out = torch.nn.functional.scaled_dot_product_attention(self.rope(q, pos), self.rope(k, pos), v)
        return self.proj(out.transpose(1, 2).reshape(1, -1, 8))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()
        self.norm1, self.norm2 = nn.LayerNorm(8), nn.LayerNorm(8)
        self.ls1, self.ls2 = nn.Identity(), nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def forward(self, tokens, pos):
        tokens = tokens + self.attn(self.norm1(tokens), pos)
        return tokens + self.mlp(self.norm2(tokens))


class TinyEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 2, 2)

    def forward(self, image):
        return self.conv(image).flatten(2).transpose(1, 2)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = SimpleNamespace(
            training=False, aa_order=['frame', 'global'], aa_block_size=1,
            patch_start_idx=2, patch_size=2, rope=True,
            _resnet_mean=torch.zeros(1, 1, 3, 1, 1),
            _resnet_std=torch.ones(1, 1, 3, 1, 1),
            patch_embed=TinyEmbed().eval(),
            camera_token=torch.randn(1, 2, 1, 8), register_token=torch.randn(1, 2, 1, 8),
            frame_blocks=nn.ModuleList([TinyBlock().eval() for _ in range(3)]),
            global_blocks=nn.ModuleList([TinyBlock().eval() for _ in range(3)]),
            position_getter=lambda batch, h, w, device: torch.cartesian_prod(
                torch.arange(h, device=device), torch.arange(w, device=device))[None])
        self.images = torch.rand(18, 1, 1, 3, 8, 8)

    def stream(self, config, count=12):
        session = research.ResearchCache(self.model, research.ResearchConfig(**config))
        output, events = [], []
        with torch.no_grad():
            for index, image in enumerate(self.images[:count]):
                output.append(session.forward(image, index)[-1].clone())
                session.maintain(index)
                session.refresh(index)
                events.append(dict(session.event))
        return session, output, events

    def test_context_identity_and_foreground_preservation(self):
        patches = torch.randn(1, 16, 8)
        positions = torch.cartesian_prod(torch.arange(4), torch.arange(4))[None]
        mask = torch.zeros(16, dtype=torch.bool)
        mask[:4] = True
        for mode in ('spatial', 'appearance', 'drop'):
            sparse, pos, inverse = research.context_tokens(patches, positions, mask, 3, mode)
            self.assertEqual(sparse.shape, (1, 7, 8))
            self.assertEqual(pos.dtype, torch.long)
            torch.testing.assert_close(sparse[:, :4], patches[:, :4], rtol=0, atol=0)
            self.assertEqual(len(inverse.unique()), 7)
            identity, identity_pos, mapping = research.context_tokens(patches, positions, mask, 16, mode)
            torch.testing.assert_close(identity, patches, rtol=0, atol=0)
            torch.testing.assert_close(identity_pos, positions)
            torch.testing.assert_close(mapping, torch.arange(16))

    def test_merging_preserves_weighted_mean_across_repeated_compression(self):
        values = torch.tensor([[1.], [3.], [8.], [12.]])
        means, mass = research.merge_rows(values, torch.tensor([0, 0, 1, 1]), torch.ones(4), 2)
        final, final_mass = research.merge_rows(means, torch.zeros(2, dtype=torch.long), mass, 1)
        torch.testing.assert_close(final, values.mean(0, keepdim=True))
        self.assertEqual(float(final_mass), 4.)

    def test_dense_context_equals_native_and_sparse_heads_keep_dense_layout(self):
        native = research.ResearchCache(self.model, research.ResearchConfig())
        dense = research.ResearchCache(self.model, research.ResearchConfig(method='context', context='dense'))
        sparse = research.ResearchCache(self.model, research.ResearchConfig(method='context', context_tokens=2))
        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[:2, :2] = True
        with torch.no_grad():
            for index, image in enumerate(self.images[:3]):
                expected = native.forward(image, index)
                actual = dense.forward(image, index, mask)
                merged = sparse.forward(image, index, mask)
                for a, b, c in zip(expected, actual, merged):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(a.shape, c.shape)
                    self.assertTrue(bool(torch.isfinite(c).all()))
                self.assertEqual(len(sparse.records[index]['mass']), 5)

    def test_temporal_actual_budget_and_recent_protection(self):
        for allocation in ('coverage', 'fifo', 'uniform'):
            session, _, events = self.stream(dict(method='temporal', budget_frames=3,
                                                  recent_frames=1, min_patches=2,
                                                  allocation=allocation), count=18)
            self.assertLessEqual(session.memory()['state_bytes'], session.byte_budget)
            self.assertTrue(any(row['allocation_actions'] for row in events))
            self.assertIn(0, session.records)
            self.assertEqual(len(session.records[17]['mass']), 18)
            for record in session.records.values():
                for pair in record['kv']:
                    self.assertEqual(pair[0].shape[2], record['positions'].shape[1])
                    self.assertEqual(pair[0].shape[2], len(record['mass']))
                    for value in pair:
                        self.assertEqual(value.untyped_storage().nbytes(), research.nbytes(value))
            self.assertEqual(session.memory()['state_bytes'], research.nbytes(session.records))

    def test_coverage_compresses_redundant_view_before_old_unique_view(self):
        session = research.ResearchCache(self.model, research.ResearchConfig(
            method='temporal', budget_frames=3, recent_frames=1, min_patches=2))
        with torch.no_grad():
            for index in range(4):
                session.forward(self.images[index], index)
            for index, row in session.records.items():
                row['descriptor'] = torch.eye(8)[min(index, 2)]
            event = session.maintain(3)
        self.assertEqual(event['allocation_actions'][0]['frame'], 2)
        self.assertEqual(len(session.records[0]['mass']), 18)

    def test_prefix_invariance_all_methods(self):
        for config in (dict(method='native'), dict(method='temporal', budget_frames=3,
                       recent_frames=1, min_patches=2), dict(method='refresh', frame_budget=4,
                       refresh_every=2, refresh='selective')):
            short, prefix, short_events = self.stream(config, 6)
            _, complete, events = self.stream(config, 12)
            for a, b in zip(prefix, complete):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(short_events, events[:6])

    def test_refresh_replaces_only_selected_groups_without_growth(self):
        session, _, _ = self.stream(dict(method='refresh', frame_budget=4,
                                         refresh_every=8, refresh_frames=1), 8)
        with torch.no_grad():
            session.forward(self.images[8], 8)
            session.maintain(8)
            before = {key: [tuple(value.clone() for value in pair) for pair in row['kv']]
                      for key, row in session.records.items()}
            size = session.memory()
            session.refresh(8)
        chosen = session.event['refreshed_frames']
        self.assertEqual(len(chosen), 1)
        self.assertNotIn(8, chosen)
        self.assertEqual(size, session.memory())
        for key, row in session.records.items():
            difference = sum(float((a - b).abs().sum()) for pair, prior in zip(row['kv'], before[key])
                             for a, b in zip(pair, prior))
            self.assertGreater(difference, 0) if key in chosen else self.assertEqual(difference, 0)

    def test_oracle_scores_do_not_commit_unselected_candidates(self):
        session, _, _ = self.stream(dict(method='refresh', frame_budget=4,
                                         refresh='oracle', refresh_every=8), 8)
        with torch.no_grad():
            session.forward(self.images[8], 8)
            session.maintain(8)
            before = {key: row['kv'] for key, row in session.records.items()}
            session.refresh(8)
        selected = session.event['refreshed_frames']
        self.assertEqual(len(session.event['refresh_scores']), 3)
        for key in before:
            if key not in selected:
                self.assertIs(before[key], session.records[key]['kv'])

    def test_refresh_none_matches_native_with_auxiliary_cost_reported(self):
        native, reference, _ = self.stream(dict(method='native', frame_budget=4))
        refresh, actual, events = self.stream(dict(method='refresh', frame_budget=4, refresh='none'))
        for a, b in zip(reference, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(native.memory()['aggregator_bytes'], refresh.memory()['aggregator_bytes'])
        self.assertGreater(refresh.memory()['auxiliary_bytes'], native.memory()['auxiliary_bytes'])
        self.assertTrue(all(not row['refreshed_frames'] for row in events))

    def test_random_refresh_reproducible(self):
        config = dict(method='refresh', frame_budget=4, refresh='random', refresh_every=2, seed=19)
        _, first, events = self.stream(config)
        _, second, repeat = self.stream(config)
        self.assertEqual(events, repeat)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_configuration_and_mask_contracts(self):
        with self.assertRaises(AssertionError):
            research.ResearchConfig(budget_frames=2, recent_frames=2)
        session = research.ResearchCache(self.model, research.ResearchConfig(method='context'))
        with self.assertRaises(ValueError):
            session.forward(self.images[0], 0)


if __name__ == '__main__':
    unittest.main()
