"""Shared label/scorer contracts; execute on CAMP, never on the editing Mac."""
import unittest

import torch

from kv_tracker.adaptive_tokens import select
from kv_tracker.atd_training import PatchScorer, patch_features, ranking_loss, scorer_keep


class TrainingContracts(unittest.TestCase):
    def test_unknown_and_gauge_invariant_features(self):
        rgb = torch.zeros(3, 28, 42)
        maps = dict(support=torch.tensor([1, 2, 0, 1, 0, 1]),
            front_depth=torch.tensor([2., 4., 0., 6., 0., 8.]),
            confidence=torch.tensor([.8, .7, 0., .9, 0., .6]), secondary_gap=torch.zeros(6))
        a = patch_features(rgb)
        b = patch_features(rgb, maps)
        self.assertEqual(tuple(a.shape), (7, 2, 3))
        self.assertEqual(tuple(b.shape), (12, 2, 3))
        scaled = dict(maps, front_depth=maps['front_depth'] * 17.)
        torch.testing.assert_close(b, patch_features(rgb, scaled))
        self.assertTrue((b[7:, 0, 2] == 0).all())
        empty = {key: torch.zeros_like(value) for key, value in maps.items()}
        self.assertTrue((patch_features(rgb, empty)[7:] == 0).all())

    def test_ranking_direction_and_frame_isolation(self):
        target = torch.tensor([[0., 1., 2.], [2., 1., 0.]])
        self.assertLess(float(ranking_loss(target, target)),
                        float(ranking_loss(-target, target)))
        torch.testing.assert_close(ranking_loss(target, target),
                                   ranking_loss(target, target + torch.tensor([[100.], [-100.]])))
        predicted = torch.zeros_like(target, requires_grad=True)
        ranking_loss(predicted, target).backward()
        self.assertLess(float(predicted.grad[0, 2]), 0.)
        self.assertGreater(float(predicted.grad[0, 0]), 0.)

    def test_teacher_and_scorer_share_spread_rule(self):
        grid = (19, 25)
        score = torch.arange(475).float()
        chosen = scorer_keep(score, grid, 7)
        reference = select(score, torch.zeros(475, dtype=torch.bool), 'scene', 'heuristic', grid, 7)
        self.assertTrue(torch.equal(chosen, reference))
        self.assertEqual(int(chosen.sum()), 238)
        self.assertTrue(chosen[-119:].all())
        self.assertTrue(chosen[:200].any())
        self.assertFalse(torch.equal(chosen, select(score, torch.zeros_like(chosen),
                                                   'scene', 'oracle', grid, 7)))
        for candidate, channels in (('A', 7), ('B', 12)):
            model = PatchScorer(candidate)
            self.assertLess(sum(p.numel() for p in model.parameters()), 1000000)
            self.assertEqual(tuple(model(torch.zeros(1, channels, *grid)).shape), (1, 475))


if __name__ == '__main__':
    unittest.main()
