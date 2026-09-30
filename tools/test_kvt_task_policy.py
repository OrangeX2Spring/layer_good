"""Task-policy mathematical/state contracts; execute inside the CAMP image only."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kvt_task_policy import trajectory_cost
from kv_tracker.atd_training import PatchScorer, scorer_keep
from kv_tracker.task_token_policy import ordered_log_probability, sample_keep, TaskTokenPolicy


class TaskPolicyContracts(unittest.TestCase):
    def test_exact_distribution_and_score_function_gradient(self):
        logits = torch.tensor([-.3, .2, .7, -.1], requires_grad=True)
        orders = [torch.tensor(order) for order in itertools.permutations(range(4), 2)]
        logp = torch.stack([ordered_log_probability(logits, order) for order in orders])
        probability = logp.exp()
        torch.testing.assert_close(probability.sum(), torch.tensor(1.))
        # Independent enumeration of sequential categorical sampling.
        expected = []
        costs = []
        for first, second in itertools.permutations(range(4), 2):
            remaining = [i for i in range(4) if i != first]
            expected.append(logits.softmax(0)[first] * logits[remaining].softmax(0)[remaining.index(second)])
            costs.append(float((first + second) ** 2))
        torch.testing.assert_close(probability, torch.stack(expected))
        costs = torch.tensor(costs)
        exact, = torch.autograd.grad((probability * costs).sum(), logits, retain_graph=True)
        estimate, = torch.autograd.grad((probability.detach() * (costs - 3.) * logp).sum(), logits)
        torch.testing.assert_close(exact, estimate, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(exact.norm()), 0.)

    def test_sampler_allocation_and_deployment(self):
        logits = torch.arange(475).float().div(475).requires_grad_()
        generator = torch.Generator().manual_seed(17)
        keep, order, logp = sample_keep(logits, (19, 25), generator)
        self.assertEqual(len(order), 119)
        self.assertEqual(int(keep.sum()), 238)
        self.assertTrue(keep[order].all())
        self.assertEqual(len(order.unique()), 119)
        logp.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        repeated, repeated_order, _ = sample_keep(logits, (19, 25), torch.Generator().manual_seed(17))
        self.assertTrue(torch.equal(repeated, keep) and torch.equal(repeated_order, order))
        deployed = scorer_keep(logits, (19, 25), 0)
        self.assertTrue(deployed[-119:].all())
        self.assertEqual(int(deployed.sum()), 238)

    def test_cost_is_gauge_invariant_and_penalizes_pose_error(self):
        reference = np.tile(np.eye(4), (12, 1, 1))
        x = np.linspace(0., 1., len(reference))
        reference[:, :3, 3] = np.stack((x, x * x, .1 * x), 1)
        times = np.arange(len(reference)) * .03
        estimate = reference.copy()
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        estimate[:, :3, :3] = rotation @ reference[:, :3, :3]
        estimate[:, :3, 3] = 7 * (reference[:, :3, 3] @ rotation.T) + [2., 3., 4.]
        cost = trajectory_cost(estimate, reference, 2., times)
        self.assertLess(cost['cost'], 1e-20)
        estimate[6, :3, 3] += [0., .5, 0.]
        damaged = trajectory_cost(estimate, reference, 2., times)
        self.assertGreater(damaged['cost'], cost['cost'] + 1e-8)
        self.assertGreater(damaged['tail_pair_cost'], damaged['mean_pair_cost'])
        collapsed = np.tile(np.eye(4), reference.shape[:1] + (1, 1))
        self.assertTrue(trajectory_cost(collapsed, reference, 2., times)['collapsed_motion'])

    def test_state_reuse_sparse_points_and_actor_only_gradient(self):
        # Explicit fake geometry isolates policy contracts from model accuracy.
        model = torch.nn.Linear(1, 1)
        model.cache = {}
        model.kept_cache = None
        scorer = PatchScorer('B')
        intrinsics = torch.tensor([[14., 0., 20.5], [0., 14., 13.5], [0., 0., 1.]])
        actor = TaskTokenPolicy('actor', intrinsics, scorer=scorer, sample=True)
        rgb = torch.rand(1, 2, 3, 28, 42)
        masks = torch.ones(2, 28, 42, dtype=torch.bool)
        seen = []

        def geometry(model, imgs, keep, cam_only, store_cache, use_cache):
            seen.append(keep.clone())
            n, h, w = imgs.shape[1], imgs.shape[-2], imgs.shape[-1]
            pose = torch.eye(4).repeat(1, n, 1, 1)
            pose[..., :3, 3] = torch.tensor([10., 20., 30.])
            coverage = keep.reshape(n, h // 14, 1, w // 14, 1).expand(n, h // 14, 14, w // 14, 14)
            coverage = coverage.reshape(1, n, h, w, 1)
            local = torch.zeros(1, n, h, w, 3)
            local[..., 2] = 2.
            local *= coverage
            points = (local + pose[..., :3, 3][:, :, None, None]) * coverage
            confidence = torch.where(coverage, 0., -1e4)
            if store_cache:
                count = int(keep.sum()) + n * 5
                model.cache = {1: dict(k=torch.zeros(1, 1, count, 1), v=torch.zeros(1, 1, count, 1))}
                model.kept_cache = dict(labels=torch.zeros(count))
            return dict(points=points, local_points=local, conf=confidence, camera_poses=pose)

        with patch('kv_tracker.task_token_policy.forward_kept', geometry), torch.no_grad():
            first = actor.forward(model, rgb, masks, [0, 0], False, True, False)
            self.assertEqual(len(actor.log_probabilities), 1)
            self.assertTrue(torch.equal(seen[0][0], seen[0][1]))
            self.assertEqual(len(actor.points), 6)
            self.assertTrue((actor.points[:, 2] == 32.).all())
            first['camera_poses'].zero_()
            torch.testing.assert_close(actor.pose[:3, 3], torch.tensor([10., 20., 30.]))
            actor.forward(model, rgb[:, :1], masks[:1], [1], True, False, True)
            prior = actor.saved[1].clone()
            actor.forward(model, rgb, masks, [0, 1], False, True, False)
            self.assertTrue(torch.equal(actor.saved[1], prior))
            self.assertEqual(len(actor.log_probabilities), 2)
            self.assertEqual(actor.events[1]['map_cache_ids'], [0, 0])
            self.assertTrue((actor.inputs[0][7:] == 0).all())
            self.assertFalse(actor.inputs[1].requires_grad)
        torch.stack(actor.log_probabilities).sum().backward()
        self.assertTrue(any(p.grad is not None for p in scorer.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
