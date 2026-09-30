"""ATD contracts; run in CAMP, never on the editing Mac."""
import math
import json
import unittest

import torch

from kv_tracker.adaptive_tokens import keep_set, energy, select, merge_groups, sensitivity
from kv_tracker.token_drop import background_keep, forward_kept


class SelectionTests(unittest.TestCase):
    def test_exact_count_and_protected_region(self):
        grid = (17, 29)
        dense = torch.zeros(math.prod(grid), dtype=torch.bool)
        dense[4:39] = True
        for count in (35, 64, 247, len(dense)):
            chosen = keep_set(dense, count, grid)
            self.assertEqual(int(chosen.sum()), count)
            self.assertTrue(chosen[dense].all())
            self.assertTrue(torch.equal(chosen, keep_set(dense, count, grid)))

    def test_bg64_unchanged(self):
        grid = (17, 29)
        mask = torch.zeros(math.prod(grid), dtype=torch.bool)
        mask[100:200] = True
        score = torch.arange(len(mask)).float()
        chosen = select(score, mask, 'object', 'heuristic', grid, 0)
        self.assertTrue(torch.equal(chosen, background_keep(mask[None], 64)[0][0]))

    def test_all_policies_exact(self):
        grid = (11, 13)
        mask = torch.zeros(math.prod(grid), dtype=torch.bool)
        mask[20:80] = True
        score = torch.linspace(0., 1., len(mask))
        for task in ('object', 'scene'):
            for policy in ('uniform', 'random17', 'random29', 'heuristic', 'oracle', 'oracle_spread'):
                chosen = select(score, mask, task, policy, grid, 19)
                self.assertEqual(int(chosen.sum()), 124 if task == 'object' else 72)
                self.assertTrue(torch.equal(chosen, select(score, mask, task, policy, grid, 19)))
                if task == 'object':
                    self.assertTrue(chosen[mask].all())
        self.assertFalse(torch.equal(select(score, mask, 'scene', 'random17', grid, 19),
                                     select(score, mask, 'scene', 'random29', grid, 19)))

    def test_homogeneous_and_textured(self):
        flat = torch.zeros(3, 28, 42)
        self.assertEqual(float(energy(flat).sum()), 0.)
        flat[:, ::2, :] = 1.
        self.assertTrue((energy(flat) > 0).all())

    def test_spread_teacher_matches_learned_allocation(self):
        from kv_tracker.atd_training import scorer_keep
        grid = (19, 25)
        score = torch.arange(math.prod(grid)).float()
        mask = torch.zeros_like(score, dtype=torch.bool)
        spread = select(score, mask, 'scene', 'oracle_spread', grid, 0)
        top = select(score, mask, 'scene', 'oracle', grid, 0)
        self.assertEqual(int(spread.sum()), 238)
        self.assertTrue(spread[-119:].all())
        self.assertTrue(spread[:237].any())
        self.assertFalse(torch.equal(spread, top))
        self.assertTrue(torch.equal(spread, scorer_keep(score, grid, 0)))

    def test_merging_coverage_count_and_positions(self):
        for grid in ((11, 13), (37, 37)):
            p = math.prod(grid)
            mask = torch.zeros(p, dtype=torch.bool)
            mask[20:40] = True
            for task in ('object', 'scene'):
                keep, groups = merge_groups(torch.linspace(0., 1., p), mask, task, grid)
                self.assertTrue((groups >= 0).all())
                self.assertEqual(int(groups.max()) + 1, int(keep.sum()))
                representatives = keep.nonzero().flatten()
                for group in range(len(representatives)):
                    members = (groups == group).nonzero().flatten()
                    self.assertIn(int(representatives[group]), members.tolist())
                    self.assertLessEqual(len(members), 9)
                    self.assertEqual(int((members // grid[1] // 3).unique().numel()), 1)
                    self.assertEqual(int((members % grid[1] // 3).unique().numel()), 1)
                if task == 'scene':
                    self.assertEqual(int(keep.sum()), math.ceil(p / 2))
                else:
                    self.assertTrue(keep[mask].all())
                    self.assertTrue((torch.bincount(groups)[groups[mask]] == 1).all())
        self.assertGreater(int(merge_groups(torch.zeros(1369), torch.zeros(1369, dtype=torch.bool),
                                           'object', (37, 37))[0].sum()), 64)


class DecisionTests(unittest.TestCase):
    def test_headroom_and_all_three_speed_sizes(self):
        from kvt_adaptive_tokens import CLIPS, CONTROLS, decide
        results = {}
        for clip in CLIPS:
            results[clip] = {name: dict(rpe_translation_m=1., translation_p99_m=2.,
                tracking_seconds=1., mean_kept_patches=100., mismatched_budget_frames=0)
                for name in CONTROLS + ('heuristic', 'oracle', 'come')}
            results[clip]['oracle']['rpe_translation_m'] = .94
        timing = [dict(shape=[266, 350], task='scene', resize_dim=308,
                       kind='query', bank=bank, fraction=fraction,
                       median_seconds=1. if fraction == 1. else .79)
                  for bank in (1, 10, 20) for fraction in (1., .5)]
        # A scene's longest edge need not equal its resize setting. Object
        # timings must not enter the scene gate, even with a coincident edge.
        timing.extend(dict(shape=[308, 700], task='object', resize_dim=518,
                           kind='query', bank=bank, fraction=fraction, median_seconds=100.)
                      for bank in (1, 10, 20) for fraction in (1., .5))
        decision = decide(results, timing)
        self.assertTrue(decision['tasks']['object']['eligible_for_training_review'])
        self.assertTrue(decision['tasks']['scene']['eligible_for_training_review'])
        timing[5]['median_seconds'] = .81
        decision = decide(results, timing)
        self.assertFalse(decision['tasks']['scene']['eligible_for_training_review'])
        self.assertTrue(decision['tasks']['object']['eligible_for_training_review'])
        for clip in ('espresso', 'ketchup'):
            results[clip]['oracle']['translation_p99_m'] = 2.01
        self.assertFalse(decide(results, timing)['tasks']['object']['oracle_headroom_pass'])
        scenes = {c: s for c, s in CLIPS.items() if s['task'] == 'scene'}
        decision = decide({c: results[c] for c in scenes}, None, scenes)
        self.assertEqual(list(decision['tasks']), ['scene'])
        self.assertTrue(decision['tasks']['scene']['oracle_headroom_pass'])
        self.assertIsNone(decision['scene_speed_pass'])
        self.assertFalse(decision['tasks']['scene']['eligible_for_training_review'])


class GPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
        assert torch.cuda.is_available(), 'CAMP GPU required'
        cls.model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained('cuda').eval())
        cls.model.requires_grad_(False)

    def test_oracle_repeatability(self):
        # Recorded 26010 office/espresso grids, including the failing object size.
        # Baseline variability is measured, not required: nondeterminism need not
        # manifest on every device/run. The fixed path must be bit-identical.
        torch.manual_seed(17)
        model = self.model
        initial_mode = torch.are_deterministic_algorithms_enabled()
        initial_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        for task, height, width in (('scene', 266, 350), ('object', 434, 616)):
            grid = (height // 14, width // 14)
            patches = math.prod(grid)
            imgs = torch.rand(1, 1, 3, height, width, device='cuda')
            mask = torch.zeros(patches, dtype=torch.bool, device='cuda')
            if task == 'object':
                mask[patches // 3:patches // 2] = True
            for cached in (False, True):
                model.cache = {}
                model.kept_cache = None
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    if cached:
                        forward_kept(model, imgs,
                            torch.ones(1, patches, dtype=torch.bool, device='cuda'),
                            store_cache=True, cam_only=True)
                    before = {i: {k: v.clone() for k, v in layer.items()}
                              for i, layer in model.cache.items()}
                    for deterministic in (False, True):
                        scores = [sensitivity(model, imgs, cached,
                            deterministic_backward=deterministic) for _ in range(3)]
                        choices = [select(score[0], mask, task, 'oracle', grid, 0)
                                   for score in scores]
                        delta = max(float((s - scores[0]).abs().max()) for s in scores[1:])
                        changed = max(int((k != choices[0]).sum()) for k in choices[1:])
                        print('ATD ORACLE REPEAT', json.dumps(dict(task=task,
                            shape=[height, width], cached=cached, deterministic=deterministic,
                            score_max_abs=delta, selection_changes=changed)), flush=True)
                        if deterministic:
                            for score, choice in zip(scores[1:], choices[1:]):
                                self.assertTrue(torch.equal(score, scores[0]))
                                self.assertTrue(torch.equal(choice, choices[0]))
                        self.assertEqual(torch.are_deterministic_algorithms_enabled(), initial_mode)
                        self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), initial_warn)
                    for i, layer in before.items():
                        for key, value in layer.items():
                            self.assertTrue(torch.equal(model.cache[i][key], value))
                model.cache = {}
                model.kept_cache = None
        print('ATD ORACLE REPEATABILITY OK', flush=True)

    def test_rope_strided_gradient(self):
        from kv_tracker.oracle_rope import OracleRoPE, cuRoPE2D
        torch.manual_seed(29)
        tokens = torch.randn(1, 2, 7, 16, device='cuda', requires_grad=True)
        positions = torch.arange(14, device='cuda').reshape(1, 7, 2)
        upstream = cuRoPE2D()
        adapter = OracleRoPE()
        with torch.no_grad():
            expected_output = upstream(tokens.detach().clone(), positions)
        output = adapter(tokens, positions)
        self.assertTrue(torch.equal(output, expected_output))
        gradient = torch.randn(1, 7, 2, 16, device='cuda').transpose(1, 2)
        self.assertFalse(gradient.is_contiguous())
        unchanged = gradient.clone()
        actual, = torch.autograd.grad(output, tokens, gradient)
        with torch.no_grad():
            expected = cuRoPE2D(F0=-1.)(gradient.clone(), positions)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(gradient, unchanged))

    def test_dense_sparse_cache_and_oracle(self):
        torch.manual_seed(17)
        model = self.model
        imgs = torch.rand(1, 2, 3, 56, 70, device='cuda')
        query = torch.rand(1, 1, 3, 56, 70, device='cuda')
        full = torch.ones(2, 20, dtype=torch.bool, device='cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            native = model(imgs, store_cache=True)['camera_poses']
            native_query = model(query, use_cache=True, cam_only=True)['camera_poses']
            model.cache = {}
            dense = forward_kept(model, imgs, full, store_cache=True)['camera_poses']
            dense_query = forward_kept(model, query, full[:1], use_cache=True, cam_only=True)['camera_poses']
            self.assertLess(float((dense - native).abs().max()), 1e-2)
            self.assertLess(float((dense_query - native_query).abs().max()), 1e-2)
            before = {i: {k: t.clone() for k, t in v.items()} for i, v in model.cache.items()}
            score = sensitivity(model, query, True)
            self.assertEqual(tuple(score.shape), (1, 20))
            self.assertTrue(torch.isfinite(score).all() and (score > 0).any())
            for i, layer in before.items():
                for key, value in layer.items():
                    self.assertTrue(torch.equal(model.cache[i][key], value))
            keep = full.clone()
            keep[:, 1::2] = False
            model.cache = {}
            encoder_inputs, positions = [], []
            encoder_hook = model.encoder.blocks[0].register_forward_pre_hook(
                lambda module, args: encoder_inputs.append(args[0].clone()))
            position_hook = model.decoder[0].register_forward_pre_hook(
                lambda module, args, kwargs: positions.append(kwargs['xpos'].clone()), with_kwargs=True)
            forward_kept(model, imgs, keep, store_cache=True, cam_only=True)
            encoder_hook.remove()
            position_hook.remove()
            start = 1 + model.encoder.num_register_tokens
            for frame in range(2):
                prepared = model.encoder.prepare_tokens_with_masks(
                    (imgs[0, frame:frame + 1] - model.image_mean) / model.image_std)
                expected = torch.cat((prepared[:, :start], prepared[:, start:][:, keep[frame]]), 1)
                self.assertTrue(torch.equal(encoder_inputs[frame], expected))
                grid = model.position_getter(1, 4, 5, imgs.device)[0] + 1
                self.assertTrue(torch.equal(positions[frame][0, 5:], grid[keep[frame]]))
            for layer in model.cache.values():
                self.assertEqual(layer['k'].shape[2], 2 * (5 + 10))
            sparse = forward_kept(model, query, keep[:1], use_cache=True, cam_only=True)
            self.assertTrue(torch.isfinite(sparse['camera_poses']).all())
            for layer in model.cache.values():
                layer['k'].zero_()
                layer['v'].zero_()
            empty_context = forward_kept(model, query, keep[:1], use_cache=True, cam_only=True)
            self.assertFalse(torch.equal(sparse['camera_poses'], empty_context['camera_poses']))
            mask = torch.zeros(20, dtype=torch.bool, device='cuda')
            chosen, groups = merge_groups(torch.arange(20, device='cuda').float(), mask, 'scene', (4, 5))
            model.cache = {}
            merged = forward_kept(model, imgs, chosen[None].expand(2, -1),
                                  groups=[groups, groups], store_cache=True)
            self.assertTrue(torch.isfinite(merged['points']).all())
            self.assertEqual(tuple(merged['points'].shape), (1, 2, 56, 70, 3))
        print('ATD GPU CONTRACTS OK', flush=True)


if __name__ == '__main__':
    unittest.main()
