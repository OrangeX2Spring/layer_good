"""Meaningful tensor/native/cache/gradient contracts; run only on CAMP."""
from copy import deepcopy
import gc
import json
import subprocess
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from kv_tracker.loop_models import (LoopConfig, LoopedPi3, LowRankLinear, VARIANTS,
                                    active_rows, selected_block, se3_update, load_looped, TrainingRoPE)
from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32


class PrimitiveContracts(unittest.TestCase):
    def test_low_rank_reconstruction_and_unmerged_storage(self):
        torch.manual_seed(17)
        base, original = nn.Linear(8, 12), nn.Linear(8, 12)
        low_rank = LowRankLinear(base, original, 8)
        x = torch.randn(3, 8)
        torch.testing.assert_close(low_rank(x), original(x), rtol=1e-5, atol=1e-5)
        self.assertIs(low_rank.base, base)
        low_rank(x).square().mean().backward()
        self.assertTrue(torch.isfinite(low_rank.down.grad).all())
        self.assertTrue(torch.isfinite(low_rank.up.grad).all())

    def test_invalid_zero_short_budget(self):
        with self.assertRaises(AssertionError):
            LoopConfig('elastic', short_loops=0)

    def test_selection_registers_and_spatial_coverage(self):
        rows = active_rows(torch.arange(24).float(), .5)
        self.assertEqual(len(rows), 17)
        self.assertEqual(rows[:5].tolist(), list(range(5)))
        self.assertEqual(len(rows.unique()), len(rows))
        self.assertTrue((rows[1:] > rows[:-1]).all())
        self.assertTrue((rows[5:] < 17).any())
        self.assertEqual(active_rows(torch.arange(24).float(), 1.).tolist(), list(range(29)))

    def test_se3_zero_rotation_gradient_and_valid_rotation(self):
        twist = torch.zeros(6, requires_grad=True)
        pose = se3_update(twist, torch.tensor(2.))
        torch.testing.assert_close(pose, torch.eye(4))
        pose[:3, 3].sum().backward()
        self.assertTrue(torch.isfinite(twist.grad).all())
        torch.testing.assert_close(twist.grad[:3], torch.full((3,), 2.))
        pose = se3_update(torch.tensor([.1, .2, -.1, .3, -.2, .4]), torch.tensor(2.))
        torch.testing.assert_close(pose[:3, :3].T @ pose[:3, :3], torch.eye(3), atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(float(torch.det(pose[:3, :3])), 1., places=5)


class DriverContracts(unittest.TestCase):
    def test_recovery_accumulates_queries_and_gates_on_free_running_development(self):
        from loop_transformer import train_one, SPLITS

        class ModelFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2., device='cuda'))
                self.config = LoopConfig('relaxed', core_pairs=8, full_loops=2,
                                         short_loops=1, rank=64, recovery=True)

            def save(self, path, provenance):
                torch.save(dict(state_dict=self.state_dict(), provenance=provenance), path)

            def storage_report(self):
                return {}

        for development_improves in (True, False):
            with self.subTest(development_improves=development_improves), \
                 tempfile.TemporaryDirectory(dir='/tmp') as temporary:
                work = Path(temporary)
                (work / 'labels').mkdir()
                for split, scene in SPLITS.items():
                    labels = [dict(frame=i, split=split) for i in range(8)]
                    torch.save(dict(scene=scene, manifest_sha256='fixture', checkpoint_weights={},
                                    labels=labels), work / 'labels' / f'{split}.pt')
                model = ModelFixture()
                calls = []

                def loss_fixture(model, images, label, budget, teacher_mix=0., components=None):
                    calls.append((torch.is_grad_enabled(), label['split'], teacher_mix, label['frame']))
                    if label['split'] == 'development' and not development_improves:
                        return (4 - model.weight).square()
                    return model.weight.square()

                class OptimizerFixture:
                    param_groups = [dict(lr=1e-5)]

                    def zero_grad(self, **kwargs):
                        model.weight.grad = None

                    def step(self):
                        with torch.no_grad():
                            model.weight.fill_(1.)

                # Capture the accumulated gradient before clipping.
                def clip_fixture(parameters, maximum):
                    self.assertAlmostEqual(float(model.weight.grad), 2 * float(model.weight))
                    return model.weight.grad.abs()

                with patch('loop_transformer.build_model', return_value=(model, None)), \
                     patch('loop_transformer.load_images', return_value=({}, [torch.zeros(3, 14, 14)])), \
                     patch('loop_transformer.sha256', return_value='fixture'), \
                     patch('loop_transformer.checkpoint_identity', return_value={}), \
                     patch('loop_transformer.calibration', return_value=None), \
                     patch('loop_transformer.query_loss', side_effect=loss_fixture), \
                     patch('loop_transformer.torch.nn.utils.clip_grad_norm_', side_effect=clip_fixture), \
                     patch('loop_transformer.torch.optim.AdamW', return_value=OptimizerFixture()):
                    train_one(work, 'relaxed', steps=8, aggregate_gate=True, recovery=True)
                trained = [row for row in calls if row[0]]
                self.assertEqual(len(trained), 64)
                self.assertEqual([row[3] for row in trained[:8]], list(range(8)))
                self.assertEqual([trained[i * 8][2] for i in range(8)], [1., .5, 0., 0., 0., 0., 0., 0.])
                self.assertTrue(all(row[2] == 0 for row in calls if row[1] == 'development'))
                summary = json.loads((work / 'models/relaxed/summary.json').read_text())
                self.assertEqual(summary['pilot_eligible'], development_improves)
                self.assertEqual(summary['development_loss_decreased'], development_improves)
                self.assertEqual(summary['selected_step'], 8)

    def test_normalized_geometry_gradient_includes_scale(self):
        from loop_transformer import distillation_loss, trajectory_loss
        scale = torch.tensor(2., device='cuda', requires_grad=True)
        pose = torch.eye(4, device='cuda')
        pose = pose + torch.nn.functional.pad(scale.expand(3, 1), (3, 0, 0, 1))
        label = dict(origin=torch.eye(4), pose=torch.eye(4).reshape(1, 1, 4, 4), scale=torch.tensor(1.))
        loss = distillation_loss(dict(camera_poses=pose.reshape(1, 1, 4, 4)), label,
                                 torch.eye(4, device='cuda'), scale)
        gradient, = torch.autograd.grad(loss, scale)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-6, rtol=0)
        target = torch.ones(2, 9, 4, device='cuda')
        features = target.clone().requires_grad_()
        torch.testing.assert_close(trajectory_loss({17: features}, {17: target.reshape(1, 18, 4)}),
                                   torch.zeros((), device='cuda'))

    def test_training_exports_best_measured_state_and_rejects_no_improvement(self):
        from loop_transformer import train_one, SPLITS

        class ModelFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2., device='cuda'))
                self.config = LoopConfig('refiner')

            def save(self, path, provenance):
                torch.save(dict(state_dict=self.state_dict(), provenance=provenance), path)

            def storage_report(self):
                return {}

        for improves, steps in ((True, 16), (False, 16), (True, 10)):
            with self.subTest(improves=improves, steps=steps), tempfile.TemporaryDirectory(dir='/tmp') as temporary:
                work = Path(temporary)
                (work / 'labels').mkdir()
                labels = [dict(frame=i) for i in range(8)]
                for split, scene in SPLITS.items():
                    torch.save(dict(scene=scene, manifest_sha256='fixture', checkpoint_weights={},
                                    labels=labels), work / 'labels' / f'{split}.pt')
                model = ModelFixture()

                class OptimizerFixture:
                    step_count = 0

                    def zero_grad(self, **kwargs):
                        model.weight.grad = None

                    def step(self):
                        self.step_count += 1
                        with torch.no_grad():
                            model.weight.fill_(1. if improves and self.step_count <= 8 else 3.)

                with patch('loop_transformer.build_model', return_value=(model, None)), \
                     patch('loop_transformer.load_images', return_value=({}, [torch.zeros(3, 14, 14)])), \
                     patch('loop_transformer.sha256', return_value='fixture'), \
                     patch('loop_transformer.checkpoint_identity', return_value={}), \
                     patch('loop_transformer.calibration', return_value=None), \
                     patch('loop_transformer.query_loss', side_effect=lambda *args: model.weight.square()), \
                     patch('loop_transformer.torch.optim.AdamW', return_value=OptimizerFixture()):
                    train_one(work, 'refiner', steps=steps, aggregate_gate=True)
                result = work / 'models' / 'refiner'
                summary = json.loads((result / 'summary.json').read_text())
                saved = torch.load(result / 'model.pt', weights_only=True)
                self.assertEqual(summary['selected_step'], 8 if improves else 0)
                self.assertEqual(summary['final_loss'], 1. if improves else 4.)
                self.assertEqual(summary['last_update_loss'], 9.)
                self.assertEqual(summary['pilot_eligible'], improves)
                self.assertEqual(float(saved['state_dict']['weight']), 1. if improves else 2.)
                self.assertEqual(saved['provenance']['selected_step'], summary['selected_step'])

    def test_overnight_preparation_gates_failure_isolation_and_full_order(self):
        from loop_transformer import main
        for failures in (False, True, 'all_blocked'):
            with self.subTest(failures=failures), tempfile.TemporaryDirectory(dir='/tmp') as temporary:
                calls = []

                def invoke_fixture(*arguments):
                    options = list(map(str, arguments))
                    stage = options[options.index('--stage') + 1]
                    variant = options[options.index('--variant') + 1]
                    work = Path(options[options.index('--work') + 1])
                    calls.append((stage, variant, options))
                    if failures is True and ((stage == 'train' and variant == 'elastic') or
                                     (stage == 'full' and variant == 'shared_kv')):
                        raise subprocess.CalledProcessError(7, options)
                    if stage == 'train':
                        initialized = '--initialize-only' in options
                        self.assertEqual(initialized, variant not in ('elastic', 'refiner'))
                        result = work / 'models' / variant
                        result.mkdir(parents=True)
                        (result / 'summary.json').write_text(json.dumps(dict(
                            pilot_eligible=not (failures == 'all_blocked' or (failures and variant == 'adaptive')))))
                    elif stage == 'pilot':
                        (work / 'pilot_gate.json').write_text(json.dumps(dict(
                            quality_pass=not (failures and variant == 'relaxed'))))

                args = SimpleNamespace(arm=None, labels=None, train_one=None,
                                       work=Path(temporary), stage='overnight', variant='all')
                with patch('loop_transformer.invoke', side_effect=invoke_fixture):
                    main(args)
                results = json.loads((Path(temporary) / 'overnight_results.json').read_text())
                self.assertEqual(set(results), set(VARIANTS) - {'combined'})
                self.assertEqual(sum(stage == 'train' for stage, _, _ in calls), 7)
                first_full = next((i for i, (stage, _, _) in enumerate(calls) if stage == 'full'), len(calls))
                self.assertTrue(all(stage == 'full' for stage, _, _ in calls[first_full:]))
                if failures == 'all_blocked':
                    self.assertEqual(first_full, len(calls))
                    self.assertTrue(all(row['status'] == 'blocked_preparation' for row in results.values()))
                elif failures:
                    self.assertEqual(results['elastic']['status'], 'process_failed')
                    self.assertEqual(results['adaptive']['status'], 'blocked_preparation')
                    self.assertEqual(results['relaxed']['status'], 'blocked_pilot')
                    self.assertEqual(results['shared_kv']['status'], 'process_failed')
                    self.assertEqual(results['nested']['status'], 'full_complete')
                    self.assertFalse(any(stage == 'full' and variant in ('elastic', 'adaptive', 'relaxed')
                                         for stage, variant, _ in calls))
                else:
                    self.assertTrue(all(row['status'] == 'full_complete' for row in results.values()))

    def test_pose_distillation_is_independent_of_outer_autocast(self):
        from loop_transformer import distillation_loss
        origin = torch.eye(4, device='cuda')
        origin[:3, 3] = torch.tensor([1.003, 2.007, 3.011], device='cuda')
        pose = torch.eye(4, device='cuda').reshape(1, 1, 4, 4)
        pose[0, 0, :3, 3] = torch.tensor([.013, .017, .019], device='cuda')
        label = dict(origin=origin.cpu(), pose=torch.eye(4).reshape(1, 1, 4, 4),
                     scale=torch.tensor(1.))
        output = dict(camera_poses=pose)
        expected = distillation_loss(output, label, origin, torch.tensor(1., device='cuda'))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = distillation_loss(output, label, origin, torch.tensor(1., device='cuda'))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_refiner_training_replays_causal_history_in_graph(self):
        from loop_transformer import query_loss

        class RefinerFixture:
            config = LoopConfig('refiner')
            map_scale = torch.tensor(1., device='cuda')

            def __init__(self):
                self.calls = []

            def __call__(self, image, **kwargs):
                self.calls.append((int(image.flatten()[0]), torch.is_grad_enabled(), kwargs))
                return {}

        model = RefinerFixture()
        images = [torch.full((3, 14, 14), float(i)) for i in range(6)]
        with patch('loop_transformer.rebuild', return_value=torch.eye(4, device='cuda')), \
             patch('loop_transformer.distillation_loss', return_value=torch.tensor(0., device='cuda')):
            query_loss(model, images, dict(bank=[0, 2], frame=5), 4)
        self.assertEqual([(frame, grad) for frame, grad, _ in model.calls],
                         [(3, True), (4, True), (5, True)])
        self.assertTrue(all(options == dict(cam_only=True, use_cache=True)
                            for _, _, options in model.calls))

    def test_adaptive_calibration_rejects_unsafe_or_uncovered_validation(self):
        from loop_transformer import calibrate_halting

        class ExitFixture:
            config = LoopConfig('adaptive')

            def __init__(self, unsafe, uncovered):
                self.unsafe, self.uncovered = unsafe, uncovered
                self.map_scale = torch.tensor(1., device='cuda')
                self.halt_thresholds = torch.full((3,), -1., device='cuda')

            def eval(self):
                return self

            def __call__(self, image, **kwargs):
                frame = int(image.flatten()[0])
                pose = torch.eye(4, device='cuda')
                if frame >= 4 and self.unsafe:
                    pose[0, 3] = 1.
                signal = 2. if frame >= 4 and self.uncovered else 1.
                self.exit_records = [dict(loop=loop, pose=pose,
                    signals=torch.full((3,), signal, device='cuda')) for loop in (2, 3)]

        images = [torch.full((3, 14, 14), float(i)) for i in range(8)]
        labels = [dict(frame=i, bank=[0, 0], origin=torch.eye(4),
                       pose=torch.eye(4).reshape(1, 1, 4, 4), scale=torch.tensor(1.))
                  for i in range(8)]
        for unsafe, uncovered, expected in ((False, False, True), (True, False, False),
                                             (False, True, False)):
            with self.subTest(unsafe=unsafe, uncovered=uncovered), \
                 tempfile.TemporaryDirectory(dir='/tmp') as temporary, \
                 patch('loop_transformer.rebuild', return_value=torch.eye(4, device='cuda')):
                model = ExitFixture(unsafe, uncovered)
                self.assertEqual(calibrate_halting(model, images, labels, Path(temporary)), expected)
                self.assertEqual(bool((model.halt_thresholds >= 0).all()), expected)


class Pi3Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available(), 'CAMP GPU contracts'
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.native = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())
        cls.native.requires_grad_(False)
        cls.images = torch.rand(1, 3, 3, 56, 84, device='cuda')

    def test_recovery_teacher_boundaries_scale_gradient_and_round_trip(self):
        from loop_transformer import teacher_labels, query_loss
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            with patch('loop_transformer.load_pi3_from_pretrained', return_value=self.native), \
                 patch('loop_transformer.load_images', return_value=(dict(scene='fixture'), list(self.images[0]))), \
                 patch('loop_transformer.QUERY_FRAMES', (2,)), \
                 patch('loop_transformer.sha256', return_value='fixture'), \
                 patch('loop_transformer.checkpoint_identity', return_value={}):
                teacher_labels(root, root / 'labels.pt', recovery=True)
            label = torch.load(root / 'labels.pt', weights_only=True)['labels'][0]
            self.assertEqual(set(label['trajectory']), {1, 17, 33})
            self.assertEqual(set(label['bank_trajectory']), {1, 17, 33})
            config = LoopConfig('relaxed', core_pairs=8, full_loops=2, short_loops=1,
                                rank=64, recovery=True)
            model = LoopedPi3(self.native, config, initialize=False).train()
            self.assertFalse(any(p.requires_grad for p in model.condition.parameters()))
            for mixture in (1., 0.):
                model.zero_grad(set_to_none=True)
                inputs = []
                handle = model.depth_blocks[len(model.core)].register_forward_pre_hook(
                    lambda module, args: inputs.append(args[0].detach().clone()))
                loss = query_loss(model, list(self.images[0]), label, 2, teacher_mix=mixture)
                handle.remove()
                self.assertEqual(len(inputs), 2)
                if mixture == 1.:
                    for actual_input, targets in zip(inputs, (label['bank_trajectory'], label['trajectory'])):
                        torch.testing.assert_close(actual_input, targets[17].to(actual_input).reshape_as(actual_input),
                                                   rtol=0, atol=0)
                self.assertEqual(set(model.loop_features), {17, 33})
                self.assertEqual(len(model.cache), 18)
                self.assertTrue(model.map_scale.requires_grad)
                self.assertIsNone(model.teacher_states)
                scale_gradient, = torch.autograd.grad(model.map_scale, model.core[0].attn.qkv.weight,
                                                      retain_graph=True)
                self.assertTrue(torch.isfinite(scale_gradient).all())
                self.assertGreater(float(scale_gradient.abs().sum()), 0.)
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                model(self.images[:, :2], store_cache=True)
                expected = model(self.images[:, 2:], cam_only=True, use_cache=True)['camera_poses']
            model.save(root / 'model.pt', dict(recovery=True))
            reloaded, provenance = load_looped(self.native, root / 'model.pt')
            self.assertTrue(provenance['recovery'])
            self.assertEqual(reloaded.config, config)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                reloaded(self.images[:, :2], store_cache=True)
                actual = reloaded(self.images[:, 2:], cam_only=True, use_cache=True)['camera_poses']
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            del model, reloaded, loss, scale_gradient, inputs
            gc.collect()
            torch.cuda.empty_cache()

    def test_rope_backward_strided_and_broadcast_gradients(self):
        native = self.native.decoder[0].attn.rope
        rope = TrainingRoPE(native.base, native.F0)
        inverse = type(native)(native.base, -native.F0)
        pos = self.native.position_getter(1, 4, 6, 'cuda') + 1
        for dtype in (torch.float32, torch.bfloat16):
            tokens = torch.randn(1, 16, 24, 64, device='cuda', dtype=dtype, requires_grad=True)
            original = tokens.detach().clone()
            with torch.no_grad():
                expected = native(tokens.detach().clone(), pos)
            actual = rope(tokens, pos)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(tokens, original, rtol=0, atol=0)
            strided = torch.randn(1, 24, 16, 64, device='cuda', dtype=dtype).transpose(1, 2)
            broadcast = torch.ones((), device='cuda', dtype=dtype).expand_as(tokens)
            for gradient in (strided, broadcast):
                self.assertFalse(gradient.is_contiguous())
                original_gradient = gradient.clone()
                with torch.no_grad():
                    expected_gradient = inverse(gradient.clone(), pos) + gradient
                result, = torch.autograd.grad(actual + tokens, tokens, gradient, retain_graph=True)
                torch.testing.assert_close(result, expected_gradient, rtol=0, atol=0)
                torch.testing.assert_close(gradient, original_gradient, rtol=0, atol=0)

    def test_native_wrapper_fidelity(self):
        for variant in ('native', 'native_compact'):
            wrapped = LoopedPi3(self.native, LoopConfig(variant)).eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                a = self.native(self.images[:, :2], store_cache=True)
                b = wrapped(self.images[:, :2], store_cache=True)
                for key in a:
                    torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
                for slot in self.native.cache:
                    for key in ('k', 'v'):
                        torch.testing.assert_close(self.native.cache[slot][key],
                            wrapped.cache[slot][key], atol=0, rtol=0)
                a = self.native(self.images[:, 2:], cam_only=True, use_cache=True)
                b = wrapped(self.images[:, 2:], cam_only=True, use_cache=True)
                torch.testing.assert_close(a['camera_poses'], b['camera_poses'], atol=0, rtol=0)
            del wrapped

    def test_compact_queries_keep_inactive_context_and_match_full_rows(self):
        block = deepcopy(self.native.decoder[0]).float().eval()
        x = torch.randn(1, 29, 1024, device='cuda', requires_grad=True)
        pos = self.native.position_getter(1, 4, 6, x.device) + 1
        pos = torch.cat((pos.new_zeros(1, 5, 2), pos), 1)
        rows = torch.arange(29, device=x.device)
        torch.testing.assert_close(selected_block(block, x, pos, rows), block(x, xpos=pos),
                                   rtol=1e-5, atol=1e-5)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            _, k, v = block(x.detach(), xpos=pos, ret_kv=True)
            history = dict(k=k, v=v)
            torch.testing.assert_close(selected_block(block, x, pos, rows, history),
                block(x, xpos=pos, kv_cache=history), rtol=2e-2, atol=2e-2)
        rows = active_rows(torch.arange(24, device=x.device).float(), .5)
        compact = selected_block(block, x, pos, rows)
        inactive = torch.ones(29, device=x.device, dtype=torch.bool)
        inactive[rows] = False
        torch.testing.assert_close(compact[:, inactive], x[:, inactive], rtol=0, atol=0)
        compact[:, rows].square().mean().backward()
        self.assertGreater(float(x.grad[:, inactive].abs().sum()), 0., 'Inactive keys must remain context')
        self.assertTrue(torch.isfinite(x.grad).all())
        del block, x, compact

    def test_training_gradients_include_bank_and_refiner_history(self):
        for variant in ('elastic', 'refiner'):
            with self.subTest(variant=variant):
                torch.manual_seed(17)
                model = LoopedPi3(self.native, LoopConfig(variant), initialize=False).train()
                parameter = (model.refiner.readout[-1].weight if variant == 'refiner'
                             else model.core[0].attn.qkv.weight)
                gradients = []
                for bank_grad in (True, False):
                    model.zero_grad(set_to_none=True)
                    # One autocast region per call, backward outside, as in the driver:
                    # a shared region reuses weight casts cached under no_grad.
                    with torch.set_grad_enabled(bank_grad and variant != 'refiner'), \
                         torch.autocast('cuda', dtype=torch.bfloat16):
                        model(self.images[:, :2], store_cache=True)
                    if variant == 'refiner':
                        with torch.set_grad_enabled(bank_grad), torch.autocast('cuda', dtype=torch.bfloat16):
                            model(self.images[:, 2:], cam_only=True, use_cache=True)
                        self.assertEqual(model.refiner_state.requires_grad, bank_grad)
                    else:
                        self.assertEqual(model.cache['core_0_0']['k'].requires_grad, bank_grad)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(self.images[:, 2:], cam_only=True, use_cache=True)
                    output['camera_poses'][0, 0, :3, 3].sum().backward()
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    gradients.append(parameter.grad.clone())
                self.assertFalse(torch.equal(*gradients), 'Bank/history must contribute gradient')
                del model, parameter, gradients, output
                gc.collect()
                torch.cuda.empty_cache()

    def test_every_variant_cache_readonly_gradients_and_reload(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                # Initialization SVD is covered separately; this checks live wiring.
                model = LoopedPi3(self.native, LoopConfig(variant), initialize=False).train()
                model.record_details = True
                if variant != 'refiner':
                    self.assertIsNot(model.camera_decoder, self.native.camera_decoder)
                    self.assertIs(next(model.camera_decoder.parameters()),
                                  next(self.native.camera_decoder.parameters()))
                    self.assertFalse(any(isinstance(m, TrainingRoPE) for m in self.native.modules()))
                if variant == 'adaptive':
                    model.geometry_calibration = (torch.tensor([[50., 0., 42.], [0., 50., 28.],
                        [0., 0., 1.]], device='cuda'), None)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    rebuilt = model(self.images[:, :2], store_cache=True)
                self.assertEqual(rebuilt['camera_poses'].shape, (1, 2, 4, 4))
                expected_slots = 10 if variant in ('shared_kv', 'combined', 'nested') else 18
                self.assertEqual(len(model.cache), expected_slots)
                cache = {slot: {key: value.clone() for key, value in row.items()}
                         for slot, row in model.cache.items()}
                pointers = {slot: {key: value.data_ptr() for key, value in row.items()}
                            for slot, row in model.cache.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(self.images[:, 2:], cam_only=variant == 'refiner', use_cache=True)
                    self.assertTrue(all(torch.isfinite(t).all() for t in output.values()))
                    loss = output['camera_poses'][0, 0, :3, 3].sum()
                    if variant != 'refiner':
                        loss = loss + model.last_features.float().square().mean()
                        loss = loss + output['local_points'].square().mean() / model.map_scale.square()
                    for features in model.exit_features.values():
                        loss = loss + features.float().square().mean()
                    loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                for slot, row in cache.items():
                    for key, value in row.items():
                        self.assertEqual(model.cache[slot][key].data_ptr(), pointers[slot][key])
                        torch.testing.assert_close(value, model.cache[slot][key], rtol=0, atol=0)
                if variant == 'nested':
                    self.assertEqual(model.last_execution['global_blocks'], 8)
                if variant == 'token':
                    self.assertEqual(len(model.last_execution['active_rows']), 3)
                    self.assertEqual(len(model.last_execution['active_rows'][0]), 17)
                if variant == 'adaptive':
                    self.assertEqual(set(model.exit_features), {2, 3})
                    self.assertTrue(all(features.requires_grad for features in model.exit_features.values()))
                    model.halt_thresholds.fill_(0.)
                model.zero_grad(set_to_none=True)
                model.last_features = model.routing_features = model.routing_scores = None
                model.exit_features = {}
                model.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    model(self.images[:, :2], store_cache=True)
                    expected = model(self.images[:, 2:], cam_only=True, use_cache=True)['camera_poses'].clone()
                with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
                    path = Path(temporary) / 'model.pt'
                    model.save(path, dict(contract_fixture=True))
                    reloaded, provenance = load_looped(self.native, path)
                    self.assertTrue(provenance['contract_fixture'])
                    reloaded.geometry_calibration = model.geometry_calibration
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        reloaded(self.images[:, :2], store_cache=True)
                        actual = reloaded(self.images[:, 2:], cam_only=True, use_cache=True)['camera_poses']
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    del reloaded, actual
                del model, cache, rebuilt, output, expected, loss, grads
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
