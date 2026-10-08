"""Meaningful tensor/native/cache/gradient contracts; run only on CAMP."""
from copy import deepcopy
import gc
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

    def test_refiner_training_replays_only_causal_history(self):
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
                         [(3, False), (4, False), (5, True)])
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
