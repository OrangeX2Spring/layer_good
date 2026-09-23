"""Contract tests for the TUM sweep. Run on CAMP, never on the editing Mac."""
import io
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from kvt_tum_run import associate_gt, evaluate, periodic_indices
from kvt_tum_selector import TumSelector, patch_novelty
from kvt_tum_sweep import Sweep, compare_prefix, semantic_configs


class TumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import actual entry points, including their transitive/native deps,
        # before staging data. Importing torch alone does not check this closure.
        for name in ('main', 'kv_tracker.dataloaders.tum',
                     'kv_tracker.eval_tools.evo_utils', 'pi3.models.pi3',
                     'pi3.curope.curope2d', 'sam2.build_sam', 'safetensors.torch',
                     'kvt_tum_viz'):
            print('CHECK IMPORT', name, flush=True)
            module = importlib.import_module(name)
            print('IMPORT OK', name, module.__file__, flush=True)
        from pi3.models.layers.pos_embed import RoPE2D
        assert RoPE2D is not None, 'Pi3 compiled RoPE backend is unavailable'
        print('DEPENDENCY IMPORTS OK', flush=True)

    def test_original_counter_phase(self):
        self.assertEqual(periodic_indices(2600, 50, 20), [0] + list(range(49, 950, 50)))
        self.assertEqual(periodic_indices(2600, 50, 64)[-1], 2599)
        self.assertEqual(len(periodic_indices(2600, 50, 64)), 53)

    def test_fixed_arrival_replay(self):
        selector = TumSelector(dict(policy='fixed', insertion_indices=[2, 5],
                                    frames=7, cap=3), io.StringIO(), io.StringIO())
        selector.model = SimpleNamespace(cache={})
        with patch('torch.cuda.synchronize'):
            for index in range(1, 7):
                chosen = selector.select(dict(idx=index), None, None, None,
                                         list(selector.inserted), original=True)
                self.assertEqual(chosen, index in (2, 5))
        self.assertEqual(selector.inserted, [0, 2, 5])
        self.assertEqual(selector.last_index, 6)

    def test_fixed_rejects_invalid_selection(self):
        for indices in ([0, 2], [2, 2], [5, 2], [2, 7], [2.0, 5], [True, 5], [2]):
            with self.subTest(indices=indices), self.assertRaises(AssertionError):
                TumSelector(dict(policy='fixed', insertion_indices=indices,
                                 frames=7, cap=3), io.StringIO(), io.StringIO())

    def test_gt_gap_rejected(self):
        rgb = np.array([.001, .034, .067, 1., 9.999])
        gt = np.array([0., .033, .066, 10.])
        nearest, valid, difference = associate_gt(rgb, gt, .02)
        np.testing.assert_array_equal(valid, [True, True, True, False, True])
        np.testing.assert_array_equal(nearest, [0, 1, 2, 2, 3])
        self.assertGreater(difference[3], .9)
        self.assertEqual(np.diff(np.flatnonzero(valid)).tolist(), [1, 1, 2])

    def test_duplicate_gt_timestamp(self):
        # freiburg2_large_no_loop repeats one GT timestamp. Association must still
        # pick a nearest pose, whichever side of the pair the search lands on.
        rgb = np.array([.03, .033, .05])
        gt = np.array([0., .033, .033, .066])
        nearest, valid, difference = associate_gt(rgb, gt, .02)
        np.testing.assert_array_equal(valid, [True, True, True])
        np.testing.assert_array_equal(nearest, [1, 1, 3])
        np.testing.assert_allclose(difference, [.003, 0., .016], atol=1e-12)

    def test_evaluation_excludes_gap_and_rpe_bridge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, result = root / 'scene', root / 'result'
            scene.mkdir()
            result.mkdir()
            times = [0., .033, .066, 1., 10., 10.033, 10.066]
            positions = np.array([[0., 0., 0.], [.1, 0., 0.], [.1, .1, 0.],
                                  [999., 999., 999.], [0., 0., .1], [.1, 0., .1], [.1, .1, .1]])
            poses = np.tile(np.eye(4), (7, 1, 1))
            poses[:, :3, 3] = positions
            np.save(result / 'traj.npy', poses)
            indices = [0, 1, 2, 4, 5, 6]
            gt = np.zeros((6, 8))
            gt[:, 0] = np.array(times)[indices]
            gt[:, 1:4] = positions[indices]
            gt[:, 7] = 1
            np.savetxt(scene / 'groundtruth.txt', gt)
            (scene / 'manifest.json').write_text(json.dumps(dict(inputs=[
                dict(timestamp=t) for t in times])))
            metrics = evaluate(scene, result, .02)
            self.assertEqual(metrics['evaluated_frames'], 6)
            self.assertEqual(metrics['rpe_pairs'], 4)
            self.assertLess(metrics['ate_m'], 1e-8)
            self.assertLess(metrics['rpe_translation_m'], 1e-8)

    def test_patch_identity_permutation_and_union(self):
        a = torch.eye(3)
        for score in ('coverage', 'chamfer'):
            self.assertAlmostEqual(patch_novelty(a, [a.flip(0)], score, .95), 0)
        b = torch.tensor([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        c = torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 1.]])
        self.assertAlmostEqual(patch_novelty(a, [b], 'coverage', .95), 2/3, places=6)
        self.assertEqual(patch_novelty(a, [b, c], 'coverage', .95), 0)
        # Coverage can use different frames; Chamfer must choose one entire set.
        self.assertGreater(patch_novelty(a, [b, c], 'chamfer', .95), 0)

    def test_chamfer_against_direct_squared_distance(self):
        torch.manual_seed(0)
        a = F.normalize(torch.randn(257, 8), dim=1)
        b = F.normalize(torch.randn(257, 8), dim=1)
        squared = ((a[:, None] - b[None]) ** 2).sum(-1)
        expected = (squared.min(0).values.mean() + squared.min(1).values.mean()) / 2
        self.assertAlmostEqual(patch_novelty(a, [b], 'chamfer', .95), float(expected), places=5)
        score, details = patch_novelty(a, [b], 'chamfer', .95, return_details=True)
        self.assertAlmostEqual(score, float(expected), places=5)
        self.assertAlmostEqual((float(details['map'].mean()) + details['reverse_mean']) / 2,
                               score, places=6)

    def test_coverage_map_reconstructs_native_score(self):
        a = torch.eye(3)
        b = a[0].repeat(3, 1)
        score, details = patch_novelty(a, [b], 'coverage', .95, return_details=True)
        self.assertAlmostEqual(score, float((details['map'] < .95).float().mean()))

    @patch('torch.cuda.synchronize')
    def test_arrival_features_are_archived(self, synchronize):
        log = io.StringIO()
        config = dict(policy='semantic', score='cosine', threshold=.05, cap=2)
        selector = TumSelector(config, log, io.StringIO())
        selector.model = SimpleNamespace(cache={})
        frame = dict(idx=0, resized_mask_np=np.ones((14, 14), dtype=bool))
        selector.tokens = torch.eye(1, 1024)
        selector.bootstrap(frame)
        for i in range(1, 4):
            selector.tokens = torch.zeros(1, 1024)
            selector.tokens[0, i] = 1
            selector.select(dict(frame, idx=i), None, None, None, selector.inserted.copy(), False)
        with tempfile.TemporaryDirectory() as temporary:
            selector.close(Path(temporary))
            with np.load(Path(temporary) / 'frame_features.npz') as saved:
                np.testing.assert_array_equal(saved['frame_ids'], [0, 1, 2, 3])
                np.testing.assert_array_equal(saved['descriptors'], np.eye(4, 1024))
        self.assertEqual(selector.inserted, [0, 1])

    def test_four_thresholds_per_family(self):
        configs = semantic_configs()
        self.assertEqual(len(configs), 16)
        self.assertEqual(len({c['name'] for c in configs}), 16)
        self.assertEqual({c['layer'] for c in configs}, {'encoder', 'decoder0'})

    def test_short_runs_do_not_request_gt_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / 'archives'
            out.mkdir()
            sweep = Sweep(SimpleNamespace(work=root, out=out, tag='test'))
            sweep.manifests['synthetic'] = dict(frames=256)

            def complete(command, **kwargs):
                result = Path(command[-1]).parent
                (result / 'metrics.json').write_text('{}')
                return SimpleNamespace(returncode=0)

            with patch('kvt_tum_sweep.subprocess.run', side_effect=complete):
                for name, prefix, preflight, expected in (
                        ('prefix', True, False, False), ('memory', False, True, False),
                        ('full', False, False, True)):
                    result, _ = sweep.run_one('synthetic', dict(name=name, policy='original'),
                                              prefix=prefix, preflight=preflight)
                    config = json.loads((result / 'config.json').read_text())
                    self.assertEqual(config['evaluate_trajectory'], expected)

    def test_feature_preflight_prefix_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep = Sweep(SimpleNamespace(work=root, out=root, tag='test'))
            short, full = root / 'short', root / 'full'
            sweep.prefixes[('synthetic', 'semantic')] = short
            with patch.object(sweep, 'run_one', return_value=(full, {})) as run_one, \
                    patch('kvt_tum_sweep.compare_prefix') as compare, \
                    patch('kvt_tum_sweep.archive_directory'):
                settings = dict(name='semantic', policy='semantic')
                sweep.paired('synthetic', settings)
                run_one.assert_called_once_with('synthetic', settings, archive=False)
                compare.assert_called_once_with(short, full)

    @patch('torch.cuda.synchronize')
    def test_periodic_phase_cap_and_future_rejected(self, synchronize):
        config = dict(policy='periodic', interval=3, cap=3)
        log = io.StringIO()
        selector = TumSelector(config, log, io.StringIO())
        selector.model = SimpleNamespace(cache={})
        for i in range(1, 12):
            selector.select(dict(idx=i), None, None, None, selector.inserted.copy(), False)
        self.assertEqual(selector.inserted, [0, 2, 5])
        rows = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertTrue(rows[7]['cap_blocked'])
        with self.assertRaises(AssertionError):
            selector.select(dict(idx=12), None, None, None, [0, 2, 12], False)

    def test_encoder_and_decoder_contract(self):
        for layer in ('encoder', 'decoder0'):
            selector = TumSelector(dict(layer=layer, score='cosine'), io.StringIO(), io.StringIO())
            selector.model = SimpleNamespace(patch_start_idx=5)
            selector.kind = 'query'
            patches = torch.randn(1, 2, 1024)
            output = {'x_norm_patchtokens': patches} if layer == 'encoder' else (
                torch.cat([torch.zeros(1, 5, 1024), patches], dim=1))
            selector.capture(None, None, output)
            torch.testing.assert_close(selector.tokens, patches[0])
            kept = selector.tokens.clone()
            selector.kind = 'rebuild'
            selector.capture(None, None, output)
            torch.testing.assert_close(selector.tokens, kept)

    def test_prefix_gate_rejects_changed_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix, full = Path(temporary) / 'prefix', Path(temporary) / 'full'
            prefix.mkdir()
            full.mkdir()
            for destination, length in ((prefix, 3), (full, 4)):
                np.save(destination / 'traj.npy', np.tile(np.eye(4), (length, 1, 1)))
                rows = [dict(frame=i, candidate=False, selected=False, capped=False,
                             cap_blocked=False, cache_frame_ids=[0], score=.01)
                        for i in range(1, length)]
                (destination / 'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            compare_prefix(prefix, full)
            rows[0]['selected'] = True
            (full / 'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(AssertionError):
                compare_prefix(prefix, full)

    def test_visualization_and_video_decode(self):
        import cv2
        from kvt_tum_viz import feature_views, patch_overlays, visualize_run

        with tempfile.TemporaryDirectory() as temporary:
            inputs, result = Path(temporary) / 'inputs', Path(temporary) / 'result'
            (inputs / 'model_rgb').mkdir(parents=True)
            result.mkdir()
            manifest = dict(inputs=[])
            for i in range(4):
                filename = f'{i}.png'
                pixels = np.full((2, 2, 3), (30 + i*40, 80, 180), np.uint8)
                self.assertTrue(cv2.imwrite(str(inputs / 'model_rgb' / filename), pixels))
                manifest['inputs'].append(dict(file=f'rgb/{filename}', timestamp=i/15))
            (inputs / 'manifest.json').write_text(json.dumps(manifest))
            config = dict(scene='synthetic', name='decoder0_cosine_0.3', policy='semantic',
                          layer='decoder0', score='cosine', threshold=.3, cap=2, interval=50)
            (result / 'config.json').write_text(json.dumps(config))
            (result / 'metrics.json').write_text(json.dumps(dict(frames=4, ate_m=0.,
                evaluated_fraction=.75, cap_reached=True)))
            np.save(result / 'kf_idx.npy', np.array([0, 2]))
            np.savez(result / 'frame_features.npz', descriptors=np.eye(4), frame_ids=np.arange(4))
            positions = np.array([[i/10, 0., 0.] for i in range(4)])
            reference = np.tile(np.eye(4), (3, 1, 1))
            reference[:, :3, 3] = positions[[0, 1, 3]]
            np.savez(result / 'evaluation.npz', rgb_indices=np.array([0, 1, 3]),
                     ate_per_frame_m=np.zeros(3), reference=reference,
                     alignment_rotation=np.eye(3), alignment_translation=np.zeros(3),
                     alignment_scale=1., full_aligned_positions=positions,
                     rpe_pair_start_indices=np.array([0]), rpe_translation_per_pair_m=np.zeros(1),
                     rpe_rotation_per_pair_deg=np.zeros(1))
            cloud = np.array([[[-.5, -.5, 1.], [.5, -.5, 1.]],
                              [[-.5, .5, 1.], [.5, .5, 1.]]])
            np.savez(result / 'final_scene.npz', xyz=np.stack([cloud, cloud]),
                     confidence=np.ones((2, 2, 2)), frame_ids=np.array([0, 2]), threshold=.5)
            rows = [dict(frame=i, score=i*.2, cap_blocked=i==3,
                         cache_bytes_before=1024, feature_bytes_before=512,
                         selector_seconds=.001) for i in range(1, 4)]
            (result / 'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            rows = [dict(frame=i, kind='query', seconds=.01) for i in range(1, 4)]
            (result / 'inference.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            visualize_run(inputs, result, video=True)
            video = json.loads((result / 'viz' / 'video_frames.json').read_text())
            self.assertEqual(video['decoded_frames'], len(video['source_indices']))
            self.assertEqual(video['source_indices'][-1], 3)
            self.assertTrue((result / 'viz' / 'scene_gt_aligned.ply').is_file())
            self.assertTrue((result / 'viz' / 'feature_pca.png').is_file())
            with np.load(result / 'viz' / 'feature_projection.npz') as projection:
                self.assertTrue(np.isnan(projection['similarity'][0]).all())
                self.assertTrue(np.isnan(projection['similarity'][:3, 1]).all())
                self.assertTrue(np.isfinite(projection['similarity'][3]).all())
            prefix = result.parent / 'prefix'
            (prefix / 'viz').mkdir(parents=True)
            np.savez(prefix / 'frame_features.npz', descriptors=np.eye(4), frame_ids=np.arange(4))
            feature_views(inputs, prefix, manifest, dict(config, name=config['name']+'_prefix'),
                          [json.loads(line) for line in (result / 'decisions.jsonl').read_text().splitlines()],
                          np.array([0, 2]), np.arange(4)/15)
            with np.load(prefix / 'viz' / 'feature_projection.npz') as projection:
                self.assertEqual(str(projection['basis_condition']), config['name']+'_prefix')
            maps = np.array([[1., 1., 1., 1.], [1., 1., .8, .8], [.8, .8, .8, .8]], dtype=np.float32)
            np.savez(result / 'patch_novelty.npz', maps=maps, frame_ids=np.arange(1, 4),
                     patch_grid=np.array([2, 2]))
            patch_rows = [dict(frame=i, score=score, candidate=score>.3, selected=i==2,
                               cap_blocked=i==3) for i, score in enumerate((0., .5, 1.), 1)]
            patch_overlays(inputs, result, manifest, dict(config, score='coverage', similarity_floor=.95),
                           patch_rows, np.arange(4)/15)
            self.assertTrue((result / 'viz' / 'patch_novelty.png').is_file())


if __name__ == '__main__':
    unittest.main()
