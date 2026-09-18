"""Contract tests for the TUM sweep. Run on CAMP, never on the editing Mac."""
import io
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
from kvt_tum_sweep import compare_prefix, semantic_configs


class TumTests(unittest.TestCase):
    def test_original_counter_phase(self):
        self.assertEqual(periodic_indices(2600, 50, 20), [0] + list(range(49, 950, 50)))
        self.assertEqual(periodic_indices(2600, 50, 64)[-1], 2599)
        self.assertEqual(len(periodic_indices(2600, 50, 64)), 53)

    def test_gt_gap_rejected(self):
        rgb = np.array([.001, .034, .067, 1., 9.999])
        gt = np.array([0., .033, .066, 10.])
        nearest, valid, difference = associate_gt(rgb, gt, .02)
        np.testing.assert_array_equal(valid, [True, True, True, False, True])
        np.testing.assert_array_equal(nearest, [0, 1, 2, 2, 3])
        self.assertGreater(difference[3], .9)
        self.assertEqual(np.diff(np.flatnonzero(valid)).tolist(), [1, 1, 2])

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

    def test_four_thresholds_per_family(self):
        configs = semantic_configs()
        self.assertEqual(len(configs), 16)
        self.assertEqual(len({c['name'] for c in configs}), 16)
        self.assertEqual({c['layer'] for c in configs}, {'encoder', 'decoder0'})

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
        from kvt_tum_viz import visualize_run

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
                          threshold=.3, cap=2, interval=50)
            (result / 'config.json').write_text(json.dumps(config))
            (result / 'metrics.json').write_text(json.dumps(dict(frames=4, ate_m=0.,
                evaluated_fraction=.75, cap_reached=True)))
            np.save(result / 'kf_idx.npy', np.array([0, 2]))
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


if __name__ == '__main__':
    unittest.main()
