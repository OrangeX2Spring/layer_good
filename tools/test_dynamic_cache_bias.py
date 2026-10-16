"""CAMP tests for motion-mask geometry and sequence-wide pose evaluation."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from dynamic_cache_bias import metrics, prepare, write_json


class DynamicInputContracts(unittest.TestCase):
    def test_mask_transform_preserves_static_dynamic_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8)).save(root / 'rgb.png')
            labels = np.zeros((28, 28), dtype=np.uint8)
            labels[:, 14:] = 255
            labels[14:, :14] = 128
            Image.fromarray(labels).save(root / 'mask.png')
            frames = []
            for index in range(16):
                pose = np.eye(4)
                pose[:3, 3] = [index * .01, index ** 2 * .001, 0]
                frames.append(dict(rgb='rgb.png', motion_mask='mask.png', timestamp=index * .04,
                                   gt_timestamp=index * .04, gt_c2w=pose.tolist()))
            manifest = dict(sequence='contract', role='primary_dynamic', pose_units='metres',
                            mask_source=dict(kind='gt_motion', description='synthetic contract',
                                             reference='test fixture, not benchmark evidence'), frames=frames)
            write_json(root / 'manifest.json', manifest)
            prepare(root / 'manifest.json', root / 'inputs')
            coverage = np.load(root / 'inputs/coverage.npy').reshape(16, 22, 22)
            np.testing.assert_array_equal(coverage[:, :11, :11], 0)
            np.testing.assert_array_equal(coverage[:, :, 11:], 1)
            np.testing.assert_array_equal(coverage[:, 11:, :11], -1)
            manifest['mask_source']['kind'] = 'semantic'
            write_json(root / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'not semantic'):
                prepare(root / 'manifest.json', root / 'invalid')

    def test_global_sim3_and_missing_gt_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inputs').mkdir()
            gt = np.tile(np.eye(4), (8, 1, 1))
            t = np.arange(8) * .1
            gt[:, :3, 3] = np.stack((t, t ** 2, t ** 3), axis=1)
            frames = [dict(timestamp=i * .04, gt_c2w=pose.tolist() if i != 3 else None)
                      for i, pose in enumerate(gt)]
            write_json(root / 'inputs/manifest.json', dict(frames=frames))
            arms = ['full', 'native', 'oracle', 'random0', 'random1', 'random2']
            estimate = gt.copy()
            estimate[:, :3, 3] = (gt[:, :3, 3] - [1, 2, 3]) / 2
            for arm in arms:
                (root / arm).mkdir()
                np.save(root / arm / 'c2w.npy', estimate)
            metrics(root, arms)
            summary = json.loads((root / 'summary.json').read_text())
            for row in summary['arms'].values():
                self.assertLess(row['ate_m'], 1e-12)
                self.assertLess(row['rpe_translation_rmse_m'], 1e-12)
                self.assertEqual(row['rpe_pairs'], 5)
                self.assertAlmostEqual(row['fitted_scale'], 2)


if __name__ == '__main__':
    unittest.main()
