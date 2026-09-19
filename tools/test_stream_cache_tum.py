"""TUM manifest contract. Standard library only, so this also runs on the Mac:
`python -m unittest discover -s tools -p 'test_stream_cache_tum.py'`.

It is discovered by the job's `test_*cache*.py` pattern, so a broken TUM clip
stops the allocation before the model is loaded.
"""
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stream_cache_tum import quaternion_to_c2w, nearest_gt  # noqa: E402

SCRIPT = Path(__file__).with_name('stream_cache_tum.py')
PNG = bytes.fromhex(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000'
    '000a49444154789c63000100000500010d0a2db40000000049454e44ae426082')
T0 = 1341847980.722988
GT_HOLE = range(60, 132)


def build_zip(directory, frames=50):
    """A TUM-shaped ZIP with a deliberate 1.2 s GT hole part way through."""
    path = directory / 'rgbd_dataset_freiburg3_long_office_household.zip'
    prefix = path.stem + '/'
    with zipfile.ZipFile(path, 'w') as packed:
        for index in range(frames):
            packed.writestr(f'{prefix}rgb/{T0 + index * .0333:.6f}.png', PNG)
        rows = ['# ground truth trajectory', '# timestamp tx ty tz qx qy qz qw']
        for j in range(120):
            stamp = T0 + j * .0166 + (1.2 if j in GT_HOLE else 0.)
            angle = (j / 120) * math.pi / 2
            rows.append(f'{stamp:.6f} {j * .01:.4f} 0.0 0.0 '
                        f'0.0 0.0 {math.sin(angle / 2):.6f} {math.cos(angle / 2):.6f}')
        packed.writestr(prefix + 'groundtruth.txt', '\n'.join(rows) + '\n')
    return path


def run(zip_path, out, *arguments):
    return subprocess.run([sys.executable, str(SCRIPT), '--zip', str(zip_path),
                           '--out', str(out), *arguments],
                          capture_output=True, text=True)


class RotationTests(unittest.TestCase):
    def assert_rotation(self, matrix):
        rotation = [row[:3] for row in matrix[:3]]
        for i in range(3):
            for j in range(3):
                dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
                self.assertAlmostEqual(dot, 1. if i == j else 0., places=9)
        determinant = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]))
        self.assertAlmostEqual(determinant, 1., places=9)
        self.assertEqual(matrix[3], [0., 0., 0., 1.])

    def test_identity_quaternion_is_identity(self):
        matrix = quaternion_to_c2w(0, 0, 0, 0, 0, 0, 1)
        self.assert_rotation(matrix)
        self.assertEqual([row[:3] for row in matrix[:3]],
                         [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def test_ninety_degrees_about_z_maps_x_to_y_and_keeps_translation(self):
        half = math.pi / 4
        matrix = quaternion_to_c2w(1, 2, 3, 0, 0, math.sin(half), math.cos(half))
        self.assert_rotation(matrix)
        self.assertAlmostEqual(matrix[1][0], 1., places=9)
        self.assertEqual([matrix[i][3] for i in range(3)], [1, 2, 3])

    def test_unnormalized_quaternion_is_normalized_not_rejected(self):
        half = math.pi / 4
        self.assert_rotation(quaternion_to_c2w(
            0, 0, 0, 0, 0, 2 * math.sin(half), 2 * math.cos(half)))

    def test_nearest_gt_picks_the_closer_side_and_clamps_at_both_ends(self):
        times = [1., 2., 3.]
        self.assertEqual(nearest_gt(1.4, times), 0)
        self.assertEqual(nearest_gt(1.6, times), 1)
        self.assertEqual(nearest_gt(-5., times), 0)
        self.assertEqual(nearest_gt(99., times), 2)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.zip = build_zip(self.directory)

    def prepared(self, *arguments):
        out = self.directory / f'out{len(list(self.directory.iterdir()))}'
        result = run(self.zip, out, *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((out / 'manifest.json').read_text()), out

    def test_schema_intrinsics_and_written_pixels(self):
        manifest, out = self.prepared('--count', '32')
        self.assertEqual(len(manifest['frames']), 32)
        self.assertEqual(manifest['sequence'], 'freiburg3_long_office_household')
        self.assertEqual(len(list((out / 'rgb').glob('*.png'))), 32)
        first = manifest['frames'][0]
        # Original resolution: prepare() derives model_intrinsics from these.
        self.assertEqual(first['intrinsics'], [[535.4, 0., 320.1], [0., 539.2, 247.6], [0., 0., 1.]])
        self.assertEqual(first['rgb'], 'rgb/000000.png')
        self.assertIn('gt_c2w', first)
        self.assertIn('gt_timestamp', first)

    def test_a_gt_hole_becomes_null_never_an_extrapolated_pose(self):
        manifest, _ = self.prepared('--count', '32')
        missing = [row for row in manifest['frames'] if row['gt_c2w'] is None]
        self.assertTrue(missing, 'the injected hole should leave at least one null')
        self.assertTrue(all(row['gt_timestamp'] is None for row in missing))
        self.assertEqual(manifest['gt_valid_frames'],
                         sum(1 for row in manifest['frames'] if row['gt_c2w'] is not None))
        for row in manifest['frames']:
            if row['gt_c2w'] is not None:
                self.assertLessEqual(abs(row['gt_timestamp'] - row['timestamp']),
                                     manifest['max_gt_delta_seconds'])

    def test_stride_widens_the_span_and_is_recorded(self):
        consecutive, _ = self.prepared('--count', '10', '--stride', '1')
        strided, _ = self.prepared('--count', '10', '--stride', '4')
        self.assertGreater(strided['span_seconds'], consecutive['span_seconds'])
        self.assertEqual(strided['selection'], {'start': 0, 'count': 10, 'stride': 4})

    def test_requesting_more_frames_than_exist_fails_loudly(self):
        result = run(self.zip, self.directory / 'too_many', '--count', '999')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('only', result.stderr)

    def test_existing_output_directory_is_not_silently_reused(self):
        target = self.directory / 'twice'
        self.assertEqual(run(self.zip, target, '--count', '5').returncode, 0)
        self.assertNotEqual(run(self.zip, target, '--count', '5').returncode, 0)


if __name__ == '__main__':
    unittest.main()
