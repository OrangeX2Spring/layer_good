"""CAMP routing and RoPE stream contracts; model graph checks run per fresh case."""
import hashlib
import json
import os
from pathlib import Path
import time
import unittest
import torch
from kv_tracker.scene_routing import project_cache


@unittest.skipUnless('KVT_GRAPH_ROPE_BUILD' in os.environ, 'graph pilot only')
class GraphContracts(unittest.TestCase):
    def test_rope_stream_capture(self):
        from kv_tracker.graph_rope import GraphRoPE, load_graph_rope, cuRoPE2D
        build = Path(os.environ['KVT_GRAPH_ROPE_BUILD'])
        started = time.perf_counter()
        backend = load_graph_rope(build)
        build_load_seconds = time.perf_counter() - started
        native, corrected = cuRoPE2D(), GraphRoPE(backend)
        rows = []
        torch.manual_seed(23)
        with torch.no_grad():
            for dtype in (torch.float32, torch.bfloat16):
                for count in (480, 243):
                    inputs = [torch.randn(1, 16, count, 64, device='cuda', dtype=dtype)
                              for _ in range(2)]
                    positions = [torch.randint(0, 26, (1, count, 2), device='cuda')
                                 for _ in range(2)]
                    expected = [native(x, p) for x, p in zip(inputs, positions)]
                    torch.cuda.synchronize()
                    errors = dict(corrected_eager_max_abs=max(
                        float((corrected(x, p) - target).abs().max())
                        for x, p, target in zip(inputs, positions, expected)))
                    for name, rope in (('native', native), ('corrected', corrected)):
                        x_buffer, p_buffer = inputs[0].clone(), positions[0].clone()
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                eager = rope(x_buffer, p_buffer)
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        errors[f'{name}_side_stream_max_abs'] = float((eager - expected[0]).abs().max())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            output = rope(x_buffer, p_buffer)
                        torch.cuda.synchronize()
                        differences = []
                        for x, p, target in zip(inputs, positions, expected):
                            x_buffer.copy_(x)
                            p_buffer.copy_(p)
                            graph.replay()
                            torch.cuda.synchronize()
                            differences.append(float((output - target).abs().max()))
                        errors[f'{name}_replay_max_abs'] = max(differences)
                    row = dict(dtype=str(dtype), tokens=count, **errors)
                    rows.append(row)
                    (build / 'probe.json').write_text(json.dumps(rows, indent=2) + '\n')
                    print('SCENE ROPE STREAM PROBE', json.dumps(row), flush=True)
                    self.assertEqual(errors['corrected_eager_max_abs'], 0.)
                    self.assertEqual(errors['corrected_side_stream_max_abs'], 0.)
                    self.assertEqual(errors['corrected_replay_max_abs'], 0.)
        (build / 'backend.json').write_text(json.dumps(dict(
            path=backend.__file__, sha256=hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(),
            torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
            build_load_seconds=build_load_seconds,
            arch=os.environ['TORCH_CUDA_ARCH_LIST'], cuda_flags=['-O3', '--use_fast_math']), indent=2) + '\n')
        print('SCENE ROPE GRAPH CONTRACTS OK', flush=True)


class RoutingContracts(unittest.TestCase):
    def test_freiburg2_distortion_matches_opencv(self):
        import cv2
        import numpy as np
        points = np.array([[.55, .3, 1.], [1.1, .6, 2.], [-.55, -.3, 1.],
                           [.15, -.25, 1.], [-.32, .35, 1.], [.4, -.2, 1.]], np.float32)
        h, w = 266, 350
        k = np.array([[520.9*w/640, 0, (325.1+.5)*w/640-.5],
                      [0, 521.*h/480, (249.7+.5)*h/480-.5], [0, 0, 1]], np.float32)
        distortion = np.array([.2312, -.7849, -.0033, -.0001, .9172], np.float32)
        confidence = torch.arange(1, 7, device='cuda', dtype=torch.float32) / 10
        tensors = [torch.as_tensor(a, device='cuda') for a in (points, k, distortion)]
        pose = torch.eye(4, device='cuda')
        maps = project_cache(tensors[0], confidence, pose, tensors[1], h, w, tensors[2])
        uv = cv2.projectPoints(points, np.zeros(3), np.zeros(3), k, distortion)[0][:, 0]
        cells = np.floor((uv + .5) / 14).astype(int)
        cell = cells[:, 1] * (w // 14) + cells[:, 0]
        self.assertTrue(maps['valid_samples'].all())
        np.testing.assert_array_equal(maps['support'].cpu().numpy(),
                                      np.bincount(cell, minlength=h*w//196))
        for c in np.unique(cell):
            ids = np.flatnonzero(cell == c)
            front = ids[np.argmin(points[ids, 2])]
            self.assertEqual(int(maps['source_index'][c]), front)
            self.assertEqual(float(maps['front_depth'][c]), float(points[front, 2]))
            self.assertEqual(maps['confidence'][c], confidence[front])
        self.assertEqual(float(maps['secondary_gap'][cell[0]]), 1.)
        plain = project_cache(tensors[0], confidence, pose, tensors[1], h, w)
        self.assertFalse(torch.equal(maps['support'], plain['support']))
        zero = project_cache(tensors[0], confidence, pose, tensors[1], h, w,
                             torch.zeros(5, device='cuda'))
        for key in plain:
            self.assertTrue(torch.equal(plain[key], zero[key]), key)

    def test_projection_visibility_and_gauge(self):
        assert torch.cuda.is_available(), 'CAMP GPU required'
        points = torch.tensor([[7., 7., 1.], [14., 14., 2.], [-7., 0., 1.],
                               [0., 0., -1.]], device='cuda')
        conf = torch.tensor([.9, .7, .5, .4], device='cuda')
        pose = torch.eye(4, device='cuda')
        k = torch.eye(3, device='cuda')
        maps = project_cache(points, conf, pose, k, 28, 28)
        self.assertEqual(maps['support'].tolist(), [2, 0, 0, 0])
        self.assertEqual(maps['front_depth'].tolist(), [1., 0., 0., 0.])
        self.assertEqual(maps['secondary_gap'].tolist(), [1., 0., 0., 0.])
        self.assertEqual(maps['source_index'].tolist(), [0, -1, -1, -1])
        self.assertEqual(maps['valid_samples'].tolist(), [True, True, False, False])
        self.assertEqual(maps['confidence'][0], conf[0])
        pose[:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device='cuda')
        pose[:3, 3] = torch.tensor([2., 3., 4.], device='cuda')
        moved = points @ pose[:3, :3].T + pose[:3, 3]
        for _ in range(3):
            repeat = project_cache(moved, conf, pose, k, 28, 28)
            for key in maps:
                self.assertTrue(torch.equal(maps[key], repeat[key]), key)

    def test_empty_support_and_tie(self):
        points = torch.tensor([[7., 7., 1.], [7., 7., 1.]], device='cuda')
        confidence = torch.tensor([.6, .9], device='cuda')
        pose, k = torch.eye(4, device='cuda'), torch.eye(3, device='cuda')
        maps = project_cache(points, confidence, pose, k, 28, 28)
        self.assertEqual(maps['source_index'][0], 0)
        self.assertEqual(maps['confidence'][0], confidence[0])
        pose[2, 3] = 3.
        maps = project_cache(points, confidence, pose, k, 28, 28)
        self.assertEqual(maps['support'].sum(), 0)
        for key in ('front_depth', 'confidence', 'secondary_gap'):
            self.assertEqual(maps[key].sum(), 0)


if __name__ == '__main__':
    unittest.main()
