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
