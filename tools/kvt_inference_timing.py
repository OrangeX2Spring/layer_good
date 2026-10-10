"""Optional synchronized model-forward timing for CAMP replay comparisons.

Separate dense and pose-only queries; neither is whole-pipeline throughput.
Hooks add synchronization overhead. Report profiled runs as such.
"""
import time

import torch


def benchmark_cached_heads(model, rgb):
    """Same final frame/cache for both heads; separate from trajectory timing."""
    image = torch.tensor(rgb.copy(), device=next(model.parameters()).device,
                         dtype=torch.float32).permute(2, 0, 1)[None, None] / 255.
    rows, poses = [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for pose_only in (True, False):
            for _ in range(5):
                output = model(image, cam_only=pose_only, use_cache=True, store_cache=False)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            start.record()
            for _ in range(20):
                output = model(image, cam_only=pose_only, use_cache=True, store_cache=False)
            end.record()
            end.synchronize()
            rows.append(dict(heads='pose' if pose_only else 'dense', iterations=20,
                             gpu_seconds=start.elapsed_time(end) / 1000,
                             wall_seconds=time.perf_counter() - started))
            poses.append(output['camera_poses'].detach().clone())
    torch.testing.assert_close(poses[0], poses[1], rtol=1e-4, atol=1e-4)
    return dict(scope='Offline repeated final frame against final frozen cache; five warmups '
                'then twenty forwards per head. No mapping, input transfer, SAM or postprocessing. '
                'Not online sequence FPS; pose/dense pose outputs must agree.',
                shape=list(image.shape), calls=rows, gpu=torch.cuda.get_device_name(),
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                autocast_dtype='bfloat16', matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cache_tokens_by_layer={str(k): int(v['k'].shape[2]) for k, v in model.cache.items()})


class InferenceTiming:
    def __init__(self, model):
        self.rows = []
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True),
                        model.register_forward_hook(self.after, with_kwargs=True)]

    def before(self, model, args, kwargs):
        torch.cuda.synchronize()
        self.started = time.perf_counter()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def after(self, model, args, kwargs, output):
        self.end.record()
        self.end.synchronize()
        images = args[0]
        self.rows.append(dict(
            kind='mapping' if kwargs.get('store_cache', False) else 'query',
            heads='pose' if kwargs.get('cam_only', False) else 'dense',
            images=int(images.shape[1]), height=int(images.shape[-2]),
            width=int(images.shape[-1]),
            gpu_seconds=self.start.elapsed_time(self.end) / 1000,
            wall_seconds=time.perf_counter() - self.started))

    def finish(self):
        for handle in self.handles:
            handle.remove()
        return dict(scope='Synchronized model.forward only; excludes input conversion, '
                    'postprocessing, detector, I/O and serialization. Includes first calls; '
                    'hooks add synchronization overhead. Dense/pose heads are not equivalent work.',
                    calls=self.rows)
