"""ARCTIC instrumentation for the two correspondence-cache pilot variants."""
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch

from kv_tracker.correspondence_cache import CorrespondenceCache
from kvt_online_selector import OnlineSelector


class CorrespondenceSelector(OnlineSelector):
    def __init__(self, config, log, result):
        super().__init__(config, log)
        assert config['policy'] == 'interval'
        self.cache_policy = CorrespondenceCache(config['cache_policy'],
            budget=config['max_keyframes'], interval=config['interval'])
        self.result = result
        self.inference_log = (result / 'inference.jsonl').open('w')
        self.initialized = False

    def attach(self, model):
        from pi3.models.layers.attention import FlashAttentionRope
        assert hashlib.sha256(Path(inspect.getfile(type(model))).read_bytes()).hexdigest() == (
            'cbcf68b3c05baab7680f6e24afda42501dc0ba799e85ca18668ba3e8a5812979')
        assert hashlib.sha256(Path(inspect.getfile(FlashAttentionRope)).read_bytes()).hexdigest() == (
            'af0c54181f341bcd047dcd591bb1ff0919acf71c1fb77307d2996867e54533f6')
        super().attach(model)
        self.hooks = [model.register_forward_pre_hook(self.before_inference, with_kwargs=True),
                      model.register_forward_hook(self.after_inference, with_kwargs=True)]

    def before_inference(self, module, args, kwargs):
        self.kind = 'query' if kwargs.get('use_cache', False) else (
            'rebuild' if self.initialized else 'bootstrap')
        self.inference_frame = self.last_index + 1 if self.kind == 'query' else self.last_index
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def after_inference(self, module, args, kwargs, output):
        torch.cuda.synchronize()
        self.inference_log.write(json.dumps(dict(frame=self.inference_frame, kind=self.kind,
            seconds=time.perf_counter() - self.started,
            cache_bytes=self.cache_policy.cache_bytes(),
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())) + '\n')
        self.inference_log.flush()

    def bootstrap(self, frame):
        super().bootstrap(frame)
        self.initialized = True

    def select(self, frame, centre, pose, cached_poses, cached_ids, original):
        index = int(frame['idx'])
        assert index == self.last_index + 1
        assert sorted(set(cached_ids)) == list(self.cache_policy.records)
        self.last_index = index
        selected = self.cache_policy.select(index, cached_ids)
        row = dict(frame=index, policy='interval', cache_frame_ids=sorted(set(cached_ids)),
                   cache_bytes_before=self.cache_policy.cache_bytes(), angle_degrees=None,
                   novelty=None, object_patches=None, original_decision=bool(original),
                   candidate=bool(selected), capped=False, selected=bool(selected),
                   evicted=list(self.cache_policy.evicted) if selected else [])
        self.log.write(json.dumps(row, allow_nan=False) + '\n')
        self.log.flush()
        if selected:
            self.inserted.append(index)
        return selected

    def close(self):
        super().close()
        for hook in self.hooks:
            hook.remove()
        self.cache_policy.close()
        self.inference_log.close()
        with (self.result / 'cache_events.jsonl').open('w') as stream:
            for event in self.cache_policy.events:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
