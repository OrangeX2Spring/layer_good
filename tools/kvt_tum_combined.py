"""TUM instrumentation for the single combined KV-Tracker cache policy."""
import json
import hashlib
import inspect
from pathlib import Path
import time

import torch

from kv_tracker.combined_cache import CombinedCache
from kvt_tum_selector import TumSelector


class CombinedSelector(TumSelector):
    def __init__(self, config, log, inference_log):
        super().__init__(config, log, inference_log)
        self.cache_policy = CombinedCache(config['cap'], config['interval'],
                                          config.get('patch_fraction', 0.5))

    def attach(self, model):
        from pi3.models.layers.attention import FlashAttentionRope
        assert hashlib.sha256(Path(inspect.getfile(FlashAttentionRope)).read_bytes()).hexdigest() == (
            'af0c54181f341bcd047dcd591bb1ff0919acf71c1fb77307d2996867e54533f6'
        ), 'Pi3 attention differs from the audited post-RoPE cache implementation'
        super().attach(model)

    def bootstrap(self, frame):
        assert frame['idx'] == 0
        self.initialized = True

    def select(self, frame, centre, pose, cached_poses, cached_ids, original):
        index = int(frame['idx'])
        assert index == self.last_index + 1
        self.last_index = index
        started = time.perf_counter()
        selected = self.cache_policy.select(index, cached_ids)
        torch.cuda.synchronize()
        row = dict(frame=index, score=None, candidate=selected, selected=selected,
                   capped=len(set(cached_ids)) >= self.config['cap'], cap_blocked=False,
                   original_decision=bool(original), cache_frame_ids=sorted(set(cached_ids)),
                   cache_bytes_before=self.cache_bytes(),
                   feature_bytes_before=self.cache_policy.feature_bytes(),
                   feature_export_seconds=0., selector_seconds=time.perf_counter() - started)
        self.log.write(json.dumps(row, allow_nan=False) + '\n')
        self.log.flush()
        if selected:
            self.inserted.append(index)
        return selected

    def close(self, result):
        self.cache_policy.close()
        with (result / 'cache_events.jsonl').open('w') as stream:
            for event in self.cache_policy.events:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
        super().close(result)
