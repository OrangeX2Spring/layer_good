"""Scene-level selectors observing arrival-time Pi3 features; no extra forwards.

The existing ARCTIC selector is deliberately unchanged. All scene patches count.
Pi3 encoder output and decoder[0] precede cross-frame attention at the pinned hash.
"""
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

PI3_SHA256 = 'cbcf68b3c05baab7680f6e24afda42501dc0ba799e85ca18668ba3e8a5812979'


def patch_novelty(current, retained, score, similarity_floor):
    """Inputs are float32 L2-normalized patch rows; match all spatial positions.

    Coverage matches against the union of retained patches. Chamfer takes the
    nearest retained FRAME under symmetric mean squared Euclidean patch distance.
    Chunking bounds temporary pairwise storage without subsampling patches.
    """
    assert current.ndim == 2 and current.dtype == torch.float32
    assert retained and score in ('coverage', 'chamfer')
    covered_similarity = torch.full((len(current),), -1., device=current.device)
    distances = []
    for reference in retained:
        assert reference.shape == current.shape and reference.dtype == current.dtype
        forward = []
        backward = torch.full((len(reference),), -1., device=current.device)
        for chunk in current.split(128):
            similarities = (chunk @ reference.T).clamp(-1, 1)
            forward.append(similarities.max(dim=1).values)
            if score == 'chamfer':
                backward = torch.maximum(backward, similarities.max(dim=0).values)
        forward = torch.cat(forward)
        if score == 'coverage':
            covered_similarity = torch.maximum(covered_similarity, forward)
        else:
            # ||u-v||² = 2-2 cos(u,v); symmetric mean averages the two directions.
            distances.append(((2 - 2 * forward).mean() + (2 - 2 * backward).mean()) / 2)
    if score == 'coverage':
        return float((covered_similarity < similarity_floor).float().mean())
    return float(torch.stack(distances).min())


class TumSelector:
    def __init__(self, config, log, inference_log):
        self.config = config
        self.log = log
        self.inference_log = inference_log
        self.inserted = [0]
        self.last_index = 0
        self.retained = []
        self.tokens = None
        self.initialized = False
        self.hooks = []

    def attach(self, model):
        self.model = model
        assert hashlib.sha256(Path(inspect.getfile(type(model))).read_bytes()).hexdigest() == PI3_SHA256
        assert model.patch_size == 14 and model.patch_start_idx == 5
        self.hooks.append(model.register_forward_pre_hook(self.before_inference, with_kwargs=True))
        self.hooks.append(model.register_forward_hook(self.after_inference, with_kwargs=True))
        if self.config['policy'] in ('semantic', 'probe'):
            layer = self.config['layer']
            assert layer in ('encoder', 'decoder0')
            module = model.encoder if layer == 'encoder' else model.decoder[0]
            self.hooks.append(module.register_forward_hook(self.capture))

    def before_inference(self, module, args, kwargs):
        self.kind = 'query' if kwargs.get('use_cache', False) else (
            'rebuild' if self.initialized else 'bootstrap')
        self.inference_frame = self.last_index + 1 if self.kind == 'query' else self.last_index
        torch.cuda.synchronize()
        self.inference_started = time.perf_counter()

    def after_inference(self, module, args, kwargs, output):
        torch.cuda.synchronize()
        record = dict(frame=self.inference_frame, kind=self.kind,
                      seconds=time.perf_counter() - self.inference_started,
                      cache_bytes=self.cache_bytes(), allocated_bytes=torch.cuda.memory_allocated(),
                      reserved_bytes=torch.cuda.memory_reserved())
        self.inference_log.write(json.dumps(record) + '\n')
        self.inference_log.flush()

    def capture(self, module, args, output):
        if self.kind == 'rebuild':
            return  # Never replace retained arrival-time features with joint features.
        if self.config['layer'] == 'encoder':
            tokens = output['x_norm_patchtokens']
        else:
            tokens = output[:, self.model.patch_start_idx:]
        assert tokens.ndim == 3 and tokens.shape[-1] == 1024
        # Bootstrap duplicates frame 0; queries have exactly one frame.
        self.tokens = tokens[0].detach().float().clone()

    def feature(self, frame):
        h, w = frame['resized_mask_np'].shape
        assert frame['resized_mask_np'].all() and h % 14 == w % 14 == 0
        assert self.tokens.shape == (h // 14 * (w // 14), 1024)
        if self.config['score'] == 'cosine':
            vector = self.tokens.mean(dim=0)
            assert vector.norm() > 0
            return F.normalize(vector, dim=0)
        assert (self.tokens.norm(dim=1) > 0).all()
        return F.normalize(self.tokens, dim=1)

    def bootstrap(self, frame):
        assert frame['idx'] == 0
        if self.config['policy'] in ('semantic', 'probe'):
            self.retained.append(self.feature(frame))
        self.tokens = None
        self.initialized = True

    def cache_bytes(self):
        return sum(v.numel() * v.element_size()
                   for layer in self.model.cache.values() for v in layer.values())

    def select(self, frame, centre, pose, cached_poses, cached_ids, original):
        index = int(frame['idx'])
        assert index == self.last_index + 1
        assert sorted(set(cached_ids)) == self.inserted and max(cached_ids) < index
        self.last_index = index
        started = time.perf_counter()
        config = self.config
        score = None
        value = None
        if config['policy'] == 'original':
            candidate = bool(original)
        elif config['policy'] == 'periodic':
            # Preserve upstream's internal counter phase for every interval.
            candidate = (index + 1) % config['interval'] == 0
        else:
            assert config['policy'] in ('semantic', 'probe')
            value = self.feature(frame)
            if config['score'] == 'cosine':
                score = float(1 - (torch.stack(self.retained) @ value).max().clamp(-1, 1))
            else:
                score = patch_novelty(value, self.retained, config['score'],
                                      config['similarity_floor'])
            candidate = config['policy'] == 'probe' or score > config['threshold']
        capped = len(self.inserted) >= config['cap']
        selected = bool(candidate and not capped)
        torch.cuda.synchronize()
        row = dict(frame=index, score=score, candidate=bool(candidate), selected=selected,
                   capped=capped, cap_blocked=bool(candidate and capped),
                   original_decision=bool(original), cache_frame_ids=list(self.inserted),
                   cache_bytes_before=self.cache_bytes(),
                   feature_bytes_before=sum(v.numel() * v.element_size() for v in self.retained),
                   selector_seconds=time.perf_counter() - started)
        self.log.write(json.dumps(row, allow_nan=False) + '\n')
        self.log.flush()
        if selected:
            self.inserted.append(index)
            if value is not None:
                self.retained.append(value)
        self.tokens = None
        return selected

    def close(self):
        for hook in self.hooks:
            hook.remove()
