"""Causal keyframe policies. No trajectory files, GT, or sequence length input.

Pi3 pin 27e96ce: decoder[0] is frame-local, decoder[1] first uses cached K/V.
The hook observes decoder[0] without replacing its output or running an encoder.
"""
import json
import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class OnlineSelector:
    def __init__(self, config, log):
        self.config = config
        self.log = log
        self.rng = np.random.default_rng(config['seed'])
        self.embeddings = []
        self.inserted = [0]
        self.last_index = 0
        self.tokens = None

    def attach(self, model):
        self.model = model
        assert model.patch_size == 14 and model.patch_start_idx == 5
        if self.config['policy'] == 'semantic':
            source = Path(inspect.getfile(type(model)))
            assert hashlib.sha256(source.read_bytes()).hexdigest() == (
                'cbcf68b3c05baab7680f6e24afda42501dc0ba799e85ca18668ba3e8a5812979'
            ), 'Pi3 source differs from the inspected pin; re-audit feature hook'
            self.hook = model.decoder[0].register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        assert output.ndim == 3 and output.shape[-1] == 1024
        self.tokens = output.detach()[:, self.model.patch_start_idx:]

    def embedding(self, frame):
        mask = torch.as_tensor(frame['resized_mask_np'], device=self.tokens.device)
        assert mask.ndim == 2 and mask.dtype == torch.bool
        h, w = mask.shape
        assert h % 14 == 0 and w % 14 == 0
        keep = F.avg_pool2d(mask.float()[None, None], 14, 14).flatten() >= 0.5
        assert self.tokens.shape[1] == keep.numel()
        count = int(keep.sum())
        if count == 0:
            return None, count
        vector = self.tokens[0, keep].float().mean(dim=0)
        assert vector.norm() > 0
        return F.normalize(vector, dim=0), count

    def bootstrap(self, frame):
        assert frame['idx'] == 0
        if self.config['policy'] == 'semantic':
            vector, count = self.embedding(frame)
            assert count > 0, 'Bootstrap has no majority-object patches'
            self.embeddings.append(vector)
        self.tokens = None

    def select(self, frame, centre, pose, cached_poses, cached_ids, original):
        index = int(frame['idx'])
        assert index == self.last_index + 1
        assert all(i < index for i in cached_ids)
        assert sorted(set(cached_ids)) == self.inserted
        assert centre.shape == (3,) and pose.shape == (4, 4)
        assert cached_poses.ndim == 3 and cached_poses.shape[1:] == (4, 4)
        self.last_index = index
        cfg = self.config
        policy = cfg['policy']
        angle = novelty = patches = None
        vector = None
        if policy == 'original':
            candidate = original
        elif policy == 'angular':
            directions = centre - cached_poses[:, :3, 3]
            current = centre - pose[:3, 3]
            assert (directions.norm(dim=-1) > 0).all() and current.norm() > 0
            similarity = (F.normalize(directions, dim=-1) @ F.normalize(current, dim=0)).max()
            angle = float(torch.rad2deg(torch.acos(similarity.clamp(-1, 1))))
            candidate = angle > cfg['angle_degrees']
        elif policy == 'interval':
            candidate = index % cfg['interval'] == 0
        elif policy == 'random':
            candidate = self.rng.random() < 1.0 / cfg['interval']
        elif policy == 'semantic':
            vector, patches = self.embedding(frame)
            candidate = False
            if patches:
                novelty = float(1 - (torch.stack(self.embeddings) @ vector).max().clamp(-1, 1))
                candidate = novelty > cfg['novelty_threshold']
        else:
            raise ValueError(policy)
        capped = cfg['max_keyframes'] > 0 and len(self.inserted) >= cfg['max_keyframes']
        selected = bool(candidate and not capped)
        cache_bytes = sum(v.numel() * v.element_size()
                          for layer in self.model.cache.values() for v in layer.values())
        row = dict(frame=index, policy=policy, cache_frame_ids=sorted(set(cached_ids)),
                   cache_bytes_before=cache_bytes, angle_degrees=angle, novelty=novelty,
                   object_patches=patches, original_decision=original,
                   candidate=bool(candidate), capped=capped, selected=selected)
        self.log.write(json.dumps(row, allow_nan=False) + '\n')
        self.log.flush()
        if selected:
            self.inserted.append(index)
            if policy == 'semantic':
                self.embeddings.append(vector)
        self.tokens = None
        return selected

    def close(self):
        if self.config['policy'] == 'semantic':
            self.hook.remove()
