"""Causal feature admission, frame eviction, and patch selection for the sweep."""

from dataclasses import dataclass, asdict
import math

import torch
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    feature: str = 'encoder'
    score: str = 'pooled'
    spatial_grid: int = 1
    threshold: float = 0.05
    similarity_floor: float = 0.9
    admission: str = 'all'
    eviction: str = 'fifo'
    frame_budget: int = 8
    patch_policy: str = 'all'
    patch_fraction: float = 1.0
    seed: int = 0

    def __post_init__(self):
        assert self.feature in ('encoder', 'frame0')
        assert self.score in ('pooled', 'centered', 'coverage', 'chamfer', 'q90')
        assert self.admission in ('all', 'novelty')
        assert self.eviction in ('fifo', 'redundancy', 'random')
        assert self.patch_policy in ('all', 'uniform', 'random', 'confidence',
                                     'novelty', 'confidence_novelty', 'mask')
        assert self.frame_budget >= 2 and self.spatial_grid >= 1
        assert 0 < self.patch_fraction <= 1
        assert -1 <= self.similarity_floor <= 1 and self.threshold >= 0


class CachePolicy:
    """Anchor + one live recent frame + admitted history, bounded by frame_budget.

    Decisions happen after prediction. Rejected frames still serve as the recent
    frame for the next query. Features are CPU float32, counted separately from KV.
    Historical patch descriptors are retained only for surviving patch tokens.
    """

    def __init__(self, config):
        self.config = config
        self.generator = torch.Generator().manual_seed(config.seed)
        self.clear()

    def clear(self):
        self.records = {}
        self.center = None

    def update(self, frame_id, patches, grid_shape, confidence, mask=None):
        assert patches.ndim == 2 and patches.device.type == 'cpu'
        assert confidence.shape == (len(patches),)
        assert math.prod(grid_shape) == len(patches)
        cfg = self.config
        patches = patches.float()
        if self.center is None:
            self.center = patches.mean(0)
        feature_map = patches.T.reshape(1, patches.shape[1], *grid_shape)
        pooled = F.adaptive_avg_pool2d(feature_map, cfg.spatial_grid).flatten()
        if cfg.score == 'centered':
            pooled = pooled.reshape(-1, cfg.spatial_grid ** 2) - self.center[:, None]
            pooled = pooled.flatten()
        descriptor = F.normalize(pooled, dim=0)
        normalized = F.normalize(patches, dim=-1)
        old_ids = list(self.records)
        if old_ids:
            descriptors = torch.stack([self.records[i]['descriptor'] for i in old_ids])
            # A zero centred anchor has similarity zero (novelty one), explicitly.
            pooled_novelty = float((1 - descriptors @ descriptor).min())
            best = torch.full((len(patches),), -1.0)
            reverse = []
            for record in self.records.values():
                history = record['patches']
                reverse_best = torch.full((len(history),), -1.0)
                for start in range(0, len(patches), 128):
                    similarity = normalized[start:start + 128] @ history.T
                    best[start:start + 128] = torch.maximum(
                        best[start:start + 128], similarity.max(1).values)
                    reverse_best = torch.maximum(reverse_best, similarity.max(0).values)
                reverse.append(1 - reverse_best)
            novelty = (1 - best).clamp(0, 2)
            scores = {
                'pooled': pooled_novelty, 'centered': pooled_novelty,
                'coverage': float((best < cfg.similarity_floor).float().mean()),
                'chamfer': float((novelty.mean() + torch.cat(reverse).mean()) / 2),
                'q90': float(torch.quantile(novelty, 0.9)),
            }
            value = scores[cfg.score]
        else:
            novelty = torch.ones(len(patches))
            value = None
        accepted = not old_ids or cfg.admission == 'all' or value > cfg.threshold
        anchor = not old_ids
        count = len(patches) if anchor or cfg.patch_policy == 'all' else max(
            1, math.ceil(len(patches) * cfg.patch_fraction))
        if count == len(patches):
            picked = torch.arange(count)
        elif cfg.patch_policy == 'uniform':
            picked = torch.linspace(0, len(patches) - 1, count).round().long()
        elif cfg.patch_policy == 'random':
            picked = torch.randperm(len(patches), generator=self.generator)[:count].sort().values
        else:
            if cfg.patch_policy == 'confidence':
                priority = confidence
            elif cfg.patch_policy == 'novelty':
                priority = novelty
            elif cfg.patch_policy == 'mask':
                assert mask is not None and mask.dtype == torch.bool and mask.shape == confidence.shape
                priority = mask.float()
            else:
                # Equal-weight rank fusion avoids assuming calibrated confidence.
                priority = (confidence.argsort(stable=True).argsort().float()
                            + novelty.argsort(stable=True).argsort().float())
            picked = priority.argsort(descending=True, stable=True)[:count].sort().values
        removed = []
        for previous in old_ids:
            if not self.records[previous]['accepted']:
                removed.append(previous)
                del self.records[previous]
        self.records[frame_id] = dict(descriptor=descriptor, patches=normalized[picked],
                                      accepted=accepted, anchor=anchor)
        evicted = []
        while len(self.records) > cfg.frame_budget:
            eligible = [i for i, r in self.records.items() if not r['anchor'] and i != frame_id]
            if cfg.eviction == 'fifo':
                victim = eligible[0]
            elif cfg.eviction == 'random':
                victim = eligible[int(torch.randint(len(eligible), (1,), generator=self.generator))]
            else:
                ids = list(self.records)
                descriptors = torch.stack([self.records[i]['descriptor'] for i in ids])
                similarities = descriptors @ descriptors.T
                similarities.fill_diagonal_(-float('inf'))
                redundancy = similarities.max(1).values
                victim = max(eligible, key=lambda i: float(redundancy[ids.index(i)]))
            del self.records[victim]
            evicted.append(victim)
        stats = {
            'frame': frame_id, 'score': value, 'accepted': accepted,
            'retained_frames': list(self.records), 'evicted': evicted,
            'expired_recent': removed, 'patch_indices': picked.tolist(),
            'novelty_quantiles': torch.quantile(novelty, torch.tensor([0., .5, .9, 1.])).tolist(),
            'confidence_quantiles': torch.quantile(confidence.float(), torch.tensor([0., .5, .9, 1.])).tolist(),
        }
        return picked, list(self.records), stats

    def feature_bytes(self):
        return (0 if self.center is None else self.center.numel() * self.center.element_size()) + sum(
            r[k].numel() * r[k].element_size() for r in self.records.values()
            for k in ('descriptor', 'patches'))

    def configuration(self):
        return asdict(self.config)
