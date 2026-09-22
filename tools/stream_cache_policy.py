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
        assert self.patch_policy in ('all', 'uniform', 'spatial_uniform', 'random', 'confidence',
                                     'novelty', 'confidence_novelty', 'mask',
                                     'correspondence', 'semantic_correspondence')
        assert self.frame_budget >= 2 and self.spatial_grid >= 1
        assert 0 < self.patch_fraction <= 1
        assert -1 <= self.similarity_floor <= 1 and self.threshold >= 0
        if self.patch_policy in ('correspondence', 'semantic_correspondence'):
            assert self.admission == 'all' and self.eviction == 'fifo'


def reciprocal_links(features, points, history_features, history_points, radius,
                     labels=None, history_labels=None):
    assert features.ndim == history_features.ndim == 2
    assert points.shape == (len(features), 3)
    assert history_points.shape == (len(history_features), 3)
    best = torch.full((len(features),), -float('inf'))
    links = torch.full((len(features),), -1, dtype=torch.long)
    reverse_score = torch.full((len(history_features),), -float('inf'))
    reverse = torch.full((len(history_features),), -1, dtype=torch.long)
    for start in range(0, len(features), 128):
        stop = min(start + 128, len(features))
        similarity = features[start:stop] @ history_features.T
        valid = (torch.cdist(points[start:stop], history_points) <= radius) & (radius > 0)
        if labels is not None:
            valid &= labels[start:stop, None] == history_labels[None]
        similarity.masked_fill_(~valid, -float('inf'))
        best[start:stop], links[start:stop] = similarity.max(1)
        score, index = similarity.max(0)
        improved = score > reverse_score
        reverse[improved] = index[improved] + start
        reverse_score = torch.maximum(reverse_score, score)
    valid = (best >= .9) & (reverse[links] == torch.arange(len(features)))
    links[~valid] = -1
    return links, best.masked_fill(~valid, -1.)


def spatial_track_pick(grid, count, links, scores, track_ids, labels=None):
    height, width = grid
    total = height * width
    assert links.shape == scores.shape == track_ids.shape == (total,)
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    groups = ((y * 4 // height) * 4 + x * 4 // width).flatten()
    if labels is not None:
        assert labels.shape == (total,) and labels.dtype == torch.bool
        groups = groups * 2 + labels.long()
    unique, sizes = torch.unique(groups, sorted=True, return_counts=True)
    quotas = sizes * count // total
    remainder = sizes * count % total
    quotas[remainder.argsort(descending=True, stable=True)[:count - int(quotas.sum())]] += 1
    picked = []
    used_tracks = set()
    for group, quota in zip(unique.tolist(), quotas.tolist()):
        candidates = (groups == group).nonzero().flatten()
        chosen = []
        order = candidates[scores[candidates].argsort(descending=True, stable=True)]
        for index in order.tolist():
            if len(chosen) == quota:
                break
            track = int(track_ids[index])
            if links[index] >= 0 and track not in used_tracks:
                chosen.append(index)
                used_tracks.add(track)
        remaining = candidates[~torch.isin(candidates, torch.tensor(chosen, dtype=torch.long))]
        missing = quota - len(chosen)
        if missing:
            offsets = torch.linspace(0, len(remaining) - 1, missing).round().long()
            chosen.extend(remaining[offsets].tolist())
        picked.extend(chosen)
    assert len(picked) == len(set(picked)) == count
    return torch.tensor(sorted(picked), dtype=torch.long)


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

    def update(self, frame_id, patches, grid_shape, confidence, mask=None, points=None):
        assert patches.ndim == 2 and patches.device.type == 'cpu'
        assert confidence.shape == (len(patches),)
        assert math.prod(grid_shape) == len(patches)
        cfg = self.config
        correspondence = cfg.patch_policy in ('correspondence', 'semantic_correspondence')
        if correspondence:
            assert points is not None and points.shape == (len(patches), 3)
            assert points.device.type == 'cpu' and torch.isfinite(points).all()
        if cfg.patch_policy == 'semantic_correspondence':
            assert mask is not None and mask.shape == (len(patches),) and mask.dtype == torch.bool
            assert mask.any() and (~mask).any()
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
        links = torch.full((len(patches),), -1, dtype=torch.long)
        track_ids = frame_id * len(patches) + torch.arange(len(patches))
        radius = None
        if count == len(patches):
            picked = torch.arange(count)
        elif correspondence:
            # Match only history that survives the FIFO replacement.
            history_ids = old_ids if len(old_ids) < cfg.frame_budget else [old_ids[0], *old_ids[2:]]
            history = [self.records[i] for i in history_ids]
            history_features = torch.cat([row['patches'] for row in history])
            history_points = torch.cat([row['points'] for row in history])
            history_tracks = torch.cat([row['tracks'] for row in history])
            history_labels = (torch.cat([row['labels'] for row in history])
                              if cfg.patch_policy == 'semantic_correspondence' else None)
            current = points.reshape(*grid_shape, 3)
            spacing = torch.cat(((current[1:] - current[:-1]).norm(dim=-1).flatten(),
                                 (current[:, 1:] - current[:, :-1]).norm(dim=-1).flatten()))
            spacing = spacing[spacing > 0]
            radius = 0. if not len(spacing) else 2 * float(spacing.median())
            links, scores = reciprocal_links(normalized, points, history_features,
                                              history_points, radius,
                                              mask if history_labels is not None else None,
                                              history_labels)
            matched = links >= 0
            track_ids[matched] = history_tracks[links[matched]]
            picked = spatial_track_pick(grid_shape, count, links, scores, track_ids,
                                        mask if history_labels is not None else None)
        elif cfg.patch_policy == 'spatial_uniform':
            picked = spatial_track_pick(grid_shape, count, links,
                                        torch.full((len(patches),), -1.), track_ids)
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
        if correspondence:
            self.records[frame_id].update(points=points[picked], tracks=track_ids[picked],
                                          indices=picked)
            if cfg.patch_policy == 'semantic_correspondence':
                self.records[frame_id]['labels'] = mask[picked]
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
        if correspondence:
            stats.update(geometry_radius=radius, matched_kept=int((links[picked] >= 0).sum()),
                         distinct_tracks_kept=len(torch.unique(track_ids[picked])))
            if cfg.patch_policy == 'semantic_correspondence':
                stats['object_matched_kept'] = int(((links[picked] >= 0) & mask[picked]).sum())
        return picked, list(self.records), stats

    def feature_bytes(self):
        fields = ('descriptor', 'patches', 'points', 'tracks', 'indices', 'labels')
        return (0 if self.center is None else self.center.numel() * self.center.element_size()) + sum(
            r[k].numel() * r[k].element_size() for r in self.records.values()
            for k in fields if k in r)

    def configuration(self):
        return asdict(self.config)
