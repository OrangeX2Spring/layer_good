"""Fixed INT2 sensitivity diagnostic; reconstructed storage, analytical bytes only."""
import json
from pathlib import Path

import torch

from stream_cache_quantization import fake_quantize

GROUPS = {'layer': 6, 'head': 4, 'channel': 4}


class CacheCompression:
    def __init__(self, specification):
        self.bits = specification['bits']
        self.family = specification.get('family')
        self.group = specification.get('group')
        self.profile = specification.get('profile', False)
        assert self.bits in (0, 2, 4)
        assert self.family is None or self.family in GROUPS
        assert (self.group is None if self.family is None else
                type(self.group) is int and 0 <= self.group < GROUPS[self.family])
        assert self.family is None or self.bits == 2
        self.variance = {key: [0.] * count for key, count in GROUPS.items()}
        self.profile_calls = 0
        self.calls = 0
        self.frame_bytes = {}
        self.events = []

    def apply(self, pairs):
        """Modify fresh K/V only. pairs contain [batch, head, token, channel] views."""
        depth = len(pairs)
        assert depth > 0 and depth % GROUPS['layer'] == 0
        packed, actual, protected, changed = 0, 0, 0, False
        for layer, pair in enumerate(pairs):
            assert len(pair) == 2
            for which, value in enumerate(pair):
                assert value.ndim == 4 and value.shape[0] == 1
                assert value.shape[1] % 4 == 0 and value.shape[3] % 4 == 0
                assert value.is_floating_point() and torch.isfinite(value).all()
                size = value.numel() * value.element_size()
                actual += size
                if self.profile:
                    # Fixed token subsample; population channel variance over tokens.
                    ids = torch.linspace(0, value.shape[2] - 1,
                                         min(128, value.shape[2]), device=value.device).long()
                    variance = value[0].index_select(1, ids).float().var(1, unbiased=False)
                    self.variance['layer'][layer * 6 // depth] += float(variance.sum())
                    for group in range(4):
                        self.variance['head'][group] += float(variance.chunk(4, 0)[group].sum())
                        self.variance['channel'][group] += float(variance.chunk(4, 1)[group].sum())
                if not self.bits:
                    packed += size
                    continue
                restored, payload = fake_quantize(value, self.bits, 2 if which == 0 else 3)
                selection = None
                if self.family == 'layer' and layer * 6 // depth == self.group:
                    selection = (slice(None),) * 4
                elif self.family == 'head':
                    width = value.shape[1] // 4
                    selection = (slice(None), slice(self.group * width, (self.group + 1) * width),
                                 slice(None), slice(None))
                elif self.family == 'channel':
                    width = value.shape[3] // 4
                    selection = (slice(None), slice(None), slice(None),
                                 slice(self.group * width, (self.group + 1) * width))
                if selection is not None:
                    restored[selection] = value[selection]
                    protected += value[selection].numel() * value.element_size()
                changed = bool(torch.any(restored != value)) or changed
                value.copy_(restored)
                packed += payload
        self.calls += 1
        self.profile_calls += int(self.profile)
        # Explicit representation: complete low-bit stream plus native overwrite values.
        # One 8-byte family/group descriptor is counted by callers per resident cache.
        return dict(actual_bytes=actual, analytical_payload_bytes=packed + protected,
                    protected_bytes=protected, changed=changed, layers=depth)

    def stream(self, adapter, frame, refresh=False):
        cache = (adapter.aggregator_cache if adapter.host == 'streamvggt'
                 else adapter.session.aggregator_kv_cache_list)
        count = int((adapter.token_frames == frame).sum())
        assert count > 0
        pairs = [tuple((t.flatten(2, 3) if adapter.host == 'streamvggt' else t)[:, :, -count:]
                       for t in pair) for pair in cache]
        report = self.apply(pairs)
        retained = set(adapter.token_frames.tolist())
        if refresh:
            self.frame_bytes = {}
        else:
            self.frame_bytes = {f: b for f, b in self.frame_bytes.items() if f in retained}
        assert frame not in self.frame_bytes
        self.frame_bytes[frame] = report['analytical_payload_bytes']
        return dict(compression_bits=self.bits, protection_family=self.family,
                    protection_group=self.group, compressed_entry=report,
                    analytical_compression_bytes=sum(self.frame_bytes.values())
                    + (8 if self.family is not None else 0))

    def kvt(self, cache, frame_ids):
        # Pi3's odd decoder layers cache normalized post-RoPE keys.
        layers = sorted(cache)
        assert layers == list(range(1, 2 * len(layers), 2))
        pairs = [(cache[i]['k'], cache[i]['v']) for i in layers]
        report = self.apply(pairs)
        report['analytical_payload_bytes'] += 8 if self.family is not None else 0
        self.events.append(dict(frame=frame_ids[-1], frame_ids=list(frame_ids),
                                layer_ids=layers, dtype=str(pairs[0][0].dtype), **report))

    def save(self, path):
        Path(path).write_text(json.dumps(dict(bits=self.bits, family=self.family,
            group=self.group, calls=self.calls, variance=self.variance,
            profile_calls=self.profile_calls, events=self.events,
            interpretation='Fake quantization. Full low-bit payload plus native protected '
            'overwrite values and 8-byte descriptor; no actual packed allocation.'),
            indent=2, allow_nan=False) + '\n')
