"""B1–B4 exploratory StreamVGGT interventions; native heads are untouched."""

import torch


def attention_bins(frame_ids, current):
    """Disjoint own / anchor / previous four / older bins."""
    own = frame_ids == current
    anchor = (frame_ids == 0) & ~own
    recent = (frame_ids >= current - 4) & ~own & ~anchor
    return torch.stack((own, anchor, recent, ~(own | anchor | recent)))


def retention_mask(frame_ids, current, local_heads, layers=False):
    bins = attention_bins(frame_ids, current)
    if layers:
        keep = bins[0] if bool(local_heads.all()) else torch.ones_like(bins[0])
        return keep[None, None, None, :]
    return (~local_heads[:, None] | bins[:3].any(0)[None, :])[None, :, None, :]


def channel_basis(value):
    """Uncentred channel PCA per head from at most 2048 evenly spaced tokens."""
    assert value.ndim == 4 and value.shape[0] == 1
    indices = torch.linspace(0, value.shape[2] - 1, min(2048, value.shape[2]),
                             device=value.device).long()
    sample = value[0].index_select(1, indices).float()
    eigenvalues, basis = torch.linalg.eigh(sample.transpose(-1, -2) @ sample)
    return eigenvalues.flip(-1).clamp_min(0).cpu(), basis.flip(-1).cpu()


class StructureExperiment:
    def __init__(self, adapter, mode, target):
        assert adapter.host == 'streamvggt'
        self.adapter, self.mode, self.target = adapter, mode, target
        self.samples = [[] for _ in adapter.original_attention]
        self.current = 0
        self.profile = None
        if mode in ('heads', 'layers', 'rank'):
            self.profile = torch.load(target.parent / 'b_profile' / 'profile.pt',
                                      map_location='cpu', weights_only=True)
            assert len(self.profile['mass']) == len(self.samples)
        for layer, (attention, _) in enumerate(adapter.original_attention):
            attention.structure_experiment = self
            attention.structure_layer = layer

    def attend(self, layer, q, k, mask):
        frames = torch.cat((self.adapter.token_frames,
                            torch.full((q.shape[2],), self.current))).to(q.device)
        assert len(frames) == k.shape[2]
        if self.mode == 'profile' and self.current % 10 == 9:
            # All special queries plus 32 evenly spaced patch queries.
            special = self.adapter.special
            indices = torch.cat((torch.arange(special, device=q.device),
                torch.linspace(special, q.shape[2] - 1, min(32, q.shape[2] - special),
                               device=q.device).long()))
            logits = q.index_select(2, indices).float() @ k.float().transpose(-1, -2)
            probability = (logits / q.shape[-1] ** .5).softmax(-1)
            bins = attention_bins(frames, self.current)
            mass = torch.stack([probability[..., group].sum(-1).mean((0, 2))
                                for group in bins], -1)
            self.samples[layer].append(mass.cpu())
        if self.mode in ('heads', 'layers'):
            assert mask is None
            local = (self.profile['mass'][layer][:, 3] < .05).to(q.device)
            mask = retention_mask(frames, self.current, local, self.mode == 'layers')
        return mask

    def after_prune(self):
        adapter = self.adapter
        report = dict(b_experiment=self.mode)
        if self.mode == 'registers':
            # Every frame retains its camera/register tokens; only recent four
            # retain patches. Token rows and RoPE positions are pruned together.
            frames = adapter.token_frames
            starts = torch.cat((torch.tensor([0]), (frames[1:] != frames[:-1]).nonzero().flatten() + 1))
            special = torch.zeros_like(frames, dtype=torch.bool)
            for start in starts.tolist():
                special[start:start + adapter.special] = True
            indices = (special | (frames > self.current - 4)).nonzero().flatten()
            adapter.aggregator_cache = [tuple(t.index_select(3, indices.to(t.device)) for t in pair)
                                        for pair in adapter.aggregator_cache]
            for attention, _ in adapter.original_attention:
                attention.cache_positions = attention.cache_positions.index_select(
                    1, indices.to(attention.cache_positions.device))
            adapter.token_frames = frames[indices]
        if self.mode == 'rank':
            count = int((adapter.token_frames == self.current).sum())
            analytical_bytes = 0
            for layer, pair in enumerate(adapter.aggregator_cache):
                for which, tensor in enumerate(pair):
                    newest = tensor.flatten(2, 3)[:, :, -count:]
                    basis = self.profile['basis'][layer][which].to(tensor.device)
                    basis = basis[..., :basis.shape[-1] // 2]
                    analytical_bytes += tensor.numel() // 2 * tensor.element_size()
                    analytical_bytes += basis.numel() * basis.element_size()
                    value = newest.float()
                    newest.copy_(((value @ basis) @ basis.transpose(-1, -2)).to(tensor.dtype))
            report['analytical_rank_aggregator_bytes'] = analytical_bytes
        return report

    def finish(self):
        if self.mode == 'profile':
            assert all(len(samples) == 20 for samples in self.samples)
            mass = torch.stack([torch.stack(samples).mean(0) for samples in self.samples])
            spectra, bases = [], []
            for pair in self.adapter.aggregator_cache:
                results = [channel_basis(t.flatten(2, 3)) for t in pair]
                spectra.append(torch.stack([result[0] for result in results]))
                bases.append(torch.stack([result[1] for result in results]))
            torch.save(dict(mass=mass, samples=torch.stack([torch.stack(s) for s in self.samples]),
                            spectrum=torch.stack(spectra), basis=torch.stack(bases)),
                       self.target / 'profile.pt')
        for attention, _ in self.adapter.original_attention:
            del attention.structure_experiment, attention.structure_layer
