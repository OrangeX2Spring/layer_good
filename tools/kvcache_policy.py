"""Token-level KV-cache selection policies for streaming geometry transformers.

Model-agnostic on purpose. Everything here returns *indices into a flat cache of
`frames * tokens_per_frame` positions*, so it serves all three hosts:

  STream3R    stream3r/stream3r/stream_session.py            `_update_cache`
  LongStream  longstream/longstream/streaming/stream_session.py  `_update_cache`
              (same method name and same flat [B, heads, T*P, D] layout; it
              vendors STream3R's attention, so RoPE is applied to k BEFORE the
              cache concatenation and dropping tokens is positionally exact)
  StreamVGGT  streamvggt/src/streamvggt/models/aggregator.py  `past_key_values`
              (5-D [B, heads, S, N, D], and RoPE is applied AFTER concatenation
              from `pos_k = pos.repeat(1, c, 1)` -- that repeat assumes every
              cached frame still holds all N tokens, so a StreamVGGT adapter
              must carry a parallel position tensor. These policies still apply;
              the adapter gathers positions with the same indices.)

No model, no GPU, no file I/O. Every function is testable on CPU.

Image resize/crop mapping is NOT here. The pooling helpers accept masks and
confidence already aligned to the exact model-input pixels.
"""
import numpy as np
import torch
import torch.nn.functional as F

POLICIES = ('causal', 'window', 'semantic', 'confidence', 'random', 'uniform')


def pool_mask(pixels, grid_shape, patch_start_idx, threshold=0.0):
    """Pool an already resized/cropped boolean mask; never select special tokens."""
    assert pixels.dtype == torch.bool and pixels.ndim == 3
    rows, cols = grid_shape
    assert rows > 0 and cols > 0 and patch_start_idx >= 0
    assert pixels.shape[1] % rows == 0 and pixels.shape[2] % cols == 0
    assert 0 <= threshold <= 1
    coverage = F.adaptive_avg_pool2d(pixels[:, None].float(), (rows, cols)).flatten(1)
    selected = coverage > 0 if threshold == 0 else coverage >= threshold
    specials = pixels.new_zeros((pixels.shape[0], patch_start_idx))
    return torch.cat((specials, selected), dim=1)


def pool_confidence(confidence, grid_shape, patch_start_idx):
    """Average already aligned confidence pixels within each patch cell."""
    assert confidence.ndim == 3 and confidence.is_floating_point()
    rows, cols = grid_shape
    assert rows > 0 and cols > 0 and patch_start_idx >= 0
    assert confidence.shape[1] % rows == 0 and confidence.shape[2] % cols == 0
    patches = F.adaptive_avg_pool2d(confidence[:, None], (rows, cols)).flatten(1)
    specials = confidence.new_full((confidence.shape[0], patch_start_idx), float('inf'))
    return torch.cat((specials, patches), dim=1)


def frames_in_cache(cache_length, tokens_per_frame):
    assert cache_length > 0 and tokens_per_frame > 0
    assert cache_length % tokens_per_frame == 0, (cache_length, tokens_per_frame)
    return cache_length // tokens_per_frame


def semantic_budget(mask, patch_start_idx):
    """Patch tokens the object occupies in each frame; the budget every null matches.

    `mask` is a boolean [frames, tokens_per_frame] over the full token row,
    special tokens included and required to be False so a mask cannot smuggle
    extra budget through them.
    """
    assert mask.dtype == torch.bool and mask.ndim == 2
    assert not mask[:, :patch_start_idx].any(), 'mask must be False on special tokens'
    return mask[:, patch_start_idx:].sum(dim=1)


def select(policy, *, frames, tokens_per_frame, patch_start_idx, mask=None,
           score=None, budget=None, window_size=5, anchor=True, keep_special=True,
           generator=None, window_counts_anchor=True):
    """Cache positions to retain, sorted ascending.

    frames            number of frames currently in the cache
    tokens_per_frame  P, including the `patch_start_idx` special tokens
    mask              bool [frames, P], object tokens; required by 'semantic'
    score             float [frames, P], higher is kept; required by 'confidence'
    budget            long [frames], patch tokens to keep per frame; required by
                      'confidence'/'random'/'uniform' so every arm is matched to
                      the semantic arm frame by frame
    anchor            keep frame 0 entire, as upstream `window` mode does
    keep_special      always retain each surviving frame's special tokens
    window_counts_anchor  True: LongStream's total-frame budget; False:
                          STream3R's anchor plus window_size recent frames

    'window' requires the host's explicit anchor-counting convention.
    """
    assert policy in POLICIES, policy
    assert frames >= 1 and tokens_per_frame > patch_start_idx >= 0
    patches = tokens_per_frame - patch_start_idx
    offsets = torch.arange(frames) * tokens_per_frame

    if policy == 'causal':
        return torch.arange(frames * tokens_per_frame)

    if policy == 'window':
        assert window_size >= 1
        recent = window_size - int(anchor and window_counts_anchor)
        kept = list(range(max(0, frames - recent), frames))
        if anchor:
            kept = [0] + kept
        rows = [offsets[f] + torch.arange(tokens_per_frame) for f in sorted(set(kept))]
        return torch.cat(rows)

    if policy == 'semantic':
        assert mask is not None and mask.shape == (frames, tokens_per_frame)
        chosen = mask.clone()
    else:
        assert budget is not None and budget.shape == (frames,)
        assert (budget >= 0).all() and (budget <= patches).all()
        chosen = torch.zeros(frames, tokens_per_frame, dtype=torch.bool)
        for frame in range(frames):
            count = int(budget[frame])
            if count == 0:
                continue
            if policy == 'confidence':
                assert score is not None and score.shape == (frames, tokens_per_frame)
                order = torch.argsort(score[frame, patch_start_idx:], descending=True,
                                      stable=True)
                picked = order[:count]
            elif policy == 'random':
                picked = torch.randperm(patches, generator=generator)[:count]
            else:  # 'uniform': evenly spaced on the patch row, deterministic
                picked = torch.from_numpy(
                    np.linspace(0, patches - 1, count).round().astype(np.int64))
                assert len(torch.unique(picked)) == count, (patches, count)
            chosen[frame, patch_start_idx + picked] = True

    if keep_special and patch_start_idx:
        chosen[:, :patch_start_idx] = True
    if anchor:
        chosen[0] = True
    return torch.nonzero(chosen.reshape(-1), as_tuple=False).squeeze(1)


def retained_bytes(indices, heads, head_dim, depth, element_size, tensors_per_layer=2):
    """Cache bytes the kept positions cost, counting keys and values in every layer.

    Special tokens and the frame-0 anchor are inside `indices`, so they are
    counted, as docs/semantic-kv-tracking-plan.md requires.
    """
    assert indices.ndim == 1
    return int(len(indices)) * heads * head_dim * depth * tensors_per_layer * element_size
