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

Mask -> token mapping is NOT here: `tools/stream3r_ycbv_tokens.py` already has
`token_grid_shape` and `mask_to_token_mask`, which replicate the loader's resize
and centre crop. Callers pass the token mask in; do not re-derive it.
"""
import numpy as np
import torch

POLICIES = ('causal', 'window', 'semantic', 'confidence', 'random', 'uniform')


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
           generator=None):
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

    'causal' and 'window' reproduce the upstream policies exactly and exist as
    self-checks, not as experimental arms.
    """
    assert policy in POLICIES, policy
    assert frames >= 1 and tokens_per_frame > patch_start_idx >= 0
    patches = tokens_per_frame - patch_start_idx
    offsets = torch.arange(frames) * tokens_per_frame

    if policy == 'causal':
        return torch.arange(frames * tokens_per_frame)

    if policy == 'window':
        kept = list(range(max(0, frames - window_size), frames))
        if anchor and 0 not in kept:
            kept = [0] + kept[1:] if len(kept) == window_size else [0] + kept
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
