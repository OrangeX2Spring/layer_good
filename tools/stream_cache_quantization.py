"""Frame-entry fake quantization; analytical packed bytes, no packed storage."""

import math

import torch


def fake_quantize(value, bits, axis, group_size=32):
    """Affine groups with FP32 scale/minimum; short final groups are unpadded.

    Keys group tokens separately per channel; values group channels per token.
    Return dequantized values in the input dtype and the hypothetical byte cost.
    """
    assert value.ndim == 4 and value.is_floating_point()
    assert bits in (4, 8) and axis in (2, 3) and group_size > 0
    rows = value.movedim(axis, -1)
    length = rows.shape[-1]
    groups, tail = divmod(length, group_size)
    result = torch.empty_like(rows)
    packed_bytes = 0
    for start, stop, width in ((0, groups * group_size, group_size),
                                (groups * group_size, length, tail)):
        if start == stop:
            continue
        block = rows[..., start:stop].float().reshape(*rows.shape[:-1], -1, width)
        minimum = block.amin(-1, keepdim=True)
        scale = (block.amax(-1, keepdim=True) - minimum) / (2 ** bits - 1)
        # Constant groups reconstruct their minimum exactly, without division by zero.
        divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = ((block - minimum) / divisor).round().clamp(0, 2 ** bits - 1)
        result[..., start:stop] = (codes * scale + minimum).reshape(
            *rows.shape[:-1], stop - start).to(value.dtype)
        packed_bytes += minimum.numel() * (math.ceil(width * bits / 8) + 8)
    return result.movedim(-1, axis), packed_bytes
