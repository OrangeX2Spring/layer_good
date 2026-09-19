"""Single-frame adapters; model files and LongStream pose references stay unchanged."""

from types import MethodType

import torch
import torch.nn.functional as F


def tensor_bytes(tree):
    if isinstance(tree, torch.Tensor):
        return tree.numel() * tree.element_size()
    if isinstance(tree, dict):
        return sum(tensor_bytes(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(tensor_bytes(value) for value in tree)
    return 0


def sparse_vggt_attention(self, x, pos=None, attn_mask=None,
                          past_key_values=None, use_cache=False):
    """Inference-only global attention, with raw K/V and explicit key positions.

    Keep the upstream pair interface and 5-D layout, using one ragged token row.
    Aggregator's S_true is used only to choose first/other special-token embeddings.
    A nonempty cache therefore still selects the exact upstream 'other' embedding.
    """
    assert use_cache and not self.training and self.fused_attn
    batch, tokens, channels = x.shape
    q, k, v = self.qkv(x).reshape(
        batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    if past_key_values is not None:
        old_k, old_v = past_key_values
        k = torch.cat((old_k.flatten(2, 3), k), dim=2)
        v = torch.cat((old_v.flatten(2, 3), v), dim=2)
        key_positions = torch.cat((self.cache_positions, pos), dim=1)
    else:
        key_positions = pos
    self.cache_positions = key_positions
    cache = (k.unsqueeze(2), v.unsqueeze(2))
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None:
        q, k = self.rope(q, pos), self.rope(k, key_positions)
    value = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.)
    value = value.transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(value)), cache


class StreamAdapter:
    def __init__(self, model, host, feature='encoder', sparse=True,
                 keyframe_stride=8, refresh=4):
        assert host in ('stream3r', 'streamvggt', 'longstream')
        assert keyframe_stride >= 1 and refresh >= 2
        self.model, self.host = model, host
        self.core = model.longstream if host == 'longstream' else model
        self.sparse = sparse
        self.keyframe_stride, self.refresh = keyframe_stride, refresh
        self.special = self.core.aggregator.patch_start_idx + int(
            host == 'longstream' and self.core.enable_scale_token)
        self.patch_size = self.core.aggregator.patch_size
        self.original_attention = []
        self.features = None
        if host == 'streamvggt' and sparse:
            for block in self.core.aggregator.global_blocks:
                attention = block.attn
                assert attention.rope is not None
                self.original_attention.append((attention, attention.forward))
                attention.cache_positions = None
                attention.forward = MethodType(sparse_vggt_attention, attention)
        if host == 'stream3r':
            from stream3r.stream_session import StreamSession
            self.session = StreamSession(model, mode='causal')
        elif host == 'longstream':
            from longstream.streaming.stream_session import StreamSession
            self.session = StreamSession(model, mode='causal')
            assert not self.core.aggregator.use_3d_rope
        else:
            self.session = None
        target = (self.core.aggregator.patch_embed if feature == 'encoder'
                  else self.core.aggregator.frame_blocks[0])
        self.feature_kind = feature
        self.hook = target.register_forward_hook(self.capture)
        self.clear()

    def capture(self, module, inputs, output):
        if self.feature_kind == 'encoder':
            value = output['x_norm_patchtokens'] if isinstance(output, dict) else output
        else:
            value = output[:, self.special:]
        assert value.ndim == 3 and value.shape[0] == 1
        self.features = value[0].detach()

    def clear(self):
        self.segment_start = 0
        self.token_frames = torch.empty(0, dtype=torch.long)
        if self.session is not None:
            self.session.clear()
        else:
            self.aggregator_cache = [None] * self.core.aggregator.depth
            self.camera_cache = [None] * self.core.camera_head.trunk_depth
            for attention, _ in self.original_attention:
                attention.cache_positions = None

    def forward(self, image, frame_id, reference_self=False):
        assert image.ndim == 5 and image.shape[:3] == (1, 1, 3)
        if self.host == 'stream3r':
            # Direct call avoids StreamSession's GPU prediction accumulation.
            output = self.model(images=image, mode='causal',
                                aggregator_kv_cache_list=self.session.aggregator_kv_cache_list,
                                camera_head_kv_cache_list=self.session.camera_head_kv_cache_list)
            self.session._update_cache(output['aggregator_kv_cache_list'],
                                       output['camera_head_kv_cache_list'])
        elif self.host == 'longstream':
            is_keyframe = frame_id % self.keyframe_stride == 0
            # At a keyframe upstream references the PREVIOUS keyframe.
            reference = max(0, ((frame_id - 1) // self.keyframe_stride) * self.keyframe_stride)
            reference = 0 if reference_self else max(0, reference - self.segment_start)
            output = self.session.forward_stream(
                image, is_keyframe=torch.tensor([[is_keyframe]], device=image.device),
                keyframe_indices=torch.tensor([[reference]], device=image.device), record=False)
        else:
            tokens, special, self.aggregator_cache = self.model.aggregator(
                image, past_key_values=self.aggregator_cache, use_cache=True,
                past_frame_idx=frame_id)
            assert special == self.special
            with torch.autocast(device_type=image.device.type, enabled=False):
                pose, self.camera_cache = self.model.camera_head(
                    tokens, past_key_values_camera=self.camera_cache, use_cache=True)
                depth, confidence = self.model.depth_head(tokens, images=image, patch_start_idx=special)
                points, points_conf = self.model.point_head(tokens, images=image, patch_start_idx=special)
            output = dict(pose_enc=pose[-1], depth=depth, depth_conf=confidence,
                          world_points=points, world_points_conf=points_conf)
        patches = (image.shape[-2] // self.patch_size) * (image.shape[-1] // self.patch_size)
        assert self.features.shape[0] == patches
        self.token_frames = torch.cat((self.token_frames,
                                       torch.full((patches + self.special,), frame_id)))
        return output

    def prune(self, patch_indices, retained_frames):
        current = int(self.token_frames[-1])
        current_start = int((self.token_frames != current).sum())
        keep = torch.isin(self.token_frames, torch.tensor(retained_frames))
        keep[current_start:] = False
        keep[current_start:current_start + self.special] = True
        keep[current_start + self.special + patch_indices] = True
        indices = keep.nonzero().flatten()
        if self.host == 'streamvggt':
            assert self.sparse
            self.aggregator_cache = [tuple(t.index_select(3, indices.to(t.device)) for t in pair)
                                     for pair in self.aggregator_cache]
            for attention, _ in self.original_attention:
                pos = attention.cache_positions
                attention.cache_positions = pos.index_select(1, indices.to(pos.device))
        else:
            self.session.aggregator_kv_cache_list = [
                [t.index_select(2, indices.to(t.device)) for t in pair]
                for pair in self.session.aggregator_kv_cache_list]
        self.token_frames = self.token_frames[indices]
        # Heads deliberately retain native causal history. LongStream's reference
        # mask indexes _frame_info by cache offset, so pruning it changes semantics.

    def needs_refresh(self, frame_id):
        return (self.host == 'longstream' and frame_id > 0
                and frame_id % (self.keyframe_stride * (self.refresh - 1)) == 0)

    def reset_segment(self, frame_id):
        assert self.host == 'longstream'
        self.session.clear_cache_only()
        self.segment_start = frame_id
        self.token_frames = torch.empty(0, dtype=torch.long)

    def memory(self):
        if self.host == 'streamvggt':
            aggregator, camera, relative, references = self.aggregator_cache, self.camera_cache, None, None
        else:
            aggregator = self.session.aggregator_kv_cache_list
            camera = self.session.camera_head_kv_cache_list
            relative = self.session.rel_pose_kv_cache_list if self.host == 'longstream' else None
            references = (self.core.rel_pose_head._keyframe_tokens_cache
                          if self.host == 'longstream' and self.core.rel_pose_head is not None else None)
        positions = sum(tensor_bytes(a.cache_positions) for a, _ in self.original_attention)
        return dict(aggregator_bytes=tensor_bytes(aggregator), camera_bytes=tensor_bytes(camera),
                    relative_pose_bytes=tensor_bytes(relative), reference_bytes=tensor_bytes(references),
                    position_bytes=positions, token_index_bytes=tensor_bytes(self.token_frames))

    def close(self):
        self.hook.remove()
        for attention, forward in self.original_attention:
            attention.forward = forward
            del attention.cache_positions
        self.features = None
