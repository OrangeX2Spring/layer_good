"""Token-drop contract tests; run on the CAMP GPU before any token-drop tracking.

The patch mapping is checked on CPU. On the GPU, the real Pi3 checkpoint is run
through upstream ``Pi3.forward`` and through ``forward_kept`` with every patch
kept: a single-frame forward must match exactly, and a keyframe rebuild plus a
cached query must agree to within bf16 batch-shape noise. A partial mask must
cache only kept tokens and leave dropped pixels with no prediction.
"""
import unittest

import torch

from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32, pi3_inference
from kv_tracker.token_drop import patch_keep

H = W = 518  # 37 x 37 patches, the ARCTIC protocol resolution
MODEL = None


def model():
    global MODEL
    if MODEL is None:
        MODEL = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained("cuda:0").eval())
    return MODEL


def frames(n, seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(1, n, H, W, 3, generator=generator).cuda()
    mask = torch.zeros(n, H, W, dtype=torch.bool, device="cuda")
    for i in range(n):
        # A different object box per frame: 20 patch rows, a ragged right edge.
        mask[i, 70 + 14 * i:350 + 14 * i, 100:230 + 5 * i] = True
    return images * mask[None, ..., None], mask


def run(images, keep, query, query_keep):
    net = model()
    net.cache = {}
    rebuild = pi3_inference(net, images, "cuda:0", store_cache=True, keep=keep)
    cache = {i: {k: t.clone() for k, t in entry.items()} for i, entry in net.cache.items()}
    tracked = pi3_inference(net, query, "cuda:0", use_cache=True, keep=query_keep)
    return rebuild, cache, tracked


class PatchKeepTests(unittest.TestCase):
    def test_touching_patches(self):
        mask = torch.zeros(1, 28, 42, dtype=torch.bool)
        mask[0, 13, 14] = True  # one pixel in row 0, column 1
        mask[0, 20, 41] = True  # one pixel in row 1, column 2
        self.assertEqual(patch_keep(mask).tolist(), [[False, True, False, False, False, True]])

    def test_empty_frame_rejected(self):
        with self.assertRaises(AssertionError):
            patch_keep(torch.zeros(2, 14, 14, dtype=torch.bool))

    def test_size_must_be_patch_multiple(self):
        with self.assertRaises(AssertionError):
            patch_keep(torch.ones(1, 15, 14, dtype=torch.bool))


@unittest.skipUnless(torch.cuda.is_available(), "needs the CAMP GPU")
class ForwardTests(unittest.TestCase):
    def test_single_frame_all_kept_is_exact(self):
        images, _ = frames(1, 0)
        net = model()
        full = torch.ones(1, (H // 14) * (W // 14), dtype=torch.bool, device="cuda")
        net.cache = {}
        reference = pi3_inference(net, images, "cuda:0")
        kept = pi3_inference(net, images, "cuda:0", keep=full)
        for a, b in zip(reference[:3], kept[:3]):
            self.assertTrue(torch.equal(a, b), float((a - b).abs().max()))

    def test_rebuild_and_query_all_kept(self):
        images, _ = frames(4, 1)
        full = torch.ones(3, (H // 14) * (W // 14), dtype=torch.bool, device="cuda")
        reference = run(images[:, :3], None, images[:, 3:], None)
        kept = run(images[:, :3], full, images[:, 3:], full[:1])
        pose = max(float((reference[0][1] - kept[0][1]).abs().max()),
                   float((reference[2][1] - kept[2][1]).abs().max()))
        cache = max(float((reference[1][i][k].float() - kept[1][i][k].float()).abs().max())
                    for i in reference[1] for k in ("k", "v"))
        print(f"ALL-KEPT max |pose diff| {pose:.2e}, max |cache diff| {cache:.2e}", flush=True)
        self.assertEqual(reference[1].keys(), kept[1].keys())
        self.assertLess(pose, 1e-2)

    def test_partial_mask(self):
        images, mask = frames(4, 2)
        keep = patch_keep(mask)
        net = model()
        rebuild, cache, tracked = run(images[:, :3], keep[:3], images[:, 3:], keep[3:])
        kept = int(keep[:3].sum()) + 3 * net.patch_start_idx
        for entry in cache.values():
            self.assertEqual(entry["k"].shape[2], kept)
        pixel = keep.reshape(4, 37, 1, 37, 1).expand(4, 37, 14, 37, 14).reshape(4, H, W)
        _, poses, conf = rebuild[:3]
        local_points = rebuild[4]
        self.assertTrue(torch.isfinite(poses).all() and torch.isfinite(tracked[1]).all())
        self.assertTrue((local_points[0][~pixel[:3]] == 0).all())
        self.assertTrue((conf[0][~pixel[:3]] == 0).all())
        self.assertTrue((conf[0][pixel[:3]] > 0).all())
        print(f"PARTIAL kept {keep.float().mean():.3f} of patches; cache tokens {kept}", flush=True)

    def test_partial_mask_is_cheaper(self):
        images, mask = frames(8, 3)
        keep = patch_keep(mask)
        spent = {}
        # The first pass warms kernels for both shapes; the second is measured.
        for name, k in 2 * (("full", None), ("kept", keep)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model().cache = {}
            pi3_inference(model(), images, "cuda:0", store_cache=True, keep=k)
            end.record()
            torch.cuda.synchronize()
            spent[name] = (start.elapsed_time(end), torch.cuda.max_memory_allocated())
        print(f"8-FRAME REBUILD full {spent['full'][0]:.0f} ms {spent['full'][1] / 2**30:.2f} GiB; "
              f"kept {keep.float().mean():.3f}: {spent['kept'][0]:.0f} ms {spent['kept'][1] / 2**30:.2f} GiB",
              flush=True)
        self.assertLess(spent["kept"][1], spent["full"][1])


if __name__ == "__main__":
    unittest.main()
