"""Token-drop contract tests; run on the CAMP GPU before any token-drop tracking.

The patch mapping is checked on CPU. On the GPU, the real Pi3 checkpoint is run
through upstream ``Pi3.forward`` and through ``forward_kept`` with every patch
kept: a single-frame forward must match exactly, and a keyframe rebuild plus a
cached query must agree to within bf16 batch-shape noise. A partial mask must
cache only kept tokens and leave dropped pixels with no prediction.

A2: kept background selection, the log-mass bias (a representative with bias
log n equals n identical keys), all-background equivalence with and without the
bias path, and the attention probe's records.
"""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from kv_tracker import token_drop
from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32, pi3_inference
from kv_tracker.token_drop import background_keep, patch_keep
from pi3.models.layers.attention import FlashAttentionRope
from pi3.models.layers.block import BlockRope

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


def run(images, keep, query, query_keep, background=None, query_background=None):
    net = model()
    net.cache = {}
    rebuild = pi3_inference(net, images, "cuda:0", store_cache=True, keep=keep,
                            background=background)
    cache = {i: {k: t.clone() for k, t in entry.items()} for i, entry in net.cache.items()}
    tracked = pi3_inference(net, query, "cuda:0", use_cache=True, keep=query_keep,
                            background=query_background)
    return rebuild, cache, tracked


def configure(test, **options):
    token_drop.config.update({"mass": False, "probe": None, **options})
    test.addCleanup(token_drop.config.update, mass=False, probe=None)


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


class BackgroundTests(unittest.TestCase):
    def test_background_keep_selects_spread_background_only(self):
        keep = torch.zeros(2, 100, dtype=torch.bool)
        keep[0, 40:50] = True
        keep[1, :97] = True
        kept, background = background_keep(keep, 4)
        self.assertEqual(background.sum(dim=1).tolist(), [4, 3])  # frame 1 has 3 free
        self.assertFalse((background & keep).any())
        self.assertTrue(torch.equal(kept, keep | background))
        self.assertEqual(background[0].nonzero()[:, 0].tolist(), [0, 30, 69, 99])
        _, every = background_keep(keep, None)
        self.assertTrue(torch.equal(every, ~keep))

    def test_mass_bias_equals_duplicated_keys(self):
        # A representative with additive bias log n attends like n identical keys,
        # through the real BlockRope and the (1, 1, 1, L) mask shape forward_kept uses.
        torch.manual_seed(5)
        block = BlockRope(dim=16, num_heads=2, qk_norm=True,
                          attn_class=FlashAttentionRope).eval()
        queries, dup = torch.randn(1, 6, 16), torch.randn(1, 1, 16)
        with torch.no_grad():
            dense = block(torch.cat([queries] + [dup] * 7, dim=1))[:, :6]
            bias = torch.zeros(1, 1, 1, 7)
            bias[..., 6] = math.log(7)
            merged = block(torch.cat([queries, dup], dim=1), attn_mask=bias)[:, :6]
            unweighted = block(torch.cat([queries, dup], dim=1))[:, :6]
        torch.testing.assert_close(merged, dense, atol=1e-5, rtol=1e-5)
        self.assertGreater(float((unweighted - dense).abs().max()), 1e-3)


@unittest.skipUnless(torch.cuda.is_available(), "needs the CAMP GPU")
class ForwardTests(unittest.TestCase):
    def test_single_frame_all_kept_is_exact(self):
        images, _ = frames(1, 0)
        net = model()
        full = torch.ones(1, (H // 14) * (W // 14), dtype=torch.bool, device="cuda")
        net.cache = {}
        reference = pi3_inference(net, images, "cuda:0")
        kept = pi3_inference(net, images, "cuda:0", keep=full)
        # Bit-exact on the A5000 (jobs 25837, 25902). The RTX 4090 selects different
        # bf16 kernels for the two paths (jobs 25903/25904), so there the rebuild
        # test's contract applies: poses within 1e-2; points and confidence reported.
        gpu = torch.cuda.get_device_name()
        diffs = {name: float((a - b).abs().max())
                 for name, a, b in zip(("points", "poses", "conf"), reference[:3], kept[:3])}
        print(f"SINGLE-FRAME {gpu} max |diff| {diffs}", flush=True)
        if gpu == "NVIDIA RTX A5000":
            for a, b in zip(reference[:3], kept[:3]):
                self.assertTrue(torch.equal(a, b), float((a - b).abs().max()))
        else:
            self.assertLess(diffs["poses"], 1e-2)

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

    def test_all_background_kept_matches_upstream_with_and_without_bias(self):
        images, mask = frames(4, 4)
        keep, background = background_keep(patch_keep(mask), None)
        self.assertTrue(keep.all())
        reference = run(images[:, :3], None, images[:, 3:], None)
        for mass in (False, True):
            configure(self, mass=mass)
            kept = run(images[:, :3], keep[:3], images[:, 3:], keep[3:],
                       background[:3], background[3:])
            pose = max(float((reference[0][1] - kept[0][1]).abs().max()),
                       float((reference[2][1] - kept[2][1]).abs().max()))
            print(f"ALL-BACKGROUND mass={mass} max |pose diff| {pose:.2e}", flush=True)
            self.assertLess(pose, 1e-2)
            bias = model().kept_cache["bias"]
            self.assertTrue(bias is None if not mass else not bias.any())

    def test_partial_background_mass_caches_weighted_keys(self):
        images, mask = frames(4, 5)
        keep, background = background_keep(patch_keep(mask), 4)
        configure(self, mass=True)
        net = model()
        rebuild, cache, tracked = run(images[:, :3], keep[:3], images[:, 3:], keep[3:],
                                      background[:3], background[3:])
        tokens = int(keep[:3].sum()) + 3 * net.patch_start_idx
        for entry in cache.values():
            self.assertEqual(entry["k"].shape[2], tokens)
        bias = net.kept_cache["bias"].float().reshape(-1)
        labels = net.kept_cache["labels"]
        self.assertEqual(bias.shape, (tokens,))
        self.assertEqual(int((labels == token_drop.BACKGROUND).sum()), 12)
        expected = math.log((37 * 37 - int((keep[0] & ~background[0]).sum())) / 4)
        first = bias[:net.patch_start_idx + int(keep[0].sum())]
        self.assertAlmostEqual(float(first.max()), expected, delta=0.05)  # bf16
        self.assertTrue((bias[labels != token_drop.BACKGROUND] == 0).all())
        self.assertTrue(torch.isfinite(rebuild[1]).all() and torch.isfinite(tracked[1]).all())
        print(f"MASS k=4 bias {float(first.max()):.3f} (log weight {expected:.3f}); "
              f"cache tokens {tokens}", flush=True)

    def test_probe_records_every_global_layer(self):
        images, mask = frames(4, 6)
        keep, background = background_keep(patch_keep(mask), 16)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.jsonl"
            probe = token_drop.Probe(path)
            configure(self, mass=True, probe=probe)
            run(images[:, :3], keep[:3], images[:, 3:], keep[3:], background[:3], background[3:])
            probe.file.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
        net = model()
        layers = list(range(1, len(net.decoder), 2))
        self.assertEqual([r["layer"] for r in records if r["kind"] == "rebuild"], layers)
        self.assertEqual([r["layer"] for r in records if r["kind"] == "query"], layers)
        cached = int(keep[:3].sum()) + 3 * net.patch_start_idx
        for r in records:
            self.assertEqual(r["cached_keys"], cached if r["kind"] == "query" else 0)
            for queries in ("object_queries", "register_queries"):
                entry = r[queries]
                total = entry["register"] + entry["object"] + entry["background"]
                self.assertAlmostEqual(total, 1, delta=1e-3)
                self.assertTrue(all(d >= 1 for d in entry["top16_distance"]))
        print(f"PROBE {len(records)} records; layer 1 object->background "
              f"{records[0]['object_queries']['background']:.3f}", flush=True)

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
