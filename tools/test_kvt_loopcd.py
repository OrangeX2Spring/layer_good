"""CAMP contracts for paired-depth readout and native-cache preservation."""
import unittest

import numpy as np
import torch

from kv_tracker.depth_contrast import DepthContrastQueries
from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32, pi3_inference


class DepthContrastContracts(unittest.TestCase):
    def test_zero_guidance_native_and_hidden_contrast(self):
        torch.manual_seed(17)
        model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained().eval())
        assert len(model.decoder) == 36
        decoder = model.decoder
        images = np.random.default_rng(17).integers(0, 256, (2, 266, 350, 3), dtype=np.uint8)
        with torch.inference_mode():
            pi3_inference(model, [images], 'cuda', store_cache=True)
            cache = {i: {k: v.clone() for k, v in row.items()} for i, row in model.cache.items()}
            pointers = {i: {k: v.data_ptr() for k, v in row.items()} for i, row in model.cache.items()}
            expected = pi3_inference(model, [images[:1]], 'cuda', cam_only=True, use_cache=True)
            actual = DepthContrastQueries(36).forward(model, images[0], 'cuda')
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            seen, counts = [], []
            def camera_input(module, args):
                seen.append(args[0].detach().clone())
            def block_call(module, args, output):
                counts.append(1)
            handles = [model.camera_decoder.register_forward_pre_hook(camera_input)]
            handles += [block.register_forward_hook(block_call) for block in model.decoder]
            try:
                for executor, depth in ((DepthContrastQueries(12), 12),
                                        (DepthContrastQueries(24), 24),
                                        (DepthContrastQueries(24, 12, .5), 24)):
                    counts.clear()
                    pose = executor.forward(model, images[0], 'cuda')
                    self.assertIs(model.decoder, decoder)
                    self.assertEqual(len(model.decoder), 36)
                    self.assertEqual(len(counts), depth)
                    self.assertTrue(torch.isfinite(pose).all())
                weak, strong, guided = seen
                torch.testing.assert_close(guided,
                    (strong.float() + .5 * (strong.float() - weak.float())).to(strong.dtype),
                    rtol=0, atol=0)
            finally:
                for handle in handles:
                    handle.remove()
            restored = DepthContrastQueries(36).forward(model, images[0], 'cuda')
            torch.testing.assert_close(restored, expected, rtol=1e-4, atol=1e-4)
            for i, row in cache.items():
                for key, tensor in row.items():
                    self.assertTrue(torch.equal(model.cache[i][key], tensor))
                    self.assertEqual(model.cache[i][key].data_ptr(), pointers[i][key])
        print('LOOPCD DEPTH CONTRAST CONTRACT OK', flush=True)


if __name__ == '__main__':
    unittest.main()
