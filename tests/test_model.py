import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from kvalloc.model import KVModel, ModelConfig  # noqa: E402


class TestModelConfig(unittest.TestCase):
    def test_owner_maps(self):
        self.assertEqual(ModelConfig(128, kv_layers=4).kv_owner, (0, 1, 2, 3))
        self.assertEqual(ModelConfig(128, kv_layers=2).kv_owner, (0, 0, 2, 2))
        self.assertEqual(ModelConfig(128, kv_layers=1).kv_owner, (0, 0, 0, 0))

    def test_invalid_configs_raise(self):
        with self.assertRaises(ValueError):
            ModelConfig(128, n_kv_heads=3)     # 3 does not divide 8
        with self.assertRaises(ValueError):
            ModelConfig(128, kv_layers=3)      # 3 does not divide 4

    def test_rel_bytes_ladder(self):
        self.assertAlmostEqual(ModelConfig(128).rel_bytes, 1.0)
        self.assertAlmostEqual(ModelConfig(128, n_kv_heads=2, kv_layers=2).rel_bytes, 4 / 32)
        self.assertAlmostEqual(ModelConfig(128, n_kv_heads=1, kv_layers=1).rel_bytes, 1 / 32)


class TestKVModel(unittest.TestCase):
    def test_param_count_decreases_with_sharing(self):
        full = KVModel(ModelConfig(128)).n_params()
        gqa = KVModel(ModelConfig(128, n_kv_heads=2)).n_params()
        cla = KVModel(ModelConfig(128, kv_layers=2)).n_params()
        both = KVModel(ModelConfig(128, n_kv_heads=1, kv_layers=1)).n_params()
        self.assertGreater(full, gqa)
        self.assertGreater(full, cla)
        self.assertGreater(gqa, both)
        self.assertGreater(cla, both)

    def test_nonowner_layers_have_no_kv_weights(self):
        m = KVModel(ModelConfig(128, kv_layers=2))
        for i, blk in enumerate(m.blocks):
            if i in (0, 2):
                self.assertTrue(hasattr(blk, "wk"))
            else:
                self.assertFalse(hasattr(blk, "wk"))
                self.assertFalse(hasattr(blk, "wv"))

    def test_shared_layers_consume_identical_kv(self):
        torch.manual_seed(0)
        m = KVModel(ModelConfig(64, kv_layers=1))
        x = torch.randint(0, 8192, (2, 16))
        m(x, debug_kv=True)
        # single owner layer 0; store holds exactly one entry
        self.assertEqual(set(m._debug_kv.keys()), {0})

    def test_forward_shapes_all_doses(self):
        for n_kv in (8, 2, 1):
            for kvl in (4, 2, 1):
                m = KVModel(ModelConfig(64, n_kv_heads=n_kv, kv_layers=kvl))
                out = m(torch.randint(0, 8192, (2, 32)))
                self.assertEqual(out.shape, (2, 32, 8192))
                self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
