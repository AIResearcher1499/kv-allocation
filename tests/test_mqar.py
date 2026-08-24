import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from kvalloc.mqar import (  # noqa: E402
    FILLER, IGNORE, KEY_HI, KEY_LO, VAL_HI, VAL_LO, VOCAB, build_examples,
    is_valid_cell, query_accuracy,
)


class TestMQAR(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(0)

    def test_invalid_cell_raises(self):
        self.assertFalse(is_valid_cell(512, 256))
        with self.assertRaises(ValueError):
            build_examples(2, 512, 256, self.gen)

    def test_zoology_layout_and_mapping(self):
        b, L, n = 8, 256, 32
        x, y = build_examples(b, L, n, self.gen)
        self.assertEqual(x.shape, (b, L))
        ctx = 2 * n
        keys = x[:, 0:ctx:2]
        vals = x[:, 1:ctx:2]
        for r in range(b):
            # keys and values each distinct within an example (Zoology sampling)
            self.assertEqual(len(set(keys[r].tolist())), n)
            self.assertEqual(len(set(vals[r].tolist())), n)
        self.assertTrue(((keys >= KEY_LO) & (keys < KEY_HI)).all())
        self.assertTrue(((vals >= VAL_LO) & (vals < VAL_HI)).all())
        for r in range(b):
            kv = dict(zip(keys[r].tolist(), vals[r].tolist()))
            q_positions = (y[r] != IGNORE).nonzero().flatten().tolist()
            self.assertEqual(len(q_positions), n)
            for p in q_positions:
                # label sits AT the query-key position (next-token shift)
                self.assertIn(x[r, p].item(), kv)
                self.assertEqual(y[r, p].item(), kv[x[r, p].item()])
                self.assertGreaterEqual(p, ctx)  # queries live in the tail
            # non-query tail positions are filler (random_non_queries=False)
            tail = x[r, ctx:]
            self.assertTrue(((tail == FILLER) | (tail >= KEY_LO)).all())

    def test_power_a_places_queries_near_context(self):
        b, L, n = 64, 512, 32
        x_near, y_near = build_examples(b, L, n, torch.Generator().manual_seed(1),
                                        power_a=0.01)
        x_uni, y_uni = build_examples(b, L, n, torch.Generator().manual_seed(1),
                                      power_a=1.0)
        mean_near = (y_near != IGNORE).nonzero()[:, 1].float().mean().item()
        mean_uni = (y_uni != IGNORE).nonzero()[:, 1].float().mean().item()
        self.assertLess(mean_near, mean_uni)  # 0.01 skews toward the KV block

    def test_query_accuracy_oracle(self):
        b, L, n = 2, 128, 16
        x, y = build_examples(b, L, n, self.gen)
        oracle = torch.zeros(b, L, VOCAB)
        oracle.scatter_(2, torch.clamp(y, min=0).unsqueeze(-1), 1.0)
        self.assertAlmostEqual(query_accuracy(oracle, y), 1.0)


if __name__ == "__main__":
    unittest.main()
