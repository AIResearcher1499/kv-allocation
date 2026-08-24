import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from kvalloc.mqar import (  # noqa: E402
    FILLER, IGNORE, N_KEYS, VALUE_LO, VOCAB, build_batch, is_valid_cell,
    query_accuracy,
)


class TestMQAR(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(0)

    def test_invalid_cell_raises(self):
        self.assertFalse(is_valid_cell(512, 256))
        with self.assertRaises(ValueError):
            build_batch(2, 512, 256, self.gen)

    def test_layout_and_mapping(self):
        b, L, n = 4, 256, 32
        x, y = build_batch(b, L, n, self.gen)
        self.assertEqual(x.shape, (b, L))
        keys = x[:, 0:2 * n:2]
        vals = x[:, 1:2 * n:2]
        # keys distinct per row, in key range; values in value range
        for r in range(b):
            self.assertEqual(len(set(keys[r].tolist())), n)
        self.assertTrue(((keys >= 1) & (keys <= N_KEYS)).all())
        self.assertTrue(((vals >= VALUE_LO) & (vals < VOCAB)).all())
        # filler gap is FILLER
        qs = L - 2 * n
        self.assertTrue((x[:, 2 * n:qs] == FILLER).all())
        # targets only at query positions
        mask = y != IGNORE
        expect = torch.zeros_like(mask)
        expect[:, qs::2] = True
        self.assertTrue((mask == expect).all())
        # each query's target equals the value paired with that key
        for r in range(b):
            kv = dict(zip(keys[r].tolist(), vals[r].tolist()))
            q_keys = x[r, qs::2].tolist()
            q_tgts = y[r, qs::2].tolist()
            # teacher-forced value token sits right after the query key
            q_next = x[r, qs + 1::2].tolist()
            self.assertEqual(sorted(q_keys), sorted(kv))
            for qk, qt, qn in zip(q_keys, q_tgts, q_next):
                self.assertEqual(qt, kv[qk])
                self.assertEqual(qn, kv[qk])

    def test_query_accuracy_oracle_and_chance(self):
        b, L, n = 2, 128, 16
        x, y = build_batch(b, L, n, self.gen)
        oracle = torch.zeros(b, L, VOCAB)
        oracle.scatter_(2, torch.clamp(y, min=0).unsqueeze(-1), 1.0)
        self.assertAlmostEqual(query_accuracy(oracle, y), 1.0)
        wrong = torch.zeros(b, L, VOCAB)
        wrong[:, :, FILLER] = 1.0
        self.assertAlmostEqual(query_accuracy(wrong, y), 0.0)


if __name__ == "__main__":
    unittest.main()
