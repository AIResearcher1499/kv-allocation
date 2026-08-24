import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict, fields, replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from kvalloc.a0 import (  # noqa: E402
    BASE_SEED, DOSES, JITTER_SEED_OFFSET, RunConfig, analyse,
    build_calibration_plan, build_plan, eval_cells, load_done_keys, train_one,
)


def _rc(**kw):
    base = dict(dim=64, n_kv=8, kv_layers=4, seq_len=64, num_pairs=8,
                lr=1e-3, seed=BASE_SEED)
    base.update(kw)
    return RunConfig(**base)


class TestPlanAndResume(unittest.TestCase):
    def test_resume_key_covers_every_field(self):
        rc = _rc()
        bumps = {int: lambda v: v + 1, float: lambda v: v * 2}
        for f in fields(RunConfig):
            old = getattr(rc, f.name)
            mutated = replace(rc, **{f.name: bumps[type(old)](old)})
            self.assertNotEqual(rc.key(), mutated.key(),
                                f"field {f.name} missing from resume key")

    def test_plan_counts(self):
        plan = build_plan()
        # 2 dims x 9 doses x 11 valid cells x 4 LRs + 4 jitter repeats
        self.assertEqual(len(eval_cells()), 11)
        self.assertEqual(len(plan), 2 * 9 * 11 * 4 + 4)
        self.assertEqual(len({rc.key() for rc in plan}), len(plan))
        # jitter repeats carry the offset seed
        self.assertEqual(sum(rc.seed == BASE_SEED + JITTER_SEED_OFFSET for rc in plan), 4)

    def test_calibration_plan_is_anchor_dose_only(self):
        plan = build_calibration_plan(steps=100)
        self.assertEqual(len(plan), 4)
        for rc in plan:
            # prereg hygiene: only the top dose may be trained pre-gate
            self.assertEqual((rc.n_kv, rc.kv_layers), (8, 4))

    def test_merge_semantics_skip_done(self):
        plan = build_plan(steps=1)[:3]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a0.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"config": asdict(plan[0]), "evals": {}}) + "\n")
            done = load_done_keys(path)
            todo = [rc for rc in plan if rc.key() not in done]
            self.assertEqual(len(todo), 2)


class TestTinyTraining(unittest.TestCase):
    def test_loss_starts_at_chance_and_decreases(self):
        torch.manual_seed(0)
        rc = _rc(steps=200, lr=3e-3, tokens_per_batch=1024, eval_seqs=8)
        _, first_loss, last_loss = train_one(rc, "cpu")
        # correct init: step-0 loss must sit at the ln(8192)=9.01 chance floor
        # (the pre-init-fix model started at ~106 — see docs/a0-diagnostics)
        self.assertLess(abs(first_loss - 9.01), 0.15)
        self.assertLess(last_loss, first_loss - 0.4)


class TestAnalyse(unittest.TestCase):
    def _rec(self, n_kv, kv_layers, seed, acc, dim=128, l=512, n=32, lr=1e-4):
        cfg = asdict(_rc(dim=dim, n_kv=n_kv, kv_layers=kv_layers,
                         seq_len=l, num_pairs=n, lr=lr, seed=seed))
        evals = {f"L{el}_N{en}": acc for (el, en) in eval_cells()}
        return {"config": cfg, "evals": evals}

    def test_gate_math_on_synthetic_records(self):
        hi = max(DOSES, key=lambda d: d[0] * d[1])
        lo = min(DOSES, key=lambda d: d[0] * d[1])
        recs = []
        # two DISTINCT task cells (same cell at two dims must NOT count as 2)
        for (l, n) in ((512, 32), (2048, 64)):
            recs.append(self._rec(hi[0], hi[1], BASE_SEED, 0.90, l=l, n=n))
            recs.append(self._rec(lo[0], lo[1], BASE_SEED, 0.40, l=l, n=n))
        # jitter pair on its designated cell (dose 2x2, dim 128, L2048 N128)
        recs.append(self._rec(2, 2, BASE_SEED, 0.70, l=2048, n=128))
        recs.append(self._rec(2, 2, BASE_SEED + JITTER_SEED_OFFSET, 0.72,
                              l=2048, n=128))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a0.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            s = analyse(path)
        self.assertAlmostEqual(s["jitter"], 0.02, places=6)
        self.assertEqual(s["cells_passing"], 2)
        self.assertTrue(s["gate_a0"])
        self.assertFalse(s["all_saturated"])
        # range 0.50 on every populated cell, all passing
        self.assertTrue(all(abs(r["range"] - 0.5) < 1e-9 for r in s["rows"]))

    def test_same_cell_two_dims_counts_once(self):
        hi = max(DOSES, key=lambda d: d[0] * d[1])
        lo = min(DOSES, key=lambda d: d[0] * d[1])
        recs = []
        for dim in (128, 256):
            recs.append(self._rec(hi[0], hi[1], BASE_SEED, 0.90, dim=dim))
            recs.append(self._rec(lo[0], lo[1], BASE_SEED, 0.40, dim=dim))
        recs.append(self._rec(2, 2, BASE_SEED, 0.70, l=2048, n=128))
        recs.append(self._rec(2, 2, BASE_SEED + JITTER_SEED_OFFSET, 0.72,
                              l=2048, n=128))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a0.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            s = analyse(path)
        self.assertEqual(s["cells_passing"], 1)
        self.assertFalse(s["gate_a0"])


if __name__ == "__main__":
    unittest.main()
