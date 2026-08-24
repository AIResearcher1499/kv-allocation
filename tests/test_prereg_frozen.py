"""Guards against silent edits to the frozen pre-registration.

Rule (docs/prereg-g0a.md header): once any data/a0_*.jsonl or data/a1_*.jsonl
exists, the prereg content hash must not change. Editing the prereg therefore
requires consciously updating PREREG_SHA256 here — which shows up in the diff.
"""

import glob
import hashlib
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "docs", "prereg-g0a.md")

# sha256 of docs/prereg-g0a.md at freeze time (2026-08-24).
PREREG_SHA256 = "f2a6f570c2af6685d62a0adad0b219d09d559447f9f556131a946b3116aaadae"


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestPreregFrozen(unittest.TestCase):
    def test_prereg_hash_matches(self):
        self.assertEqual(
            _sha256(PREREG),
            PREREG_SHA256,
            "docs/prereg-g0a.md changed. If data/a0_*|a1_* files exist, this "
            "edit is prohibited (append a dated amendment instead and only "
            "then update PREREG_SHA256 in the same commit).",
        )

    def test_thresholds_present_verbatim(self):
        # Anchor phrases for the frozen gate thresholds; renaming/removing any
        # of these is a gate change and must fail loudly.
        with open(PREREG, encoding="utf-8") as f:
            text = f.read()
        for phrase in (
            "range ≥ 15 accuracy points",
            "≥ 3× the pooled 2-seed spread",
            "NOT required for GO",
            "No rank-correlation statistics in any gate",
        ):
            self.assertIn(phrase, text)

    def test_data_files_imply_hash_lock(self):
        produced = glob.glob(os.path.join(ROOT, "data", "a0_*.jsonl")) + glob.glob(
            os.path.join(ROOT, "data", "a1_*.jsonl")
        )
        if produced:
            self.assertNotEqual(
                PREREG_SHA256, "__FILL_AT_COMMIT__",
                "Gate data exists but the prereg hash was never locked.",
            )


if __name__ == "__main__":
    unittest.main()
