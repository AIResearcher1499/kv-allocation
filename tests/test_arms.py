import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kvalloc.arms import ARMS, FUNGIBILITY_PAIRS, arm  # noqa: E402


class TestArms(unittest.TestCase):
    def test_fungibility_pairs_collide_exactly(self):
        for a_name, b_name in FUNGIBILITY_PAIRS:
            self.assertEqual(arm(a_name).rel_bytes, arm(b_name).rel_bytes)
            self.assertEqual(
                arm(a_name).kv_bytes_per_token_bf16,
                arm(b_name).kv_bytes_per_token_bf16,
            )

    def test_ladder_matches_prereg_table(self):
        expected = {
            "mha": 1.0,
            "gqa4": 0.25,
            "mqa": 1 / 12,
            "cla2": 0.5,
            "cla4": 0.25,
            "cla12": 1 / 12,
            "gqa4_cla2": 0.125,
            "gqa2_cla3": 24 / 144,
        }
        for a in ARMS:
            self.assertAlmostEqual(a.rel_bytes, expected[a.name], places=10)

    def test_names_unique(self):
        names = [a.name for a in ARMS]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
