"""Fast frozen-grid check for the held-out RA-SNR confirmation."""

import unittest
from pathlib import Path

from scripts import run_fuzzy_v4_confirm as sweep


class V4ConfirmTests(unittest.TestCase):
    def test_real_grid_has_four_arms_on_five_unseen_seeds(self):
        primary, comparators = sweep.load_sweep(Path(__file__).resolve().parents[1])
        configs = primary + comparators
        self.assertEqual((len(primary), len(comparators)), (10, 10))
        self.assertEqual({config["seed"] for _, config in configs}, sweep.SEEDS)
        self.assertEqual(len({config["run_id"] for _, config in configs}), 20)
        rasnr = [c for _, c in configs if c["recycling"]["kind"] == "snr_recovery"]
        self.assertEqual(len(rasnr), 5)
        for config in rasnr:
            self.assertEqual(config["recycling"]["recovery_boost"], 2.0)
            self.assertEqual(config["recycling"]["recovery_steps"], 100)


if __name__ == "__main__":
    unittest.main()
