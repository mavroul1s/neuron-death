"""Fast integrity checks for the RA-SNR V4 GPU entry point."""

import unittest
from pathlib import Path

from scripts import run_fuzzy_v4_sweep as sweep


class V4SweepTests(unittest.TestCase):
    def test_frozen_real_grid_contains_only_twelve_rasnr_runs(self):
        root = Path(__file__).resolve().parents[1]
        configs = sweep.load_sweep(root)
        self.assertEqual(len(configs), 12)
        self.assertEqual({config["seed"] for _, config in configs}, {15, 16, 17})
        self.assertEqual(
            {
                (
                    config["recycling"]["recovery_boost"],
                    config["recycling"]["recovery_steps"],
                )
                for _, config in configs
            },
            sweep.EXPECTED_VARIANTS,
        )
        for _, config in configs:
            recycling = config["recycling"]
            self.assertEqual(recycling["kind"], "snr_recovery")
            self.assertEqual(recycling["snr_eta"], 0.08)
            self.assertEqual(recycling["snr_tau_max"], 20_000)
            self.assertTrue(recycling["zero_outgoing_after_event"])


if __name__ == "__main__":
    unittest.main()
