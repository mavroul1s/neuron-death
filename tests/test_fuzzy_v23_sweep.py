"""Fast integrity and phase-order checks for the V2.3 GPU entry point."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_fuzzy_v23_sweep as sweep


class V23SweepTests(unittest.TestCase):
    def test_frozen_real_grid_contains_only_twelve_targets_and_twelve_yokes(self):
        root = Path(__file__).resolve().parents[1]
        targets, yokes = sweep.load_sweep(root)
        self.assertEqual((len(targets), len(yokes)), (12, 12))
        self.assertEqual({config["seed"] for _, config in targets}, {15, 16, 17})
        self.assertEqual(
            {(config["recycling"]["learning_degree"]["degree_threshold"],
              config["recycling"]["learning_degree"]["saliency_full_ratio"])
             for _, config in targets},
            sweep.EXPECTED_VARIANTS,
        )
        for _, config in targets:
            learning = config["recycling"]["learning_degree"]
            self.assertEqual(learning["patience"], 1)
            self.assertEqual(learning["cooldown_steps"], 500)

    def test_yokes_cannot_launch_before_all_targets_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "mnist.npz").touch()
            targets = [(Path("target.json"), {"run_id": "target"})] * 12
            yokes = [(Path("yoke.json"), {"run_id": "yoke"})] * 12
            commands = []
            with patch.object(sweep, "load_sweep", return_value=(targets, yokes)), \
                    patch.object(sweep, "_run_command", side_effect=lambda command, *_: commands.append(command)), \
                    patch.object(sweep, "check_complete", side_effect=RuntimeError("target incomplete")), \
                    patch.object(sweep, "archive_results"):
                with self.assertRaisesRegex(RuntimeError, "target incomplete"):
                    sweep.run_sweep(root, data, root / "output")
            launches = [command for command in commands if "scripts/launch_pair.py" in command]
            self.assertEqual(len(launches), 1)
            self.assertEqual(launches[0][2], "target.json")


if __name__ == "__main__":
    unittest.main()
