"""Fast integrity and phase-order checks for the V3 GPU entry point."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_fuzzy_v3_sweep as sweep


class V3SweepTests(unittest.TestCase):
    def test_frozen_real_grid_contains_nine_targets_and_nine_yokes(self):
        root = Path(__file__).resolve().parents[1]
        targets, yokes = sweep.load_sweep(root)
        self.assertEqual((len(targets), len(yokes)), (9, 9))
        self.assertEqual({config["seed"] for _, config in targets}, {15, 16, 17})
        self.assertEqual({config["recycling"]["learning_degree"]["replacement_rate"]
                          for _, config in targets}, sweep.EXPECTED_RATES)
        for _, config in targets:
            learning = config["recycling"]["learning_degree"]
            self.assertEqual(learning["monitor_every"], 100)
            self.assertEqual(learning["cooldown_steps"], 500)
            self.assertEqual(learning["task_grace_steps"], 100)

    def test_yokes_cannot_launch_before_all_targets_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "mnist.npz").touch()
            targets = [(Path("target.json"), {"run_id": "target"})] * 9
            yokes = [(Path("yoke.json"), {"run_id": "yoke"})] * 9
            commands = []
            with patch.object(sweep, "load_sweep", return_value=(targets, yokes)), \
                    patch.object(sweep, "_run_command", side_effect=lambda command, *_: commands.append(command)), \
                    patch.object(sweep, "check_complete", side_effect=RuntimeError("target incomplete")), \
                    patch.object(sweep, "archive_results"):
                with self.assertRaisesRegex(RuntimeError, "target incomplete"):
                    sweep.run_sweep(root, data, root / "output")
            launches = [command for command in commands if "scripts/launch_pair.py" in command]
            self.assertEqual(len(launches), 1)


if __name__ == "__main__":
    unittest.main()
