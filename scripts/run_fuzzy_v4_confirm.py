"""Run the frozen held-out RA-SNR confirmation in two phases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from scripts.run_fuzzy_v21_sweep import PREFLIGHT_TESTS, _run_command, check_complete


SEEDS = {18, 19, 20, 21, 22}
PREFLIGHT = [
    *PREFLIGHT_TESTS,
    "tests/test_interventions.py::test_snr_recovery_boosts_only_recovering_outgoing_columns",
    "tests/test_interventions.py::test_standard_snr_has_no_recovery_gradient_side_effect",
    "tests/test_fuzzy_v4_confirm.py",
]


def load_sweep(source: Path):
    plan = json.loads((source / "configs/fuzzy_v4_confirm_plan.json").read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_FIRST_V4_CONFIRM_RUN":
        raise RuntimeError("V4 confirmation plan is not frozen")
    pinned = dict(plan["integrity"]["core_source_sha256"])
    pinned[plan["integrity"]["original_frozen_plan_path"]] = plan["integrity"]["original_frozen_plan_sha256"]
    pinned["configs/fuzzy_v4_dev_plan.json"] = plan["integrity"]["v4_development_plan_sha256"]
    pinned.update({row["config_path"]: row["config_file_sha256"] for row in plan["run_manifest"]})
    for name, expected in pinned.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source/config differs from frozen V4 confirmation: {name}")
    configs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(
        (source / "configs/fuzzy_v4_confirm").glob("*.json"))]
    if len(configs) != 20 or {int(c["seed"]) for _, c in configs} != SEEDS:
        raise RuntimeError("Expected four arms on held-out seeds 18-22")
    arms = {row["run_id"]: row["arm"] for row in plan["run_manifest"]}
    if {c["run_id"] for _, c in configs} != set(arms):
        raise RuntimeError("Confirmation run identities differ from frozen manifest")
    primary = [pair for pair in configs if arms[pair[1]["run_id"]] in {"rasnr", "snr"}]
    comparators = [pair for pair in configs if arms[pair[1]["run_id"]] in {"redo", "regrama"}]
    primary_order = {rid: i for i, rid in enumerate(plan["execution"]["primary_run_ids"])}
    comparator_order = {rid: i for i, rid in enumerate(plan["execution"]["comparator_run_ids"])}
    primary.sort(key=lambda pair: primary_order[pair[1]["run_id"]])
    comparators.sort(key=lambda pair: comparator_order[pair[1]["run_id"]])
    if len(primary) != 10 or len(comparators) != 10:
        raise RuntimeError("Expected ten primary and ten comparator runs")
    return primary, comparators


def archive_results(source, runs, output, deadline):
    _run_command([sys.executable, "-c",
                  "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2])",
                  str(output / "fuzzy_v4_confirm_full_runs"), str(runs)], source, deadline)
    if any(runs.glob("*/config.json")):
        _run_command([sys.executable, "scripts/make_analysis_extract.py", "--runs-root", str(runs),
                      "--pattern", "fuzzy_v4_confirm_*", "--out", str(output / "fuzzy_v4_confirm_extract"),
                      "--with-c4", "--zip"], source, deadline)


def run_sweep(source: Path, data: Path, output: Path = Path("/kaggle/working")):
    started = time.monotonic()
    deadline = started + 10 * 3600
    training_deadline = deadline - 30 * 60
    runs = output / "fuzzy_v4_confirm_runs"
    runs.mkdir(parents=True, exist_ok=True)
    report, error = {"status": "running", "complete_runs": 0}, None
    try:
        primary, comparators = load_sweep(source)
        if not (data / "mnist.npz").is_file():
            raise FileNotFoundError("Attached runtime must contain cached MNIST")
        _run_command([sys.executable, "-c", "import torch; assert torch.cuda.device_count() == 2"],
                     source, min(training_deadline, time.monotonic() + 60))
        _run_command([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *PREFLIGHT],
                     source, min(training_deadline, time.monotonic() + 5 * 60))
        print("FUZZY_V4_CONFIRM_PREFLIGHT_PASSED", flush=True)
        for name, configs in (("PRIMARY_RASNR_SNR", primary), ("REDO_REGRAMA", comparators)):
            print(f"FUZZY_V4_CONFIRM_{name}_LAUNCHING", flush=True)
            _run_command([sys.executable, "scripts/launch_pair.py", *[str(p) for p, _ in configs],
                          "--gpus", "0,1", "--runs-root", str(runs), "--data-root", str(data),
                          "--budget-hours", str(max(0, training_deadline - time.monotonic()) / 3600)],
                         source, training_deadline)
            check_complete(configs, runs)
            report["complete_runs"] += len(configs)
            print(f"FUZZY_V4_CONFIRM_{name}_COMPLETE", flush=True)
        report["status"] = "complete"
    except BaseException as exc:
        error = exc
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        plan = source / "configs/fuzzy_v4_confirm_plan.json"
        if plan.exists():
            (runs / plan.name).write_bytes(plan.read_bytes())
        report["elapsed_seconds_before_archive"] = time.monotonic() - started
        (runs / "session_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        try:
            archive_results(source, runs, output, deadline)
            report["artifacts_preserved"] = True
        except BaseException as archive_error:
            report.update(artifacts_preserved=False,
                          archive_error=f"{type(archive_error).__name__}: {archive_error}")
            if error is None:
                error = archive_error
        report["status"] = "failed" if error else "complete"
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "fuzzy_v4_confirm_session_status.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if error:
        raise error
    print("FUZZY_V4_HELDOUT_CONFIRMATION_COMPLETE", flush=True)
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_sweep(root, Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data")
