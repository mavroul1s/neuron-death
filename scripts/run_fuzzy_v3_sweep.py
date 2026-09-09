"""Run the frozen BT-FR replacement-rate sweep in targeted/yoked phases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from scripts.run_fuzzy_v21_sweep import PREFLIGHT_TESTS, _run_command, check_complete, verify_yokes


EXPECTED_RATES = {0.00030, 0.00040, 0.00048}
V3_PREFLIGHT_TESTS = [*PREFLIGHT_TESTS, "tests/test_fuzzy_v3_sweep.py"]


def load_sweep(source: Path):
    plan_path = source / "configs/fuzzy_v3_dev_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_FIRST_V3_RUN":
        raise RuntimeError("V3 plan must be frozen before running")
    pinned = dict(plan["integrity"]["core_source_sha256"])
    pinned[plan["integrity"]["original_frozen_plan_path"]] = plan["integrity"]["original_frozen_plan_sha256"]
    pinned["configs/fuzzy_v23_dev_plan.json"] = plan["integrity"]["prior_v23_plan_sha256"]
    pinned.update({row["config_path"]: row["config_file_sha256"] for row in plan["run_manifest"]})
    for name, expected in pinned.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source/config differs from frozen V3 plan: {name}")
    configs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(
        (source / "configs/fuzzy_v3_dev").glob("*.json"))]
    targets = [pair for pair in configs if pair[1]["recycling"]["kind"] == "fuzzy_budget"]
    yokes = [pair for pair in configs if pair[1]["recycling"]["kind"] == "fuzzy_budget_yoked_random"]
    if len(configs) != 18 or len(targets) != 9 or len(yokes) != 9:
        raise RuntimeError("Expected nine BT-FR targets and nine exact-yoked controls")
    if {config["run_id"] for _, config in configs} != {row["run_id"] for row in plan["run_manifest"]}:
        raise RuntimeError("V3 config identities differ from frozen manifest")
    grid = {(float(config["recycling"]["learning_degree"]["replacement_rate"]), int(config["seed"]))
            for _, config in targets}
    if grid != {(rate, seed) for rate in EXPECTED_RATES for seed in (15, 16, 17)}:
        raise RuntimeError("V3 rate/seed grid is incomplete or duplicated")
    target_configs = {config["run_id"]: config for _, config in targets}
    sources = []
    for _, config in yokes:
        source_id = config["recycling"]["learning_degree"]["yoked_from_run_id"]
        if source_id not in target_configs or config["seed"] != target_configs[source_id]["seed"]:
            raise RuntimeError("Invalid V3 target/yoke pairing")
        sources.append(source_id)
    if len(set(sources)) != 9:
        raise RuntimeError("Each V3 target must have exactly one yoke")
    target_order = {run_id: i for i, run_id in enumerate(plan["execution"]["targeted_run_ids"])}
    yoke_order = {run_id: i for i, run_id in enumerate(plan["execution"]["yoked_run_ids"])}
    targets.sort(key=lambda pair: target_order[pair[1]["run_id"]])
    yokes.sort(key=lambda pair: yoke_order[pair[1]["run_id"]])
    return targets, yokes


def archive_results(source: Path, runs: Path, output: Path, deadline: float):
    _run_command([sys.executable, "-c",
                  "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2])",
                  str(output / "fuzzy_v3_dev_full_runs"), str(runs)], source, deadline)
    if any(runs.glob("*/config.json")):
        _run_command([sys.executable, "scripts/make_analysis_extract.py", "--runs-root", str(runs),
                      "--pattern", "fuzzy_v3_dev_*", "--out", str(output / "fuzzy_v3_dev_extract"),
                      "--with-c4", "--zip"], source, deadline)


def run_sweep(source: Path, data: Path, output: Path = Path("/kaggle/working")):
    started = time.monotonic()
    deadline = started + 10 * 3600
    training_deadline = deadline - 30 * 60
    runs = output / "fuzzy_v3_dev_runs"
    runs.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "complete_runs": 0, "yoked_validation": []}
    error = None
    try:
        targets, yokes = load_sweep(source)
        if not (data / "mnist.npz").is_file():
            raise FileNotFoundError("Attached runtime must contain cached MNIST")
        _run_command([sys.executable, "-c", "import torch; assert torch.cuda.device_count() == 2"],
                     source, min(training_deadline, time.monotonic() + 60))
        _run_command([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                      *V3_PREFLIGHT_TESTS], source, min(training_deadline, time.monotonic() + 5 * 60))
        print("FUZZY_V3_PREFLIGHT_PASSED", flush=True)
        for phase, configs in (("TARGETED", targets), ("YOKED_RANDOM", yokes)):
            print(f"FUZZY_V3_{phase}_PHASE_LAUNCHING", flush=True)
            _run_command([sys.executable, "scripts/launch_pair.py", *[str(path) for path, _ in configs],
                          "--gpus", "0,1", "--runs-root", str(runs), "--data-root", str(data),
                          "--budget-hours", str(max(0, training_deadline - time.monotonic()) / 3600)],
                         source, training_deadline)
            check_complete(configs, runs)
            report["complete_runs"] += len(configs)
            print(f"FUZZY_V3_{phase}_PHASE_COMPLETE", flush=True)
        report["yoked_validation"] = verify_yokes(yokes, runs)
        report["status"] = "complete"
    except BaseException as exc:
        error = exc
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        plan = source / "configs/fuzzy_v3_dev_plan.json"
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
        (output / "fuzzy_v3_dev_session_status.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if error:
        raise error
    print("FUZZY_V3_DEVELOPMENT_COMPLETE", flush=True)
    return report


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1]
    run_sweep(source, Path(sys.argv[1]) if len(sys.argv) > 1 else source / "data")
