"""Freeze the held-out RA-SNR confirmation before its first run."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

from make_fuzzy_v22_configs import ROOT, default_config, resolved_hash, sha, write_json


CONFIG_DIR = ROOT / "configs/fuzzy_v4_confirm"
PLAN_PATH = ROOT / "configs/fuzzy_v4_confirm_plan.json"
SEEDS = (18, 19, 20, 21, 22)
BASES = {
    "rasnr": ROOT / "configs/fuzzy_v4_dev/fuzzy_v4_dev_rasnr_b2p0_h100_lr0p1_s15.json",
    "snr": ROOT / "configs/fuzzy_v2_dev/fuzzy_v2_dev_snr_eta0p08_lr0p1_s15.json",
    "redo": ROOT / "configs/fuzzy_v2_dev/fuzzy_v2_dev_redo_t0p1_lr0p1_s15.json",
    "regrama": ROOT / "configs/fuzzy_v2_dev/fuzzy_v2_dev_regrama_t0p01_lr0p1_s15.json",
}


def main() -> None:
    if PLAN_PATH.exists() or CONFIG_DIR.exists():
        raise FileExistsError("V4 confirmation artifacts already exist; never overwrite them")
    defaults = default_config()
    bases = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in BASES.items()}
    CONFIG_DIR.mkdir(parents=True)
    manifest, primary_ids, comparator_ids = [], [], []
    for seed in SEEDS:
        for arm in ("rasnr", "snr", "redo", "regrama"):
            config = deepcopy(bases[arm])
            run_id = f"fuzzy_v4_confirm_{arm}_lr0p1_s{seed}"
            config.update(
                run_id=run_id,
                seed=seed,
                notes="Held-out RA-SNR confirmation; see configs/fuzzy_v4_confirm_plan.json.",
            )
            path = CONFIG_DIR / f"{run_id}.json"
            write_json(path, config)
            manifest.append(
                {
                    "run_id": run_id,
                    "arm": arm,
                    "seed": seed,
                    "config_path": path.relative_to(ROOT).as_posix(),
                    "config_hash": resolved_hash(config, defaults),
                    "config_file_sha256": sha(path),
                }
            )
            (primary_ids if arm in {"rasnr", "snr"} else comparator_ids).append(run_id)

    analysis = ROOT / "remote_runs/neuron-death-fuzzy-v1-v4rasnr0909-a36f1ab2/v4_analysis.json"
    plan = {
        "status": "FROZEN_BEFORE_FIRST_V4_CONFIRM_RUN",
        "frozen_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method_name": "Recovery-Accelerated SNR (RA-SNR)",
        "scope": "Held-out confirmation separate from configs/analysis_plan.json.",
        "trigger": {
            "v4_development_analysis_sha256": sha(analysis),
            "locked_winner": {"recovery_boost": 2.0, "recovery_steps": 100},
            "development_accuracy_pct": 92.74598684210528,
            "development_vs_snr_pp": {
                "point": 0.12212719298249075,
                "ci_lo": 0.08370614035089208,
                "ci_hi": 0.1530482456140847,
            },
        },
        "heldout_seeds": list(SEEDS),
        "arms": {
            "rasnr": "locked boost=2.0, recovery_steps=100, SNR eta=0.08",
            "snr": "eta=0.08",
            "redo": "tau=0.1",
            "regrama": "tau=0.01",
        },
        "new_run_count": 20,
        "run_manifest": manifest,
        "execution": {
            "primary_run_ids": primary_ids,
            "comparator_run_ids": comparator_ids,
            "phase_order": "Complete the ten paired RA-SNR/SNR runs before ReDo/ReGraMa.",
            "gpu_mode": "Two independent runs, one per T4",
            "session_budget_hours": 10.0,
            "hard_limit_hours": 12,
            "checkpoint_every_tasks": 10,
            "dataset": "nmavros/neuron-death-code",
            "visibility": "private",
        },
        "primary_outcome": {
            "column": "online_accuracy",
            "probe_point": "task_end",
            "task_window_zero_indexed_inclusive": [150, 199],
            "estimator": "IQM over the 5x50 held-out seed-by-task matrix",
        },
        "bootstrap": {
            "unit": "whole paired seed trajectory",
            "resample_tasks": False,
            "n_bootstrap": 20_000,
            "confidence": 0.95,
        },
        "comparisons": {
            "primary": "RA-SNR minus SNR on seeds 18-22",
            "secondary": ["RA-SNR minus ReDo", "RA-SNR minus ReGraMa"],
            "supportive": "Combined development plus held-out estimate over seeds 15-22",
        },
        "decision": {
            "confirm": "Held-out RA-SNR-minus-SNR CI lower bound above zero.",
            "best_method": "RA-SNR point estimate above all three comparators on held-out seeds.",
            "locked_parameters": "No parameter or seed changes after this freeze.",
        },
        "integrity": {
            "original_frozen_plan_path": "configs/analysis_plan.json",
            "original_frozen_plan_sha256": sha(ROOT / "configs/analysis_plan.json"),
            "v4_development_plan_sha256": sha(ROOT / "configs/fuzzy_v4_dev_plan.json"),
            "core_source_sha256": {
                name: sha(ROOT / name)
                for name in (
                    "src/probes.py",
                    "src/interventions.py",
                    "src/train.py",
                    "src/models.py",
                    "src/config.py",
                )
            },
        },
    }
    write_json(PLAN_PATH, plan)
    print(f"wrote {len(manifest)} configs and {PLAN_PATH}")
    print("plan_sha256", sha(PLAN_PATH))


if __name__ == "__main__":
    main()
