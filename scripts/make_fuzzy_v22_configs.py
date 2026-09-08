"""Generate and freeze the targeted V2.2 controller sweep before its first run."""

from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/fuzzy_v22_dev"
PLAN_PATH = ROOT / "configs/fuzzy_v22_dev_plan.json"
SEEDS = (15, 16, 17)
VARIANTS = (
    {"id": "t0p55_p2_c1000", "degree_threshold": 0.55, "patience": 2, "cooldown_steps": 1000,
     "counterfactual_mean_resets": 9321},
    {"id": "t0p60_p2_c1000", "degree_threshold": 0.60, "patience": 2, "cooldown_steps": 1000,
     "counterfactual_mean_resets": 10623},
    {"id": "t0p60_p2_c500", "degree_threshold": 0.60, "patience": 2, "cooldown_steps": 500,
     "counterfactual_mean_resets": 13596},
    {"id": "t0p60_p1_c500", "degree_threshold": 0.60, "patience": 1, "cooldown_steps": 500,
     "counterfactual_mean_resets": 18520},
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_config() -> dict:
    tree = ast.parse((ROOT / "src/config.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "DEFAULT_CONFIG":
            return ast.literal_eval(node.value)
    raise RuntimeError("DEFAULT_CONFIG not found")


def deep_merge(base: dict, override: dict) -> dict:
    value = deepcopy(base)
    for key, item in override.items():
        value[key] = deep_merge(value[key], item) if isinstance(item, dict) and isinstance(value.get(key), dict) \
            else deepcopy(item)
    return value


def resolved_hash(raw: dict, defaults: dict) -> str:
    config = deep_merge(defaults, raw)
    config["data"]["seed"] = int(config["seed"])
    config["data"]["n_probe"] = int(config["probe"]["n_probe"])
    config["data"].pop("root", None)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    if PLAN_PATH.exists() or CONFIG_DIR.exists():
        raise FileExistsError("V2.2 plan/config directory already exists; frozen artifacts are never overwritten")
    base_target = json.loads((ROOT / "configs/fuzzy_v21_dev/fuzzy_v21_dev_fuzzy_v2_d0p5_lr0p1_s15.json").read_text())
    base_random = json.loads((ROOT / "configs/fuzzy_v21_dev/fuzzy_v21_dev_yoked_random_d0p5_lr0p1_s15.json").read_text())
    defaults = default_config()
    CONFIG_DIR.mkdir(parents=True)
    manifest = []
    targeted_ids, yoked_ids = [], []
    for variant in VARIANTS:
        for seed in SEEDS:
            target_id = f"fuzzy_v22_dev_target_{variant['id']}_lr0p1_s{seed}"
            random_id = f"fuzzy_v22_dev_yoked_{variant['id']}_lr0p1_s{seed}"
            target = deepcopy(base_target)
            target.update(run_id=target_id, seed=seed,
                          notes="Pre-run V2.2 controller sweep; see configs/fuzzy_v22_dev_plan.json.")
            target["recycling"]["learning_degree"].update(
                degree_threshold=variant["degree_threshold"], patience=variant["patience"],
                cooldown_steps=variant["cooldown_steps"])
            random = deepcopy(base_random)
            random.update(run_id=random_id, seed=seed,
                          notes="Pre-run exact-yoked control for V2.2; see configs/fuzzy_v22_dev_plan.json.")
            random["recycling"]["learning_degree"]["yoked_from_run_id"] = target_id
            for kind, config, run_id in (("fuzzy_v2", target, target_id),
                                         ("fuzzy_v2_yoked_random", random, random_id)):
                path = CONFIG_DIR / f"{run_id}.json"
                write_json(path, config)
                manifest.append({
                    "run_id": run_id, "kind": kind, "variant": variant["id"], "seed": seed,
                    "degree_threshold": variant["degree_threshold"], "patience": variant["patience"],
                    "cooldown_steps": variant["cooldown_steps"],
                    "config_path": path.relative_to(ROOT).as_posix(),
                    "config_hash": resolved_hash(config, defaults), "config_file_sha256": sha(path),
                    "yoked_from_run_id": target_id if kind == "fuzzy_v2_yoked_random" else None,
                })
            targeted_ids.append(target_id)
            yoked_ids.append(random_id)

    attached = ROOT / "remote_runs/user_results_v21_20260908"
    plan = {
        "status": "FROZEN_BEFORE_FIRST_V22_RUN",
        "frozen_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "Supervisor-authorized exploratory method development, separate from configs/analysis_plan.json.",
        "objective": "Find a temporal-fuzzy setting whose late online accuracy exceeds ReDo, ReGraMa and SNR while retaining an exact-dose selection advantage.",
        "evidence_before_freeze": {
            "v21_analysis_sha256": sha(attached / "attached_v21_analysis.json"),
            "controller_diagnosis_sha256": sha(attached / "v21_controller_diagnosis.json"),
            "counterfactual_dose_sha256": sha(attached / "v21_counterfactual_dose.json"),
            "v21_best": {"variant": "threshold 0.50, patience 2, cooldown 1000",
                         "late_accuracy_pct": 90.58813596491228, "mean_resets": 3673,
                         "selection_advantage_pp": 0.5031578947368387},
            "reused_comparator_points_pct": {"none": 87.520, "redo_tau0.1": 92.346,
                                               "regrama_tau0.01": 92.293, "snr_eta0.08": 92.624},
            "diagnosis": "A global score cliff just above 0.50 suppresses layer-1 recycling. Threshold 0.55 is the smallest change that crosses it. Threshold 0.60 plus controlled persistence/cooldown changes spans approximately 9k-19k resets on fixed V2.1 trajectories.",
            "counterfactual_limit": "Dose projections hold logged V2.1 degree trajectories fixed. New resets change subsequent trajectories, so they select a safe sweep range rather than predict outcomes.",
        },
        "setting": {"dataset": "online permuted MNIST", "n_tasks": 200, "batch_size": 128,
                    "optimizer": "SGD momentum 0.9", "learning_rate": 0.1,
                    "hidden_dims": [500, 500, 500], "seeds": list(SEEDS)},
        "variants": list(VARIANTS),
        "unchanged_fields": {"ewma_beta": 0.9, "task_grace_steps": 100, "monitor_every": 100,
                             "warmup_steps": 1000, "max_reset_fraction": 0.075,
                             "activity_full": 0.1, "saliency_quantile": 0.75,
                             "scale_quantile": 0.75, "update_full_ratio": 0.1,
                             "saliency_full_ratio": 0.1,
                             "score": "V2 TOPSIS-like process health with saliency importance guard",
                             "reset": "unchanged incoming reinitialization, outgoing zeroing and optimizer-state reset"},
        "new_run_count": 24,
        "run_manifest": manifest,
        "execution": {"targeted_run_ids": targeted_ids, "yoked_run_ids": yoked_ids,
                      "stage_order": "All twelve targeted runs complete before any of the twelve exact-yoked random controls start.",
                      "gpu_mode": "Two independent runs, one per T4", "session_budget_hours": 10.0,
                      "hard_limit_hours": 12, "checkpoint_every_tasks": 10,
                      "reuse_baselines": True, "rerun_existing_baselines": False,
                      "dataset": "nmavros/neuron-death-code", "visibility": "private"},
        "primary_outcome": {"column": "online_accuracy", "probe_point": "task_end",
                            "task_window_zero_indexed_inclusive": [150, 199],
                            "estimator": "IQM over the 3x50 seed-by-task matrix"},
        "bootstrap": {"unit": "whole paired seed trajectory", "resample_tasks": False,
                      "n_bootstrap": 10000, "confidence": 0.95},
        "planned_comparisons": {"each_target": ["paired exact-yoked random", "V2.1 threshold 0.50",
                                                  "none", "ReDo tau=0.1", "ReGraMa tau=0.01", "SNR eta=0.08"],
                                "report_all_variants": True},
        "decision": {"development_winner": "Highest target late-window IQM; ties broken by fewer resets.",
                     "accuracy_goal": "Target point estimate above SNR 92.624%, followed by a held-out paired-seed confirmation before any SOTA claim.",
                     "selection_goal": "Paired target-minus-yoked CI lower bound above zero.",
                     "no_posthoc_changes": "No added variants, seed changes or in-run parameter adjustment after freeze."},
        "integrity": {"original_frozen_plan_path": "configs/analysis_plan.json",
                      "original_frozen_plan_sha256": sha(ROOT / "configs/analysis_plan.json"),
                      "prior_v21_plan_sha256": sha(ROOT / "configs/fuzzy_v21_dev_plan.json"),
                      "core_source_sha256": {name: sha(ROOT / name) for name in (
                          "src/temporal_fuzzy.py", "src/probes.py", "src/interventions.py", "src/train.py", "src/config.py")}},
    }
    write_json(PLAN_PATH, plan)
    print(f"wrote {len(manifest)} configs and {PLAN_PATH}")
    print("plan_sha256", sha(PLAN_PATH))


if __name__ == "__main__":
    main()
