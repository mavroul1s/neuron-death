"""Generate and freeze the V2.3 process/saliency dose sweep before its first run."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

from make_fuzzy_v22_configs import (
    ROOT,
    default_config,
    resolved_hash,
    sha,
    write_json,
)


CONFIG_DIR = ROOT / "configs/fuzzy_v23_dev"
PLAN_PATH = ROOT / "configs/fuzzy_v23_dev_plan.json"
SEEDS = (15, 16, 17)
VARIANTS = (
    {"id": "t0p80_es0p80", "degree_threshold": 0.80, "saliency_full_ratio": 0.1000000000,
     "effective_saliency_threshold": 0.80, "counterfactual_mean_resets": 8559},
    {"id": "t0p90_es0p80", "degree_threshold": 0.90, "saliency_full_ratio": 0.0888888889,
     "effective_saliency_threshold": 0.80, "counterfactual_mean_resets": 9274},
    {"id": "t0p99_es0p80", "degree_threshold": 0.99, "saliency_full_ratio": 0.0808080808,
     "effective_saliency_threshold": 0.80, "counterfactual_mean_resets": 12700},
    {"id": "t0p99_es0p90", "degree_threshold": 0.99, "saliency_full_ratio": 0.0909090909,
     "effective_saliency_threshold": 0.90, "counterfactual_mean_resets": 17000},
)


def main() -> None:
    if PLAN_PATH.exists() or CONFIG_DIR.exists():
        raise FileExistsError("V2.3 plan/config directory already exists; frozen artifacts are never overwritten")
    base_target = json.loads((
        ROOT / "configs/fuzzy_v22_dev/fuzzy_v22_dev_target_t0p60_p1_c500_lr0p1_s15.json"
    ).read_text(encoding="utf-8"))
    base_random = json.loads((
        ROOT / "configs/fuzzy_v22_dev/fuzzy_v22_dev_yoked_t0p60_p1_c500_lr0p1_s15.json"
    ).read_text(encoding="utf-8"))
    defaults = default_config()
    CONFIG_DIR.mkdir(parents=True)
    manifest = []
    targeted_ids, yoked_ids = [], []
    for variant in VARIANTS:
        for seed in SEEDS:
            target_id = f"fuzzy_v23_dev_target_{variant['id']}_lr0p1_s{seed}"
            random_id = f"fuzzy_v23_dev_yoked_{variant['id']}_lr0p1_s{seed}"
            target = deepcopy(base_target)
            target.update(run_id=target_id, seed=seed,
                          notes="Pre-run V2.3 process/saliency dose sweep; see configs/fuzzy_v23_dev_plan.json.")
            target["recycling"]["learning_degree"].update(
                degree_threshold=variant["degree_threshold"],
                saliency_full_ratio=variant["saliency_full_ratio"],
            )
            random = deepcopy(base_random)
            random.update(run_id=random_id, seed=seed,
                          notes="Pre-run exact-yoked control for V2.3; see configs/fuzzy_v23_dev_plan.json.")
            random["recycling"]["learning_degree"]["yoked_from_run_id"] = target_id
            for kind, config, run_id in (("fuzzy_v2", target, target_id),
                                         ("fuzzy_v2_yoked_random", random, random_id)):
                path = CONFIG_DIR / f"{run_id}.json"
                write_json(path, config)
                manifest.append({
                    "run_id": run_id,
                    "kind": kind,
                    "variant": variant["id"],
                    "seed": seed,
                    "degree_threshold": variant["degree_threshold"],
                    "saliency_full_ratio": variant["saliency_full_ratio"],
                    "effective_saliency_threshold": variant["effective_saliency_threshold"],
                    "patience": 1,
                    "cooldown_steps": 500,
                    "config_path": path.relative_to(ROOT).as_posix(),
                    "config_hash": resolved_hash(config, defaults),
                    "config_file_sha256": sha(path),
                    "yoked_from_run_id": target_id if kind == "fuzzy_v2_yoked_random" else None,
                })
            targeted_ids.append(target_id)
            yoked_ids.append(random_id)

    evidence = ROOT / "remote_runs/neuron-death-fuzzy-v1-v22sota0908-78924178"
    plan = {
        "status": "FROZEN_BEFORE_FIRST_V23_RUN",
        "frozen_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "Supervisor-authorized exploratory method development, separate from configs/analysis_plan.json.",
        "objective": "Exceed ReDo, ReGraMa and SNR while retaining an exact-dose temporal-fuzzy selection advantage.",
        "evidence_before_freeze": {
            "v22_analysis_sha256": sha(evidence / "v22_analysis.json"),
            "controller_diagnosis_sha256": sha(evidence / "v22_controller_diagnosis.json"),
            "decoupled_counterfactual_sha256": sha(evidence / "v22_decoupled_counterfactual.json"),
            "quota_counterfactual_sha256": sha(evidence / "v22_quota_counterfactual.json"),
            "v22_best": {"variant": "threshold 0.60, patience 1, cooldown 500",
                         "late_accuracy_pct": 90.96256578947369, "mean_resets": 4889,
                         "selection_advantage_pp": 0.3876535087719235},
            "reused_comparator_points_pct": {"none": 87.520, "redo_tau0.1": 92.346,
                                               "regrama_tau0.01": 92.293, "snr_eta0.08": 92.624},
            "diagnosis": "All V2.2 candidates were selected and cap saturation was zero. The shared degree threshold jointly limits process health and the saliency guard, leaving only 4.1k-4.9k resets. V2.3 raises the process threshold while controlling the equivalent saliency cutoff separately through saliency_full_ratio.",
            "parameter_mapping": "With degree=max(process_health,saliency_health), the equivalent saliency cutoff under the V2.2 normalization is degree_threshold*saliency_full_ratio/0.1.",
            "counterfactual_limit": "Dose projections hold V2.2 trajectories fixed; they select a range and do not predict accuracy.",
        },
        "setting": {"dataset": "online permuted MNIST", "n_tasks": 200, "batch_size": 128,
                    "optimizer": "SGD momentum 0.9", "learning_rate": 0.1,
                    "hidden_dims": [500, 500, 500], "seeds": list(SEEDS)},
        "variants": list(VARIANTS),
        "unchanged_fields": {"ewma_beta": 0.9, "patience": 1, "cooldown_steps": 500,
                             "task_grace_steps": 100, "monitor_every": 100,
                             "warmup_steps": 1000, "max_reset_fraction": 0.075,
                             "activity_full": 0.1, "saliency_quantile": 0.75,
                             "scale_quantile": 0.75, "update_full_ratio": 0.1,
                             "score": "V2 TOPSIS-like process health with saliency importance guard",
                             "reset": "unchanged incoming reinitialization, outgoing zeroing and optimizer-state reset"},
        "new_run_count": 24,
        "run_manifest": manifest,
        "execution": {"targeted_run_ids": targeted_ids, "yoked_run_ids": yoked_ids,
                      "stage_order": "All twelve targeted runs complete before any exact-yoked control starts.",
                      "gpu_mode": "Two independent runs, one per T4", "session_budget_hours": 10.0,
                      "hard_limit_hours": 12, "checkpoint_every_tasks": 10,
                      "reuse_baselines": True, "rerun_existing_baselines": False,
                      "dataset": "nmavros/neuron-death-code", "visibility": "private"},
        "primary_outcome": {"column": "online_accuracy", "probe_point": "task_end",
                            "task_window_zero_indexed_inclusive": [150, 199],
                            "estimator": "IQM over the 3x50 seed-by-task matrix"},
        "bootstrap": {"unit": "whole paired seed trajectory", "resample_tasks": False,
                      "n_bootstrap": 10000, "confidence": 0.95},
        "planned_comparisons": {"each_target": ["paired exact-yoked random", "V2.2 winner",
                                                  "none", "ReDo tau=0.1", "ReGraMa tau=0.01", "SNR eta=0.08"],
                                "report_all_variants": True},
        "decision": {"development_winner": "Highest target late-window IQM; ties broken by fewer resets.",
                     "accuracy_goal": "Point estimate above SNR 92.624%, followed by held-out paired-seed confirmation before a SOTA claim.",
                     "selection_goal": "Paired target-minus-yoked CI lower bound above zero.",
                     "no_posthoc_changes": "No added variants, seed changes or in-run parameter adjustment after freeze."},
        "integrity": {"original_frozen_plan_path": "configs/analysis_plan.json",
                      "original_frozen_plan_sha256": sha(ROOT / "configs/analysis_plan.json"),
                      "prior_v22_plan_sha256": sha(ROOT / "configs/fuzzy_v22_dev_plan.json"),
                      "core_source_sha256": {name: sha(ROOT / name) for name in (
                          "src/temporal_fuzzy.py", "src/probes.py", "src/interventions.py", "src/train.py", "src/config.py")}},
    }
    write_json(PLAN_PATH, plan)
    print(f"wrote {len(manifest)} configs and {PLAN_PATH}")
    print("plan_sha256", sha(PLAN_PATH))


if __name__ == "__main__":
    main()
