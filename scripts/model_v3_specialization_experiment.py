"""Localized Predictive Model V3 milestone, Task #41 -- model
specialization experiment (unified vs Door/Exit/Stair-specific models).

One controlled experiment, run once: same scenario split, same feature
schema, same XGBoost architecture/hyperparameters as the winning
unified model in train_localized_predictive_model_v3.py. The ONLY
difference is that each specialized model is trained on train rows of
ONE candidate_type only, then evaluated against that same type's test
rows -- directly comparable to the unified model's existing
test_metrics_by_candidate_type in training_report_v3.json.

Per the milestone charter: prefer the unified model unless a
specialized model shows a clear, robust benefit. This script does not
retrain the unified model -- it reuses the already-written
training_report_v3.json for that side of the comparison.
"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import train_localized_predictive_model_v3 as v3
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios

OUTPUT_PATH = v3.OUTPUT_DIR / "model_specialization_experiment.json"


def main() -> None:
    t_start = time.time()

    print("Loading Target V2 relabeled dataset (chunked)...")
    frame = v3._load_dataset_chunked(v3.CSV_PATH)

    split = split_scenarios(frame["scenario_id"].astype(str).unique().tolist(), seed=v3.SEED)
    train_df, val_df, test_df = apply_split(frame, split)
    assert_no_scenario_overlap(split, train_df, val_df, test_df)
    del frame
    gc.collect()

    train_trainable = v3._trainable_rows(train_df)
    test_trainable = v3._trainable_rows(test_df)
    del train_df, val_df, test_df
    gc.collect()

    with open(v3.OUTPUT_DIR / "training_report_v3.json", "r", encoding="utf-8") as f:
        unified_report = json.load(f)
    unified_by_type = unified_report["models"]["xgboost"]["test_metrics_by_candidate_type"]

    results = {}

    for candidate_type in ("Door", "Exit", "Stair"):

        print(f"Training specialized model for candidate_type={candidate_type} ...")
        t0 = time.time()

        type_train = train_trainable[train_trainable["candidate_type"] == candidate_type]
        type_test = test_trainable[test_trainable["candidate_type"] == candidate_type]

        train_feat = v3.build_experimental_feature_matrix(type_train)
        test_feat = v3.build_experimental_feature_matrix(type_test)
        del type_train, type_test
        gc.collect()

        model = v3._fresh_model("xgboost")
        class_weight_map = compute_class_weight_map(train_feat.y)
        sample_weight = sample_weights_from_class_weight(train_feat.y, class_weight_map)
        model.fit(train_feat.X, train_feat.y, sample_weight=sample_weight)

        test_prob = model.predict_proba(test_feat.X)
        metrics = compute_metrics(test_feat.y, test_prob, threshold=0.5)

        specialized_pr_auc = metrics.pr_auc
        unified_pr_auc = unified_by_type[candidate_type]["pr_auc"]

        results[candidate_type] = {
            "train_row_count": int(len(train_feat.y)),
            "train_positive_rate": float(train_feat.y.mean()),
            "test_row_count": int(len(test_feat.y)),
            "specialized_test_metrics": metrics.to_dict(),
            "unified_test_metrics": unified_by_type[candidate_type],
            "specialized_pr_auc": specialized_pr_auc,
            "unified_pr_auc": unified_pr_auc,
            "pr_auc_delta_specialized_minus_unified": specialized_pr_auc - unified_pr_auc,
            "train_seconds": time.time() - t0,
        }

        print(f"    {candidate_type}: specialized PR-AUC={specialized_pr_auc:.4f} "
              f"vs unified PR-AUC={unified_pr_auc:.4f} "
              f"(delta={specialized_pr_auc - unified_pr_auc:+.4f}) ({time.time() - t0:.1f}s)")

        del model, train_feat, test_feat, test_prob
        gc.collect()

    mean_delta = float(np.mean([r["pr_auc_delta_specialized_minus_unified"] for r in results.values()]))
    any_clear_win = any(r["pr_auc_delta_specialized_minus_unified"] > 0.03 for r in results.values())
    verdict = (
        "SPECIALIZATION_SHOWS_CLEAR_BENEFIT" if (mean_delta > 0.03 and any_clear_win)
        else "PREFER_UNIFIED_MODEL"
    )

    report = {
        "experiment": "model_specialization_v3",
        "seed": v3.SEED,
        "architecture": "xgboost (same hyperparameters as unified winning model)",
        "by_candidate_type": results,
        "mean_pr_auc_delta_specialized_minus_unified": mean_delta,
        "verdict": verdict,
        "verdict_rule": "SPECIALIZATION_SHOWS_CLEAR_BENEFIT requires mean PR-AUC delta > 0.03 AND at least one type with delta > 0.03; otherwise PREFER_UNIFIED_MODEL (charter default)",
        "total_wall_seconds": time.time() - t_start,
    }

    v3.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"VERDICT: {verdict} (mean delta={mean_delta:+.4f})")
    print(f"Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
