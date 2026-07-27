"""Localized Predictive Model V3.1 milestone, Phase 12-14 -- model
comparison for GENERALIZATION (not just aggregate PR-AUC), calibration
under topology distribution shift, and the deterministic-current-state
baseline repeated under the exact same topology holdouts.

For each of the 4 topology-holdout folds:
  - the holdout's own 3-family train scenario population is further
    split 85/15 (scenario-level, fresh split, seed=SEED+1) into a
    sub-train (for fitting) and sub-val (for fitting an isotonic
    calibrator) -- both entirely from KNOWN/seen topology families.
  - XGBoost, HistGradientBoosting, and LogisticRegression are each
    fit on the sub-train and evaluated on the held-out (UNSEEN)
    family's real test rows -- Phase 12's question is which model
    generalizes best, not which wins the usual random-split race.
  - XGBoost's raw + isotonic-calibrated (calibrator fit on sub-val,
    entirely known-topology) probabilities are both evaluated for
    ECE/Brier on the unseen family's test rows -- Phase 13's "does
    calibration trained on known topology transfer" question.
  - DeterministicCurrentStateBaseline (no-op fit, no training data
    needed) is scored on the same held-out test rows for direct
    comparison -- Phase 14's "does ML still beat the deterministic
    baseline on UNSEEN topology" question.
"""
import gc
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from xgboost import XGBClassifier

import train_localized_predictive_model_v3 as v3
from predictive_model.baselines import DeterministicCurrentStateBaseline
from predictive_model.calibration import IsotonicCalibrator, expected_calibration_error
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios
from predictive_model.topology_holdout import apply_topology_holdout, assert_no_holdout_overlap, build_topology_holdout_splits
from predictive_model.tree_models import HistGradientBoostingModel
from sklearn.metrics import brier_score_loss

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "localized_predictive_model_v3_1"
OUTPUT_PATH = OUTPUT_DIR / "generalization_model_comparison_report.json"

SUBSPLIT_SEED = v3.SEED + 1


def _fit_xgboost(X, y, sample_weight):
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
        eval_metric="logloss", scale_pos_weight=1.0, random_state=v3.SEED, n_jobs=v3.N_JOBS,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, lambda X_: model.predict_proba(X_)[:, 1]


def _fit_histgb(X, y, sample_weight):
    model = HistGradientBoostingModel(seed=v3.SEED)
    model.fit(X, y, sample_weight=sample_weight)
    return model, model.predict_proba


def _fit_logreg(X, y, sample_weight):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    model = SkLogisticRegression(max_iter=2000, class_weight="balanced", random_state=v3.SEED)
    model.fit((X - mean) / std, y, sample_weight=sample_weight)
    return model, lambda X_: model.predict_proba((X_ - mean) / std)[:, 1]


def main() -> None:

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v3._watchdog_log_path = str(OUTPUT_DIR / "memory_watchdog_generalization_comparison.log")
    with open(v3._watchdog_log_path, "w", encoding="utf-8") as f:
        f.write(f"watchdog started {time.time():.1f}\n")
    threading.Thread(target=v3._memory_watchdog, daemon=True).start()

    with open(v3.SCENARIO_METADATA_PATH, "r", encoding="utf-8") as f:
        scenario_metadata = json.load(f)

    print("Loading dataset...")
    frame = v3._load_dataset_chunked(v3.CSV_PATH)
    full_trainable = v3._trainable_rows(frame)
    del frame
    gc.collect()

    holdout_splits = build_topology_holdout_splits(scenario_metadata)

    report = {}

    for holdout in holdout_splits:

        print(f"=== Held out: {holdout.held_out_family} ===")
        family_report = {}

        holdout_train_df, holdout_test_df = apply_topology_holdout(full_trainable, holdout)
        assert_no_holdout_overlap(holdout, holdout_train_df, holdout_test_df)

        # sub-split the holdout's OWN (known-topology) train scenarios
        # 85/15 for calibrator fitting -- entirely separate from the
        # primary V3 split/seed, a fresh partition of this fold's own
        # population.
        sub_split = split_scenarios(
            holdout_train_df["scenario_id"].astype(str).unique().tolist(),
            seed=SUBSPLIT_SEED, ratios=(0.85, 0.15, 0.0),
        )
        sub_train_ids = set(sub_split.train_scenario_ids)
        sub_val_ids = set(sub_split.val_scenario_ids)
        sub_train_df = holdout_train_df[holdout_train_df["scenario_id"].astype(str).isin(sub_train_ids)]
        sub_val_df = holdout_train_df[holdout_train_df["scenario_id"].astype(str).isin(sub_val_ids)]

        train_feat = v3.build_experimental_feature_matrix(sub_train_df)
        val_feat = v3.build_experimental_feature_matrix(sub_val_df)
        test_feat = v3.build_experimental_feature_matrix(holdout_test_df)

        class_weight_map = compute_class_weight_map(train_feat.y)
        sample_weight = sample_weights_from_class_weight(train_feat.y, class_weight_map)

        # --- Phase 12: model comparison for generalization ---
        model_results = {}
        xgb_test_prob = None
        xgb_val_prob = None

        for model_name, fit_fn in (
            ("xgboost", _fit_xgboost), ("histgradientboosting", _fit_histgb), ("logistic_regression", _fit_logreg),
        ):
            t0 = time.time()
            model, predict_fn = fit_fn(train_feat.X, train_feat.y, sample_weight)
            test_prob = predict_fn(test_feat.X)
            metrics = compute_metrics(test_feat.y, test_prob, threshold=0.5)
            model_results[model_name] = {"test_metrics": metrics.to_dict(), "seconds": time.time() - t0}
            print(f"  [{model_name}] PR-AUC={metrics.pr_auc:.4f} ROC-AUC={metrics.roc_auc:.4f} ({time.time()-t0:.1f}s)")

            if model_name == "xgboost":
                xgb_test_prob = test_prob
                xgb_val_prob = predict_fn(val_feat.X)

            del model
            gc.collect()

        family_report["model_comparison"] = model_results
        best_generalizing = max(model_results.items(), key=lambda kv: kv[1]["test_metrics"]["pr_auc"])[0]
        family_report["best_generalizing_model"] = best_generalizing

        # --- Phase 13: calibration under shift (XGBoost, the winning
        # production architecture) ---
        before_ece = expected_calibration_error(test_feat.y, xgb_test_prob)
        before_brier = float(brier_score_loss(test_feat.y, xgb_test_prob))

        calibrator = IsotonicCalibrator().fit(xgb_val_prob, val_feat.y)  # fit on KNOWN-topology val only
        calibrated_test_prob = calibrator.transform(xgb_test_prob)
        after_ece = expected_calibration_error(test_feat.y, calibrated_test_prob)
        after_brier = float(brier_score_loss(test_feat.y, calibrated_test_prob))

        family_report["calibration_under_shift"] = {
            "before_calibration": {"ece": before_ece, "brier": before_brier},
            "after_isotonic_fit_on_known_topology_val": {"ece": after_ece, "brier": after_brier},
            "calibration_improves_under_shift": after_ece < before_ece,
        }
        print(f"  calibration: ECE {before_ece:.4f} -> {after_ece:.4f} (known-topology-fit calibrator)")

        # --- Phase 14: deterministic baseline under the SAME holdout ---
        det_model = DeterministicCurrentStateBaseline(test_feat.feature_names)
        det_prob = det_model.predict_proba(test_feat.X)
        det_metrics = compute_metrics(test_feat.y, det_prob, threshold=0.5)
        family_report["deterministic_baseline"] = {"test_metrics": det_metrics.to_dict()}
        xgb_pr_auc = model_results["xgboost"]["test_metrics"]["pr_auc"]
        family_report["ml_vs_deterministic_on_unseen_topology"] = {
            "xgboost_pr_auc": xgb_pr_auc,
            "deterministic_pr_auc": det_metrics.pr_auc,
            "ml_beats_deterministic": xgb_pr_auc > det_metrics.pr_auc,
            "relative_lift": (xgb_pr_auc / det_metrics.pr_auc) if det_metrics.pr_auc > 0 else None,
        }
        print(f"  deterministic baseline PR-AUC={det_metrics.pr_auc:.4f} "
              f"(xgboost {xgb_pr_auc:.4f}, beats={family_report['ml_vs_deterministic_on_unseen_topology']['ml_beats_deterministic']})")

        report[holdout.held_out_family] = family_report

        del (holdout_train_df, holdout_test_df, sub_train_df, sub_val_df, train_feat, val_feat, test_feat,
             xgb_test_prob, xgb_val_prob, calibrator, calibrated_test_prob, det_prob)
        gc.collect()

    report["total_wall_seconds"] = time.time() - t_start

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Done in {time.time() - t_start:.1f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
