"""Localized Predictive Model V3.1 milestone, Phase 9 -- holdout failure
analysis for the two topology-holdout families that fail (multi_exit_
wide, twin_stair_highrise -- PR-AUC roughly half of in-distribution in
V3's original leave-one-topology-family-out evaluation).

Refits the canonical production XGBoost config (same hyperparameters
V3 used) on the same 3-of-4-family holdout train set, then breaks down
false positives/negatives on the held-out family's test rows by
candidate type, occupancy range, queue range, alternative-route count,
flow rate, congestion trend, multi-bottleneck status, and lead-time
range -- looking for an engineering explanation for WHY unseen
topology fails, not just confirming that it does.
"""
import gc
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import train_localized_predictive_model_v3 as v3
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics
from predictive_model.topology_holdout import apply_topology_holdout, assert_no_holdout_overlap, build_topology_holdout_splits

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "localized_predictive_model_v3_1"
OUTPUT_PATH = OUTPUT_DIR / "holdout_failure_analysis_report.json"

FAILING_FAMILIES = ("multi_exit_wide", "twin_stair_highrise")

QUEUE_BUCKETS = ((0, 0), (1, 1), (2, 3), (4, 999))
FLOW_BUCKETS = ((0.0, 0.0), (0.01, 2.0), (2.01, 10.0), (10.01, 1e9))
ALT_ROUTE_BUCKETS = ((0, 0), (1, 1), (2, 999))
LEAD_TIME_BUCKETS = v3.LEAD_TIME_BUCKETS


def _fit_xgboost(X, y, sample_weight):
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
        eval_metric="logloss", scale_pos_weight=1.0, random_state=v3.SEED, n_jobs=v3.N_JOBS,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _bucket_label(value, buckets, fmt="{:g}-{:g}"):
    for lo, hi in buckets:
        if lo <= value <= hi:
            return fmt.format(lo, hi)
    return "other"


def _rate_by(analysis: pd.DataFrame, column: str) -> dict:
    report = {}
    for value, group in analysis.groupby(column, observed=True):
        n = len(group)
        report[str(value)] = {
            "n": int(n),
            "false_positive_rate": float(group["is_fp"].sum() / n) if n else None,
            "false_negative_rate": float(group["is_fn"].sum() / n) if n else None,
            "positive_rate": float(group["y_true"].mean()) if n else None,
        }
    return report


def main() -> None:

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v3._watchdog_log_path = str(OUTPUT_DIR / "memory_watchdog_holdout_failure.log")
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

    holdout_splits = {s.held_out_family: s for s in build_topology_holdout_splits(scenario_metadata)}

    report = {}

    for family in FAILING_FAMILIES:

        print(f"=== {family} ===")
        holdout = holdout_splits[family]

        holdout_train_df, holdout_test_df = apply_topology_holdout(full_trainable, holdout)
        assert_no_holdout_overlap(holdout, holdout_train_df, holdout_test_df)

        train_feat = v3.build_experimental_feature_matrix(holdout_train_df)
        test_feat = v3.build_experimental_feature_matrix(holdout_test_df)
        del holdout_train_df
        gc.collect()

        class_weight_map = compute_class_weight_map(train_feat.y)
        sample_weight = sample_weights_from_class_weight(train_feat.y, class_weight_map)
        model = _fit_xgboost(train_feat.X, train_feat.y, sample_weight)

        prob = model.predict_proba(test_feat.X)[:, 1]
        metrics = compute_metrics(test_feat.y, prob, threshold=0.5)
        y_pred = (prob >= 0.5).astype(int)

        analysis = holdout_test_df.reset_index(drop=True).copy()
        analysis["y_true"] = test_feat.y
        analysis["y_pred"] = y_pred
        analysis["y_prob"] = prob
        analysis["is_fp"] = (analysis["y_true"] == 0) & (analysis["y_pred"] == 1)
        analysis["is_fn"] = (analysis["y_true"] == 1) & (analysis["y_pred"] == 0)

        analysis["queue_bucket"] = analysis["candidate_queue_length"].apply(lambda v: _bucket_label(v, QUEUE_BUCKETS))
        analysis["flow_bucket"] = analysis["candidate_recent_flow_rate"].apply(lambda v: _bucket_label(v, FLOW_BUCKETS))
        analysis["alt_route_bucket"] = analysis["candidate_alternative_route_count"].apply(
            lambda v: _bucket_label(v, ALT_ROUTE_BUCKETS))

        # multi-bottleneck: reuse the same simultaneous-bottleneck-count
        # logic as V3's own error_analysis, computed from this family's
        # own full test frame (ground truth, not prediction-dependent).
        positive_rows = holdout_test_df[holdout_test_df["target"] == True]  # noqa: E712
        bottleneck_counts = positive_rows.groupby(["scenario_id", "observation_time"])["candidate_id"].nunique()
        analysis["simultaneous_bottleneck_count"] = analysis.apply(
            lambda row: int(bottleneck_counts.get((row["scenario_id"], row["observation_time"]), 0)), axis=1,
        )
        analysis["multiple_simultaneous_bottlenecks"] = analysis["simultaneous_bottleneck_count"] >= 2

        family_report = {
            "test_metrics": metrics.to_dict(),
            "overall_false_positive_rate": float(analysis["is_fp"].mean()),
            "overall_false_negative_rate": float(analysis["is_fn"].mean()),
            "by_candidate_type": _rate_by(analysis, "candidate_type"),
            "by_congestion_trend": _rate_by(analysis, "candidate_congestion_trend"),
            "by_queue_bucket": _rate_by(analysis, "queue_bucket"),
            "by_flow_bucket": _rate_by(analysis, "flow_bucket"),
            "by_alt_route_bucket": _rate_by(analysis, "alt_route_bucket"),
            "by_multiple_simultaneous_bottlenecks": _rate_by(analysis, "multiple_simultaneous_bottlenecks"),
        }

        # occupancy bucket, reusing the same canonical thresholds V3 used
        from predictive_dataset.label_analysis import occupancy_bucket
        analysis["occupancy_bucket"] = analysis["total_active_occupant_count"].apply(occupancy_bucket)
        family_report["by_occupancy_bucket"] = _rate_by(analysis, "occupancy_bucket")

        # lead-time bucket recall for positives only
        positive_test = analysis[analysis["y_true"] == 1]
        lead_time_report = {}
        for lo, hi in LEAD_TIME_BUCKETS:
            bucket = positive_test[(positive_test["lead_time_seconds_v2"] > lo) & (positive_test["lead_time_seconds_v2"] <= hi)]
            n = len(bucket)
            lead_time_report[f"{lo}-{hi}s"] = {
                "n": int(n), "recall": float((bucket["y_pred"] == 1).mean()) if n else None,
            }
        family_report["by_lead_time_bucket"] = lead_time_report

        report[family] = family_report
        print(f"  PR-AUC={metrics.pr_auc:.4f} FP_rate={family_report['overall_false_positive_rate']:.4f} "
              f"FN_rate={family_report['overall_false_negative_rate']:.4f}")
        print(json.dumps(family_report["by_candidate_type"], indent=1))

        del train_feat, test_feat, model, prob, analysis, holdout_test_df
        gc.collect()

    report["total_wall_seconds"] = time.time() - t_start

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Done in {time.time() - t_start:.1f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
