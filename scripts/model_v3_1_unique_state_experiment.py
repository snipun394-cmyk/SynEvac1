"""Localized Predictive Model V3.1 milestone, Phase 6 -- unique-state
training experiments. Phase 5 (shuffle battery variant E) proved the
canonical V3 model's skill is ~fully recoverable from duplicate-feature-
group membership alone -- this phase asks the natural follow-up: what
happens to generalization (not just aggregate PR-AUC) if duplicate-
state dominance is REDUCED during training?

Three experimental TRAIN representations, canonical dataset never
overwritten, scenario-level split preserved exactly (same seed/
convention as V3):

  Canonical: full train split, unchanged (V3's own reproduction).
  A: one row per exact-duplicate feature vector (deterministic pick --
     first row by (scenario_id, observation_time, candidate_id) order,
     not truly random, for exact reproducibility).
  B: one row per (feature vector, scenario_id) pair -- collapses only
     WITHIN-scenario temporal duplication (Phase 3/4 found this is the
     minority, ~4.5%, of duplication), leaving cross-scenario repeats
     (~95.5%) intact.
  C: full rows kept, but each row's sample_weight is additionally
     divided by its duplicate-group size (on top of the usual class-
     imbalance weight) -- no row is dropped, but no duplicate group can
     dominate the loss just by raw repetition count.

Evaluated on the SAME real (non-deduplicated) val/test splits in every
case, since that is what any real deployment would actually see.
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
from predictive_model.calibration import evaluate_calibration_methods
from predictive_model.error_analysis import build_error_analysis
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics, metrics_by_group
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios
from predictive_model.topology_holdout import apply_topology_holdout, assert_no_holdout_overlap, build_topology_holdout_splits

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "localized_predictive_model_v3_1"
OUTPUT_PATH = OUTPUT_DIR / "unique_state_experiment_report.json"

LEAD_TIME_BUCKETS = v3.LEAD_TIME_BUCKETS


def _group_ids_for_matrix(X: np.ndarray) -> np.ndarray:
    view = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, inverse = np.unique(view, return_inverse=True)
    return inverse.reshape(-1)


def _fit_xgboost(X, y, sample_weight):
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
        eval_metric="logloss", scale_pos_weight=1.0, random_state=v3.SEED, n_jobs=v3.N_JOBS,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _make_variant_train(variant: str, train_trainable: pd.DataFrame) -> pd.DataFrame:
    """Returns (possibly-reduced) train_trainable plus a '_extra_weight'
    column (1.0 for canonical/A/B, 1/group_size for C) -- canonical/A/B
    use plain class-imbalance weighting, C additionally multiplies by
    this column."""

    if variant == "canonical":
        out = train_trainable.copy()
        out["_extra_weight"] = 1.0
        return out

    feat = v3.build_experimental_feature_matrix(train_trainable)
    group_id = _group_ids_for_matrix(feat.X)
    del feat
    gc.collect()

    work = train_trainable.reset_index(drop=True).copy()
    work["_group_id"] = group_id

    if variant == "A":
        # deterministic pick: first row per group, in (scenario_id,
        # observation_time, candidate_id) order.
        work = work.sort_values(["scenario_id", "observation_time", "candidate_id"])
        deduped = work.drop_duplicates(subset="_group_id", keep="first")
        deduped = deduped.drop(columns=["_group_id"])
        deduped["_extra_weight"] = 1.0
        return deduped

    if variant == "B":
        work = work.sort_values(["scenario_id", "observation_time", "candidate_id"])
        deduped = work.drop_duplicates(subset=["_group_id", "scenario_id"], keep="first")
        deduped = deduped.drop(columns=["_group_id"])
        deduped["_extra_weight"] = 1.0
        return deduped

    if variant == "C":
        group_sizes = work.groupby("_group_id")["_group_id"].transform("size")
        work["_extra_weight"] = 1.0 / group_sizes
        work = work.drop(columns=["_group_id"])
        return work

    raise ValueError(variant)


def _evaluate(model, test_feat, test_trainable, val_feat, scenario_metadata, full_horizon_test_frame, threshold=0.5):

    test_prob = model.predict_proba(test_feat.X)[:, 1]
    val_prob = model.predict_proba(val_feat.X)[:, 1]
    metrics = compute_metrics(test_feat.y, test_prob, threshold=threshold)
    by_type = metrics_by_group(test_feat.y, test_prob, test_feat.candidate_types, threshold=threshold)

    calib = evaluate_calibration_methods(val_feat.y, val_prob, test_feat.y, test_prob)

    y_pred = (test_prob >= threshold).astype(int)
    error_analysis = build_error_analysis(
        test_trainable, test_feat.y, y_pred, test_prob, scenario_metadata, full_horizon_test_frame,
    )

    test_analysis = test_trainable.reset_index(drop=True).copy()
    test_analysis["y_true"] = test_feat.y
    test_analysis["y_prob"] = test_prob
    test_analysis["y_pred"] = y_pred
    positive_test = test_analysis[test_analysis["y_true"] == 1]
    lead_time_report = {}
    for lo, hi in LEAD_TIME_BUCKETS:
        bucket = positive_test[(positive_test["lead_time_seconds_v2"] > lo) & (positive_test["lead_time_seconds_v2"] <= hi)]
        n = len(bucket)
        lead_time_report[f"{lo}-{hi}s"] = {
            "n": int(n),
            "recall": float((bucket["y_pred"] == 1).mean()) if n else None,
        }

    return {
        "test_metrics": metrics.to_dict(),
        "test_metrics_by_candidate_type": by_type,
        "calibration": calib.to_dict(),
        "error_analysis": {
            "overall_false_positive_rate": error_analysis["overall_false_positive_rate"],
            "overall_false_negative_rate": error_analysis["overall_false_negative_rate"],
            "by_candidate_type": error_analysis["by_candidate_type"],
            "by_occupancy_bucket": error_analysis["by_occupancy_bucket"],
            "by_multiple_simultaneous_bottlenecks": error_analysis["by_multiple_simultaneous_bottlenecks"],
        },
        "lead_time_analysis": lead_time_report,
    }


def main() -> None:

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v3._watchdog_log_path = str(OUTPUT_DIR / "memory_watchdog_unique_state.log")
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

    # full_trainable (ALL 2500 scenarios, ALL 4 topology families) is
    # kept alive for the WHOLE script -- topology holdout (below) needs
    # every family's full scenario population, exactly like V3's own
    # apply_topology_holdout(frame, holdout) call used the pre-split
    # frame, not the primary split's already-reduced train portion.
    split = split_scenarios(full_trainable["scenario_id"].astype(str).unique().tolist(), seed=v3.SEED)
    train_df, val_df, test_df = apply_split(full_trainable, split)
    assert_no_scenario_overlap(split, train_df, val_df, test_df)

    train_trainable_full = train_df
    val_trainable = val_df
    test_trainable = test_df

    val_feat = v3.build_experimental_feature_matrix(val_trainable)
    test_feat = v3.build_experimental_feature_matrix(test_trainable)
    full_horizon_test_frame = test_trainable  # this split's own full trainable frame

    report = {"variants": {}, "topology_holdout": {}}

    for variant in ("canonical", "A", "B", "C"):

        print(f"=== Variant {variant} ===")
        t0 = time.time()

        variant_train = _make_variant_train(variant, train_trainable_full)
        variant_feat = v3.build_experimental_feature_matrix(variant_train)

        class_weight_map = compute_class_weight_map(variant_feat.y)
        base_weight = sample_weights_from_class_weight(variant_feat.y, class_weight_map)
        sample_weight = base_weight * variant_train["_extra_weight"].to_numpy()

        model = _fit_xgboost(variant_feat.X, variant_feat.y, sample_weight)

        result = _evaluate(model, test_feat, test_trainable, val_feat, scenario_metadata, full_horizon_test_frame)
        result["train_row_count"] = int(len(variant_feat.y))
        result["train_positive_rate"] = float(variant_feat.y.mean())
        result["seconds"] = time.time() - t0

        report["variants"][variant] = result
        print(f"  rows={result['train_row_count']} PR-AUC={result['test_metrics']['pr_auc']:.4f} "
              f"ROC-AUC={result['test_metrics']['roc_auc']:.4f} ({result['seconds']:.1f}s)")

        del variant_train, variant_feat, model, base_weight, sample_weight
        gc.collect()

    del test_feat, val_feat, train_trainable_full, val_trainable, test_trainable, full_horizon_test_frame
    gc.collect()

    # --- topology holdout for each variant (uses full_trainable, ALL
    # 2500 scenarios across all 4 families, kept alive since the start) ---
    holdout_splits = build_topology_holdout_splits(scenario_metadata)

    for variant in ("canonical", "A", "B", "C"):

        report["topology_holdout"][variant] = {}

        for holdout in holdout_splits:

            print(f"=== Variant {variant}, topology holdout [{holdout.held_out_family}] ===")
            t0 = time.time()

            holdout_train_df, holdout_test_df = apply_topology_holdout(full_trainable, holdout)
            assert_no_holdout_overlap(holdout, holdout_train_df, holdout_test_df)
            # canonical/A/B/C dedup only applies within the holdout's
            # OWN train rows (never touches the held-out family's test
            # rows, which stay full/real for evaluation).
            holdout_train_trainable = holdout_train_df
            del holdout_train_df
            gc.collect()

            variant_holdout_train = _make_variant_train(variant, holdout_train_trainable)
            variant_holdout_feat = v3.build_experimental_feature_matrix(variant_holdout_train)
            holdout_test_feat = v3.build_experimental_feature_matrix(holdout_test_df)

            class_weight_map = compute_class_weight_map(variant_holdout_feat.y)
            base_weight = sample_weights_from_class_weight(variant_holdout_feat.y, class_weight_map)
            sample_weight = base_weight * variant_holdout_train["_extra_weight"].to_numpy()

            model = _fit_xgboost(variant_holdout_feat.X, variant_holdout_feat.y, sample_weight)
            prob = model.predict_proba(holdout_test_feat.X)[:, 1]
            metrics = compute_metrics(holdout_test_feat.y, prob, threshold=0.5)

            report["topology_holdout"][variant][holdout.held_out_family] = {
                "train_row_count": int(len(variant_holdout_feat.y)),
                "test_row_count": int(len(holdout_test_feat.y)),
                "test_metrics": metrics.to_dict(),
                "seconds": time.time() - t0,
            }
            print(f"  PR-AUC={metrics.pr_auc:.4f} ROC-AUC={metrics.roc_auc:.4f} ({time.time()-t0:.1f}s)")

            del (holdout_train_trainable, variant_holdout_train, variant_holdout_feat, holdout_test_feat,
                 model, prob, base_weight, sample_weight, holdout_test_df)
            gc.collect()

    report["total_wall_seconds"] = time.time() - t_start

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Done in {time.time() - t_start:.1f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
