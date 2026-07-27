"""Localized Predictive Model V3.1 milestone, Phase 5 -- shuffled-label
failure investigation. Reproduces V3's original anomalous result
(ROC-AUC ~0.378) then runs 9 controlled variants (A-I) to determine
EXACTLY which combination of factors causes it, rather than accepting
the prior milestone's explanation without rigorous evidence.

Every variant trains on TRAIN split labels (real X, some form of
shuffled y) and evaluates against REAL VAL labels -- identical protocol
to predictive_model.sanity_checks.label_shuffle_test, just with the
shuffle mechanism varied per letter. Does NOT modify sanity_checks.py
itself (that shared module works correctly -- near chance -- in every
OTHER prior milestone; this investigation is specific to V3's exact
production configuration).
"""
import gc
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import train_localized_predictive_model_v3 as v3
from predictive_model.baselines import LogisticRegressionBaseline
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios
from predictive_model.tree_models import HistGradientBoostingModel

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "localized_predictive_model_v3_1"
OUTPUT_PATH = OUTPUT_DIR / "shuffle_battery_report.json"


def _group_ids_for_matrix(X: np.ndarray) -> np.ndarray:
    view = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, inverse = np.unique(view, return_inverse=True)
    return inverse.reshape(-1)


def _shuffle_global(y, rng):
    y2 = y.copy()
    rng.shuffle(y2)
    return y2


def _shuffle_within_groups(y, group_keys, rng):
    """Shuffle y independently within each distinct value of group_keys
    -- cross-group structure (which group has how many positives) is
    preserved exactly; only within-group row assignment is randomized."""

    y2 = y.copy()
    order = np.argsort(group_keys, kind="stable")
    sorted_keys = group_keys[order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(sorted_keys)]))

    y_sorted = y2[order]
    for start, end in zip(starts, ends):
        segment = y_sorted[start:end].copy()
        rng.shuffle(segment)
        y_sorted[start:end] = segment

    result = np.empty_like(y2)
    result[order] = y_sorted
    return result


def _xgb_full(X, y):
    """production XGBoost config -- _fit_with_weight recomputes class
    weights internally from y (see scripts/train_localized_predictive_
    model_v3.py), so no external sample_weight needs to be passed."""
    model = v3._fresh_model("xgboost")
    return v3._fit_with_weight(model, X, y, "xgboost")


def _xgb_no_reweight(X, y):
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
        eval_metric="logloss", scale_pos_weight=1.0, random_state=v3.SEED, n_jobs=v3.N_JOBS,
    )
    model.fit(X, y)
    return model, lambda Xte: model.predict_proba(Xte)[:, 1]


def _xgb_reduced_capacity(X, y, class_weight_map, sample_weight):
    model = XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, tree_method="hist",
        eval_metric="logloss", scale_pos_weight=1.0, random_state=v3.SEED, n_jobs=v3.N_JOBS,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, lambda Xte: model.predict_proba(Xte)[:, 1]


def main() -> None:

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v3._watchdog_log_path = str(OUTPUT_DIR / "memory_watchdog_shuffle_battery.log")
    with open(v3._watchdog_log_path, "w", encoding="utf-8") as f:
        f.write(f"watchdog started {time.time():.1f}\n")
    threading.Thread(target=v3._memory_watchdog, daemon=True).start()

    print("Loading scenario_metadata.json...")
    with open(v3.SCENARIO_METADATA_PATH, "r", encoding="utf-8") as f:
        scenario_metadata = json.load(f)
    topology_by_scenario = {m["scenario_id"]: m["topology_family"] for m in scenario_metadata}

    print("Loading dataset...")
    frame = v3._load_dataset_chunked(v3.CSV_PATH)
    split = split_scenarios(frame["scenario_id"].astype(str).unique().tolist(), seed=v3.SEED)
    train_df, val_df, test_df = apply_split(frame, split)
    assert_no_scenario_overlap(split, train_df, val_df, test_df)
    del frame, test_df
    gc.collect()

    train_trainable = v3._trainable_rows(train_df)
    val_trainable = v3._trainable_rows(val_df)
    del train_df, val_df
    gc.collect()

    train_trainable["topology_family"] = train_trainable["scenario_id"].astype(str).map(topology_by_scenario)

    train_feat = v3.build_experimental_feature_matrix(train_trainable)
    val_feat = v3.build_experimental_feature_matrix(val_trainable)

    y_train = train_feat.y
    scenario_ids = train_trainable["scenario_id"].astype(str).to_numpy()
    candidate_types = train_trainable["candidate_type"].astype(str).to_numpy()
    topology_families = train_trainable["topology_family"].to_numpy()
    group_ids = _group_ids_for_matrix(train_feat.X)

    del train_trainable, val_trainable
    gc.collect()

    results = {}

    def _record(letter, description, y_shuffled, fit_fn):
        t0 = time.time()
        model_or_pair = fit_fn(y_shuffled)
        if isinstance(model_or_pair, tuple):
            model, predict_fn = model_or_pair
        else:
            model, predict_fn = model_or_pair, model_or_pair.predict_proba
        prob = predict_fn(val_feat.X)
        roc_auc = float(roc_auc_score(val_feat.y, prob))
        pr_auc = float(average_precision_score(val_feat.y, prob))
        results[letter] = {
            "description": description,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "near_chance": abs(roc_auc - 0.5) < 0.05,
            "seconds": time.time() - t0,
        }
        print(f"  [{letter}] {description}: ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} "
              f"near_chance={results[letter]['near_chance']} ({time.time()-t0:.1f}s)")
        del model, prob
        gc.collect()

    rng_a = np.random.default_rng(v3.SEED)
    y_a = _shuffle_global(y_train, rng_a)

    print("A. Global row-level shuffle, full XGBoost (reproduction)...")
    _record("A_global_shuffle_full_xgboost", "global row-level shuffle, production XGBoost config", y_a,
             lambda y: _xgb_full(train_feat.X, y))

    print("B. Scenario-level (block-within-scenario) shuffle, full XGBoost...")
    rng_b = np.random.default_rng(v3.SEED + 1)
    y_b = _shuffle_within_groups(y_train, scenario_ids, rng_b)
    _record("B_scenario_block_shuffle_full_xgboost", "shuffle within each scenario only", y_b,
             lambda y: _xgb_full(train_feat.X, y))

    print("C. Within-candidate-type shuffle, full XGBoost...")
    rng_c = np.random.default_rng(v3.SEED + 2)
    y_c = _shuffle_within_groups(y_train, candidate_types, rng_c)
    _record("C_within_candidate_type_shuffle_full_xgboost", "shuffle within each candidate_type only", y_c,
             lambda y: _xgb_full(train_feat.X, y))

    print("D. Within-topology-family shuffle, full XGBoost...")
    rng_d = np.random.default_rng(v3.SEED + 3)
    y_d = _shuffle_within_groups(y_train, topology_families, rng_d)
    _record("D_within_topology_family_shuffle_full_xgboost", "shuffle within each topology_family only", y_d,
             lambda y: _xgb_full(train_feat.X, y))

    print("E. Within-duplicate-feature-group shuffle, full XGBoost...")
    rng_e = np.random.default_rng(v3.SEED + 4)
    y_e = _shuffle_within_groups(y_train, group_ids, rng_e)
    _record("E_within_duplicate_feature_group_shuffle_full_xgboost",
             "shuffle within each exact-duplicate-feature-vector group only "
             "(preserves each group's true empirical positive rate exactly)", y_e,
             lambda y: _xgb_full(train_feat.X, y))

    print("F. Global shuffle, XGBoost WITHOUT class reweighting...")
    _record("F_global_shuffle_no_reweighting", "global shuffle, XGBoost with scale_pos_weight=1.0, no sample_weight",
             y_a, lambda y: _xgb_no_reweight(train_feat.X, y))

    print("G. Global shuffle, XGBoost REDUCED capacity (depth=3, 50 trees)...")
    _record("G_global_shuffle_reduced_capacity", "global shuffle, XGBoost depth=3/50 trees, WITH reweighting",
             y_a, lambda y: _xgb_reduced_capacity(
                 train_feat.X, y, compute_class_weight_map(y),
                 sample_weights_from_class_weight(y, compute_class_weight_map(y))))

    print("H. Global shuffle, LogisticRegression control...")

    def _lr_fit(y):
        model = LogisticRegressionBaseline(seed=v3.SEED)
        class_weight_map = compute_class_weight_map(y)
        sample_weight = sample_weights_from_class_weight(y, class_weight_map)
        model.fit(train_feat.X, y, sample_weight=sample_weight)
        return model

    _record("H_global_shuffle_logistic_regression", "global shuffle, LogisticRegression", y_a, _lr_fit)

    print("I. Global shuffle, HistGradientBoosting control...")

    def _histgb_fit(y):
        model = HistGradientBoostingModel(seed=v3.SEED)
        class_weight_map = compute_class_weight_map(y)
        sample_weight = sample_weights_from_class_weight(y, class_weight_map)
        model.fit(train_feat.X, y, sample_weight=sample_weight)
        return model

    _record("I_global_shuffle_histgradientboosting", "global shuffle, HistGradientBoosting", y_a, _histgb_fit)

    report = {
        "experiment": "shuffle_battery_v3_1",
        "seed": v3.SEED,
        "train_row_count": int(len(y_train)),
        "val_row_count": int(len(val_feat.y)),
        "results": results,
        "total_wall_seconds": time.time() - t_start,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Done in {time.time() - t_start:.1f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
