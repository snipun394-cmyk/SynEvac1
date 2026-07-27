"""Localized Predictive Model V3.1 milestone, Phase 7-8 -- topology
representation audit and feature distribution shift.

Phase 7: classify every V3 feature by what it represents (candidate-
local dynamic/structural, whole-building dynamic/structural, topology
proxy, candidate identity proxy), then train a simple topology-family
classifier using ONLY the V3 feature vector to quantify how strongly
topology is already encoded in the feature space.

Phase 8: for each topology family vs. the rest, quantify feature
distribution shift using a metric appropriate to each feature's type
(standardized mean difference for continuous features, a proportion
difference for the one-hot/categorical ones -- PSI/KS are documented
but not blindly applied where inappropriate, per the milestone's own
explicit instruction).

No re-simulation -- reuses the same already-loaded eligible dataset.
"""
import gc
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import train_localized_predictive_model_v3 as v3

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "localized_predictive_model_v3_1"
OUTPUT_PATH = OUTPUT_DIR / "topology_representation_and_shift_report.json"

# Phase 7 -- feature classification. Static, based on established
# feature semantics (docs/architecture/localized_predictive_model_v2_2.md
# and predictive_congestion_target_v2.md's own live-parity audits) --
# no computation needed, this is a documentation/reasoning table.
FEATURE_CLASSIFICATION = {
    "total_active_occupant_count": "whole_building_dynamic",
    "candidate_capacity": "candidate_local_structural",
    "candidate_walking_distance": "candidate_local_structural",
    "candidate_traversable": "candidate_local_dynamic",
    "candidate_adjacent_zone_occupancy": "whole_building_dynamic",
    "candidate_queue_length": "candidate_local_dynamic",
    "candidate_approaching_count": "candidate_local_dynamic",
    "candidate_congestion_level": "candidate_local_dynamic",
    "candidate_recent_flow_rate": "candidate_local_dynamic",
    "candidate_congestion_trend": "candidate_local_dynamic",
    "candidate_alternative_route_count": "topology_proxy",
    "candidate_type": "candidate_identity_proxy",
}

CONTINUOUS_FEATURES = (
    "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_adjacent_zone_occupancy", "candidate_queue_length", "candidate_approaching_count",
    "candidate_recent_flow_rate", "candidate_alternative_route_count",
)
CATEGORICAL_FEATURES = (
    "candidate_traversable", "candidate_congestion_level", "candidate_congestion_trend", "candidate_type",
)


def _standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """(mean_a - mean_b) / pooled_std -- Cohen's-d-style SMD, the
    standard, defensible metric for comparing two groups' means on a
    CONTINUOUS feature. |SMD| >= 0.8 is conventionally 'large'."""

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import ks_2samp
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def _categorical_proportion_shift(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    """Per-category |proportion_a - proportion_b| -- the appropriate,
    simple metric for categorical/one-hot features (PSI is designed for
    this too, but with this few categories a direct proportion-gap table
    is more transparent and equally defensible)."""

    props_a = a.value_counts(normalize=True)
    props_b = b.value_counts(normalize=True)
    categories = sorted(set(props_a.index) | set(props_b.index))
    return {
        str(cat): abs(float(props_a.get(cat, 0.0)) - float(props_b.get(cat, 0.0)))
        for cat in categories
    }


def main() -> None:

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v3._watchdog_log_path = str(OUTPUT_DIR / "memory_watchdog_topology_audit.log")
    with open(v3._watchdog_log_path, "w", encoding="utf-8") as f:
        f.write(f"watchdog started {time.time():.1f}\n")
    threading.Thread(target=v3._memory_watchdog, daemon=True).start()

    with open(v3.SCENARIO_METADATA_PATH, "r", encoding="utf-8") as f:
        scenario_metadata = json.load(f)
    topology_by_scenario = {m["scenario_id"]: m["topology_family"] for m in scenario_metadata}

    print("Loading dataset...")
    frame = v3._load_dataset_chunked(v3.CSV_PATH)
    trainable = v3._trainable_rows(frame)
    del frame
    gc.collect()
    trainable["topology_family"] = trainable["scenario_id"].astype(str).map(topology_by_scenario)

    # =====================================================
    # Phase 7 -- topology-family classifier from V3 features only
    # =====================================================

    print("Building feature matrix for topology classifier...")
    prepared = v3.build_experimental_feature_matrix(trainable)
    families = trainable["topology_family"].astype(str).to_numpy()
    scenario_ids = trainable["scenario_id"].astype(str).to_numpy()

    # scenario-level split for the classifier itself (never split by row
    # -- a topology-family classifier trained/tested on rows from the
    # SAME scenario would trivially memorize scenario_id-correlated
    # noise, not a fair test of "is topology encoded in the FEATURES").
    unique_scenarios = sorted(set(scenario_ids))
    rng = np.random.default_rng(v3.SEED)
    shuffled_scenarios = list(unique_scenarios)
    rng.shuffle(shuffled_scenarios)
    n_test = int(0.2 * len(shuffled_scenarios))
    test_scenario_set = set(shuffled_scenarios[:n_test])

    is_test = np.array([sid in test_scenario_set for sid in scenario_ids])

    X_train, X_test = prepared.X[~is_test], prepared.X[is_test]
    fam_train, fam_test = families[~is_test], families[is_test]

    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
        random_state=v3.SEED, n_jobs=v3.N_JOBS,
    )
    # multi-class label encode
    family_names = sorted(set(families))
    family_to_int = {f: i for i, f in enumerate(family_names)}
    y_train_int = np.array([family_to_int[f] for f in fam_train])
    y_test_int = np.array([family_to_int[f] for f in fam_test])

    t0 = time.time()
    clf.fit(X_train, y_train_int)
    pred_test = clf.predict(X_test)
    topology_classifier_accuracy = float(accuracy_score(y_test_int, pred_test))
    topology_classifier_f1_macro = float(f1_score(y_test_int, pred_test, average="macro"))

    # majority-class baseline for comparison
    majority_family = pd.Series(fam_train).value_counts().idxmax()
    majority_pred = np.full_like(y_test_int, family_to_int[majority_family])
    majority_accuracy = float(accuracy_score(y_test_int, majority_pred))

    print(f"Topology classifier: accuracy={topology_classifier_accuracy:.4f} "
          f"(majority baseline={majority_accuracy:.4f}) f1_macro={topology_classifier_f1_macro:.4f} "
          f"({time.time()-t0:.1f}s)")

    del X_train, X_test, clf
    gc.collect()

    # =====================================================
    # Phase 8 -- feature distribution shift per topology family
    # =====================================================

    print("Computing feature distribution shift...")
    shift_report = {}

    for family in family_names:

        family_mask = trainable["topology_family"].astype(str).to_numpy() == family
        shift_report[family] = {"continuous_smd": {}, "continuous_ks": {}, "categorical_proportion_shift": {}}

        for feature in CONTINUOUS_FEATURES:
            col = trainable[feature].to_numpy(dtype=float)
            a = col[family_mask]
            b = col[~family_mask]
            shift_report[family]["continuous_smd"][feature] = _standardized_mean_difference(a, b)
            shift_report[family]["continuous_ks"][feature] = _ks_statistic(a, b)

        for feature in CATEGORICAL_FEATURES:
            a = trainable.loc[family_mask, feature].astype(str)
            b = trainable.loc[~family_mask, feature].astype(str)
            shift_report[family]["categorical_proportion_shift"][feature] = _categorical_proportion_shift(a, b)

    # rank features by max |SMD| across families -- "strongest shifts"
    max_abs_smd = {}
    for feature in CONTINUOUS_FEATURES:
        values = [abs(shift_report[fam]["continuous_smd"][feature]) for fam in family_names
                  if not np.isnan(shift_report[fam]["continuous_smd"][feature])]
        max_abs_smd[feature] = max(values) if values else float("nan")
    ranked_continuous = sorted(max_abs_smd.items(), key=lambda kv: -kv[1] if not np.isnan(kv[1]) else 0)

    report = {
        "phase7_feature_classification": FEATURE_CLASSIFICATION,
        "phase7_topology_classifier": {
            "family_names": family_names,
            "test_scenario_count": len(test_scenario_set),
            "train_scenario_count": len(unique_scenarios) - len(test_scenario_set),
            "accuracy": topology_classifier_accuracy,
            "f1_macro": topology_classifier_f1_macro,
            "majority_class_baseline_accuracy": majority_accuracy,
            "interpretation_note": (
                "High accuracy here means topology family is strongly encoded/inferable from the "
                "feature space -- NOT automatically evidence of leakage, but a structural explanation "
                "for why leave-one-topology-out generalization is hard."
            ),
        },
        "phase8_distribution_shift": shift_report,
        "phase8_features_ranked_by_max_abs_smd": ranked_continuous,
        "total_wall_seconds": time.time() - t_start,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Done in {time.time() - t_start:.1f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
