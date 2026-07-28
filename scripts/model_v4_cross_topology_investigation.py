"""Cross-Topology Generalization Investigation -- WHY does an entirely
UNSEEN topology family (not just an unseen structural variant) fail to
transfer, per Predictive Dataset V3's own Phase 20 finding (commit
ecf3f9b, verdict B)? INVESTIGATION ONLY. No Model V4 is trained/
exported here, nothing is wired into LiveRuntime/Recommendation/
Guidance/Decision Policy, no new Designer asset, no GNN. Every
experiment reuses Dataset V3 (data/predictive_dataset_campaign_v3) and
the SAME XGBoost configuration prior milestones selected
(predictive_model.tree_models.build_tree_models, no hyperparameter
search) -- this is a causal-comparison investigation, not a leaderboard.

Phases implemented here (see docs/architecture/
cross_topology_generalization_investigation.md for the full narrative):

  1  Family-holdout reproduction (+ FPR/FNR, deterministic baseline)
  2  Feature distribution shift (SMD) + family classifier
  3  Conditional target-rate analysis (covariate vs concept shift)
  4  Failure-region analysis (false positive / false negative slices)
  5  Candidate-type-specific models (Door/Exit/Stair-only) -- diagnostic
  6  Scale-normalization experiment (experimental features, Phase 6)
  8  Graph-context experiment (experimental features, Phase 8): A/B/C/D
  9  Family-ID diagnostic (NEVER for production)
 10  Leave-multiple-families-out scaling test
 12  Model robustness (XGBoost vs HistGradientBoosting vs LogisticRegression)
 13  Target V2 stability across families

Usage: python scripts/model_v4_cross_topology_investigation.py
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_dataset.experimental_features_v4 import (
    GRAPH_CONTEXT_FEATURE_NAMES,
    NORMALIZED_FEATURE_NAMES,
    add_graph_context_features,
    add_normalized_features,
    build_graph_context_table,
)
from predictive_dataset.topologies_v3 import CANONICAL_FAMILIES
from predictive_model.baselines import DeterministicCurrentStateBaseline, LogisticRegressionBaseline
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix, trainable_rows
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics, metrics_by_group
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios
from predictive_model.topology_holdout import apply_topology_holdout, build_topology_holdout_splits
from predictive_model.tree_models import build_tree_models

REPO_ROOT = Path(__file__).resolve().parent.parent
V3_CAMPAIGN_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v3"
CSV_PATH = V3_CAMPAIGN_DIR / "candidate_dataset_v3.csv"
SCENARIO_METADATA_PATH = V3_CAMPAIGN_DIR / "scenario_metadata.json"

OUTPUT_DIR = REPO_ROOT / "data" / "cross_topology_generalization_investigation"

SEED = 20260728  # today's date, this milestone's own master seed
N_JOBS = 2

BASE_FEATURE_NAMES = (
    "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_traversable", "candidate_adjacent_zone_occupancy", "candidate_queue_length",
    "candidate_approaching_count", "candidate_congestion_level",
)
EXPERIMENTAL_FEATURE_NAMES = ("candidate_recent_flow_rate", "candidate_congestion_trend", "candidate_alternative_route_count")

DISTRIBUTION_FEATURES = (
    "candidate_queue_length", "candidate_recent_flow_rate", "candidate_walking_distance",
    "candidate_alternative_route_count", "candidate_adjacent_zone_occupancy",
    "total_active_occupant_count",
)

_COMPACT_DTYPES = {
    "observation_time": "float32",
    "total_active_occupant_count": "int32",
    "candidate_capacity": "float32",
    "candidate_walking_distance": "float32",
    "candidate_traversable": "bool",
    "candidate_adjacent_zone_occupancy": "float32",
    "candidate_queue_length": "float32",
    "candidate_approaching_count": "float32",
    "candidate_recent_flow_rate": "float32",
    "candidate_alternative_route_count": "int32",
    "lead_time_seconds_v2": "float32",
}

MIN_AVAILABLE_MEMORY_BYTES = 300_000_000


def _mem_log(label: str, memory_log: list) -> None:
    vm = psutil.virtual_memory()
    available_mb = vm.available / 1e6
    print(f"    [mem] {label}: available={available_mb:.0f}MB ({vm.percent:.0f}% used)", flush=True)
    memory_log.append({"label": label, "available_mb": available_mb, "percent_used": vm.percent})
    if vm.available < MIN_AVAILABLE_MEMORY_BYTES:
        raise MemoryError(f"Available system memory critically low ({available_mb:.0f}MB) at step {label!r}.")


def _load_dataset_chunked(csv_path: Path, chunksize: int = 250_000) -> pd.DataFrame:

    keep_columns = (
        ["scenario_id", "observation_time", "candidate_id", "candidate_type",
         "topology_family", "structural_variant_id"]
        + list(BASE_FEATURE_NAMES) + list(EXPERIMENTAL_FEATURE_NAMES)
        + ["currently_congested_v2", "had_any_activity_in_window_v2", "target_v2", "lead_time_seconds_v2"]
    )

    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=keep_columns):
        chunk = chunk.astype({k: v for k, v in _COMPACT_DTYPES.items() if k in chunk.columns})
        chunks.append(chunk)

    frame = pd.concat(chunks, ignore_index=True)
    del chunks

    for column in ("scenario_id", "candidate_id", "candidate_type", "candidate_congestion_level",
                    "candidate_congestion_trend", "topology_family", "structural_variant_id"):
        frame[column] = frame[column].astype("category")

    frame["currently_congested_v2"] = frame["currently_congested_v2"].astype(bool)
    frame["had_any_activity_in_window_v2"] = frame["had_any_activity_in_window_v2"].astype(bool)

    frame = frame.rename(columns={
        "target_v2": "target",
        "currently_congested_v2": "currently_congested",
        "had_any_activity_in_window_v2": "had_any_activity_in_window",
    })

    return frame


# =====================================================
# Feature-matrix assembly: canonical (V2.1/V3) 27-column base, plus
# optional additive experimental blocks. Reuses
# predictive_model.feature_prep_v2_1.build_experimental_feature_matrix
# for the canonical block verbatim -- never redefines it -- then
# horizontally stacks any requested extra numeric/bool columns already
# present on the SAME frame (added upstream by add_normalized_features/
# add_graph_context_features), so row order/identity is guaranteed
# consistent between the two halves.
# =====================================================


def build_matrix(frame: pd.DataFrame, extra_columns: Tuple[str, ...] = (), include_family_id: bool = False):

    base = build_experimental_feature_matrix(frame)

    extra_arrays = []
    extra_names = []

    for column in extra_columns:
        values = frame[column].to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=-1.0, posinf=-1.0, neginf=-1.0)
        extra_arrays.append(values)
        extra_names.append(column)

    if include_family_id:
        for family in CANONICAL_FAMILIES:
            extra_arrays.append((frame["topology_family"] == family).to_numpy(dtype=float))
            extra_names.append(f"topology_family={family}")

    if not extra_arrays:
        X = base.X.astype(np.float32)
        return base.__class__(X=X, y=base.y, scenario_ids=base.scenario_ids, candidate_types=base.candidate_types, feature_names=base.feature_names)

    X = np.column_stack([base.X] + extra_arrays).astype(np.float32)
    names = base.feature_names + tuple(extra_names)

    return base.__class__(X=X, y=base.y, scenario_ids=base.scenario_ids, candidate_types=base.candidate_types, feature_names=names)


def _fit_model(algorithm: str, X, y):

    if algorithm == "logistic_regression":
        model = LogisticRegressionBaseline(seed=SEED)
        model.fit(X, y)
        return model

    model = build_tree_models(seed=SEED, n_jobs=N_JOBS)[algorithm]
    class_weight_map = compute_class_weight_map(y)
    sample_weight = sample_weights_from_class_weight(y, class_weight_map)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _compact_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """compute_metrics().to_dict(), with the 10-bin calibration curve
    replaced by a single scalar (mean |fraction_of_positives -
    mean_predicted_value|, an ECE-style summary) plus explicit FPR/FNR
    derived from the confusion matrix -- used for the ~50 bulk
    diagnostic fits below to keep the report a manageable size; Phase 1
    keeps the full curve since that IS the reconstruction-of-record."""

    cm = metrics_dict["confusion_matrix"]
    fp, tn, fn, tp = cm["fp"], cm["tn"], cm["fn"], cm["tp"]

    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None

    curve = metrics_dict["calibration_curve"]
    if curve["fraction_of_positives"]:
        ece_proxy = float(np.mean(np.abs(
            np.array(curve["fraction_of_positives"]) - np.array(curve["mean_predicted_value"])
        )))
    else:
        ece_proxy = None

    compact = {k: v for k, v in metrics_dict.items() if k != "calibration_curve"}
    compact["false_positive_rate"] = fpr
    compact["false_negative_rate"] = fnr
    compact["calibration_ece_proxy"] = ece_proxy
    return compact


def _fpr_fnr(metrics_dict: Dict[str, Any]) -> Tuple[Any, Any]:
    cm = metrics_dict["confusion_matrix"]
    fp, tn, fn, tp = cm["fp"], cm["tn"], cm["fn"], cm["tp"]
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None
    return fpr, fnr


# =====================================================
# Phase 1 + Phase 4 -- family-holdout reproduction AND failure-region
# analysis, in one pass per family (both need the same held-out
# predictions, so this avoids refitting twice).
# =====================================================


def _failure_region_breakdown(test_df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """False-positive/false-negative RATE (not count) within slices of
    test_df, for every dimension Phase 4 lists. Rate, not count, is what
    lets slices of very different size be compared."""

    y_pred = (y_prob >= threshold).astype(int)
    d = test_df.copy()
    d["_y_true"] = y_true
    d["_y_pred"] = y_pred
    d["_is_fp"] = (d["_y_true"] == 0) & (d["_y_pred"] == 1)
    d["_is_fn"] = (d["_y_true"] == 1) & (d["_y_pred"] == 0)
    d["_is_neg"] = d["_y_true"] == 0
    d["_is_pos"] = d["_y_true"] == 1

    def rate_by(column: str, bins=None, labels=None) -> Dict[str, Any]:
        series = d[column]
        if bins is not None:
            series = pd.cut(d[column].astype(float), bins=bins, labels=labels)
        grouped = d.groupby(series, observed=True)
        fp_rate = (grouped["_is_fp"].sum() / grouped["_is_neg"].sum().replace(0, np.nan)).to_dict()
        fn_rate = (grouped["_is_fn"].sum() / grouped["_is_pos"].sum().replace(0, np.nan)).to_dict()
        n = grouped.size().to_dict()
        return {
            "false_positive_rate_by_bucket": {str(k): (float(v) if pd.notna(v) else None) for k, v in fp_rate.items()},
            "false_negative_rate_by_bucket": {str(k): (float(v) if pd.notna(v) else None) for k, v in fn_rate.items()},
            "n_by_bucket": {str(k): int(v) for k, v in n.items()},
        }

    d["_multi_bottleneck"] = d.groupby(["scenario_id", "observation_time"], observed=True)["_is_pos"].transform("sum") >= 2

    return {
        "by_candidate_type": rate_by("candidate_type"),
        "by_congestion_trend": rate_by("candidate_congestion_trend"),
        "by_structural_variant": rate_by("structural_variant_id"),
        "by_occupancy_level": rate_by("total_active_occupant_count", bins=[-1, 5, 15, 30, 1e9], labels=["LOW(<=5)", "MED(6-15)", "HIGH(16-30)", "VHIGH(>30)"]),
        "by_flow_rate_band": rate_by("candidate_recent_flow_rate", bins=[-1, 0, 2, 5, 10, 1e9], labels=["0", "1-2", "3-5", "6-10", ">10"]),
        "by_queue_length_band": rate_by("candidate_queue_length", bins=[-1, 0, 2, 5, 10, 1e9], labels=["0", "1-2", "3-5", "6-10", ">10"]),
        "by_alt_route_count": rate_by("candidate_alternative_route_count", bins=[-1, 0, 1, 2, 3, 1e9], labels=["0", "1", "2", "3", ">3"]),
        "by_walking_distance_band": rate_by("candidate_walking_distance", bins=[-1, 10, 25, 40, 1e9], labels=["<=10", "11-25", "26-40", ">40"]),
        "by_multi_bottleneck": {
            "false_positive_rate": {
                str(k): (float(v) if pd.notna(v) else None)
                for k, v in (d[d["_is_neg"]].groupby("_multi_bottleneck", observed=True)["_is_fp"].mean()).to_dict().items()
            },
            "false_negative_rate": {
                str(k): (float(v) if pd.notna(v) else None)
                for k, v in (d[d["_is_pos"]].groupby("_multi_bottleneck", observed=True)["_is_fn"].mean()).to_dict().items()
            },
        },
    }


def phase1_and_4_family_holdout(frame: pd.DataFrame, scenario_metadata: list, memory_log: list) -> Dict[str, Any]:

    print("\n=== PHASE 1 + 4: family-holdout reproduction + failure-region analysis ===", flush=True)
    splits = build_topology_holdout_splits(scenario_metadata)
    results = {}

    for split in splits:

        t0 = time.time()
        train_df, test_df = apply_topology_holdout(frame, split)
        train_trainable = trainable_rows(train_df)
        test_trainable = trainable_rows(test_df)

        train_feat = build_matrix(train_trainable)
        test_feat = build_matrix(test_trainable)

        model = _fit_model("xgboost", train_feat.X, train_feat.y)
        prob = model.predict_proba(test_feat.X)
        metrics = compute_metrics(test_feat.y, prob, threshold=0.5)
        by_type = metrics_by_group(test_feat.y, prob, test_feat.candidate_types, threshold=0.5)

        det_model = DeterministicCurrentStateBaseline(train_feat.feature_names)
        det_model.fit(train_feat.X, train_feat.y)
        det_prob = det_model.predict_proba(test_feat.X)
        det_metrics = compute_metrics(test_feat.y, det_prob, threshold=0.5)

        fpr, fnr = _fpr_fnr(metrics.to_dict())

        failure_regions = _failure_region_breakdown(test_trainable, test_feat.y, prob)

        results[split.held_out_family] = {
            "train_row_count": int(len(train_feat.y)),
            "test_row_count": int(len(test_feat.y)),
            "test_positive_rate": float(test_feat.y.mean()),
            "xgboost_test_metrics": metrics.to_dict(),
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "xgboost_test_metrics_by_candidate_type": by_type,
            "deterministic_baseline_test_metrics": det_metrics.to_dict(),
            "xgboost_vs_deterministic_relative_lift": (
                metrics.pr_auc / det_metrics.pr_auc if (metrics.pr_auc and det_metrics.pr_auc) else None
            ),
            "failure_region_analysis": failure_regions,
            "fit_seconds": time.time() - t0,
        }

        print(f"  [{split.held_out_family}] PR-AUC={metrics.pr_auc:.4f} ROC-AUC={metrics.roc_auc:.4f} "
              f"FPR={fpr:.4f} FNR={fnr:.4f} det_PR-AUC={det_metrics.pr_auc:.4f} ({time.time()-t0:.1f}s)", flush=True)

        del train_df, test_df, train_trainable, test_trainable, train_feat, test_feat, model
        gc.collect()
        _mem_log(f"after family holdout [{split.held_out_family}]", memory_log)

    return results


# =====================================================
# Phase 2 -- feature distribution shift (SMD) + family-classifier
# identifiability (repeats V3.1's own experiment, on Dataset V3, at
# full scale, restricted to the trainable-row subset this milestone
# otherwise uses throughout, for direct comparability with Phase 1).
# =====================================================


def _standardized_mean_difference(values: pd.Series, in_family_mask: pd.Series) -> float:

    in_vals = values[in_family_mask].astype(float)
    out_vals = values[~in_family_mask].astype(float)

    pooled_std = np.sqrt((in_vals.var() + out_vals.var()) / 2.0)
    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0

    return float(abs(in_vals.mean() - out_vals.mean()) / pooled_std)


def phase2_distribution_shift_and_identifiability(trainable_frame: pd.DataFrame, memory_log: list) -> Dict[str, Any]:

    print("\n=== PHASE 2: feature distribution shift + family identifiability ===", flush=True)

    smd_table: Dict[str, Dict[str, float]] = {}
    for feature in DISTRIBUTION_FEATURES:
        smd_table[feature] = {}
        for family in CANONICAL_FAMILIES:
            mask = trainable_frame["topology_family"] == family
            smd_table[feature][family] = _standardized_mean_difference(trainable_frame[feature], mask)

    max_smd_ranked = sorted(
        ((feature, max(smd_table[feature].values())) for feature in DISTRIBUTION_FEATURES),
        key=lambda kv: kv[1], reverse=True,
    )

    # candidate_type distribution by family (categorical -- reported as
    # proportions rather than SMD, which is defined for continuous features)
    type_distribution = (
        pd.crosstab(trainable_frame["topology_family"], trainable_frame["candidate_type"], normalize="index")
        .round(4).to_dict(orient="index")
    )

    # Family classifier: same discipline as V3's own Phase 10 identifiability
    # audit (model_v3_topology_variant_identifiability_audit.py) -- scenario-
    # level split, XGBoost, canonical features only, predicting topology_family.
    split = split_scenarios(trainable_frame["scenario_id"].unique(), seed=SEED, ratios=(0.8, 0.0, 0.2))
    train_df = trainable_frame[trainable_frame["scenario_id"].isin(split.train_scenario_ids)]
    test_df = trainable_frame[trainable_frame["scenario_id"].isin(split.test_scenario_ids)]

    train_feat = build_matrix(train_df)
    test_feat = build_matrix(test_df)

    family_labels = {family: i for i, family in enumerate(CANONICAL_FAMILIES)}
    y_train = train_df["topology_family"].map(family_labels).to_numpy()
    y_test = test_df["topology_family"].map(family_labels).to_numpy()

    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                         eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS)
    clf.fit(train_feat.X, y_train)
    pred = clf.predict(test_feat.X)

    accuracy = float((pred == y_test).mean())
    majority_baseline = float(pd.Series(y_test).value_counts(normalize=True).max())

    print(f"  Family classifier accuracy={accuracy:.4f} (majority baseline={majority_baseline:.4f})", flush=True)
    for feature, max_smd in max_smd_ranked:
        print(f"  SMD[{feature}] max={max_smd:.3f}", flush=True)

    del train_df, test_df, train_feat, test_feat, clf
    gc.collect()
    _mem_log("after Phase 2", memory_log)

    return {
        "standardized_mean_difference_by_feature_and_family": smd_table,
        "features_ranked_by_max_smd": max_smd_ranked,
        "candidate_type_distribution_by_family": type_distribution,
        "family_classifier_accuracy": accuracy,
        "family_classifier_majority_baseline": majority_baseline,
    }


# =====================================================
# Phase 3 -- conditional target-rate analysis: does P(target=1 | feature
# bin) change across families (concept shift), independent of whether
# the MARGINAL distribution of the feature itself changes (covariate
# shift, Phase 2's job)?
# =====================================================


CONDITIONAL_TARGET_FEATURES: Dict[str, Dict[str, Any]] = {
    "candidate_queue_length": {"bins": [-1, 0, 2, 5, 10, 1e9], "labels": ["0", "1-2", "3-5", "6-10", ">10"]},
    "candidate_recent_flow_rate": {"bins": [-1, 0, 2, 5, 10, 1e9], "labels": ["0", "1-2", "3-5", "6-10", ">10"]},
    "candidate_walking_distance": {"bins": [-1, 10, 25, 40, 1e9], "labels": ["<=10", "11-25", "26-40", ">40"]},
    "total_active_occupant_count": {"bins": [-1, 5, 15, 30, 1e9], "labels": ["LOW(<=5)", "MED(6-15)", "HIGH(16-30)", "VHIGH(>30)"]},
}


def phase3_conditional_target_relationships(trainable_frame: pd.DataFrame) -> Dict[str, Any]:

    print("\n=== PHASE 3: conditional target-rate (covariate vs concept shift) ===", flush=True)

    result: Dict[str, Any] = {}

    for feature, spec in CONDITIONAL_TARGET_FEATURES.items():

        binned = pd.cut(trainable_frame[feature].astype(float), bins=spec["bins"], labels=spec["labels"])
        table = (
            trainable_frame.assign(_bin=binned)
            .groupby(["_bin", "topology_family"], observed=True)["target"]
            .mean()
            .unstack("topology_family")
        )
        n_table = (
            trainable_frame.assign(_bin=binned)
            .groupby(["_bin", "topology_family"], observed=True)["target"]
            .size()
            .unstack("topology_family")
        )

        # Concept-shift signal: for each bin present in >=2 families with
        # >=30 rows each, the spread (max-min) of conditional positive rate.
        spread_by_bin = {}
        for bin_label in table.index:
            row = table.loc[bin_label]
            counts = n_table.loc[bin_label]
            valid = row[(counts >= 30) & row.notna()]
            if len(valid) >= 2:
                spread_by_bin[str(bin_label)] = float(valid.max() - valid.min())

        max_spread = max(spread_by_bin.values()) if spread_by_bin else None

        result[feature] = {
            "conditional_positive_rate_by_bin_and_family": {
                str(k): {fam: (float(v) if pd.notna(v) else None) for fam, v in row.items()}
                for k, row in table.iterrows()
            },
            "n_by_bin_and_family": {
                str(k): {fam: int(v) if pd.notna(v) else 0 for fam, v in row.items()}
                for k, row in n_table.iterrows()
            },
            "concept_shift_spread_by_bin": spread_by_bin,
            "max_concept_shift_spread": max_spread,
        }

        print(f"  [{feature}] max conditional-rate spread across families (well-supported bins) = {max_spread}", flush=True)

    return result


# =====================================================
# Phase 5 -- candidate-type-specific models, diagnostic only.
# =====================================================


def phase5_candidate_type_specialization(frame: pd.DataFrame, scenario_metadata: list, memory_log: list) -> Dict[str, Any]:

    print("\n=== PHASE 5: candidate-type interactions (Door/Exit/Stair-only vs unified) ===", flush=True)
    splits = build_topology_holdout_splits(scenario_metadata)
    results = {}

    for split in splits:

        train_df, test_df = apply_topology_holdout(frame, split)
        train_trainable = trainable_rows(train_df)
        test_trainable = trainable_rows(test_df)

        per_type = {}
        for candidate_type in ("Door", "Exit", "Stair"):

            train_slice = train_trainable[train_trainable["candidate_type"] == candidate_type]
            test_slice = test_trainable[test_trainable["candidate_type"] == candidate_type]

            if train_slice["target"].nunique() < 2 or len(test_slice) == 0 or test_slice["target"].nunique() < 2:
                per_type[candidate_type] = {"skipped_reason": "insufficient rows or single class", "n_train": len(train_slice), "n_test": len(test_slice)}
                continue

            train_feat = build_matrix(train_slice)
            test_feat = build_matrix(test_slice)

            model = _fit_model("xgboost", train_feat.X, train_feat.y)
            prob = model.predict_proba(test_feat.X)
            metrics = _compact_metrics(compute_metrics(test_feat.y, prob).to_dict())
            per_type[candidate_type] = {"n_train": len(train_slice), "n_test": len(test_slice), "metrics": metrics}

            del train_feat, test_feat, model
            gc.collect()

        results[split.held_out_family] = per_type
        print(f"  [{split.held_out_family}] Door PR-AUC={per_type.get('Door', {}).get('metrics', {}).get('pr_auc')} "
              f"Exit PR-AUC={per_type.get('Exit', {}).get('metrics', {}).get('pr_auc')} "
              f"Stair PR-AUC={per_type.get('Stair', {}).get('metrics', {}).get('pr_auc')}", flush=True)

        del train_df, test_df, train_trainable, test_trainable
        gc.collect()
        _mem_log(f"after Phase 5 [{split.held_out_family}]", memory_log)

    return results


# =====================================================
# Phase 6 + Phase 8 -- normalization / graph-context / combined
# experiment, run as a controlled A/B/C/D comparison at every family
# holdout. A ("canonical") is NOT refit here -- it is copied from Phase
# 1's own results so the comparison uses the IDENTICAL fit, not a
# second independently-seeded one.
# =====================================================


def phase6_8_normalization_and_graph_context(
    frame: pd.DataFrame, scenario_metadata: list, phase1_results: Dict[str, Any], memory_log: list,
) -> Dict[str, Any]:

    print("\n=== PHASE 6 + 8: normalization / graph-context experiment (A/B/C/D) ===", flush=True)
    splits = build_topology_holdout_splits(scenario_metadata)
    results = {}

    variants = {
        "B_normalized": NORMALIZED_FEATURE_NAMES,
        "C_graph_context": GRAPH_CONTEXT_FEATURE_NAMES,
        "D_normalized_plus_graph_context": NORMALIZED_FEATURE_NAMES + GRAPH_CONTEXT_FEATURE_NAMES,
    }

    for split in splits:

        train_df, test_df = apply_topology_holdout(frame, split)
        train_trainable = trainable_rows(train_df)
        test_trainable = trainable_rows(test_df)

        family_result = {
            "A_canonical_pr_auc": phase1_results[split.held_out_family]["xgboost_test_metrics"]["pr_auc"],
        }

        for label, extra_columns in variants.items():

            t0 = time.time()
            train_feat = build_matrix(train_trainable, extra_columns=extra_columns)
            test_feat = build_matrix(test_trainable, extra_columns=extra_columns)

            model = _fit_model("xgboost", train_feat.X, train_feat.y)
            prob = model.predict_proba(test_feat.X)
            metrics = _compact_metrics(compute_metrics(test_feat.y, prob).to_dict())

            family_result[label] = {"metrics": metrics, "fit_seconds": time.time() - t0}
            print(f"  [{split.held_out_family}] {label} PR-AUC={metrics['pr_auc']} ({time.time()-t0:.1f}s)", flush=True)

            del train_feat, test_feat, model
            gc.collect()

        results[split.held_out_family] = family_result
        del train_df, test_df, train_trainable, test_trainable
        gc.collect()
        _mem_log(f"after Phase 6+8 [{split.held_out_family}]", memory_log)

    return results


# =====================================================
# Phase 9 -- family-ID diagnostic. DIAGNOSTIC ONLY -- never proposed
# for production (topology_family is never a live-honest per-candidate
# feature: it is dataset/campaign bookkeeping about WHICH GENERATOR
# built a scenario, not something LiveRuntime observes about a real
# building it has never labeled).
# =====================================================


def phase9_family_id_diagnostic(frame: pd.DataFrame, scenario_metadata: list, phase1_results: Dict[str, Any], memory_log: list) -> Dict[str, Any]:

    print("\n=== PHASE 9: family-ID diagnostic (DIAGNOSTIC ONLY, never for production) ===", flush=True)

    # (a) ordinary in-distribution scenario split, with vs without family-id
    trainable = trainable_rows(frame)
    ord_split = split_scenarios(trainable["scenario_id"].unique(), seed=SEED)
    train_df, val_df, test_df = apply_split(trainable, ord_split)
    assert_no_scenario_overlap(ord_split, train_df, val_df, test_df)

    in_dist = {}
    for label, use_family_id in (("without_family_id", False), ("with_family_id", True)):
        train_feat = build_matrix(train_df, include_family_id=use_family_id)
        test_feat = build_matrix(test_df, include_family_id=use_family_id)
        model = _fit_model("xgboost", train_feat.X, train_feat.y)
        prob = model.predict_proba(test_feat.X)
        metrics = _compact_metrics(compute_metrics(test_feat.y, prob).to_dict())
        in_dist[label] = metrics
        print(f"  [in-distribution, {label}] PR-AUC={metrics['pr_auc']}", flush=True)
        del train_feat, test_feat, model
        gc.collect()

    del train_df, val_df, test_df
    gc.collect()
    _mem_log("after Phase 9 in-distribution", memory_log)

    # (b) family holdout, WITH family-id present in training (held-out
    # family's indicator column is always 0 at test time, since the
    # model never saw that family during training) -- compared against
    # Phase 1's own "without family-id" result for the SAME holdout.
    splits = build_topology_holdout_splits(scenario_metadata)
    holdout_with_id = {}

    for split in splits:

        train_df, test_df = apply_topology_holdout(frame, split)
        train_trainable = trainable_rows(train_df)
        test_trainable = trainable_rows(test_df)

        train_feat = build_matrix(train_trainable, include_family_id=True)
        test_feat = build_matrix(test_trainable, include_family_id=True)

        model = _fit_model("xgboost", train_feat.X, train_feat.y)
        prob = model.predict_proba(test_feat.X)
        metrics = _compact_metrics(compute_metrics(test_feat.y, prob).to_dict())

        holdout_with_id[split.held_out_family] = {
            "pr_auc_with_family_id": metrics["pr_auc"],
            "pr_auc_without_family_id_phase1": phase1_results[split.held_out_family]["xgboost_test_metrics"]["pr_auc"],
        }
        print(f"  [family holdout, {split.held_out_family}] with_family_id PR-AUC={metrics['pr_auc']} "
              f"vs Phase1 without={phase1_results[split.held_out_family]['xgboost_test_metrics']['pr_auc']}", flush=True)

        del train_df, test_df, train_trainable, test_trainable, train_feat, test_feat, model
        gc.collect()

    _mem_log("after Phase 9 holdout", memory_log)

    return {"in_distribution_with_vs_without_family_id": in_dist, "family_holdout_with_family_id": holdout_with_id}


# =====================================================
# Phase 10 -- leave-multiple-families-out scaling test: does transfer
# improve monotonically as MORE distinct families are represented in
# training? train-on-1 (=test on other 3, the mirror image of Phase 1's
# train-on-3/test-on-1) vs train-on-2 (3 complementary pairs, test on
# the other 2) vs train-on-3 (Phase 1 itself, reused not refit).
# =====================================================


def _fit_and_eval(frame: pd.DataFrame, train_families: Tuple[str, ...], test_families: Tuple[str, ...]) -> Dict[str, Any]:

    train_df = trainable_rows(frame[frame["topology_family"].isin(train_families)])
    test_df = trainable_rows(frame[frame["topology_family"].isin(test_families)])

    train_feat = build_matrix(train_df)
    test_feat = build_matrix(test_df)

    if train_feat.y.sum() == 0 or len(set(train_feat.y.tolist())) < 2:
        return {"skipped_reason": "single-class train set", "train_families": train_families, "test_families": test_families}

    model = _fit_model("xgboost", train_feat.X, train_feat.y)
    prob = model.predict_proba(test_feat.X)
    metrics = _compact_metrics(compute_metrics(test_feat.y, prob).to_dict())

    result = {
        "train_families": train_families, "test_families": test_families,
        "train_row_count": int(len(train_feat.y)), "test_row_count": int(len(test_feat.y)),
        "metrics": metrics,
    }

    del train_df, test_df, train_feat, test_feat, model
    gc.collect()

    return result


def phase10_leave_multiple_families_out(frame: pd.DataFrame, phase1_results: Dict[str, Any], memory_log: list) -> Dict[str, Any]:

    print("\n=== PHASE 10: leave-multiple-families-out scaling test ===", flush=True)
    families = CANONICAL_FAMILIES

    train_on_1 = []
    for family in families:
        others = tuple(f for f in families if f != family)
        r = _fit_and_eval(frame, (family,), others)
        train_on_1.append(r)
        print(f"  train_on_1=[{family}] -> avg test PR-AUC={r.get('metrics', {}).get('pr_auc')}", flush=True)
        _mem_log(f"after Phase 10 train_on_1 [{family}]", memory_log)

    # 3 complementary 2-vs-2 pairings, each covering all 4 families
    pairings = [
        (("single_exit_lowrise", "twin_stair_highrise"), ("multi_exit_wide", "v1_topology_fixed")),
        (("single_exit_lowrise", "multi_exit_wide"), ("twin_stair_highrise", "v1_topology_fixed")),
        (("single_exit_lowrise", "v1_topology_fixed"), ("twin_stair_highrise", "multi_exit_wide")),
    ]

    train_on_2 = []
    for train_families, test_families in pairings:
        r = _fit_and_eval(frame, train_families, test_families)
        train_on_2.append(r)
        print(f"  train_on_2={train_families} -> test PR-AUC={r.get('metrics', {}).get('pr_auc')}", flush=True)
        _mem_log(f"after Phase 10 train_on_2 {train_families}", memory_log)

    train_on_3_avg = float(np.mean([
        v["xgboost_test_metrics"]["pr_auc"] for v in phase1_results.values() if v["xgboost_test_metrics"]["pr_auc"] is not None
    ]))
    train_on_2_avg = float(np.mean([r["metrics"]["pr_auc"] for r in train_on_2 if "metrics" in r and r["metrics"]["pr_auc"] is not None]))
    train_on_1_avg = float(np.mean([r["metrics"]["pr_auc"] for r in train_on_1 if "metrics" in r and r["metrics"]["pr_auc"] is not None]))

    print(f"  AVG PR-AUC by training-family-count: 1->{train_on_1_avg:.4f}  2->{train_on_2_avg:.4f}  3(Phase1)->{train_on_3_avg:.4f}", flush=True)

    return {
        "train_on_1_family": train_on_1,
        "train_on_2_families": train_on_2,
        "train_on_3_families_avg_pr_auc_from_phase1": train_on_3_avg,
        "avg_pr_auc_by_training_family_count": {"1": train_on_1_avg, "2": train_on_2_avg, "3": train_on_3_avg},
        "monotonic_improvement": bool(train_on_1_avg <= train_on_2_avg <= train_on_3_avg),
    }


# =====================================================
# Phase 12 -- model robustness: does the family-holdout failure persist
# across algorithms, or is it XGBoost-specific?
# =====================================================


def phase12_model_robustness(frame: pd.DataFrame, scenario_metadata: list, phase1_results: Dict[str, Any], memory_log: list) -> Dict[str, Any]:

    print("\n=== PHASE 12: model robustness (LogisticRegression / HistGradientBoosting / XGBoost) ===", flush=True)
    splits = build_topology_holdout_splits(scenario_metadata)
    results = {}

    for split in splits:

        train_df, test_df = apply_topology_holdout(frame, split)
        train_trainable = trainable_rows(train_df)
        test_trainable = trainable_rows(test_df)

        train_feat = build_matrix(train_trainable)
        test_feat = build_matrix(test_trainable)

        family_result = {"xgboost": phase1_results[split.held_out_family]["xgboost_test_metrics"]["pr_auc"]}

        for algorithm in ("logistic_regression", "gradient_boosting"):
            t0 = time.time()
            model = _fit_model(algorithm, train_feat.X, train_feat.y)
            prob = model.predict_proba(test_feat.X)
            metrics = _compact_metrics(compute_metrics(test_feat.y, prob).to_dict())
            family_result[algorithm] = metrics["pr_auc"]
            print(f"  [{split.held_out_family}] {algorithm} PR-AUC={metrics['pr_auc']} ({time.time()-t0:.1f}s)", flush=True)
            del model
            gc.collect()

        best = max(family_result, key=lambda k: (family_result[k] if family_result[k] is not None else -1))
        family_result["best_generalizer"] = best
        results[split.held_out_family] = family_result

        del train_df, test_df, train_trainable, test_trainable, train_feat, test_feat
        gc.collect()
        _mem_log(f"after Phase 12 [{split.held_out_family}]", memory_log)

    return results


# =====================================================
# Phase 13 -- Target V2 stability across families.
# =====================================================


def phase13_target_stability(frame: pd.DataFrame) -> Dict[str, Any]:

    print("\n=== PHASE 13: Target V2 stability across families ===", flush=True)

    trainable = trainable_rows(frame)
    by_family = trainable.groupby("topology_family", observed=True)

    positive_rate = by_family["target"].mean().to_dict()
    positive_rate_by_type = (
        trainable.groupby(["topology_family", "candidate_type"], observed=True)["target"].mean()
        .unstack("candidate_type").to_dict(orient="index")
    )

    positives = trainable[trainable["target"] == True]  # noqa: E712
    lead_time_stats = positives.groupby("topology_family", observed=True)["lead_time_seconds_v2"].agg(["mean", "median", "count"]).to_dict(orient="index")

    for family in CANONICAL_FAMILIES:
        print(f"  [{family}] positive_rate={positive_rate.get(family)} median_lead_time={lead_time_stats.get(family, {}).get('median')}", flush=True)

    return {
        "positive_rate_by_family": {k: float(v) for k, v in positive_rate.items()},
        "positive_rate_by_family_and_candidate_type": positive_rate_by_type,
        "lead_time_seconds_stats_by_family": lead_time_stats,
    }


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()
    memory_log: list = []
    _mem_log("start", memory_log)

    with open(SCENARIO_METADATA_PATH, encoding="utf-8") as f:
        scenario_metadata = json.load(f)

    print("Loading Dataset V3 (chunked)...", flush=True)
    t0 = time.time()
    frame = _load_dataset_chunked(CSV_PATH)
    print(f"Loaded {len(frame)} rows, {frame['scenario_id'].nunique()} scenarios in {time.time()-t0:.1f}s", flush=True)
    _mem_log("after dataset load", memory_log)

    print("Attaching Phase 6 normalized + Phase 8 graph-context experimental columns...", flush=True)
    graph_context_table = build_graph_context_table()
    frame = add_normalized_features(frame)
    frame = add_graph_context_features(frame, graph_context_table)
    _mem_log("after experimental feature attachment", memory_log)

    trainable_frame = trainable_rows(frame)

    report: Dict[str, Any] = {
        "dataset_manifest": {"csv_path": str(CSV_PATH), "row_count": len(frame), "scenario_count": int(frame["scenario_id"].nunique())},
        "seed": SEED,
    }

    report_path = OUTPUT_DIR / "cross_topology_investigation_report.json"

    def _checkpoint() -> None:
        # Resilience against this milestone's own tight-memory environment
        # (~7.9GB total RAM, frequently <1GB available before this script
        # even starts, per this milestone's own measured baseline): if a
        # later phase hits the MemoryError floor, every EARLIER phase's
        # results are still on disk, not lost.
        report["memory_log"] = memory_log
        report["total_wall_seconds_so_far"] = time.time() - overall_start
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

    report["phase1_and_4_family_holdout_and_failure_regions"] = phase1_and_4_family_holdout(frame, scenario_metadata, memory_log)
    _checkpoint()
    report["phase2_distribution_shift_and_identifiability"] = phase2_distribution_shift_and_identifiability(trainable_frame, memory_log)
    _checkpoint()
    report["phase3_conditional_target_relationships"] = phase3_conditional_target_relationships(trainable_frame)
    _checkpoint()
    report["phase5_candidate_type_specialization"] = phase5_candidate_type_specialization(frame, scenario_metadata, memory_log)
    _checkpoint()
    report["phase6_8_normalization_and_graph_context"] = phase6_8_normalization_and_graph_context(
        frame, scenario_metadata, report["phase1_and_4_family_holdout_and_failure_regions"], memory_log,
    )
    _checkpoint()
    report["phase9_family_id_diagnostic"] = phase9_family_id_diagnostic(
        frame, scenario_metadata, report["phase1_and_4_family_holdout_and_failure_regions"], memory_log,
    )
    _checkpoint()
    report["phase10_leave_multiple_families_out"] = phase10_leave_multiple_families_out(
        frame, report["phase1_and_4_family_holdout_and_failure_regions"], memory_log,
    )
    _checkpoint()
    report["phase12_model_robustness"] = phase12_model_robustness(
        frame, scenario_metadata, report["phase1_and_4_family_holdout_and_failure_regions"], memory_log,
    )
    _checkpoint()
    report["phase13_target_v2_stability"] = phase13_target_stability(frame)

    report["memory_log"] = memory_log
    report["total_wall_seconds"] = time.time() - overall_start

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone in {report['total_wall_seconds']:.1f}s. Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
