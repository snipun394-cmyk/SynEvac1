"""Predictive Dataset V4 milestone, Phases 12-14 -- PILOT-SCALE
diagnostics, run against data/predictive_dataset_campaign_v4_pilot
(600 scenarios, 24 variants, 6 families, 770,358 rows) BEFORE
committing to the full-scale campaign.

Phase 12: graph-context distribution audit -- do the 2 NEW families
broaden the betweenness/is_bridge/upstream_catchment distributions
relative to the 4 OLD families, or do they land in the same range
(which would mean redesigning before scaling up)?

Phase 13: family identifiability -- (A) old canonical (V2.1, 12-field)
features only vs (B) new V4 (15-field, +graph-context) schema,
predicting topology_family (now 6-way, not 4-way).

Phase 14: controlled generalization pilot -- leave-one-family-out,
OLD schema vs NEW graph-context schema, same XGBoost config, no
tuning. Tests especially: a NEW family (multi_wing) held out entirely,
and an OLD family (twin_stair_highrise, V3's hardest) held out
entirely.

INVESTIGATION ONLY -- no Model V4 trained/exported, nothing wired
anywhere. Usage: python scripts/model_v4_pilot_diagnostics.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix, trainable_rows
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics
from predictive_model.scenario_split import split_scenarios
from predictive_model.topology_holdout import apply_topology_holdout, build_topology_holdout_splits
from predictive_model.tree_models import build_tree_models

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v4_pilot"
CSV_PATH = PILOT_DIR / "candidate_dataset_v4.csv"
SCENARIO_METADATA_PATH = PILOT_DIR / "scenario_metadata.json"

OUTPUT_DIR = PILOT_DIR
SEED = 20260729
N_JOBS = 2

GRAPH_CONTEXT_COLUMNS = ("candidate_betweenness_centrality", "candidate_is_bridge", "candidate_upstream_catchment_count")


def _load_pilot_dataset() -> pd.DataFrame:

    frame = pd.read_csv(CSV_PATH)
    frame = frame.rename(columns={
        "target_v2": "target",
        "currently_congested_v2": "currently_congested",
        "had_any_activity_in_window_v2": "had_any_activity_in_window",
    })
    frame["target"] = frame["target"].map({True: True, False: False, "True": True, "False": False, np.nan: None})
    return frame


def build_matrix_with_graph_context(frame: pd.DataFrame):

    base = build_experimental_feature_matrix(frame)

    extra = np.column_stack([
        frame["candidate_betweenness_centrality"].to_numpy(dtype=float),
        frame["candidate_is_bridge"].astype(bool).to_numpy(dtype=float),
        frame["candidate_upstream_catchment_count"].to_numpy(dtype=float),
    ])

    X = np.column_stack([base.X, extra])
    names = base.feature_names + GRAPH_CONTEXT_COLUMNS

    return base.__class__(X=X, y=base.y, scenario_ids=base.scenario_ids, candidate_types=base.candidate_types, feature_names=names)


def _fit_xgboost(X, y):
    model = build_tree_models(seed=SEED, n_jobs=N_JOBS)["xgboost"]
    class_weight_map = compute_class_weight_map(y)
    sample_weight = sample_weights_from_class_weight(y, class_weight_map)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def phase12_distribution_audit(frame: pd.DataFrame) -> Dict[str, Any]:

    print("\n=== PHASE 12: graph-context distribution audit (old vs new families) ===", flush=True)

    old_families = ("single_exit_lowrise", "twin_stair_highrise", "multi_exit_wide", "v1_topology_fixed")
    new_families = ("multi_wing", "ring_corridor")

    old_mask = frame["topology_family"].isin(old_families)
    new_mask = frame["topology_family"].isin(new_families)

    result = {}
    for column in GRAPH_CONTEXT_COLUMNS:
        values = frame[column].astype(float)
        old_vals = values[old_mask]
        new_vals = values[new_mask]
        result[column] = {
            "old_families": {"min": float(old_vals.min()), "max": float(old_vals.max()), "mean": float(old_vals.mean())},
            "new_families": {"min": float(new_vals.min()), "max": float(new_vals.max()), "mean": float(new_vals.mean())},
            "new_families_extend_range": bool(new_vals.max() > old_vals.max() or new_vals.min() < old_vals.min()),
        }
        print(f"  [{column}] old={result[column]['old_families']} new={result[column]['new_families']} "
              f"extends_range={result[column]['new_families_extend_range']}", flush=True)

    return result


def phase13_family_identifiability(trainable_frame: pd.DataFrame) -> Dict[str, Any]:

    print("\n=== PHASE 13: family identifiability (old canonical vs new V4 schema, 6-way) ===", flush=True)

    split = split_scenarios(trainable_frame["scenario_id"].unique(), seed=SEED, ratios=(0.8, 0.0, 0.2))
    train_df = trainable_frame[trainable_frame["scenario_id"].isin(split.train_scenario_ids)]
    test_df = trainable_frame[trainable_frame["scenario_id"].isin(split.test_scenario_ids)]

    families = sorted(trainable_frame["topology_family"].unique())
    label_map = {f: i for i, f in enumerate(families)}
    y_train = train_df["topology_family"].map(label_map).to_numpy()
    y_test = test_df["topology_family"].map(label_map).to_numpy()
    majority_baseline = float(pd.Series(y_test).value_counts(normalize=True).max())

    from xgboost import XGBClassifier

    results = {}
    for label, builder in (("A_old_canonical", build_experimental_feature_matrix), ("B_new_v4_schema", build_matrix_with_graph_context)):

        train_feat = builder(train_df)
        test_feat = builder(test_df)

        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                             eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS)
        clf.fit(train_feat.X, y_train)
        pred = clf.predict(test_feat.X)
        accuracy = float((pred == y_test).mean())

        results[label] = {"accuracy": accuracy, "majority_baseline": majority_baseline, "n_features": train_feat.X.shape[1]}
        print(f"  [{label}] accuracy={accuracy:.4f} (majority baseline={majority_baseline:.4f}, n_features={train_feat.X.shape[1]})", flush=True)

    return results


def phase14_controlled_generalization_pilot(frame: pd.DataFrame) -> Dict[str, Any]:

    print("\n=== PHASE 14: controlled generalization pilot (old vs new schema, leave-one-family-out) ===", flush=True)

    with open(SCENARIO_METADATA_PATH, encoding="utf-8") as f:
        scenario_metadata = json.load(f)

    splits = build_topology_holdout_splits(scenario_metadata)
    results = {}

    for split in splits:

        train_df, test_df = apply_topology_holdout(frame, split)
        train_trainable = trainable_rows(train_df)
        test_trainable = trainable_rows(test_df)

        family_result = {}
        for label, builder in (("old_schema", build_experimental_feature_matrix), ("new_v4_schema", build_matrix_with_graph_context)):

            if train_trainable["target"].nunique() < 2 or test_trainable["target"].nunique() < 2:
                family_result[label] = {"skipped_reason": "insufficient rows or single class"}
                continue

            train_feat = builder(train_trainable)
            test_feat = builder(test_trainable)

            t0 = time.time()
            model = _fit_xgboost(train_feat.X, train_feat.y)
            prob = model.predict_proba(test_feat.X)
            metrics = compute_metrics(test_feat.y, prob)

            family_result[label] = {
                "pr_auc": metrics.pr_auc, "roc_auc": metrics.roc_auc,
                "n_train": int(len(train_feat.y)), "n_test": int(len(test_feat.y)),
                "fit_seconds": time.time() - t0,
            }

        old_pr = family_result.get("old_schema", {}).get("pr_auc")
        new_pr = family_result.get("new_v4_schema", {}).get("pr_auc")
        family_result["relative_delta_pct"] = (
            100.0 * (new_pr - old_pr) / old_pr if (old_pr and new_pr is not None) else None
        )

        results[split.held_out_family] = family_result
        print(f"  [{split.held_out_family}] old_schema PR-AUC={old_pr} new_v4_schema PR-AUC={new_pr} "
              f"delta={family_result['relative_delta_pct']}", flush=True)

    return results


def main() -> None:

    print("Loading pilot dataset...", flush=True)
    frame = _load_pilot_dataset()
    trainable_frame = trainable_rows(frame)
    print(f"Loaded {len(frame)} rows, {len(trainable_frame)} trainable, {frame['scenario_id'].nunique()} scenarios", flush=True)

    report = {
        "dataset_manifest": {"csv_path": str(CSV_PATH), "row_count": len(frame), "trainable_row_count": len(trainable_frame)},
        "phase12_graph_context_distribution_audit": phase12_distribution_audit(frame),
        "phase13_family_identifiability": phase13_family_identifiability(trainable_frame),
        "phase14_controlled_generalization_pilot": phase14_controlled_generalization_pilot(frame),
    }

    report_path = OUTPUT_DIR / "pilot_diagnostics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
