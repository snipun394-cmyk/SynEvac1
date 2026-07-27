"""Localized Predictive Model V3.1 milestone, Phase 11 -- relational/
normalized feature experiments. Derives ratio features from EXISTING,
already-live-parity-audited raw columns only (no re-simulation, no new
information) and tests them in a controlled comparison against the
canonical V2.1 12-field schema -- primary random-split metrics AND
topology holdout, since the motivating question is specifically
whether scale-invariant (ratio) features transfer better across
topology than the raw absolute values Phase 7/8 showed are strongly
topology-shifted.

Every derived feature here satisfies the milestone's own 4 conditions:
  1. clear engineering meaning (a demand/capacity-style ratio),
  2. computable from state at time t only (no future information --
     every input is one of V3's own existing t-only features),
  3. an honest live counterpart exists (it inherits the SAME live-parity
     status as its raw components, already audited in V3's Phase 22),
  4. not a disguised target (none of these approach anything like
     y itself -- see the leakage correlation check below).

Does NOT modify predictive_model/feature_prep_v2_1.py or the production
schema -- builds an extended matrix locally in this script only.
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
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios
from predictive_model.topology_holdout import apply_topology_holdout, assert_no_holdout_overlap, build_topology_holdout_splits

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "localized_predictive_model_v3_1"
OUTPUT_PATH = OUTPUT_DIR / "relational_feature_experiment_report.json"

RELATIONAL_FEATURE_NAMES = (
    "rel_queue_to_capacity", "rel_flow_to_capacity", "rel_adjacent_occupancy_to_capacity",
    "rel_alt_route_share", "rel_adjacent_occupancy_to_building_occupancy",
)


def _add_relational_features(frame: pd.DataFrame, scenario_route_totals: dict) -> pd.DataFrame:

    out = frame.copy()
    capacity = out["candidate_capacity"].to_numpy(dtype=float)
    capacity_safe = np.where(capacity > 0, capacity, 1.0)

    out["rel_queue_to_capacity"] = out["candidate_queue_length"].to_numpy(dtype=float) / capacity_safe
    out["rel_flow_to_capacity"] = out["candidate_recent_flow_rate"].to_numpy(dtype=float) / capacity_safe
    out["rel_adjacent_occupancy_to_capacity"] = (
        out["candidate_adjacent_zone_occupancy"].to_numpy(dtype=float) / capacity_safe
    )

    total_routes = out["scenario_id"].astype(str).map(scenario_route_totals).to_numpy(dtype=float)
    total_routes_safe = np.where(total_routes > 0, total_routes, 1.0)
    out["rel_alt_route_share"] = out["candidate_alternative_route_count"].to_numpy(dtype=float) / total_routes_safe

    building_occ = out["total_active_occupant_count"].to_numpy(dtype=float)
    building_occ_safe = np.where(building_occ > 0, building_occ, 1.0)
    out["rel_adjacent_occupancy_to_building_occupancy"] = (
        out["candidate_adjacent_zone_occupancy"].to_numpy(dtype=float) / building_occ_safe
    )

    return out


def _build_extended_feature_matrix(frame: pd.DataFrame, scenario_route_totals: dict):

    extended = _add_relational_features(frame, scenario_route_totals)
    base = v3.build_experimental_feature_matrix(extended)

    extra_cols = [extended[name].to_numpy(dtype=np.float32) for name in RELATIONAL_FEATURE_NAMES]
    X_extra = np.column_stack(extra_cols)
    X_full = np.concatenate([base.X, X_extra], axis=1)
    feature_names_full = tuple(base.feature_names) + RELATIONAL_FEATURE_NAMES

    return type(base)(X=X_full, y=base.y, scenario_ids=base.scenario_ids,
                       candidate_types=base.candidate_types, feature_names=feature_names_full)


def _fit_xgboost(X, y, sample_weight):
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
        eval_metric="logloss", scale_pos_weight=1.0, random_state=v3.SEED, n_jobs=v3.N_JOBS,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def main() -> None:

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v3._watchdog_log_path = str(OUTPUT_DIR / "memory_watchdog_relational.log")
    with open(v3._watchdog_log_path, "w", encoding="utf-8") as f:
        f.write(f"watchdog started {time.time():.1f}\n")
    threading.Thread(target=v3._memory_watchdog, daemon=True).start()

    with open(v3.SCENARIO_METADATA_PATH, "r", encoding="utf-8") as f:
        scenario_metadata = json.load(f)
    scenario_route_totals = {
        m["scenario_id"]: max(1, m["door_count"] + m["exit_count"] + m["stair_count"]) for m in scenario_metadata
    }

    print("Loading dataset...")
    frame = v3._load_dataset_chunked(v3.CSV_PATH)
    full_trainable = v3._trainable_rows(frame)
    del frame
    gc.collect()

    split = split_scenarios(full_trainable["scenario_id"].astype(str).unique().tolist(), seed=v3.SEED)
    train_df, val_df, test_df = apply_split(full_trainable, split)
    assert_no_scenario_overlap(split, train_df, val_df, test_df)

    report = {"primary": {}, "topology_holdout": {}, "leakage_correlation_check": {}}

    for schema in ("canonical", "extended_relational"):

        print(f"=== Primary: {schema} ===")
        t0 = time.time()

        if schema == "canonical":
            train_feat = v3.build_experimental_feature_matrix(train_df)
            test_feat = v3.build_experimental_feature_matrix(test_df)
        else:
            train_feat = _build_extended_feature_matrix(train_df, scenario_route_totals)
            test_feat = _build_extended_feature_matrix(test_df, scenario_route_totals)

        class_weight_map = compute_class_weight_map(train_feat.y)
        sample_weight = sample_weights_from_class_weight(train_feat.y, class_weight_map)
        model = _fit_xgboost(train_feat.X, train_feat.y, sample_weight)
        prob = model.predict_proba(test_feat.X)[:, 1]
        metrics = compute_metrics(test_feat.y, prob, threshold=0.5)

        report["primary"][schema] = {"test_metrics": metrics.to_dict(), "seconds": time.time() - t0}
        print(f"  PR-AUC={metrics.pr_auc:.4f} ROC-AUC={metrics.roc_auc:.4f} ({time.time()-t0:.1f}s)")

        if schema == "extended_relational":
            # leakage correlation check for the 5 NEW columns only
            for i, name in enumerate(RELATIONAL_FEATURE_NAMES):
                col = train_feat.X[:, len(train_feat.feature_names) - len(RELATIONAL_FEATURE_NAMES) + i]
                y = train_feat.y.astype(float)
                std_c, std_y = col.std(), y.std()
                r = float(np.corrcoef(col, y)[0, 1]) if std_c > 0 and std_y > 0 else None
                report["leakage_correlation_check"][name] = r

        del train_feat, test_feat, model, prob
        gc.collect()

    holdout_splits = build_topology_holdout_splits(scenario_metadata)

    for schema in ("canonical", "extended_relational"):

        report["topology_holdout"][schema] = {}

        for holdout in holdout_splits:

            print(f"=== Topology holdout: {schema}, held out [{holdout.held_out_family}] ===")
            t0 = time.time()

            holdout_train_df, holdout_test_df = apply_topology_holdout(full_trainable, holdout)
            assert_no_holdout_overlap(holdout, holdout_train_df, holdout_test_df)

            if schema == "canonical":
                train_feat = v3.build_experimental_feature_matrix(holdout_train_df)
                test_feat = v3.build_experimental_feature_matrix(holdout_test_df)
            else:
                train_feat = _build_extended_feature_matrix(holdout_train_df, scenario_route_totals)
                test_feat = _build_extended_feature_matrix(holdout_test_df, scenario_route_totals)

            class_weight_map = compute_class_weight_map(train_feat.y)
            sample_weight = sample_weights_from_class_weight(train_feat.y, class_weight_map)
            model = _fit_xgboost(train_feat.X, train_feat.y, sample_weight)
            prob = model.predict_proba(test_feat.X)[:, 1]
            metrics = compute_metrics(test_feat.y, prob, threshold=0.5)

            report["topology_holdout"][schema][holdout.held_out_family] = {
                "test_metrics": metrics.to_dict(), "seconds": time.time() - t0,
            }
            print(f"  PR-AUC={metrics.pr_auc:.4f} ROC-AUC={metrics.roc_auc:.4f} ({time.time()-t0:.1f}s)")

            del holdout_train_df, holdout_test_df, train_feat, test_feat, model, prob
            gc.collect()

    report["relational_feature_names"] = RELATIONAL_FEATURE_NAMES
    report["total_wall_seconds"] = time.time() - t_start

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Done in {time.time() - t_start:.1f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
