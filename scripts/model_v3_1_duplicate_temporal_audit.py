"""Localized Predictive Model V3.1 milestone, Phases 2-4 -- duplicate-
structure and temporal-redundancy audit at full scale.

Operates on the SAME already-existing eligible (trainable) rows Model
V3 used (data/predictive_congestion_target_v2/candidate_dataset_
relabeled.csv, currently_congested excluded) -- no re-simulation.

Distinguishes explicitly (Phase 1's own requirement) between:
  A. exact duplicate FEATURE vectors (X only) -- what V3's 96.5%
     figure actually measured, on the encoded 27-column model-input
     matrix, TRAIN SPLIT ONLY.
  B. exact duplicate FEATURE+LABEL rows (X and y both identical).
  C. repeated states within the same scenario.
  D. repeated states across different scenarios.
  E. repeated states across different topology families.

This script computes A-E at FULL SCALE (all eligible rows, not just
train), so the Phase 1 "confirm what 96.5% means" question and the
Phase 2/3 duplicate-structure audit are answered from the same, single,
consistent computation.
"""
import gc
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_localized_predictive_model_v3 as v3
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "localized_predictive_model_v3_1"
OUTPUT_PATH = OUTPUT_DIR / "duplicate_temporal_audit.json"

TEMPORAL_LAGS_SECONDS = (5.0, 10.0, 15.0, 20.0)


def _group_ids_for_matrix(X: np.ndarray) -> np.ndarray:
    """Exact-duplicate-row group id per row, via a byte-void view (the
    same technique used for V3's original shuffle-test investigation) --
    O(n log n), no python-level row loop."""

    view = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, inverse = np.unique(view, return_inverse=True)
    return inverse.reshape(-1)


def main() -> None:

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v3._watchdog_log_path = str(OUTPUT_DIR / "memory_watchdog.log")
    with open(v3._watchdog_log_path, "w", encoding="utf-8") as f:
        f.write(f"watchdog started {time.time():.1f}\n")
    threading.Thread(target=v3._memory_watchdog, daemon=True).start()

    print("Loading scenario_metadata.json...")
    with open(v3.SCENARIO_METADATA_PATH, "r", encoding="utf-8") as f:
        scenario_metadata = json.load(f)
    topology_by_scenario = {m["scenario_id"]: m["topology_family"] for m in scenario_metadata}

    print("Loading Target V2 relabeled dataset (chunked)...")
    frame = v3._load_dataset_chunked(v3.CSV_PATH)
    trainable = v3._trainable_rows(frame)
    del frame
    gc.collect()
    print(f"Eligible (trainable) rows: {len(trainable)}, scenarios: {trainable['scenario_id'].nunique()}")

    trainable["topology_family"] = trainable["scenario_id"].astype(str).map(topology_by_scenario)

    # =====================================================
    # Phase 1 confirmation + Phase 2 -- duplicate-feature-vector audit,
    # FULL SCALE (all eligible rows, not just train, unlike V3's
    # original 96.5% figure which was train-split-only).
    # =====================================================

    prepared = build_experimental_feature_matrix(trainable)
    n_rows = len(prepared.X)

    group_id = _group_ids_for_matrix(prepared.X)
    n_unique_feature_vectors = int(group_id.max()) + 1

    group_frame = pd.DataFrame({
        "group_id": group_id,
        "scenario_id": trainable["scenario_id"].astype(str).to_numpy(),
        "topology_family": trainable["topology_family"].to_numpy(),
        "candidate_type": trainable["candidate_type"].astype(str).to_numpy(),
        "target": prepared.y.astype(int),
    })
    del prepared
    gc.collect()

    group_sizes = group_frame.groupby("group_id").size()
    rows_in_duplicated_groups = int(group_sizes[group_sizes > 1].sum())

    # Phase 2: conflicting-label groups -- identical X, different y.
    label_nunique = group_frame.groupby("group_id")["target"].nunique()
    conflicting_groups = label_nunique[label_nunique > 1]
    n_conflicting_groups = int(len(conflicting_groups))
    rows_in_conflicting_groups = int(group_sizes.loc[conflicting_groups.index].sum())

    # A vs B: exact-feature-vector duplicate groups (A) vs exact
    # feature+label duplicate groups (B) -- B further splits every A
    # group with conflicting labels into up to 2 (X,y) sub-groups.
    xy_group_sizes = group_frame.groupby(["group_id", "target"]).size()
    n_unique_feature_label_pairs = int(len(xy_group_sizes))
    rows_in_duplicated_xy_groups = int(xy_group_sizes[xy_group_sizes > 1].sum())

    # Phase 3: within-scenario vs cross-scenario vs cross-topology.
    scenario_nunique = group_frame.groupby("group_id")["scenario_id"].nunique()
    topology_nunique = group_frame.groupby("group_id")["topology_family"].nunique()

    dup_group_ids = group_sizes[group_sizes > 1].index
    dup_scenario_nunique = scenario_nunique.loc[dup_group_ids]
    dup_topology_nunique = topology_nunique.loc[dup_group_ids]

    within_scenario_only_groups = dup_group_ids[(dup_scenario_nunique == 1).to_numpy()]
    cross_scenario_groups = dup_group_ids[(dup_scenario_nunique > 1).to_numpy()]
    cross_topology_groups = dup_group_ids[(dup_topology_nunique > 1).to_numpy()]

    rows_within_scenario_only = int(group_sizes.loc[within_scenario_only_groups].sum())
    rows_cross_scenario = int(group_sizes.loc[cross_scenario_groups].sum())
    rows_cross_topology = int(group_sizes.loc[cross_topology_groups].sum())

    # candidate-type composition + empirical positive prob per group,
    # for the largest N groups (full per-group detail for 93K+ groups
    # would be a huge JSON -- report distributional summary + top 20).
    group_stats = group_frame.groupby("group_id").agg(
        size=("target", "size"),
        positives=("target", "sum"),
        n_scenarios=("scenario_id", "nunique"),
        n_topologies=("topology_family", "nunique"),
    )
    group_stats["positive_rate"] = group_stats["positives"] / group_stats["size"]
    group_stats = group_stats.sort_values("size", ascending=False)

    top_groups = []
    for group_id_value, row in group_stats.head(20).iterrows():
        member_types = group_frame.loc[group_frame["group_id"] == group_id_value, "candidate_type"].value_counts().to_dict()
        top_groups.append({
            "group_id": int(group_id_value),
            "size": int(row["size"]),
            "positives": int(row["positives"]),
            "positive_rate": float(row["positive_rate"]),
            "n_scenarios": int(row["n_scenarios"]),
            "n_topology_families": int(row["n_topologies"]),
            "candidate_type_composition": member_types,
        })

    phase2_3_report = {
        "definition_note": (
            "Feature-vector duplication measured on the encoded 27-column model-input "
            "matrix (predictive_model.feature_prep_v2_1.build_experimental_feature_matrix), "
            "FULL SCALE (all 1,730,976 eligible/trainable rows across train+val+test), "
            "NOT the V3 shuffle-test investigation's train-split-only 96.5% figure -- "
            "see 'v3_original_96pct_figure_context' below for that reconciliation."
        ),
        "total_eligible_rows": n_rows,
        "n_unique_feature_vectors_A": n_unique_feature_vectors,
        "pct_unique_feature_vectors_A": 100.0 * n_unique_feature_vectors / n_rows,
        "pct_rows_in_duplicated_feature_vector_groups_A": 100.0 * rows_in_duplicated_groups / n_rows,
        "largest_duplicate_group_size": int(group_sizes.max()),
        "median_duplicate_group_size_among_duplicated": float(group_sizes[group_sizes > 1].median()),
        "n_duplicate_groups_A": int(len(group_sizes[group_sizes > 1])),
        "n_conflicting_label_groups": n_conflicting_groups,
        "pct_duplicate_groups_with_conflicting_labels": (
            100.0 * n_conflicting_groups / len(group_sizes[group_sizes > 1]) if len(group_sizes[group_sizes > 1]) else 0.0
        ),
        "pct_rows_inside_conflicting_label_groups": 100.0 * rows_in_conflicting_groups / n_rows,
        "n_unique_feature_label_pairs_B": n_unique_feature_label_pairs,
        "pct_rows_in_duplicated_feature_label_pairs_B": 100.0 * rows_in_duplicated_xy_groups / n_rows,
        "within_scenario_only_duplicate_rows_C": rows_within_scenario_only,
        "pct_duplicate_rows_within_scenario_only_C": (
            100.0 * rows_within_scenario_only / rows_in_duplicated_groups if rows_in_duplicated_groups else 0.0
        ),
        "cross_scenario_duplicate_rows_D": rows_cross_scenario,
        "pct_duplicate_rows_cross_scenario_D": (
            100.0 * rows_cross_scenario / rows_in_duplicated_groups if rows_in_duplicated_groups else 0.0
        ),
        "cross_topology_duplicate_rows_E": rows_cross_topology,
        "pct_duplicate_rows_cross_topology_E": (
            100.0 * rows_cross_topology / rows_in_duplicated_groups if rows_in_duplicated_groups else 0.0
        ),
        "top_20_largest_groups": top_groups,
    }

    print(json.dumps({k: v for k, v in phase2_3_report.items() if k != "top_20_largest_groups"}, indent=2))

    del group_frame, group_stats, group_sizes, scenario_nunique, topology_nunique, xy_group_sizes
    gc.collect()

    # =====================================================
    # Phase 4 -- temporal redundancy. For each (scenario_id,
    # candidate_id) trajectory, sorted by observation_time, measure
    # exact-feature-vector match rate at 1/2/3/4-tick lags (dt=5.0s, so
    # this is exactly 5/10/15/20s). ENGINEERING PROXY, not a
    # statistically rigorous ESS calculation -- documented as such.
    # =====================================================

    print("Computing temporal redundancy (this re-derives group_id fresh to avoid holding both in memory)...")
    prepared2 = build_experimental_feature_matrix(trainable)
    group_id2 = _group_ids_for_matrix(prepared2.X)
    del prepared2
    gc.collect()

    traj_frame = pd.DataFrame({
        "scenario_id": trainable["scenario_id"].astype(str).to_numpy(),
        "candidate_id": trainable["candidate_id"].astype(str).to_numpy(),
        "observation_time": trainable["observation_time"].to_numpy(),
        "group_id": group_id2,
    })
    traj_frame = traj_frame.sort_values(["scenario_id", "candidate_id", "observation_time"])

    dt = 5.0
    lag_results = {}
    same_traj = (
        (traj_frame["scenario_id"].to_numpy()[:-1] == traj_frame["scenario_id"].to_numpy()[1:])
        & (traj_frame["candidate_id"].to_numpy()[:-1] == traj_frame["candidate_id"].to_numpy()[1:])
    )

    for seconds in TEMPORAL_LAGS_SECONDS:
        lag_ticks = round(seconds / dt)

        scenario_arr = traj_frame["scenario_id"].to_numpy()
        candidate_arr = traj_frame["candidate_id"].to_numpy()
        time_arr = traj_frame["observation_time"].to_numpy()
        group_arr = traj_frame["group_id"].to_numpy()

        n = len(traj_frame)
        if n <= lag_ticks:
            continue

        same_scenario = scenario_arr[:-lag_ticks] == scenario_arr[lag_ticks:]
        same_candidate = candidate_arr[:-lag_ticks] == candidate_arr[lag_ticks:]
        expected_time_gap = np.isclose(time_arr[lag_ticks:] - time_arr[:-lag_ticks], seconds, atol=1e-6)
        valid_pair = same_scenario & same_candidate & expected_time_gap

        n_valid = int(valid_pair.sum())
        if n_valid == 0:
            lag_results[str(seconds)] = {"n_valid_pairs": 0, "exact_match_rate": None}
            continue

        exact_match = group_arr[:-lag_ticks][valid_pair] == group_arr[lag_ticks:][valid_pair]
        lag_results[str(seconds)] = {
            "n_valid_pairs": n_valid,
            "exact_match_rate": float(exact_match.mean()),
        }
        print(f"  lag={seconds}s: n_valid_pairs={n_valid} exact_match_rate={exact_match.mean():.4f}")

    # Engineering proxies for "effective sample size" (explicitly NOT a
    # statistically rigorous ESS -- clearly labeled proxies only).
    n_unique_trajectories = int(traj_frame.groupby(["scenario_id", "candidate_id"]).ngroups)

    # collapse consecutive identical-group runs within each trajectory
    # to 1 representative row each -- a temporal-redundancy-aware proxy
    # distinct from the plain unique-feature-vector count.
    traj_frame["_run_start"] = (
        (traj_frame["group_id"] != traj_frame["group_id"].shift(1))
        | (traj_frame["scenario_id"] != traj_frame["scenario_id"].shift(1))
        | (traj_frame["candidate_id"] != traj_frame["candidate_id"].shift(1))
    )
    n_temporal_runs = int(traj_frame["_run_start"].sum())

    phase4_report = {
        "note": (
            "ENGINEERING PROXIES, not a statistically rigorous effective-sample-size "
            "estimator -- no autocorrelation-based ESS formula is fit for exact-duplicate "
            "categorical/binned features like this schema, so exact-match rate and a "
            "run-collapse row count are reported instead, both clearly labeled as proxies."
        ),
        "dt_seconds": dt,
        "exact_match_rate_by_lag": lag_results,
        "n_unique_scenario_candidate_trajectories": n_unique_trajectories,
        "n_temporal_runs_proxy": n_temporal_runs,
        "pct_rows_that_are_temporal_run_continuations": 100.0 * (1 - n_temporal_runs / n_rows),
        "n_unique_feature_vectors_for_reference": n_unique_feature_vectors,
    }

    print(json.dumps(phase4_report, indent=2))

    full_report = {
        "phase": "duplicate_structure_and_temporal_redundancy_audit_v3_1",
        "total_wall_seconds": time.time() - t_start,
        "duplicate_structure": phase2_3_report,
        "temporal_redundancy": phase4_report,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Done in {time.time() - t_start:.1f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
