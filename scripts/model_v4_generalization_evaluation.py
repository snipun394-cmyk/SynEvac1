"""Predictive Dataset V4 milestone, Phases 20-25 -- FULL-SCALE
exploratory generalization evaluation against the 3,945,171-row Dataset
V4 campaign. INVESTIGATION ONLY -- no Model V4 trained/exported,
nothing wired anywhere, same XGBoost configuration every prior
milestone selected (predictive_model.tree_models.build_tree_models, no
hyperparameter search).

Phase 20: concept-shift reanalysis -- does conditioning on graph
context shrink the cross-family conditional-target-rate spread the
Cross-Topology Generalization Investigation (e8d728a) found at matched
occupancy/flow states?

Phase 21/22: leave-one-family-out (all 6 families), 3 controlled
variants per holdout:
  A_old_family_old_schema -- train on the OTHER OLD families only
    (matching V3's own population), old 12-column schema. Only defined
    for the 4 OLD families (multi_wing/ring_corridor didn't exist in
    V3's population at all).
  B_v4_old_schema -- train on ALL other 5 V4 families (old+new mix),
    old 12-column schema. Isolates the DIVERSITY benefit (A vs B: same
    features, more/different training families).
  C_v4_new_schema -- train on ALL other 5 V4 families, full 15-column
    V4 schema (+graph context). Isolates the REPRESENTATION benefit (B
    vs C: same training population, extra features).
Every variant is also compared against the deterministic-current-state
baseline (Phase 22).

Phase 23: new-family transfer -- multi_wing/ring_corridor's own C
results, examined in detail (precision/recall/FPR/FNR/failure regions).

Phase 24: old-family regression -- compares B/C's 4 old-family numbers
against e8d728a's own Phase 1 numbers (0.254/0.513/0.286/0.497).

Phase 25: graph-context ablation -- for 4 families (2 new + the 2
historically-hardest old ones), each of the 3 graph-context features
added ALONE on top of the old schema, isolating individual contribution.

Usage: python scripts/model_v4_generalization_evaluation.py
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_model.baselines import DeterministicCurrentStateBaseline
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix, trainable_rows
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics
from predictive_model.tree_models import build_tree_models

REPO_ROOT = Path(__file__).resolve().parent.parent
V4_CAMPAIGN_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v4"
CSV_PATH = V4_CAMPAIGN_DIR / "candidate_dataset_v4.csv"

OUTPUT_DIR = V4_CAMPAIGN_DIR
SEED = 20260729
N_JOBS = 2

OLD_FAMILIES = ("single_exit_lowrise", "twin_stair_highrise", "multi_exit_wide", "v1_topology_fixed")
NEW_FAMILIES = ("multi_wing", "ring_corridor")
ALL_FAMILIES = OLD_FAMILIES + NEW_FAMILIES

E8D728A_PHASE1_PR_AUC = {
    "multi_exit_wide": 0.2540, "single_exit_lowrise": 0.5126,
    "twin_stair_highrise": 0.2858, "v1_topology_fixed": 0.4969,
}

GRAPH_CONTEXT_COLUMNS = ("candidate_betweenness_centrality", "candidate_is_bridge", "candidate_upstream_catchment_count")

BASE_FEATURE_NAMES = (
    "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_traversable", "candidate_adjacent_zone_occupancy", "candidate_queue_length",
    "candidate_approaching_count", "candidate_congestion_level",
)
EXPERIMENTAL_FEATURE_NAMES = ("candidate_recent_flow_rate", "candidate_congestion_trend", "candidate_alternative_route_count")

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
    "candidate_betweenness_centrality": "float32",
    "candidate_is_bridge": "bool",
    "candidate_upstream_catchment_count": "int32",
    "lead_time_seconds_v2": "float32",
}

MIN_AVAILABLE_MEMORY_BYTES = 300_000_000


def _mem_log(label: str, memory_log: list) -> None:
    vm = psutil.virtual_memory()
    available_mb = vm.available / 1e6
    print(f"    [mem] {label}: available={available_mb:.0f}MB ({vm.percent:.0f}% used)", flush=True)
    memory_log.append({"label": label, "available_mb": available_mb})
    if vm.available < MIN_AVAILABLE_MEMORY_BYTES:
        raise MemoryError(f"Available system memory critically low ({available_mb:.0f}MB) at step {label!r}.")


def _load_dataset_chunked(csv_path: Path, chunksize: int = 250_000) -> pd.DataFrame:

    keep_columns = (
        ["scenario_id", "observation_time", "candidate_id", "candidate_type", "topology_family", "structural_variant_id"]
        + list(BASE_FEATURE_NAMES) + list(EXPERIMENTAL_FEATURE_NAMES) + list(GRAPH_CONTEXT_COLUMNS)
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
        "target_v2": "target", "currently_congested_v2": "currently_congested",
        "had_any_activity_in_window_v2": "had_any_activity_in_window",
    })

    return frame


def build_matrix(frame: pd.DataFrame, extra_columns: Tuple[str, ...] = ()):

    base = build_experimental_feature_matrix(frame)

    if not extra_columns:
        return base.__class__(X=base.X.astype(np.float32), y=base.y, scenario_ids=base.scenario_ids,
                               candidate_types=base.candidate_types, feature_names=base.feature_names)

    extra_arrays = []
    for column in extra_columns:
        values = frame[column].to_numpy(dtype=float)
        extra_arrays.append(values)

    X = np.column_stack([base.X] + extra_arrays).astype(np.float32)
    names = base.feature_names + extra_columns

    return base.__class__(X=X, y=base.y, scenario_ids=base.scenario_ids, candidate_types=base.candidate_types, feature_names=names)


def _fit_xgboost(X, y):
    model = build_tree_models(seed=SEED, n_jobs=N_JOBS)["xgboost"]
    class_weight_map = compute_class_weight_map(y)
    sample_weight = sample_weights_from_class_weight(y, class_weight_map)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _run_holdout(train_df: pd.DataFrame, test_df: pd.DataFrame, extra_columns: Tuple[str, ...] = ()) -> Dict[str, Any]:

    train_trainable = trainable_rows(train_df)
    test_trainable = trainable_rows(test_df)

    if train_trainable["target"].nunique() < 2 or len(test_trainable) == 0 or test_trainable["target"].nunique() < 2:
        return {"skipped_reason": "insufficient rows or single class"}

    train_feat = build_matrix(train_trainable, extra_columns)
    test_feat = build_matrix(test_trainable, extra_columns)

    t0 = time.time()
    model = _fit_xgboost(train_feat.X, train_feat.y)
    prob = model.predict_proba(test_feat.X)
    metrics = compute_metrics(test_feat.y, prob)

    det_model = DeterministicCurrentStateBaseline(train_feat.feature_names)
    det_model.fit(train_feat.X, train_feat.y)
    det_prob = det_model.predict_proba(test_feat.X)
    det_metrics = compute_metrics(test_feat.y, det_prob)

    cm = metrics.confusion_matrix
    fpr = cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) else None
    fnr = cm["fn"] / (cm["fn"] + cm["tp"]) if (cm["fn"] + cm["tp"]) else None

    result = {
        "pr_auc": metrics.pr_auc, "roc_auc": metrics.roc_auc, "precision": metrics.precision,
        "recall": metrics.recall, "f1": metrics.f1, "false_positive_rate": fpr, "false_negative_rate": fnr,
        "n_train": int(len(train_feat.y)), "n_test": int(len(test_feat.y)), "test_positive_rate": float(test_feat.y.mean()),
        "deterministic_baseline_pr_auc": det_metrics.pr_auc,
        "relative_lift_vs_deterministic": (metrics.pr_auc / det_metrics.pr_auc) if det_metrics.pr_auc else None,
        "fit_seconds": time.time() - t0,
    }

    del train_feat, test_feat, model
    gc.collect()

    return result


def phase20_concept_shift_reanalysis(trainable_frame: pd.DataFrame) -> Dict[str, Any]:

    print("\n=== PHASE 20: concept-shift reanalysis (graph-context conditioning) ===", flush=True)

    occ_bins = [-1, 5, 15, 30, 1e9]
    occ_labels = ["LOW", "MED", "HIGH", "VHIGH"]
    flow_bins = [-1, 0, 2, 5, 10, 1e9]
    flow_labels = ["0", "1-2", "3-5", "6-10", ">10"]
    catch_bins = [-1, 1, 1e9]
    catch_labels = ["LOW_CATCH(<=1)", "HIGH_CATCH(>1)"]

    d = trainable_frame.copy()
    d["_occ_bin"] = pd.cut(d["total_active_occupant_count"].astype(float), bins=occ_bins, labels=occ_labels)
    d["_flow_bin"] = pd.cut(d["candidate_recent_flow_rate"].astype(float), bins=flow_bins, labels=flow_labels)
    d["_catch_bin"] = pd.cut(d["candidate_upstream_catchment_count"].astype(float), bins=catch_bins, labels=catch_labels)

    def _max_spread(group_cols):
        table = d.groupby(group_cols + ["topology_family"], observed=True)["target"].mean().unstack("topology_family")
        n_table = d.groupby(group_cols + ["topology_family"], observed=True)["target"].size().unstack("topology_family")
        spreads = []
        for idx in table.index:
            row = table.loc[idx]
            counts = n_table.loc[idx]
            valid = row[(counts >= 30) & row.notna()]
            if len(valid) >= 2:
                spreads.append(float(valid.max() - valid.min()))
        return {"max_spread": max(spreads) if spreads else None, "mean_spread": (sum(spreads) / len(spreads)) if spreads else None, "n_bins_compared": len(spreads)}

    without_context = _max_spread(["_occ_bin", "_flow_bin"])
    with_context = _max_spread(["_occ_bin", "_flow_bin", "_catch_bin"])

    print(f"  P(target|occ,flow) spread: {without_context}", flush=True)
    print(f"  P(target|occ,flow,catchment) spread: {with_context}", flush=True)

    shrinkage_pct = None
    if without_context["max_spread"] and with_context["max_spread"] is not None:
        shrinkage_pct = 100.0 * (without_context["max_spread"] - with_context["max_spread"]) / without_context["max_spread"]

    return {
        "without_graph_context": without_context,
        "with_graph_context_catchment": with_context,
        "max_spread_shrinkage_pct": shrinkage_pct,
    }


def phase21_22_family_holdout_ABC(frame: pd.DataFrame, memory_log: list) -> Dict[str, Any]:

    print("\n=== PHASE 21/22: leave-one-family-out A(V3-pop/old-schema) vs B(V4-pop/old-schema) vs C(V4-pop/new-schema) ===", flush=True)

    results = {}

    for held_out in ALL_FAMILIES:

        print(f"  --- held out: {held_out} ---", flush=True)
        family_result = {}

        test_df = frame[frame["topology_family"] == held_out]

        if held_out in OLD_FAMILIES:
            train_df_old_pop = frame[frame["topology_family"].isin([f for f in OLD_FAMILIES if f != held_out])]
            family_result["A_old_family_old_schema"] = _run_holdout(train_df_old_pop, test_df, extra_columns=())
            print(f"    A (V3-population, old schema): PR-AUC={family_result['A_old_family_old_schema'].get('pr_auc')}", flush=True)
            del train_df_old_pop
            gc.collect()

        train_df_v4 = frame[frame["topology_family"] != held_out]

        family_result["B_v4_old_schema"] = _run_holdout(train_df_v4, test_df, extra_columns=())
        print(f"    B (V4-population, old schema): PR-AUC={family_result['B_v4_old_schema'].get('pr_auc')}", flush=True)

        family_result["C_v4_new_schema"] = _run_holdout(train_df_v4, test_df, extra_columns=GRAPH_CONTEXT_COLUMNS)
        print(f"    C (V4-population, new schema): PR-AUC={family_result['C_v4_new_schema'].get('pr_auc')}", flush=True)

        a_pr = family_result.get("A_old_family_old_schema", {}).get("pr_auc")
        b_pr = family_result.get("B_v4_old_schema", {}).get("pr_auc")
        c_pr = family_result.get("C_v4_new_schema", {}).get("pr_auc")

        family_result["diversity_benefit_A_to_B_relative_pct"] = (100.0 * (b_pr - a_pr) / a_pr) if (a_pr and b_pr is not None) else None
        family_result["representation_benefit_B_to_C_relative_pct"] = (100.0 * (c_pr - b_pr) / b_pr) if (b_pr and c_pr is not None) else None

        results[held_out] = family_result

        del train_df_v4, test_df
        gc.collect()
        _mem_log(f"after family holdout [{held_out}]", memory_log)

    return results


def phase24_old_family_regression(phase21_results: Dict[str, Any]) -> Dict[str, Any]:

    print("\n=== PHASE 24: old-family regression check (vs e8d728a Phase 1) ===", flush=True)

    comparison = {}
    for family in OLD_FAMILIES:
        e8_pr = E8D728A_PHASE1_PR_AUC[family]
        b_pr = phase21_results[family]["B_v4_old_schema"].get("pr_auc")
        c_pr = phase21_results[family]["C_v4_new_schema"].get("pr_auc")

        comparison[family] = {
            "e8d728a_pr_auc": e8_pr,
            "v4_B_old_schema_pr_auc": b_pr,
            "v4_C_new_schema_pr_auc": c_pr,
            "B_relative_delta_pct": (100.0 * (b_pr - e8_pr) / e8_pr) if b_pr is not None else None,
            "C_relative_delta_pct": (100.0 * (c_pr - e8_pr) / e8_pr) if c_pr is not None else None,
        }
        print(f"  [{family}] e8d728a={e8_pr} -> V4-B={b_pr} ({comparison[family]['B_relative_delta_pct']:.1f}%) "
              f"-> V4-C={c_pr} ({comparison[family]['C_relative_delta_pct']:.1f}%)", flush=True)

    no_catastrophic_regression = all(
        c["B_relative_delta_pct"] is not None and c["B_relative_delta_pct"] > -50.0 and
        c["C_relative_delta_pct"] is not None and c["C_relative_delta_pct"] > -50.0
        for c in comparison.values()
    )

    return {"per_family": comparison, "no_catastrophic_regression_gt_50pct": no_catastrophic_regression}


def phase25_ablation(frame: pd.DataFrame, memory_log: list) -> Dict[str, Any]:

    print("\n=== PHASE 25: graph-context feature ablation (4 families) ===", flush=True)

    ablation_families = ("multi_wing", "ring_corridor", "multi_exit_wide", "twin_stair_highrise")
    results = {}

    for held_out in ablation_families:

        print(f"  --- held out: {held_out} ---", flush=True)
        train_df = frame[frame["topology_family"] != held_out]
        test_df = frame[frame["topology_family"] == held_out]

        family_result = {}
        for label, columns in (
            ("betweenness_only", ("candidate_betweenness_centrality",)),
            ("is_bridge_only", ("candidate_is_bridge",)),
            ("catchment_only", ("candidate_upstream_catchment_count",)),
            ("all_three", GRAPH_CONTEXT_COLUMNS),
        ):
            r = _run_holdout(train_df, test_df, extra_columns=columns)
            family_result[label] = r.get("pr_auc")
            print(f"    [{label}] PR-AUC={r.get('pr_auc')}", flush=True)

        results[held_out] = family_result
        del train_df, test_df
        gc.collect()
        _mem_log(f"after ablation [{held_out}]", memory_log)

    return results


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()
    memory_log: list = []
    _mem_log("start", memory_log)

    print("Loading Dataset V4 (chunked)...", flush=True)
    t0 = time.time()
    frame = _load_dataset_chunked(CSV_PATH)
    print(f"Loaded {len(frame)} rows, {frame['scenario_id'].nunique()} scenarios in {time.time()-t0:.1f}s", flush=True)
    _mem_log("after dataset load", memory_log)

    trainable_frame = trainable_rows(frame)

    report: Dict[str, Any] = {
        "dataset_manifest": {"csv_path": str(CSV_PATH), "row_count": len(frame), "scenario_count": int(frame["scenario_id"].nunique())},
        "seed": SEED,
    }

    report_path = OUTPUT_DIR / "generalization_evaluation_report.json"

    def _checkpoint():
        report["memory_log"] = memory_log
        report["total_wall_seconds_so_far"] = time.time() - overall_start
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

    report["phase20_concept_shift_reanalysis"] = phase20_concept_shift_reanalysis(trainable_frame)
    _checkpoint()

    report["phase21_22_family_holdout_ABC"] = phase21_22_family_holdout_ABC(frame, memory_log)
    _checkpoint()

    report["phase24_old_family_regression"] = phase24_old_family_regression(report["phase21_22_family_holdout_ABC"])
    _checkpoint()

    report["phase25_graph_context_ablation"] = phase25_ablation(frame, memory_log)
    _checkpoint()

    report["memory_log"] = memory_log
    report["total_wall_seconds"] = time.time() - overall_start

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone in {report['total_wall_seconds']:.1f}s. Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
