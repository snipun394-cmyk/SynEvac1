"""Predictive Dataset V3 milestone, Phase 18/19 -- EXPLORATORY
generalization proxy experiment. NOT Model V4 training -- no
hyperparameter search, no export, nothing wired anywhere. Answers one
question: does Dataset V3's added STRUCTURAL diversity improve
unseen-topology generalization relative to Model V3.1's own findings
(data/localized_predictive_model_v3/training_report_v3.json's
topology_holdout section, trained on Dataset V2's 4 fixed graphs)?

A. Leave-one-STRUCTURAL-VARIANT-out (all 16 variants) --
   predictive_model.structural_variant_holdout (new, Phase 18A).
B. Leave-one-topology-FAMILY-out (all 4 families) --
   predictive_model.topology_holdout (REUSED, unmodified, Phase 18B) --
   directly comparable to V3's own family-holdout numbers since it is
   the exact same splitting code.
C. Deterministic-current-state baseline computed at every holdout
   (Phase 19), same predictive_model.baselines.DeterministicCurrentStateBaseline
   used by scripts/train_localized_predictive_model_v3.py.

Same core algorithm as Model V3 (predictive_model.tree_models.build_tree_models's
XGBoost, no hyperparameter search) -- reused verbatim, not reconfigured.

Usage: python scripts/predictive_dataset_v3_generalization_experiment.py
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_model.baselines import DeterministicCurrentStateBaseline
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix, trainable_rows
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics, metrics_by_group
from predictive_model.structural_variant_holdout import (
    apply_structural_variant_holdout,
    assert_no_variant_holdout_overlap,
    build_structural_variant_holdout_splits,
)
from predictive_model.topology_holdout import apply_topology_holdout, assert_no_holdout_overlap, build_topology_holdout_splits
from predictive_model.tree_models import build_tree_models

REPO_ROOT = Path(__file__).resolve().parent.parent
V3_CAMPAIGN_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v3"
CSV_PATH = V3_CAMPAIGN_DIR / "candidate_dataset_v3.csv"
SCENARIO_METADATA_PATH = V3_CAMPAIGN_DIR / "scenario_metadata.json"
V31_TRAINING_REPORT_PATH = REPO_ROOT / "data" / "localized_predictive_model_v3" / "training_report_v3.json"

OUTPUT_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v3_generalization"

SEED = 20260727  # same master seed as the V3 campaign itself
N_JOBS = 2
ALGORITHM_NAME = "xgboost"  # same winning algorithm every prior V2/V2.2/V3 milestone selected

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
    "lead_time_seconds_v2": "float32",
}

MIN_AVAILABLE_MEMORY_BYTES = 300_000_000


def _mem_log(label: str) -> Dict[str, float]:
    vm = psutil.virtual_memory()
    available_mb = vm.available / 1e6
    print(f"    [mem] {label}: available={available_mb:.0f}MB ({vm.percent:.0f}% used)", flush=True)
    if vm.available < MIN_AVAILABLE_MEMORY_BYTES:
        raise MemoryError(f"Available system memory critically low ({available_mb:.0f}MB) at step {label!r}.")
    return {"label": label, "available_mb": available_mb, "percent_used": vm.percent}


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


def _fit_xgboost(X, y, feature_names):
    model = build_tree_models(seed=SEED, n_jobs=N_JOBS)[ALGORITHM_NAME]
    class_weight_map = compute_class_weight_map(y)
    sample_weight = sample_weights_from_class_weight(y, class_weight_map)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _run_one_holdout(label: str, family: str, train_df: pd.DataFrame, test_df: pd.DataFrame, memory_log: list) -> Dict[str, Any]:

    train_trainable = trainable_rows(train_df)
    test_trainable = trainable_rows(test_df)

    train_feat = build_experimental_feature_matrix(train_trainable)
    test_feat = build_experimental_feature_matrix(test_trainable)

    if len(test_feat.y) == 0 or len(train_feat.y) == 0 or len(set(train_feat.y.tolist())) < 2:
        return {"label": label, "family": family, "skipped_reason": "insufficient trainable rows or single-class train set",
                "train_row_count": int(len(train_feat.y)), "test_row_count": int(len(test_feat.y))}

    t0 = time.time()
    model = _fit_xgboost(train_feat.X, train_feat.y, train_feat.feature_names)
    fit_seconds = time.time() - t0

    prob = model.predict_proba(test_feat.X)
    metrics = compute_metrics(test_feat.y, prob, threshold=0.5)
    by_type = metrics_by_group(test_feat.y, prob, test_feat.candidate_types, threshold=0.5)

    det_model = DeterministicCurrentStateBaseline(train_feat.feature_names)
    det_model.fit(train_feat.X, train_feat.y)
    det_prob = det_model.predict_proba(test_feat.X)
    det_metrics = compute_metrics(test_feat.y, det_prob, threshold=0.5)

    result = {
        "label": label,
        "family": family,
        "train_scenario_count": int(train_df["scenario_id"].nunique()),
        "test_scenario_count": int(test_df["scenario_id"].nunique()),
        "train_row_count": int(len(train_feat.y)),
        "test_row_count": int(len(test_feat.y)),
        "test_positive_rate": float(test_feat.y.mean()),
        "xgboost_test_metrics": metrics.to_dict(),
        "xgboost_test_metrics_by_candidate_type": by_type,
        "deterministic_baseline_test_metrics": det_metrics.to_dict(),
        "xgboost_vs_deterministic": {
            "absolute_lift": (metrics.pr_auc - det_metrics.pr_auc) if (metrics.pr_auc is not None and det_metrics.pr_auc is not None) else None,
            "relative_lift": (metrics.pr_auc / det_metrics.pr_auc) if det_metrics.pr_auc else None,
        },
        "fit_seconds": fit_seconds,
    }

    del model, train_feat, test_feat, train_trainable, test_trainable
    gc.collect()
    memory_log.append(_mem_log(f"after holdout [{label}]"))

    return result


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()
    memory_log = [_mem_log("start")]

    with open(SCENARIO_METADATA_PATH, encoding="utf-8") as f:
        scenario_metadata = json.load(f)

    print("Loading Dataset V3 (chunked)...", flush=True)
    t0 = time.time()
    frame = _load_dataset_chunked(CSV_PATH)
    print(f"Loaded {len(frame)} rows, {frame['scenario_id'].nunique()} scenarios in {time.time()-t0:.1f}s", flush=True)
    memory_log.append(_mem_log("after dataset load"))

    # =====================================================
    # Phase 18A -- leave-one-structural-variant-out (all 16)
    # =====================================================

    variant_splits = build_structural_variant_holdout_splits(scenario_metadata)
    variant_results: Dict[str, Any] = {}

    for split in variant_splits:
        print(f"[variant holdout] {split.held_out_variant} (family={split.held_out_family})...", flush=True)
        train_df, test_df = apply_structural_variant_holdout(frame, split)
        assert_no_variant_holdout_overlap(split, train_df, test_df)
        result = _run_one_holdout(split.held_out_variant, split.held_out_family, train_df, test_df, memory_log)
        variant_results[split.held_out_variant] = result
        del train_df, test_df
        gc.collect()
        pr_auc = result.get("xgboost_test_metrics", {}).get("pr_auc")
        print(f"    PR-AUC={pr_auc} ({result.get('fit_seconds', 0):.1f}s)", flush=True)

    # =====================================================
    # Phase 18B -- leave-one-topology-family-out (all 4) -- REUSES
    # predictive_model.topology_holdout unmodified, directly comparable
    # to Model V3.1's own family-holdout numbers.
    # =====================================================

    family_splits = build_topology_holdout_splits(scenario_metadata)
    family_results: Dict[str, Any] = {}

    for split in family_splits:
        print(f"[family holdout] {split.held_out_family}...", flush=True)
        train_df, test_df = apply_topology_holdout(frame, split)
        assert_no_holdout_overlap(split, train_df, test_df)
        result = _run_one_holdout(split.held_out_family, split.held_out_family, train_df, test_df, memory_log)
        family_results[split.held_out_family] = result
        del train_df, test_df
        gc.collect()
        pr_auc = result.get("xgboost_test_metrics", {}).get("pr_auc")
        print(f"    PR-AUC={pr_auc} ({result.get('fit_seconds', 0):.1f}s)", flush=True)

    # =====================================================
    # Phase 18C -- comparison with V3.1's own family-holdout results
    # (trained on Dataset V2's 4 fixed graphs)
    # =====================================================

    v31_family_holdout = {}
    if V31_TRAINING_REPORT_PATH.exists():
        with open(V31_TRAINING_REPORT_PATH, encoding="utf-8") as f:
            v31_report = json.load(f)
        v31_family_holdout = v31_report.get("topology_holdout", {})

    family_comparison = {}
    for family, result in family_results.items():
        v31_pr_auc = v31_family_holdout.get(family, {}).get("test_metrics", {}).get("pr_auc")
        v3_pr_auc = result.get("xgboost_test_metrics", {}).get("pr_auc")
        family_comparison[family] = {
            "v3_dataset_v2_pr_auc": v31_pr_auc,
            "v3_dataset_v3_pr_auc": v3_pr_auc,
            "absolute_delta": (v3_pr_auc - v31_pr_auc) if (v3_pr_auc is not None and v31_pr_auc is not None) else None,
            "relative_delta_pct": (
                100.0 * (v3_pr_auc - v31_pr_auc) / v31_pr_auc
            ) if (v3_pr_auc is not None and v31_pr_auc not in (None, 0)) else None,
        }

    report = {
        "dataset_manifest": {"csv_path": str(CSV_PATH), "row_count": len(frame), "scenario_count": int(frame["scenario_id"].nunique())},
        "algorithm": ALGORITHM_NAME,
        "seed": SEED,
        "phase18a_structural_variant_holdout": variant_results,
        "phase18b_topology_family_holdout": family_results,
        "phase18c_family_holdout_comparison_vs_dataset_v2": family_comparison,
        "memory_log": memory_log,
        "total_wall_seconds": time.time() - overall_start,
    }

    report_path = OUTPUT_DIR / "generalization_experiment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone in {report['total_wall_seconds']:.1f}s. Wrote {report_path}")


if __name__ == "__main__":
    main()
