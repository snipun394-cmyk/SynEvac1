from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from predictive_dataset.label_analysis import HIGH_OCCUPANCY_MIN, LOW_OCCUPANCY_MAX, occupancy_bucket
from predictive_model.error_analysis import simultaneous_bottleneck_counts
from predictive_model.metrics import compute_metrics


# =====================================================
# Localized Predictive Model V2 milestone, Phases 8-10 -- the three
# operational slices this milestone specifically asks for beyond
# candidate type (which predictive_model.metrics.metrics_by_group
# already covers): simultaneous-bottleneck multiplicity, building
# occupancy level, and single-exit vs. multi-exit topology. Built as a
# separate, additive module rather than editing predictive_model.
# error_analysis.build_error_analysis (V1, frozen) -- reuses that
# module's simultaneous_bottleneck_counts() ground-truth computation
# (a scenario/tick property, not a per-model artifact) but computes full
# ClassificationMetrics (ROC-AUC/PR-AUC/F1/etc, not just FP/FN rates)
# per slice via predictive_model.metrics.compute_metrics, since this
# milestone's Phase 8 explicitly asks for the full metric set per slice,
# not just error rates.
# =====================================================


def annotate_operational_slices(
    test_trainable: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    scenario_metadata: Sequence[Dict[str, Any]],
    full_horizon_frame: pd.DataFrame,
) -> pd.DataFrame:
    """test_trainable is the trainable-rows subset of a test/holdout
    split, row-aligned with y_true/y_prob. full_horizon_frame is the
    FULL single-horizon frame (every scenario, not just this split) --
    simultaneous-bottleneck count is a property of the scenario/tick,
    computed from ground truth across all scenarios, not just the ones
    in this particular split."""

    analysis = test_trainable.reset_index(drop=True).copy()
    analysis["y_true"] = y_true
    analysis["y_prob"] = y_prob
    analysis["y_pred"] = (y_prob >= threshold).astype(int)

    metadata_by_scenario = {entry["scenario_id"]: entry for entry in scenario_metadata}

    analysis["total_occupants_scenario"] = analysis["scenario_id"].astype(str).map(
        lambda sid: metadata_by_scenario.get(sid, {}).get("total_occupants")
    )
    analysis["occupancy_bucket"] = analysis["total_occupants_scenario"].apply(
        lambda count: occupancy_bucket(int(count)) if count is not None else "UNKNOWN"
    )

    analysis["exit_count"] = analysis["scenario_id"].astype(str).map(
        lambda sid: metadata_by_scenario.get(sid, {}).get("exit_count")
    )
    analysis["exit_topology"] = analysis["exit_count"].apply(
        lambda count: "single_exit" if count == 1 else ("multi_exit" if count is not None else "UNKNOWN")
    )

    bottleneck_counts = simultaneous_bottleneck_counts(full_horizon_frame).rename(
        "simultaneous_bottleneck_count"
    ).reset_index()
    analysis = analysis.merge(bottleneck_counts, on=["scenario_id", "observation_time"], how="left")
    analysis["simultaneous_bottleneck_count"] = analysis["simultaneous_bottleneck_count"].fillna(0).astype(int)

    def _bottleneck_category(count: int) -> str:
        if count <= 0:
            return "none"
        if count == 1:
            return "single"
        return "multiple"

    analysis["bottleneck_category"] = analysis["simultaneous_bottleneck_count"].apply(_bottleneck_category)

    return analysis


def _metrics_with_rates(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:

    metrics = compute_metrics(y_true, y_prob, threshold=threshold).to_dict()
    cm = metrics["confusion_matrix"]

    n_pos = cm["tp"] + cm["fn"]
    n_neg = cm["tn"] + cm["fp"]

    metrics["false_negative_rate"] = (cm["fn"] / n_pos) if n_pos else None
    metrics["false_positive_rate"] = (cm["fp"] / n_neg) if n_neg else None

    return metrics


def slice_report(analysis: pd.DataFrame, slice_column: str, threshold: float) -> Dict[str, Any]:
    """Full ClassificationMetrics (+ FN/FP rate) per distinct value of
    slice_column, computed over ALL candidate types combined."""

    report = {}

    for value, group in analysis.groupby(slice_column, observed=True):
        report[str(value)] = _metrics_with_rates(
            group["y_true"].to_numpy(), group["y_prob"].to_numpy(), threshold,
        )

    return report


def slice_report_by_candidate_type(analysis: pd.DataFrame, slice_column: str, threshold: float) -> Dict[str, Any]:
    """Nested {slice_value: {candidate_type: metrics}} -- the same slice,
    broken down per Door/Exit/Stair, since Phase 7's central concern
    (does an aggregate metric hide poor Stair performance) applies to
    every other slice too."""

    report: Dict[str, Any] = {}

    for slice_value, slice_group in analysis.groupby(slice_column, observed=True):

        by_type: Dict[str, Any] = {}
        for candidate_type, type_group in slice_group.groupby("candidate_type", observed=True):
            by_type[str(candidate_type)] = _metrics_with_rates(
                type_group["y_true"].to_numpy(), type_group["y_prob"].to_numpy(), threshold,
            )

        report[str(slice_value)] = by_type

    return report


def build_operational_slice_report(
    test_trainable: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    scenario_metadata: Sequence[Dict[str, Any]],
    full_horizon_frame: pd.DataFrame,
) -> Dict[str, Any]:

    analysis = annotate_operational_slices(
        test_trainable, y_true, y_prob, threshold, scenario_metadata, full_horizon_frame,
    )

    return {
        "occupancy_thresholds": {"low_max": LOW_OCCUPANCY_MAX, "high_min": HIGH_OCCUPANCY_MIN},
        "by_bottleneck_category": slice_report(analysis, "bottleneck_category", threshold),
        "by_bottleneck_category_and_candidate_type": slice_report_by_candidate_type(analysis, "bottleneck_category", threshold),
        "by_occupancy_bucket": slice_report(analysis, "occupancy_bucket", threshold),
        "by_occupancy_bucket_and_candidate_type": slice_report_by_candidate_type(analysis, "occupancy_bucket", threshold),
        "by_exit_topology": slice_report(analysis, "exit_topology", threshold),
        "by_exit_topology_and_candidate_type": slice_report_by_candidate_type(analysis, "exit_topology", threshold),
    }
