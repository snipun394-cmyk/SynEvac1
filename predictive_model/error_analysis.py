from typing import Any, Dict, List

import numpy as np
import pandas as pd

from predictive_dataset.label_analysis import HIGH_OCCUPANCY_MIN, LOW_OCCUPANCY_MAX, occupancy_bucket


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 9 --
# error analysis. Slices false positives/false negatives by the same
# operational dimensions the task asked for: occupancy level, blocked
# routes, candidate type, temporal phase (a proxy for "late
# congestion"), and simultaneous-bottleneck count. Reuses
# predictive_dataset.label_analysis's own occupancy_bucket/threshold
# constants rather than re-tuning a second "what counts as high
# occupancy" definition.
# =====================================================


def _phase_of(observation_time: float, duration) -> str:
    """Mirrors predictive_dataset.label_analysis._phase_of's own EARLY/
    MID/LATE (each scenario's own duration thirds) definition exactly --
    not re-imported since that function is a private module-level
    helper there, but the same thresholds (1/3, 2/3) and UNKNOWN
    fallback for a missing/zero duration."""

    if not duration or duration <= 0:
        return "UNKNOWN"

    fraction = observation_time / duration

    if fraction < 1.0 / 3.0:
        return "EARLY"
    if fraction < 2.0 / 3.0:
        return "MID"
    return "LATE"


def simultaneous_bottleneck_counts(full_horizon_frame: pd.DataFrame) -> pd.Series:
    """Per (scenario_id, observation_time), how many DISTINCT candidates
    have target==True (real future congestion) -- ground truth, not a
    prediction. Indexed by (scenario_id, observation_time) for joining
    back onto any row subset of that same horizon's frame."""

    positive_rows = full_horizon_frame[full_horizon_frame["target"] == True]  # noqa: E712
    counts = positive_rows.groupby(["scenario_id", "observation_time"])["candidate_id"].nunique()
    return counts


def build_error_analysis(
    test_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    scenario_metadata: List[Dict[str, Any]],
    full_horizon_frame: pd.DataFrame,
) -> Dict[str, Any]:
    """test_frame is the trainable-rows subset of the test split, row-
    aligned with y_true/y_pred/y_prob. full_horizon_frame is that SAME
    horizon's full frame (all scenarios/ticks/candidates, not just test)
    -- used only to compute simultaneous-bottleneck counts, itself a
    property of the SCENARIO/TIME, not of the test split."""

    analysis = test_frame.copy()
    analysis["y_true"] = y_true
    analysis["y_pred"] = y_pred
    analysis["y_prob"] = y_prob
    analysis["is_false_positive"] = (y_true == 0) & (y_pred == 1)
    analysis["is_false_negative"] = (y_true == 1) & (y_pred == 0)

    metadata_by_scenario = {entry["scenario_id"]: entry for entry in scenario_metadata}

    analysis["blocked_route_count"] = analysis["scenario_id"].map(
        lambda sid: (
            metadata_by_scenario[sid]["blocked_door_count"]
            + metadata_by_scenario[sid]["blocked_exit_count"]
            + metadata_by_scenario[sid]["unavailable_stair_count"]
        ) if sid in metadata_by_scenario else None
    )
    analysis["has_blocked_route"] = analysis["blocked_route_count"] > 0

    analysis["occupancy_bucket"] = analysis["total_active_occupant_count"].apply(
        lambda count: occupancy_bucket(count)
    )

    analysis["temporal_phase"] = analysis.apply(
        lambda row: _phase_of(
            row["observation_time"],
            metadata_by_scenario.get(row["scenario_id"], {}).get("evacuation_duration"),
        ),
        axis=1,
    )

    bottleneck_counts = simultaneous_bottleneck_counts(full_horizon_frame)
    analysis["simultaneous_bottleneck_count"] = analysis.apply(
        lambda row: int(bottleneck_counts.get((row["scenario_id"], row["observation_time"]), 0)), axis=1,
    )
    analysis["multiple_simultaneous_bottlenecks"] = analysis["simultaneous_bottleneck_count"] >= 2

    def _rate_by(column: str) -> Dict[str, Any]:
        report = {}
        for value, group in analysis.groupby(column, observed=True):
            n = len(group)
            report[str(value)] = {
                "n": int(n),
                "false_positive_rate": float(group["is_false_positive"].sum() / n) if n else None,
                "false_negative_rate": float(group["is_false_negative"].sum() / n) if n else None,
            }
        return report

    return {
        "total_rows": int(len(analysis)),
        "total_false_positives": int(analysis["is_false_positive"].sum()),
        "total_false_negatives": int(analysis["is_false_negative"].sum()),
        "overall_false_positive_rate": float(analysis["is_false_positive"].mean()),
        "overall_false_negative_rate": float(analysis["is_false_negative"].mean()),
        "by_candidate_type": _rate_by("candidate_type"),
        "by_occupancy_bucket": _rate_by("occupancy_bucket"),
        "occupancy_thresholds": {"low_max": LOW_OCCUPANCY_MAX, "high_min": HIGH_OCCUPANCY_MIN},
        "by_blocked_route": _rate_by("has_blocked_route"),
        "by_temporal_phase": _rate_by("temporal_phase"),
        "by_multiple_simultaneous_bottlenecks": _rate_by("multiple_simultaneous_bottlenecks"),
    }
