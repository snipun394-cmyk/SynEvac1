from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prediction_evaluation.classification_metrics import compute_classification_metrics
from prediction_evaluation.regression_metrics import compute_regression_metrics


# =====================================================
# Predictive Model Development & Benchmark Campaign milestone, Phase 5 --
# robustness slicing. Slice membership is derived from the LEGACY
# per-scenario feature/ground-truth fields (ignition_zone, total_
# occupants, Exit_N_State, ...) -- never from the canonical live-feature
# schema a model actually trains on. This is deliberate: slicing on a
# field the model doesn't see as an input is exactly what a genuine
# robustness study needs (otherwise a slice boundary could itself leak
# into the model's own features). Reuses this milestone's own
# ai_registry-generated legacy dataset export purely as slicing
# metadata, never as a model input -- consistent with "do not modify
# feature extraction."
#
# Pipeline B (ai_registry) trains against exactly ONE fixed Building
# (ai_registry.training_scenario.make_training_building()) -- so
# "different building layouts" is NOT a testable axis here at all (there
# is only one layout). This is disclosed explicitly wherever this module
# is used, never silently omitted.
# =====================================================


def occupancy_tier(total_occupants: Optional[float]) -> str:

    if total_occupants is None:
        return "unknown"

    value = float(total_occupants)

    if value <= 10:
        return "low"
    if value <= 20:
        return "medium"
    return "high"


def floor_of_ignition(ignition_floor: Optional[str]) -> str:

    if ignition_floor is None:
        return "unknown"

    return "floor_2_upper" if str(ignition_floor) == "floor-2" else "floor_1_ground"


def exit_block_tier(legacy_row: Dict[str, Any]) -> str:

    blocked = 0
    total = 0

    for key in ("Exit_1_State", "Exit_2_State"):

        value = legacy_row.get(key)
        if value is None:
            continue

        total += 1
        if value in (False, "False", "CLOSED", 0):
            blocked += 1

    if total == 0:
        return "unknown"

    return "blocked" if blocked > 0 else "all_open"


def congestion_tier(ground_truth_row: Optional[Dict[str, Any]]) -> str:

    if not ground_truth_row:
        return "unknown"

    exits_over = ground_truth_row.get("exits_exceeding_capacity") or ()
    stairs_over = ground_truth_row.get("stairs_exceeding_capacity") or ()

    return "heavy_congestion" if (len(exits_over) + len(stairs_over)) > 0 else "normal"


def fire_origin_zone(ignition_zone: Optional[str]) -> str:

    return str(ignition_zone) if ignition_zone else "unknown"


CONDITION_AXES = (
    "occupancy_tier", "floor_of_ignition", "exit_block_tier", "congestion_tier", "fire_origin_zone",
)


def build_condition_tags(legacy_row: Dict[str, Any], ground_truth_row: Optional[Dict[str, Any]]) -> Dict[str, str]:

    return {
        "occupancy_tier": occupancy_tier(legacy_row.get("total_occupants")),
        "floor_of_ignition": floor_of_ignition(legacy_row.get("ignition_floor")),
        "exit_block_tier": exit_block_tier(legacy_row),
        "congestion_tier": congestion_tier(ground_truth_row),
        "fire_origin_zone": fire_origin_zone(legacy_row.get("ignition_zone")),
    }


@dataclass(frozen=True)
class RobustnessSliceResult:

    axis: str
    value: str
    sample_count: int
    classification_metrics: Optional[Any] = None
    regression_metrics: Optional[Any] = None


def robustness_report_classification(
    y_true: Sequence[bool], y_pred: Sequence[bool], y_proba: Sequence[float], tags: Sequence[Dict[str, str]],
) -> List[RobustnessSliceResult]:

    results = []

    for axis in CONDITION_AXES:

        values = sorted(set(tag[axis] for tag in tags))

        for value in values:

            indices = [i for i, tag in enumerate(tags) if tag[axis] == value]

            if not indices:
                continue

            metrics = compute_classification_metrics(
                [y_true[i] for i in indices], [y_pred[i] for i in indices],
                [y_proba[i] for i in indices] if y_proba else (),
            )

            results.append(RobustnessSliceResult(
                axis=axis, value=value, sample_count=len(indices), classification_metrics=metrics,
            ))

    return results


def robustness_report_regression(
    y_true: Sequence[float], y_pred: Sequence[float], tags: Sequence[Dict[str, str]],
) -> List[RobustnessSliceResult]:

    results = []

    for axis in CONDITION_AXES:

        values = sorted(set(tag[axis] for tag in tags))

        for value in values:

            indices = [i for i, tag in enumerate(tags) if tag[axis] == value]

            if not indices:
                continue

            metrics = compute_regression_metrics([y_true[i] for i in indices], [y_pred[i] for i in indices])

            results.append(RobustnessSliceResult(
                axis=axis, value=value, sample_count=len(indices), regression_metrics=metrics,
            ))

    return results
