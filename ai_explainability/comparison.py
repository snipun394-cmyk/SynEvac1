from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ai_training.dataset import TrainingDataset
from ai_training.models.bottleneck_model import NO_BOTTLENECK_LABEL, BottleneckModel


# =====================================================
# Phase 4 -- Decision Policy Comparison. Quantifies where a trained
# AI model's predictions agree or disagree with the deterministic
# Decision Policy's OWN already-computed recommendations
# (decision_policy/exit_policy.py, stair_policy.py, zone_policy.py) --
# never recomputes, overrides, or replaces a Decision Policy decision.
# =====================================================

# Decision Policy's own literal status/action vocabulary
# (decision_policy/exit_policy.py, stair_policy.py, zone_policy.py) --
# restated here as the two-way "is this location flagged as
# concerning" split this module compares an AI prediction against.
CONCERNING_EXIT_STATUSES = ("CLOSE", "HIGH_CONGESTION")
CONCERNING_STAIR_STATUSES = ("AVOID", "CONGESTED")
CONCERNING_ZONE_ACTIONS = ("EVACUATE_IMMEDIATELY", "SHELTER_IN_PLACE")


def decision_policy_flagged(
    decision_policy: Dict[str, Any], location_id: str, location_type: Optional[str],
) -> Optional[bool]:

    # Returns True/False if Decision Policy recorded an explicit
    # decision for this exact (location_id, location_type), or None if
    # it recorded none (e.g. location_type=="door" -- Decision Policy
    # makes no direct door-level decision -- or the id simply isn't
    # present in that decision list).

    if location_type == "exit":

        for decision in decision_policy.get("exit_decisions", ()):
            if decision.get("exit_id") == location_id:
                return decision.get("status") in CONCERNING_EXIT_STATUSES

    elif location_type == "stair":

        for decision in decision_policy.get("stair_decisions", ()):
            if decision.get("stair_id") == location_id:
                return decision.get("status") in CONCERNING_STAIR_STATUSES

    elif location_type == "zone":

        for decision in decision_policy.get("zone_decisions", ()):
            if decision.get("zone_id") == location_id:
                return decision.get("action") in CONCERNING_ZONE_ACTIONS

    return None


# =====================================================


@dataclass(frozen=True)
class DecisionPolicyComparisonRecord:

    scenario_id: str
    ai_prediction: str
    ai_confidence: Optional[float]
    decision_policy_flagged: Optional[bool]
    agreement: Optional[bool]  # None when there is nothing comparable for this scenario


# =====================================================


def compare_bottleneck_location_to_decision_policy(
    model: BottleneckModel, dataset: TrainingDataset,
) -> List[DecisionPolicyComparisonRecord]:

    # model must be a BottleneckModel fit with target="location". For
    # every scenario: predicts the bottleneck location, then checks
    # whether Decision Policy's own recorded decision for that exact
    # id (looked up via Ground Truth's own peak_congestion_location_type,
    # which disambiguates zone/exit/stair/door) is "concerning" --
    # agreement=True means the AI's predicted bottleneck location is
    # also a location Decision Policy independently flagged.

    X_rows, _y, _extra = BottleneckModel.build_table(dataset, target="location")
    predictions = model.predict(X_rows)
    confidences = _classification_confidences(model, X_rows)

    records: List[DecisionPolicyComparisonRecord] = []

    for scenario_id, prediction, confidence, ground_truth, decision_policy in zip(
        dataset.scenario_ids,
        predictions,
        confidences,
        dataset.ground_truth_rows(),
        dataset.decision_policy_rows(),
    ):

        if prediction == NO_BOTTLENECK_LABEL or ground_truth is None or decision_policy is None:

            records.append(
                DecisionPolicyComparisonRecord(
                    scenario_id=scenario_id, ai_prediction=prediction, ai_confidence=confidence,
                    decision_policy_flagged=None, agreement=None,
                )
            )
            continue

        location_type = ground_truth.get("peak_congestion_location_type")
        flagged = decision_policy_flagged(decision_policy, prediction, location_type)

        records.append(
            DecisionPolicyComparisonRecord(
                scenario_id=scenario_id, ai_prediction=prediction, ai_confidence=confidence,
                decision_policy_flagged=flagged, agreement=flagged,
            )
        )

    return records


# =====================================================


def _classification_confidences(model, X_rows) -> List[Optional[float]]:

    if not model.is_classifier:
        return [None] * len(X_rows)

    try:
        probabilities = model.predict_proba(X_rows)
    except (TypeError, AttributeError):
        return [None] * len(X_rows)

    return [float(np.max(row)) for row in probabilities]


# =====================================================


def summarize_comparisons(records: List[DecisionPolicyComparisonRecord]) -> Dict[str, Any]:

    comparable = [record for record in records if record.agreement is not None]
    agreements = sum(1 for record in comparable if record.agreement)

    return {
        "total": len(records),
        "comparable": len(comparable),
        "agreements": agreements,
        "disagreements": len(comparable) - agreements,
        "agreement_rate": (agreements / len(comparable)) if comparable else None,
    }
