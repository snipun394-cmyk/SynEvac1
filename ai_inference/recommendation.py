from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ai_training.models.bottleneck_model import NO_BOTTLENECK_LABEL

from ai_inference.confidence import estimate_confidence
from ai_inference.loader import LoadedModel


# =====================================================
# Phase 5 -- Recommendation Layer. Merges ONE AI prediction with an
# ALREADY-COMPUTED Decision Policy dict for the SAME scenario -- never
# recomputes, overrides, or replaces a Decision Policy decision. The
# Decision Policy remains the deterministic engineering baseline; this
# module only produces an advisory annotation on top of it.
# =====================================================

# Decision Policy's own literal status/action vocabulary
# (decision_policy/exit_policy.py, stair_policy.py, zone_policy.py) --
# restated here (independently of ai_explainability.comparison, which
# ai_inference must not depend on) as the two-way "is this location
# flagged as concerning" split an AI prediction is compared against.
CONCERNING_EXIT_STATUSES = ("CLOSE", "HIGH_CONGESTION")
CONCERNING_STAIR_STATUSES = ("AVOID", "CONGESTED")
CONCERNING_ZONE_ACTIONS = ("EVACUATE_IMMEDIATELY", "SHELTER_IN_PLACE")

ACTION_ESCALATE = "ESCALATE"
ACTION_REVIEW = "REVIEW"
ACTION_FOLLOW_DECISION_POLICY = "FOLLOW_DECISION_POLICY"


def decision_policy_flagged(
    decision_policy: Dict[str, Any], location_id: str, location_type: Optional[str],
) -> Optional[bool]:

    # Returns True/False if Decision Policy recorded an explicit
    # decision for this exact (location_id, location_type), or None if
    # it recorded none (e.g. location_type=="door", or the id is
    # simply absent from that decision list).

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
class Recommendation:

    ai_prediction: Any
    ai_confidence: Optional[float]
    decision_policy_flagged: Optional[bool]
    agreement: Optional[bool]
    recommended_action: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:

        return {
            "ai_prediction": self.ai_prediction,
            "ai_confidence": self.ai_confidence,
            "decision_policy_flagged": self.decision_policy_flagged,
            "agreement": self.agreement,
            "recommended_action": self.recommended_action,
            "rationale": self.rationale,
        }


# =====================================================


def build_recommendation(
    loaded_model: LoadedModel,
    X_row: Dict[str, Any],
    decision_policy: Dict[str, Any],
    *,
    location_type: Optional[str] = None,
) -> Recommendation:

    # loaded_model must be a BottleneckModel fit with target="location"
    # -- the one AI prediction directly comparable to a single named
    # location the way Decision Policy itself makes decisions.

    prediction = loaded_model.model.predict([X_row])[0]
    confidence, _basis = estimate_confidence(loaded_model, X_row)

    if prediction == NO_BOTTLENECK_LABEL:
        flagged = None
    else:
        flagged = decision_policy_flagged(decision_policy, prediction, location_type)

    recommended_action, rationale = _derive_recommended_action(prediction, flagged)

    return Recommendation(
        ai_prediction=prediction,
        ai_confidence=confidence,
        decision_policy_flagged=flagged,
        agreement=flagged,
        recommended_action=recommended_action,
        rationale=rationale,
    )


# =====================================================


def _derive_recommended_action(prediction: str, flagged: Optional[bool]) -> Tuple[str, str]:

    # str(...), not repr()/!r -- model.predict() can hand back a numpy
    # scalar (np.str_) whose repr ("np.str_('zone-1')") would otherwise
    # leak into this human-readable rationale text.
    location = str(prediction)

    if prediction == NO_BOTTLENECK_LABEL:

        return (
            ACTION_FOLLOW_DECISION_POLICY,
            "AI predicts no bottleneck for this scenario -- defer entirely to Decision "
            "Policy's own recommendations.",
        )

    if flagged is None:

        return (
            ACTION_FOLLOW_DECISION_POLICY,
            f"Decision Policy records no direct decision for '{location}' -- defer to "
            "its zone/exit/stair-level recommendations.",
        )

    if flagged:

        return (
            ACTION_ESCALATE,
            f"AI and Decision Policy agree '{location}' is a concern -- treat as high priority.",
        )

    return (
        ACTION_REVIEW,
        f"AI predicts '{location}' as a potential bottleneck, but Decision Policy did "
        "not flag it -- a human should review before acting.",
    )
