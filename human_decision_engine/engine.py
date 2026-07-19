from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from hazard.severity import HazardSeverity

from behavior_library.assistance_strategies import (
    ASSISTANCE_ESCORT,
    ASSISTANCE_PUSH_WHEELCHAIR,
)

from human_decision_engine.events import DecisionEventLog, HELP_DECISION, HELP_REJECTED
from human_decision_engine.pairing import AssistancePairing, PAIRING_ASSIST, PAIRING_GROUP_FOLLOW


# =====================================================
# Phase 1/2 -- Civilian Decision Engine / Assistance Decision.
#
# One HumanDecisionEngine instance is shared across every civilian
# registered in a session (the same "one strategy instance reused
# across every occupant" convention DEFAULT_PROFILE_REGISTRY's own
# BehaviorProfileTemplate already establishes) -- this is what lets a
# later-registered occupant's decision depend on an earlier one's
# already-resolved outcome (mirroring context.decisions_so_far's own
# established role, just carrying richer, engine-owned state:
# structural pairing + hazard-aware outcome + a shared event log).
#
# Own-safety/time-penalty is modeled as a hazard-severity threshold at
# the *helper's own* zone (context.hazard_snapshot -- the Dynamic
# Hazard Layer's documented point of contact with Human Behavior;
# absent entirely when no hazard is in play, degrading to "always
# clear to help", exactly like every other hazard-aware strategy in
# this codebase already does when context.hazard_snapshot is None).
# Distance is captured structurally by pairing.py's same-zone-only
# pairing; "relationship" (Phase 2's own parenthetical "future
# extension") has no signal to read yet, so it is not modeled --
# never faked.
# =====================================================

ACTION_EVACUATE = "EVACUATE"
ACTION_HELP = "HELP"
ACTION_BE_ASSISTED = "BE_ASSISTED"
ACTION_IGNORE = "IGNORE"
ACTION_DELAY = "DELAY"

_ASSISTANCE_TYPE_BY_PAIRING_KIND = {
    PAIRING_ASSIST: ASSISTANCE_PUSH_WHEELCHAIR,
    PAIRING_GROUP_FOLLOW: ASSISTANCE_ESCORT,
}

# Own-zone hazard severity that makes a would-be helper hesitate
# (DELAY) or decline outright (IGNORE) rather than commit -- the same
# "illustrative, not validated" threshold-based honesty this codebase
# already applies to FALL_HAZARD_SCORE_THRESHOLD/RUNNING_HAZARD_SCORE_
# THRESHOLD (ground_truth.human_behavior).
_DELAY_AT_OR_ABOVE = HazardSeverity.HIGH
_IGNORE_AT_OR_ABOVE = HazardSeverity.CRITICAL

_SEVERITY_ORDINAL = {
    HazardSeverity.NONE: 0,
    HazardSeverity.LOW: 1,
    HazardSeverity.MODERATE: 2,
    HazardSeverity.HIGH: 3,
    HazardSeverity.CRITICAL: 4,
}


@dataclass(frozen=True)
class CivilianDecision:

    action: str
    target_id: Optional[str] = None
    assistance_type: Optional[str] = None
    reason: str = ""


def _own_hazard_severity(context) -> Optional[HazardSeverity]:

    if context.hazard_snapshot is None:
        return None

    return context.hazard_snapshot.node_state(context.start_id).severity


class HumanDecisionEngine:

    def __init__(self, pairings: Sequence[AssistancePairing], event_log: Optional[DecisionEventLog] = None):

        self._target_by_helper: Dict[str, str] = {p.helper_id: p.target_id for p in pairings}
        self._kind_by_helper: Dict[str, str] = {p.helper_id: p.kind for p in pairings}
        self._helper_by_target: Dict[str, str] = {p.target_id: p.helper_id for p in pairings}

        self._decisions: Dict[str, CivilianDecision] = {}

        self.event_log = event_log if event_log is not None else DecisionEventLog()

    # =====================================================

    def evaluate(self, context) -> CivilianDecision:

        occupant_id = context.profile.occupant_id

        cached = self._decisions.get(occupant_id)

        if cached is not None:
            return cached

        target_id = self._target_by_helper.get(occupant_id)
        helper_id = self._helper_by_target.get(occupant_id)

        if target_id is not None:
            decision = self._evaluate_helper(context, occupant_id, target_id)
        elif helper_id is not None:
            decision = self._evaluate_target(occupant_id, helper_id)
        else:
            decision = CivilianDecision(action=ACTION_EVACUATE, reason="not_paired_this_session")

        self._decisions[occupant_id] = decision

        return decision

    # =====================================================

    def decision_for(self, occupant_id: str) -> Optional[CivilianDecision]:

        # Read-only lookup for a strategy stage that runs *after*
        # decide() has already called evaluate() once (route choice/
        # pre-movement delay) -- never triggers a fresh evaluation, so
        # calling it before decide() simply returns None.

        return self._decisions.get(occupant_id)

    # =====================================================

    def _evaluate_helper(self, context, helper_id, target_id) -> CivilianDecision:

        severity = _own_hazard_severity(context)
        ordinal = _SEVERITY_ORDINAL.get(severity, 0)

        if ordinal >= _SEVERITY_ORDINAL[_IGNORE_AT_OR_ABOVE]:
            action, reason = ACTION_IGNORE, "own_safety_hazard_critical"
        elif ordinal >= _SEVERITY_ORDINAL[_DELAY_AT_OR_ABOVE]:
            action, reason = ACTION_DELAY, "hazard_present_hesitates"
        else:
            action, reason = ACTION_HELP, "clear_to_assist"

        kind = self._kind_by_helper.get(helper_id, PAIRING_ASSIST)
        assistance_type = _ASSISTANCE_TYPE_BY_PAIRING_KIND.get(kind, ASSISTANCE_PUSH_WHEELCHAIR)

        if action == ACTION_HELP:

            self.event_log.record(
                HELP_DECISION, helper_id, related_occupant_id=target_id,
                reason=reason, metadata={"assistance_type": assistance_type, "pairing_kind": kind},
            )

            return CivilianDecision(
                action=ACTION_HELP, target_id=target_id, assistance_type=assistance_type, reason=reason,
            )

        self.event_log.record(
            HELP_REJECTED, helper_id, related_occupant_id=target_id,
            reason=reason, metadata={"pairing_kind": kind},
        )

        return CivilianDecision(action=action, reason=reason)

    # =====================================================

    def _evaluate_target(self, occupant_id, helper_id) -> CivilianDecision:

        helper_decision = self._decisions.get(helper_id)

        if helper_decision is not None and helper_decision.action == ACTION_HELP:

            return CivilianDecision(
                action=ACTION_BE_ASSISTED,
                target_id=helper_id,
                assistance_type=helper_decision.assistance_type,
                reason="claimed_by_helper",
            )

        # The helper declined (IGNORE/DELAY), was never resolved this
        # session (ordering violation), or is otherwise unavailable --
        # the target proceeds independently rather than being left
        # stranded on an assumption nobody ever confirmed.
        return CivilianDecision(action=ACTION_EVACUATE, reason="helper_unavailable")
