from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from behaviour_profile_resolver.category import OccupantCategory, occupant_category


# =====================================================
# Phase 2 -- "Adult notices nearby wheelchair user": the *noticing* half
# of that example is a perceivable, structural fact (who is physically
# co-located with whom, and what category are they), computable
# up front from Scenario data alone, before any hazard-aware judgement
# is made. This module computes only that structural pairing -- a pure
# function of (occupant_id, zone_id, category), the same information
# any bystander could observe by looking around a room. The *evaluates/
# decides* half (distance already captured here via same-zone
# proximity; hazard, own safety, time penalty) happens later, at
# decide()-time, in human_decision_engine.engine.HumanDecisionEngine --
# using DecisionContext, which this module never sees.
#
# "Nearby" is modeled as "starts in the same zone" -- this platform's
# own established spatial granularity (Node/Zone graph, not continuous
# coordinates; see behavior_library.assistance_strategies, which never
# modeled geometric distance either). A documented simplification, not
# a validated proxemics model, matching this codebase's own "same
# honesty" convention throughout.
# =====================================================

PAIRING_ASSIST = "ASSIST"
PAIRING_GROUP_FOLLOW = "GROUP_FOLLOW"

# Capable of physically assisting another person -- Adult/Staff/Fire
# Warden are ordinary, mobile, unencumbered civilians (Wheelchair
# Users, Children, and Elderly occupants are never assigned the helper
# role here; a future revision could relax this, but this phase's own
# framing -- "Adult notices..." -- only ever gives the example from a
# physically-capable civilian's perspective).
HELPER_CAPABLE_CATEGORIES = (
    OccupantCategory.ADULT, OccupantCategory.STAFF, OccupantCategory.FIRE_WARDEN,
)

# Needs active physical assistance to evacuate at a normal pace.
ASSIST_TARGET_CATEGORIES = (OccupantCategory.WHEELCHAIR_USER,)

# Benefits from staying with a capable adult without needing to be
# physically carried/pushed -- Phase 5's own "Parent with child" /
# "Adult helping elderly" examples.
GROUP_FOLLOW_CATEGORIES = (OccupantCategory.CHILD, OccupantCategory.ELDERLY)


@dataclass(frozen=True)
class AssistancePairing:

    helper_id: str
    target_id: str
    kind: str  # PAIRING_ASSIST | PAIRING_GROUP_FOLLOW


def compute_dynamic_pairings(occupants: Sequence) -> Tuple[AssistancePairing, ...]:

    # Deterministic: same Scenario occupants always produce the same
    # pairings, regardless of call order (results are sorted by
    # zone_id/occupant_id throughout, never by iteration order of a
    # dict/set). One helper is claimed by at most one target -- a
    # structural stand-in for "time penalty" (a helper already
    # committed to one person cannot simultaneously be paired with a
    # second).

    by_zone: Dict[str, List] = {}

    for occupant in occupants:
        by_zone.setdefault(occupant.zone_id, []).append(occupant)

    pairings: List[AssistancePairing] = []
    claimed_helpers = set()
    claimed_targets = set()

    for zone_id in sorted(by_zone):

        members = sorted(by_zone[zone_id], key=lambda occupant: occupant.occupant_id)

        helpers = [
            occupant for occupant in members
            if occupant_category(occupant.behaviour_profile_id) in HELPER_CAPABLE_CATEGORIES
        ]
        assist_targets = [
            occupant for occupant in members
            if occupant_category(occupant.behaviour_profile_id) in ASSIST_TARGET_CATEGORIES
        ]
        follow_targets = [
            occupant for occupant in members
            if occupant_category(occupant.behaviour_profile_id) in GROUP_FOLLOW_CATEGORIES
        ]

        for target in assist_targets:

            helper = next(
                (h for h in helpers if h.occupant_id not in claimed_helpers), None,
            )

            if helper is None:
                continue

            claimed_helpers.add(helper.occupant_id)
            claimed_targets.add(target.occupant_id)
            pairings.append(AssistancePairing(helper.occupant_id, target.occupant_id, PAIRING_ASSIST))

        for target in follow_targets:

            if target.occupant_id in claimed_targets:
                continue

            helper = next(
                (h for h in helpers if h.occupant_id not in claimed_helpers), None,
            )

            if helper is None:
                continue

            claimed_helpers.add(helper.occupant_id)
            claimed_targets.add(target.occupant_id)
            pairings.append(
                AssistancePairing(helper.occupant_id, target.occupant_id, PAIRING_GROUP_FOLLOW),
            )

    return tuple(pairings)


# =====================================================


def ordered_for_dynamic_pairing(occupants: Sequence, pairings: Sequence[AssistancePairing]) -> List:

    # Same two-bucket stable partition behaviour_profile_resolver.
    # registrar._ordered_for_assistance() already establishes for the
    # scenario-authored case -- every helper must be registered (and
    # resolved) before the specific target they were paired with, so
    # AssistanceAwareRouteChoiceStrategy can read the helper's already-
    # resolved BehaviorDecision out of context.decisions_so_far.

    helper_ids = {pairing.helper_id for pairing in pairings}

    helpers = [occupant for occupant in occupants if occupant.occupant_id in helper_ids]
    others = [occupant for occupant in occupants if occupant.occupant_id not in helper_ids]

    return helpers + others
