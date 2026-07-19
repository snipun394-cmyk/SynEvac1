from typing import Any, Dict, Mapping, Tuple

from perception.human_inference import InferenceFlag, derive_inference_flags
from perception.models.human_observation import HumanObservation


# =====================================================
# Human Perception Integration -- additive Decision Policy module.
# Mirrors rescue_policy.py's own role (compute_rescue_priorities()
# already ranks occupants by Ground Truth risk/zone-hazard data) one
# layer over: this module ranks currently-*observed* people by
# perception.human_inference.derive_inference_flags()'s own
# HIGH_RESCUE_PRIORITY flag, never by re-deriving its own priority
# rule from HumanObservation's raw fields. Every actual judgment
# ("is this possibly an injury," "is this person high priority") is
# already made by that one derivation function -- this module's only
# job is turning the HIGH_RESCUE_PRIORITY subset of a cycle's
# human_observations into the plain-dict row shape DecisionPolicy's
# other *_decisions/*_priorities fields already use (see
# decision_policy/policy.py's own "every list-shaped field is a plain
# tuple of plain dicts" convention).
#
# Deliberately independent of zone_policy/exit_policy/stair_policy/
# rescue_policy: those all reason from Ground Truth (a completed
# simulation's engineering-known-true state); this module reasons from
# live_system-shaped, possibly-partial, possibly-live perception data.
# Combining the two into one unified priority ordering is future work,
# not attempted here -- human_rescue_priorities is reported as its own
# additive field on DecisionPolicy, alongside (never replacing)
# rescue_priorities/rescue_order.
# =====================================================


def compute_human_rescue_priorities(
    human_observations: Mapping[str, HumanObservation],
) -> Tuple[Dict[str, Any], ...]:

    rows = []

    for person_id in sorted(human_observations):

        observation = human_observations[person_id]
        flags = derive_inference_flags(observation)

        if InferenceFlag.HIGH_RESCUE_PRIORITY not in flags:
            continue

        rows.append({
            "person_id": person_id,
            "zone_id": observation.zone_id,
            "floor_id": observation.floor_id,
            "classification": observation.classification.name,
            "state": observation.state.name if observation.state is not None else None,
            "inference_flags": sorted(flag.name for flag in flags),
            "confidence": observation.confidence,
        })

    return tuple(rows)
