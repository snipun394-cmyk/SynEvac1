from enum import Enum, auto
from typing import FrozenSet

from perception.models.human_observation import BehaviorEvent, HumanClassification, HumanObservation, HumanState


class InferenceFlag(Enum):

    # Phase 5 -- higher-level, *hedged* inferences: every one of these
    # is a "possible"/"priority" judgment, never a fact. None of them
    # is ever set directly by a perception source (see
    # HumanObservation's own fields -- there is no `inference_flags`
    # constructor parameter on that type at all) -- the only way a
    # flag exists is by passing an already-built HumanObservation
    # through derive_inference_flags() below, which is the one place
    # in this codebase allowed to compute one. This is the same
    # "derived, never stored redundantly" discipline
    # HazardNodeState.severity/BuildingObservation's own severity
    # classification already apply, one level further hedged: those
    # derive a classification from a score; this derives a *maybe*
    # from a behavior.

    POSSIBLE_PRE_MOVEMENT_DELAY = auto()
    POSSIBLE_INJURY = auto()
    HIGH_RESCUE_PRIORITY = auto()


# =====================================================
# States/classifications this module treats as evidence toward a
# flag -- named constants, not inlined magic tuples, so the *reason*
# a flag fires is documented once, here, rather than re-derived by
# reading conditional logic.
# =====================================================

# NEVER_MOVING_YET is the literal, direct trigger Phase 3's own
# docstring names for this flag -- no other HumanState implies it.
_PRE_MOVEMENT_DELAY_STATES = frozenset({HumanState.NEVER_MOVING_YET})

# Every one of these describes a person no longer moving normally
# under their own power -- an honest behavioral pattern consistent
# with (never a claim of) injury. Phase 4's "do not infer medical
# conditions" rule is what keeps this a *possible* flag rather than a
# HumanState value of its own (there is no HumanState.INJURED, and
# there never should be -- only FALLEN/CRAWLING/BEING_ASSISTED, the
# literal, observable behaviors this flag is derived from).
_POSSIBLE_INJURY_STATES = frozenset({
    HumanState.FALLEN, HumanState.CRAWLING, HumanState.BEING_ASSISTED,
})

# Classifications this module treats as inherently higher rescue
# priority regardless of current state -- a child or a wheelchair user
# facing the same hazard as an unclassified adult has a documented,
# structural reason to be prioritized (reduced mobility/independence),
# not a behavioral inference about this specific instant.
_HIGH_PRIORITY_CLASSIFICATIONS = frozenset({
    HumanClassification.CHILD, HumanClassification.WHEELCHAIR_USER,
})

# A person another occupant is actively dragging, carrying, or
# assisting is -- by construction of the behavior/state being observed
# at all -- someone who needs help *right now*, independent of the
# possible-injury/classification signals above.
_HIGH_PRIORITY_STATES = frozenset({HumanState.BEING_ASSISTED})
_HIGH_PRIORITY_BEHAVIOR_EVENTS = frozenset({
    BehaviorEvent.DRAGGING_ANOTHER_PERSON, BehaviorEvent.CARRYING_ANOTHER_PERSON,
})


def derive_inference_flags(observation: HumanObservation) -> FrozenSet[InferenceFlag]:

    # The one function in this codebase allowed to produce an
    # InferenceFlag -- a pure function of an already-built
    # HumanObservation's own classification/state/behavior_events,
    # exactly Phase 5's own rule ("must always be derived from
    # lower-level observations, never directly detected"). Never reads
    # anything beyond the four fields below; in particular, never
    # reads confidence -- how much to trust a flag is a presentation/
    # policy concern for whatever renders or acts on it (see
    # command_center.human_panel and
    # decision_policy.human_priority_policy), not something this
    # function decides by silently dropping low-confidence evidence.

    flags = set()

    if observation.state in _PRE_MOVEMENT_DELAY_STATES:
        flags.add(InferenceFlag.POSSIBLE_PRE_MOVEMENT_DELAY)

    possible_injury = observation.state in _POSSIBLE_INJURY_STATES
    if possible_injury:
        flags.add(InferenceFlag.POSSIBLE_INJURY)

    high_rescue_priority = (
        possible_injury
        or observation.classification in _HIGH_PRIORITY_CLASSIFICATIONS
        or observation.state in _HIGH_PRIORITY_STATES
        or bool(observation.behavior_events & _HIGH_PRIORITY_BEHAVIOR_EVENTS)
    )
    if high_rescue_priority:
        flags.add(InferenceFlag.HIGH_RESCUE_PRIORITY)

    return frozenset(flags)
