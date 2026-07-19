from dataclasses import dataclass, field
from enum import Enum, auto
from typing import FrozenSet, Optional


class HumanClassification(Enum):

    # Phase 2 -- direct, observable classifications only. Each of these
    # is meant to be something a vision model (or a simulation-backed
    # stand-in) can assert from appearance/equipment alone, with no
    # inference about intent, capability, or medical state layered in
    # here -- "Wheelchair User" is an observation of a wheelchair being
    # used, not a claim about why. A source unable to classify a person
    # at all (not enough resolution, an unrecognized category, or a
    # simulation-backed provider with no classification signal at all)
    # reports UNKNOWN -- never a guessed default like ADULT.

    ADULT = auto()
    CHILD = auto()
    ELDERLY = auto()
    WHEELCHAIR_USER = auto()
    FIREFIGHTER = auto()
    FIRE_WARDEN = auto()
    UNKNOWN = auto()


class HumanState(Enum):

    # Phase 3 -- current observable state. Every value here describes
    # what a person is *doing* or how they are *positioned* right now,
    # never why (no "injured", no "panicking") -- that hedge belongs
    # entirely to Phase 5's InferenceFlag, derived from this state, not
    # asserted as this state. NEVER_MOVING_YET is deliberately named
    # for exactly the role its own docstring below describes: a person
    # observed at their starting location who has not yet begun to
    # move -- the direct observational trigger a pre-movement-delay
    # inference is meant to be derived from (see
    # perception.human_inference.derive_inference_flags()).

    WALKING = auto()
    RUNNING = auto()
    STANDING = auto()
    FALLEN = auto()
    CRAWLING = auto()
    WAITING = auto()
    BEING_ASSISTED = auto()
    HELPING_ANOTHER_OCCUPANT = auto()

    # A candidate for pre-movement delay -- observed at their starting
    # position, not yet in motion. Distinct from WAITING (which
    # describes someone stopped *during* their evacuation, e.g. queued
    # at a door) and from STANDING (stopped but already underway) --
    # NEVER_MOVING_YET specifically means "has not started evacuating
    # at all yet," the literal precondition Phase 3 names.
    NEVER_MOVING_YET = auto()


class BehaviorEvent(Enum):

    # Phase 4 -- directly observable interactions between people, never
    # a diagnosis of why they are happening. A person may exhibit more
    # than one of these at once (e.g. GROUPED_MOVEMENT while also
    # HELPING_ANOTHER_PERSON) -- see HumanObservation.behavior_events
    # below, a set, not a single value.

    HELPING_ANOTHER_PERSON = auto()
    DRAGGING_ANOTHER_PERSON = auto()
    CARRYING_ANOTHER_PERSON = auto()
    PUSHING_WHEELCHAIR = auto()
    GROUPED_MOVEMENT = auto()
    QUEUEING = auto()


@dataclass(frozen=True)
class HumanObservation:

    # Phase 1 -- one detected person, as of one perception cycle.
    # Backend-agnostic by construction: every field here is something a
    # vision model, a pose-estimation stage, an action-recognition
    # stage, or (today, the only implemented source) a simulation-
    # ground-truth bridge could equally well populate -- this type
    # itself has no opinion about, and no import of, any of them (see
    # tests.test_perception.PerceptionPackageDependencyDirectionTests,
    # which already enforces that for every type in this module's
    # package).
    #
    # person_id is a *tracking* identity -- stable for as long as one
    # perception source keeps tracking the same physical person across
    # cycles, never a claim of real-world identity (name, employee ID,
    # ...). A person absent from a given cycle's
    # BuildingObservation.human_observations mapping means "not
    # currently observed" (track lost, left every camera's view,
    # evacuated, ...), the same "absence means unobserved" convention
    # ObservedNodeState/ObservedOccupancy already establish -- never
    # "observed and confirmed absent."
    #
    # zone_id/floor_id are independently Optional: a source that can
    # localize a person to a floor but not (yet) a specific zone -- or
    # a person genuinely between two zones, e.g. mid-doorway -- leaves
    # the field it cannot honestly report as None, never a guessed
    # nearest zone.
    #
    # classification defaults to UNKNOWN (Phase 2's own "no signal"
    # value). state has no equally-neutral "unknown" value among Phase
    # 3's fixed list, so it is Optional instead -- None means "this
    # source does not report state at all" rather than fabricating a
    # default like STANDING.

    person_id: str

    zone_id: Optional[str] = None
    floor_id: Optional[str] = None

    confidence: Optional[float] = None

    classification: HumanClassification = HumanClassification.UNKNOWN
    state: Optional[HumanState] = None

    behavior_events: FrozenSet[BehaviorEvent] = field(default_factory=frozenset)

    last_observed_time: Optional[float] = None

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "behavior_events", frozenset(self.behavior_events))

        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):

            raise ValueError(
                f"HumanObservation.confidence must be in [0.0, 1.0], got "
                f"{self.confidence!r}."
            )
