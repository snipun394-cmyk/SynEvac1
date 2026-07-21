from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from behavior_recognition.metrics import TemporalMetrics, WorldTemporalMetrics


class RecognizedBehavior(Enum):

    # Human Behavior Recognition Framework milestone, Phase 5 -- ONLY
    # behaviors that can honestly be inferred from tracking geometry
    # (position/bounding-box history over time) alone, with NO neural
    # network, NO pose estimation, NO appearance/action-recognition
    # model of any kind. Deliberately NOT the same enum as
    # perception.models.human_observation.HumanState -- that vocabulary
    # already includes claims (FALLEN, BEING_ASSISTED,
    # HELPING_ANOTHER_OCCUPANT, ...) this milestone has no honest basis
    # to assert from a bounding-box trajectory, and command_center.
    # building_view/incident_data already treat HumanState.FALLEN as a
    # confident, operator-facing signal ("fallen_count") -- conflating
    # a hedged geometric heuristic with that would misrepresent a guess
    # as a confirmed observation. See docs/architecture/
    # behavior_recognition.md Sec 4 for the full reasoning on which of
    # these values (only WALKING/RUNNING) are honestly re-mapped onto
    # HumanState for existing downstream consumers, and which
    # (STATIONARY/POSSIBLY_FALLEN/UNKNOWN) are not.

    UNKNOWN = auto()
    STATIONARY = auto()
    WALKING = auto()
    RUNNING = auto()

    # Explicitly hedged ("POSSIBLY") -- a coarse geometric heuristic
    # (a person's bounding box becoming wide/low and staying that way
    # while stationary), with known, real false-positive causes
    # (crouching, sitting, bending down, tying a shoe, picking
    # something up all look identical to this heuristic). Disabled by
    # default in RuleBasedBehaviorRecognizer for exactly that reason --
    # see rule_based_recognizer.py's own docstring. Never fabricates
    # FALLEN (HumanState's own, stronger, unhedged member).
    POSSIBLY_FALLEN = auto()

    # Explicitly NOT included, and never will be produced by a rule-
    # based recognizer operating on tracking geometry alone (Phase 5's
    # own exclusion list): HELPING, PANIC, LEADERSHIP, DISABILITY,
    # INJURY, CONFUSION, FOLLOWING, HERDING. Every one of those is
    # either a multi-person social inference or an interpretation of
    # WHY a person is moving a certain way -- neither is available from
    # a single track's own position history.


@dataclass(frozen=True)
class BehaviorObservation:

    # One track's recognized behavior, as of one BehaviorRecognizer.
    # recognize() call -- Phase 3. Contains ONLY behavior-recognition
    # information, local to one camera's one tracked person. Deliberately
    # contains NONE of:
    #
    #   occupant_id     -- a global identity remains entirely
    #                      live_camera_pipeline.identity_resolver.
    #                      IdentityResolver's job; this package never
    #                      imports Detection or IdentityResolver.
    #   building zone/floor -- localization is RawHumanDetection's own
    #                      honestly-uncertain field, unrelated to
    #                      *behavior* geometry.
    #   AI / BuildingState fields -- this package has no dependency on
    #                      either (Phase 12 architecture guards,
    #                      tests/test_behavior_recognition_architecture_
    #                      guards.py).
    #
    # track_id is the SAME local, temporary, camera-scoped identity
    # tracking.tracked_human.TrackedHuman.track_id already is -- never a
    # global occupant_id, and never assigned or reinterpreted by this
    # package.

    camera_id: str
    track_id: str
    timestamp: float

    recognized_behavior: RecognizedBehavior
    confidence: float

    supporting_metrics: TemporalMetrics

    # Camera Calibration & World Coordinate Projection milestone, Phase
    # 6 -- None whenever no world position was supplied for this track
    # this cycle (no calibration configured for its camera -- an honest
    # "not available", never a fabricated stand-in). When present,
    # RuleBasedBehaviorRecognizer classifies using THIS metrics' own
    # world_velocity in preference to supporting_metrics.velocity
    # (pixel-space) -- see rule_based_recognizer.py's own docstring on
    # "operate using world-space motion instead of pixel-space whenever
    # calibration is available."
    world_metrics: Optional[WorldTemporalMetrics] = None

    def __post_init__(self):

        if not (0.0 <= self.confidence <= 1.0):

            raise ValueError(
                f"BehaviorObservation.confidence must be in [0.0, 1.0], got "
                f"{self.confidence!r}."
            )
