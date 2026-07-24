from dataclasses import dataclass, field
from typing import Optional, Tuple

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanClassification, HumanState

from live_occupants.history import OccupantHistory
from live_occupants.state import OccupantStatus


@dataclass(frozen=True)
class LiveOccupant:

    # Live Occupant Digital Twin milestone, Phase 3 -- the ONE canonical
    # runtime representation of one occupant, gathering what today is
    # scattered across tracking.tracked_human.TrackedHuman (per-camera,
    # ephemeral), behavior_recognition.observation.BehaviorObservation
    # (per-camera, ephemeral), cross_camera_identity's own
    # GlobalIdentityRecord (global, but thin -- only last_camera_id/
    # last_track_id/last_timestamp, no zone/behavior/position at all),
    # and virtual_camera.detection.Detection (per-camera, ephemeral,
    # recomputed fresh every cycle, thrown away once
    # MultiCameraFusionEngine consumes it). None of those objects is
    # both (a) keyed by the GLOBAL occupant_id and (b) persists across
    # cycles with first_seen/last_seen/lifecycle status -- this type is
    # the first one that is.
    #
    # Deliberately does NOT duplicate BuildingState or Detection (Phase
    # 3's own explicit instruction): no `is_false_positive`/
    # `floor_severity`/anything BuildingState's own FusedTrack/
    # hazard_summary already derives independently. Every field here is
    # either (a) genuinely new information no existing frozen, per-cycle
    # type can honestly hold (first_seen, current_status, history)
    # because it requires persisting ACROSS cycles, or (b) the CURRENT
    # single-cycle snapshot of what's already flowing through the
    # pipeline (current_camera_id/current_track_id/current_zone/
    # current_floor/world_position/world_velocity/behavior/confidence),
    # reused as-is rather than re-derived.
    #
    # Live Human State & Assistance Perception Bridge milestone --
    # human_classification/human_state (below) are a DELIBERATE, NARROW
    # REVERSAL of this type's own original "no classification" design
    # decision, driven by a genuine gap that decision left unaddressed:
    # BuildingState/FusedTrack's own `classification`/`human_state`
    # fields are a SEPARATE, per-cycle, highest-confidence-detection-
    # wins snapshot (multi_camera_fusion.engine.MultiCameraFusionEngine.
    # _fuse_group(), unchanged by this milestone), recomputed fresh every
    # cycle with no cross-cycle memory and no staleness handling --
    # exactly wrong for Emergency Response Intelligence's own need for
    # PERSISTENT, temporally-reconciled assistance evidence (a person
    # reported FALLEN one cycle must not silently read back as UNKNOWN
    # the very next cycle a single camera fails to reconfirm it). These
    # two fields are therefore genuinely NEW, cross-cycle information no
    # existing frozen per-cycle type could honestly hold either -- the
    # same justification first_seen/current_status/history already rely
    # on -- reconciled via human_evidence.reconciliation.
    # reconcile_classification()/reconcile_state() (Phase 7/8), never
    # re-deriving or duplicating BuildingState's own separate, still-
    # unmodified per-cycle view.
    #
    # Frozen, like every other value object in this codebase's pipeline
    # (TrackedHuman, BehaviorObservation, Detection, FusedTrack,
    # BuildingState) -- live_occupants.manager.LiveOccupantManager is
    # the one place that owns MUTATION, by replacing its own stored
    # reference with a new LiveOccupant instance (dataclasses.replace())
    # each update, exactly the same "immutable value + mutable owning
    # registry" pattern cross_camera_identity.identity_registry.
    # IdentityRegistry already established for GlobalIdentityRecord.

    occupant_id: str

    current_camera_id: Optional[str]
    current_track_id: Optional[str]

    current_zone_id: Optional[str]
    current_floor_id: Optional[str]

    world_position: Optional[Tuple[float, float]]
    world_velocity: Optional[float]

    behavior: Optional[RecognizedBehavior]

    confidence: float

    first_seen: float
    last_seen: float

    status: OccupantStatus

    history: OccupantHistory = field(default_factory=OccupantHistory)

    # Live Human State & Assistance Perception Bridge milestone -- see
    # this class's own docstring above. `human_classification` defaults
    # to HumanClassification.UNKNOWN (never a guessed ADULT -- the same
    # "no signal" convention perception.models.human_observation.
    # HumanObservation.classification already establishes), `human_state`
    # defaults to None (HumanState has no equally-neutral member, same
    # convention as HumanObservation.state). *_confidence/_source/
    # _last_observed_at are None until genuine evidence has ever been
    # supplied -- never fabricated.
    human_classification: HumanClassification = HumanClassification.UNKNOWN
    human_classification_confidence: Optional[float] = None
    human_classification_source: Optional[str] = None
    human_classification_last_observed_at: Optional[float] = None

    human_state: Optional[HumanState] = None
    human_state_confidence: Optional[float] = None
    human_state_source: Optional[str] = None
    human_state_last_observed_at: Optional[float] = None

    # CCTV Connection & Calibration Readiness milestone, Phase 8 -- one
    # of camera_calibration.camera_model.WORLD_POSITION_PROVENANCE_
    # {NONE,UNVALIDATED,VALIDATED} (the string constant), mirroring
    # world_position's own CURRENT-cycle-only convention (not history-
    # tracked). The Python default None here means something different
    # from the WORLD_POSITION_PROVENANCE_NONE *string* -- None means "no
    # provenance information was ever supplied to update() at all" (a
    # caller not wired through world projection at all, including every
    # existing test/demo that predates this field); the string
    # WORLD_POSITION_PROVENANCE_NONE means "world projection was
    # attempted and honestly found no calibration for this camera" --
    # both are distinguishable by the caller if it matters.
    world_position_provenance: Optional[str] = None
