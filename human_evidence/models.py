from dataclasses import dataclass
from typing import Optional

from perception.models.human_observation import HumanClassification, HumanState


# =====================================================
# Live Human State & Assistance Perception Bridge milestone -- the
# small, standalone model for PERSISTENT human classification/state
# evidence, reusing perception.models.human_observation's own
# HumanClassification/HumanState enums exactly (Phase 4/5's explicit
# "do not invent a second vocabulary" requirement). Deliberately its
# own package, not folded into live_occupants/ itself: this milestone's
# own Phase 26 requires "human evidence must not import Emergency
# Response... no circular dependency," and keeping the reconciliation
# rules (reconciliation.py) importable and unit-testable in complete
# isolation from LiveOccupant's own lifecycle/history/indexing concerns
# is the same "small, standalone, reusable" discipline crowd_
# intelligence.trends.TrendTracker already established one milestone
# over (reused directly by evacuation_progress, never duplicated).
#
# This module holds ONLY the plain-value evidence shape; live_occupants.
# manager.LiveOccupantManager is the one caller that actually applies
# reconcile_classification()/reconcile_state() (human_evidence/
# reconciliation.py) against its own stored LiveOccupant state -- this
# package itself never depends on live_occupants, never depends on
# emergency_response, and is never depended upon by cross_camera_
# identity/tracking/behavior_recognition (mechanically enforced, see
# tests/test_human_evidence_architecture_guards.py).
# =====================================================


@dataclass(frozen=True)
class HumanEvidence:

    # One occupant's own currently-held classification/state evidence,
    # as already reconciled by human_evidence.reconciliation -- a plain
    # snapshot, not itself a mutable owner of anything (LiveOccupant is
    # the one place this shape is actually stored, as four of its own
    # fields, rather than as a nested HumanEvidence instance -- see
    # live_occupants/occupant.py's own docstring on why: LiveOccupant is
    # already a frozen dataclass whose OWN fields are the canonical
    # record, exactly as world_position/world_velocity/behavior already
    # are, not a further nested value object). This type exists for
    # tests/reconciliation.py's own convenience (a self-contained
    # argument/return shape), not as a second, competing storage
    # location.
    #
    # classification_confidence/state_confidence reuse the SAME
    # per-detection presence-confidence value every other evidence field
    # in this codebase already carries (RawHumanDetection.confidence/
    # Detection.confidence) -- investigated directly (Phase 1 item 1-3):
    # no separate, classification-specific or state-specific confidence
    # signal exists anywhere in this codebase today. Documented here
    # rather than fabricating a new, invented number: "the detector's
    # own confidence that a person is there at all, carried through as
    # the best available proxy" (Phase 3's own "never fabricate
    # confidence... if unavailable: None" is honored by simply passing
    # this same real number through, or None when even that is
    # unavailable, never inventing a new one).
    #
    # classification_source/state_source is the camera_id that most
    # recently supplied the CURRENTLY-HELD value -- the only genuinely
    # known "source" concept this pipeline has (no separate detector-
    # name/model-version concept exists anywhere yet).

    occupant_id: str
    timestamp: float

    classification: HumanClassification = HumanClassification.UNKNOWN
    classification_confidence: Optional[float] = None
    classification_source: Optional[str] = None
    classification_last_observed_at: Optional[float] = None

    human_state: Optional[HumanState] = None
    state_confidence: Optional[float] = None
    state_source: Optional[str] = None
    state_last_observed_at: Optional[float] = None
