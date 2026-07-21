from dataclasses import dataclass
from typing import Optional

from behavior_recognition.observation import BehaviorObservation


@dataclass(frozen=True)
class CrossCameraObservation:

    # Cross-Camera Identity Resolution (ReID Framework) milestone,
    # Phase 2/5 -- one candidate's per-cycle input to a CrossCameraMatcher.
    # Deliberately does NOT flatten/duplicate BehaviorObservation's own
    # velocity/direction/recognized_behavior vocabulary (Phase 12's
    # explicit "depends only on ... behavior observations" allowance) --
    # `behavior` is None whenever no BehaviorRecognizer is configured
    # upstream (an honest "no evidence", never a fabricated stand-in).
    #
    # Deliberately has NO bounding_box/position field: no world-
    # coordinate calibration exists anywhere in this codebase yet (see
    # virtual_camera.detection.Detection's own "no per-occupant exact
    # position feed" disclosure) linking one camera's pixel-space
    # geometry to another's, so this package has no honest basis to
    # reason about cross-camera geometric proximity at all -- only
    # camera topology (which cameras can plausibly reach which others,
    # and how quickly) and time continuity are used instead.

    camera_id: str
    track_id: str  # LOCAL track id (tracking.tracked_human.TrackedHuman.track_id) -- never a global id
    timestamp: float

    track_confidence: float
    track_age: int

    behavior: Optional[BehaviorObservation] = None


@dataclass(frozen=True)
class ResolvedIdentity:

    # One track's resolved GLOBAL identity, as of one
    # CrossCameraIdentityResolver.resolve() call -- Phase 6/7. camera_id/
    # track_id here are the SAME local, per-cycle pairing every other
    # per-cycle output in this pipeline already uses; global_id is the
    # ONLY new piece of information -- a persistent identity that
    # survives a camera change, minted and owned exclusively by
    # cross_camera_identity.identity_registry.IdentityRegistry, in a
    # format (see IdentityRegistry.create()) that can never collide
    # with, or be mistaken for, a tracking.tracked_human.TrackedHuman.
    # track_id (Phase 6's "never expose tracker IDs as global IDs").

    camera_id: str
    track_id: str
    global_id: str
    timestamp: float
