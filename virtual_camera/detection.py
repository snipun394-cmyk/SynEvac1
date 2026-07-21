from dataclasses import dataclass
from typing import Optional, Tuple

from perception.models.human_observation import HumanClassification, HumanState


@dataclass(frozen=True)
class Detection:

    # One camera's sighting of one person at one instant -- the exact
    # shape a future real detector (RTSP frame -> YOLO -> tracker)
    # would also need to produce. Deliberately NOT the same type as
    # perception.models.human_observation.HumanObservation: a
    # Detection is raw, per-camera, unfused sensor output ("CAM-003
    # saw this at this position with this confidence"); HumanObservation
    # is already a resolved, per-person fact. Reuses HumanObservation's
    # own HumanClassification/HumanState enums as-is rather than
    # inventing a second vocabulary -- Phase 2's explicit "reuse the
    # existing Human Behaviour and Human State systems, do not
    # duplicate information" requirement.
    #
    # Every field here is something a real camera + detector could
    # honestly report; nothing is inferred or assigned that a vision
    # pipeline could not equally produce (see virtual_camera/camera.py
    # for exactly which Ground-Truth-only signal -- classification,
    # human_state -- this simulation-backed source has that a real one
    # would need YOLO/tracking to approximate instead. Everything
    # downstream of this type must never care which).

    camera_id: str
    timestamp: float

    occupant_id: str

    floor_id: Optional[str]
    zone_id: Optional[str]

    # (x, y) in the same meter-space coordinate convention
    # models.camera.Camera/models.zone.Zone already use. None only
    # when zone_id is also None (location genuinely unresolvable --
    # see virtual_camera/camera.py). At zone granularity for Simulation
    # (the occupant's own zone's center, set by virtual_camera/camera.py)
    # -- the Camera Calibration & World Coordinate Projection milestone
    # is what finally gives the LIVE/YOLO pipeline (live_camera_pipeline.
    # identity_resolver's own _to_detection()) a genuinely per-occupant
    # value here too, via camera_calibration.projection.WorldProjector,
    # rather than the coarser zone-center Simulation itself still uses.
    position: Optional[Tuple[float, float]]

    confidence: float

    classification: HumanClassification
    human_state: Optional[HumanState]

    # Phase 4 -- Camera Imperfections. True only for a synthetic
    # detection with no corresponding real occupant (see
    # virtual_camera.imperfections.DetectionImperfectionModel.
    # false_positive_rate) -- occupant_id is a synthetic id in that
    # case, never a real one, and classification/human_state are
    # always UNKNOWN/None (a false positive has no real person to
    # honestly classify).
    is_false_positive: bool = False

    # Camera Calibration & World Coordinate Projection milestone, Phase
    # 7 -- both None whenever no calibration/behavior-recognition
    # world-space data is available (Simulation's own Detection
    # construction in virtual_camera/camera.py never sets either --
    # world_velocity and projection_confidence are LIVE/YOLO-pipeline-
    # only concepts; Simulation's `position` is already an exact
    # ground-truth value with no "projection" step to have a confidence
    # about). world_velocity: meters/second, from behavior_recognition.
    # observation.BehaviorObservation.world_metrics.world_velocity.
    # projection_confidence: camera_calibration.projection.
    # WorldProjection.projection_confidence, distinct from `confidence`
    # above (which is the DETECTOR's own confidence that a person is
    # there at all, not this projection's own geometric reliability).
    world_velocity: Optional[float] = None
    projection_confidence: Optional[float] = None

    # =====================================================

    def __post_init__(self):

        if not (0.0 <= self.confidence <= 1.0):

            raise ValueError(
                f"Detection.confidence must be in [0.0, 1.0], got "
                f"{self.confidence!r}."
            )
