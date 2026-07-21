from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from perception.models.human_observation import HumanClassification, HumanState

from live_camera_pipeline.frame_source import CameraFrame


@dataclass(frozen=True)
class RawHumanDetection:

    # One frame's raw sighting of one person, BEFORE cross-camera
    # identity resolution -- the boundary Phase 2/4/8 describe. Every
    # field here is something a real detector+local-tracker could
    # honestly report; nothing here is a globally resolved identity.
    #
    # local_track_id is deliberately namespaced by camera_id, not a
    # bare string alone -- (camera_id, local_track_id) is the only
    # safe key. "CAM-A track_5" and "CAM-B track_5" are two
    # DIFFERENT camera-local ids that happen to share a string; they
    # must never be treated as the same person by default (Phase 7/16
    # -- see live_camera_pipeline/identity_resolver.py and
    # tests/test_identity_resolver.py).
    #
    # Reuses HumanClassification/HumanState from
    # perception.models.human_observation as *evidence*, not fact --
    # a real detector's classification/state guess, still subject to
    # IdentityResolver/downstream confidence handling exactly like
    # virtual_camera.detection.Detection's own classification/
    # human_state fields already are. No duplicate vocabulary.

    camera_id: str
    local_track_id: Optional[str]
    timestamp: float

    bounding_box: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 1.0

    classification_evidence: Optional[HumanClassification] = None
    state_evidence: Optional[HumanState] = None

    floor_id: Optional[str] = None
    # Zone localization was honestly uncertain before the Camera
    # Calibration & World Coordinate Projection milestone -- a camera
    # assigned to more than one zone could not know which one a raw
    # bounding box fell into without a real calibrated projection. Now
    # that camera_calibration.projection.WorldProjector exists,
    # live_camera_pipeline.pipeline.LiveCameraPipeline's own glue
    # populates this field (and world_position/projection_confidence
    # below) whenever a WorldProjector is configured for this camera --
    # None remains the honest answer whenever it is not (uncalibrated
    # camera, or a geometrically undefined projection).
    zone_id: Optional[str] = None

    # Camera Calibration & World Coordinate Projection milestone, Phase
    # 7 -- world_position is the SAME (x, y) floor-plan meter
    # convention models.zone.Zone/models.camera.Camera already use
    # (never a separately-invented coordinate system), set only when a
    # camera_calibration.projection.WorldProjector actually produced
    # one for this detection this cycle. world_velocity is NOT
    # something a single RawHumanDetection can compute in isolation (it
    # is a TEMPORAL quantity) -- it is set by the pipeline glue from
    # behavior_recognition.observation.BehaviorObservation.world_metrics.
    # world_velocity, the exact same "set downstream of detection,
    # never by the detector itself" pattern state_evidence already
    # follows for behavior. All three are None whenever the
    # corresponding upstream stage (calibration/behavior recognition)
    # is not configured -- never fabricated.
    world_position: Optional[Tuple[float, float]] = None
    world_velocity: Optional[float] = None
    projection_confidence: Optional[float] = None

    is_false_positive: bool = False


class HumanDetector(ABC):

    # The seam a real vision model plugs into: CameraFrame in,
    # RawHumanDetection(s) out. Today: no implementation. Tomorrow:
    # YOLO + a local tracker, wheelchair detection, fallen-person
    # detection, etc. all become one concrete HumanDetector without
    # CameraManager, MultiCameraFusionEngine, BuildingState, or
    # Advisory ever changing (Phase 4's own requirement).

    @abstractmethod
    def detect(self, frame: CameraFrame) -> Tuple[RawHumanDetection, ...]:
        ...
