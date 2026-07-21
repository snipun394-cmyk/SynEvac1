from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional, Sequence, Tuple

from perception.models.human_observation import HumanClassification

from virtual_camera.detection import Detection

from live_camera_pipeline.human_detector import RawHumanDetection


class IdentityResolver(ABC):

    # The seam Phase 2/7/8 exist to name: the ONE place a real
    # cross-camera re-identification model would need to live.
    # RawHumanDetection (camera-local, unresolved) in, Detection
    # (globally resolved -- Detection.occupant_id is, by every existing
    # consumer's contract, a global identity) out. MultiCameraFusion
    # Engine is never touched or imported here, and never needs to be:
    # it already fuses whatever Detections it is handed by
    # Detection.occupant_id equality (multi_camera_fusion/engine.py),
    # so resolving identity correctly BEFORE fuse() is called is
    # exactly what makes Live mode "fuse exactly as Simulation does
    # today, completely unchanged."
    #
    # Not implemented here, and out of scope for this milestone: any
    # actual computer-vision re-identification model. Both concrete
    # resolvers below are honest, non-ML strategies.

    @abstractmethod
    def resolve(
        self,
        raw_detections: Sequence[RawHumanDetection],
        time: float,
    ) -> Tuple[Detection, ...]:
        ...


class SimulationIdentityResolver(IdentityResolver):

    # Deterministic pass-through, proving the seam's parity with
    # Simulation's existing behavior: RawHumanDetection.local_track_id
    # IS already the global identity in this strategy, exactly the way
    # virtual_camera.camera.VirtualCamera's own occupant_id already is.
    #
    # NOT wired into the live-running Simulation pipeline -- VirtualCamera
    # and SimulatedDetectionProvider keep producing Detection objects
    # directly, unchanged, as they always have. This class exists only
    # to prove/exercise the IdentityResolver interface itself under
    # test with Simulation-shaped inputs, and as the natural resolver a
    # future ReplayDetectionProvider could reuse verbatim (recorded
    # identity is just as globally known as Simulation's).

    def resolve(
        self,
        raw_detections: Sequence[RawHumanDetection],
        time: float,
    ) -> Tuple[Detection, ...]:

        return tuple(
            _to_detection(raw, occupant_id=raw.local_track_id or "")
            for raw in raw_detections
        )


class MappingIdentityResolver(IdentityResolver):

    # A real, honest, non-ML resolution strategy: an explicit
    # {(camera_id, local_track_id): global_id} table, the kind a
    # manual calibration step or a simple handover rule could produce
    # without any re-identification model. Two different cameras'
    # same-numbered local track is namespaced by camera_id in the
    # lookup key, so it is NEVER treated as the same person unless the
    # mapping says so explicitly (Phase 7 C/D, Phase 16's "same local
    # track ID on different cameras does not imply same person"
    # guard) -- an unmapped pair gets a synthesized id derived from
    # the pair itself, deterministic and always distinct per camera.

    def __init__(self, mapping: Optional[Mapping[Tuple[str, str], str]] = None):

        self._mapping: Dict[Tuple[str, str], str] = dict(mapping or {})

    # =====================================================

    def set_mapping(self, camera_id: str, local_track_id: str, global_id: str) -> None:

        self._mapping[(camera_id, local_track_id)] = global_id

    # =====================================================

    def resolve(
        self,
        raw_detections: Sequence[RawHumanDetection],
        time: float,
    ) -> Tuple[Detection, ...]:

        return tuple(
            _to_detection(raw, occupant_id=self._resolve_one(raw))
            for raw in raw_detections
        )

    # =====================================================

    def _resolve_one(self, raw: RawHumanDetection) -> str:

        local_track_id = raw.local_track_id or ""

        key = (raw.camera_id, local_track_id)

        if key in self._mapping:
            return self._mapping[key]

        # No declared relationship -- default to a per-camera-namespaced
        # synthetic id, never a bare local_track_id, so a coincidental
        # string collision across cameras can never silently fuse two
        # different people.
        return f"{raw.camera_id}:{local_track_id}"


def _to_detection(raw: RawHumanDetection, occupant_id: str) -> Detection:

    return Detection(
        camera_id=raw.camera_id,
        timestamp=raw.timestamp,
        occupant_id=occupant_id,
        floor_id=raw.floor_id,
        zone_id=raw.zone_id,
        # Camera Calibration & World Coordinate Projection milestone:
        # world-coordinate calibration now exists (camera_calibration.
        # projection.WorldProjector) -- raw.world_position/world_velocity/
        # projection_confidence are None exactly when no calibration/
        # behavior-recognition world data was available upstream for
        # this detection (live_camera_pipeline.pipeline.LiveCameraPipeline's
        # own glue populates them otherwise), so this remains an honest
        # pass-through, never a fabrication, in either case.
        position=raw.world_position,
        confidence=raw.confidence,
        classification=raw.classification_evidence or HumanClassification.UNKNOWN,
        human_state=raw.state_evidence,
        is_false_positive=raw.is_false_positive,
        world_velocity=raw.world_velocity,
        projection_confidence=raw.projection_confidence,
    )
