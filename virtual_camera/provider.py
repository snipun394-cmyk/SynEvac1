from typing import Dict, Optional, Tuple

from visibility.engine import VisibilityEngine

from virtual_camera.camera import VirtualCamera
from virtual_camera.detection import Detection
from virtual_camera.imperfections import DetectionImperfectionModel


class DetectionProvider:

    # Phase 3 -- the Detection Stream's one seam. Mirrors this
    # codebase's own established "one interface, raises
    # NotImplementedError" pattern exactly (CameraProvider,
    # HumanObservationProvider, HazardProvider, OccupancyProvider,
    # PerceptionProvider all follow it) -- a downstream consumer
    # (Perception, AI, RL, Advisory System, Command Center) is meant to
    # depend on THIS class only, never on SimulatedDetectionProvider
    # (below) or on VirtualCamera directly.
    #
    # Phase 6 -- Future Compatibility. This is the exact seam a real
    # detector eventually plugs into:
    #
    #   RTSP -> YOLO -> Tracking -> Detection Stream
    #
    # becomes a second implementation of this same class -- a future
    # LiveDetectionProvider (or ReplayDetectionProvider, for recorded
    # video) that produces the identical Detection shape from a real
    # camera feed instead of Ground Truth. Nothing downstream --
    # Perception, AI, RL, Advisory System, Command Center -- imports
    # SimulatedDetectionProvider by name or branches on which
    # implementation it was handed; every one of them is written
    # against detections_at()'s return type alone, the same
    # "AIDecisionEngine/RL never know whether a HazardSnapshot came
    # from hazard_evolution/ or a replayed sensor log" substitutability
    # docs/architecture/perception_layer.md already establishes one
    # layer over. That is what makes VirtualCamera "the future
    # replacement point for Live CCTV": swap only what sits BEHIND this
    # interface, never the interface or anything built on top of it.

    def detections_at(self, camera_id: str, time: float) -> Tuple[Detection, ...]:

        raise NotImplementedError


class SimulatedDetectionProvider(DetectionProvider):

    # V1's one concrete implementation -- a simulation-backed stand-in,
    # not a vision model, exactly the same role
    # GroundTruthCameraProvider/GroundTruthHumanObservationProvider
    # already play for their own interfaces. Owns one VirtualCamera per
    # Camera Asset, built once at construction (see VirtualCamera's own
    # docstring on why its visibility polygon is computed once, not per
    # tick) and reused for every detections_at() call afterward.

    def __init__(
        self,
        cameras,
        building,
        human_observation_provider,
        visibility_engine: Optional[VisibilityEngine] = None,
        imperfection_model: Optional[DetectionImperfectionModel] = None,
    ):

        visibility_engine = visibility_engine or VisibilityEngine()

        self._virtual_cameras: Dict[str, VirtualCamera] = {
            camera.id: VirtualCamera(
                camera,
                building,
                human_observation_provider,
                visibility_engine=visibility_engine,
                imperfection_model=imperfection_model,
            )
            for camera in cameras
        }

    # =====================================================

    def detections_at(self, camera_id: str, time: float) -> Tuple[Detection, ...]:

        virtual_camera = self._virtual_cameras.get(camera_id)

        if virtual_camera is None:

            raise KeyError(
                f"SimulatedDetectionProvider was never given a camera with "
                f"id {camera_id!r}."
            )

        return virtual_camera.detections_at(time)

    # =====================================================

    def all_detections_at(self, time: float) -> Tuple[Detection, ...]:

        # A convenience aggregate across every registered camera -- the
        # same "same tick, one call per camera, concatenated" shape a
        # multi-camera Detection Stream consumer (Command Center's live
        # feed, a future dataset export) would otherwise have to
        # reimplement itself against detections_at() one camera at a
        # time.

        detections = []

        for virtual_camera in self._virtual_cameras.values():
            detections.extend(virtual_camera.detections_at(time))

        return tuple(detections)
