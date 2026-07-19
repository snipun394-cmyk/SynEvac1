from typing import Mapping

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrameSource
from live_camera_pipeline.human_detector import HumanDetector
from live_camera_pipeline.identity_resolver import IdentityResolver


class LiveCameraPipeline:

    # Phase 12's orchestrator seam: wires
    # CameraFrameSource -> HumanDetector -> IdentityResolver ->
    # LiveCameraPipelineDetectionProvider, entirely via dependency
    # injection, so this class runs today with fake/in-memory
    # implementations (see tests/test_live_camera_pipeline.py) with
    # zero real camera. It deliberately stops there -- it never calls
    # CameraManager, MultiCameraFusionEngine, or BuildingStateEstimator
    # itself. A caller registers the detection_provider it was built
    # with into CameraManager (register_detection_provider(DeviceMode.
    # LIVE, provider)) and continues with the exact same
    # CameraManager -> MultiCameraFusionEngine -> BuildingStateEstimator
    # chain Simulation already uses -- "everything downstream remains
    # unchanged" (Phase 12's own requirement).
    #
    # When real CCTV access exists, only frame_sources' values change
    # (SimulationFrameSource/fakes -> RTSPFrameSource) and human_detector
    # changes (a fake -> real YOLO+tracker). Nothing about this class,
    # or anything it is wired into, needs to change.

    def __init__(
        self,
        frame_sources: Mapping[str, CameraFrameSource],
        human_detector: HumanDetector,
        identity_resolver: IdentityResolver,
        detection_provider: LiveCameraPipelineDetectionProvider,
    ):

        self.frame_sources = dict(frame_sources)
        self.human_detector = human_detector
        self.identity_resolver = identity_resolver
        self.detection_provider = detection_provider

    # =====================================================

    def run_cycle(self, time: float) -> None:

        raw_detections = []

        for frame_source in self.frame_sources.values():

            frame = frame_source.read_frame()

            if frame is None:
                continue

            raw_detections.extend(self.human_detector.detect(frame))

        resolved = self.identity_resolver.resolve(raw_detections, time)

        self.detection_provider.publish(self.frame_sources.keys(), resolved)
