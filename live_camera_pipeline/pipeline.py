import dataclasses
from typing import Mapping, Optional

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrameSource
from live_camera_pipeline.human_detector import HumanDetector
from live_camera_pipeline.identity_resolver import IdentityResolver

from tracking.tracker import SingleCameraTracker


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
    #
    # Single-Camera Tracking Framework milestone, Phase 7: `tracker` is
    # an OPTIONAL, additive seam between human_detector.detect() and
    # identity_resolver.resolve() -- omitting it (the default,
    # unchanged behavior for every existing caller/test) reproduces
    # this class's exact pre-tracking behavior. When supplied, each
    # frame's raw detections are run through tracker.update(camera_id,
    # frame.timestamp, raw) (tracking/tracker.SingleCameraTracker),
    # and only its own per-detection positional output (never the
    # trailing MISSING/EXPIRED remainder -- there is no real
    # observation to hand IdentityResolver for a track not seen this
    # cycle) is used to replace each RawHumanDetection's own
    # per-frame-only local_track_id with the tracker's now
    # frame-to-frame-STABLE track_id. IdentityResolver.resolve() itself
    # is never modified -- it still receives plain RawHumanDetection
    # objects, just ones whose local_track_id is more stable input
    # than an unstabilized detector could honestly provide alone
    # (Phase 8's identity boundary: a tracking id is still local and
    # temporary, never a global occupant_id -- that remains entirely
    # IdentityResolver's own job, untouched here).

    def __init__(
        self,
        frame_sources: Mapping[str, CameraFrameSource],
        human_detector: HumanDetector,
        identity_resolver: IdentityResolver,
        detection_provider: LiveCameraPipelineDetectionProvider,
        tracker: Optional[SingleCameraTracker] = None,
    ):

        self.frame_sources = dict(frame_sources)
        self.human_detector = human_detector
        self.identity_resolver = identity_resolver
        self.detection_provider = detection_provider
        self.tracker = tracker

    # =====================================================

    def run_cycle(self, time: float) -> None:

        raw_detections = []

        for camera_id, frame_source in self.frame_sources.items():

            frame = frame_source.read_frame()

            if frame is None:
                continue

            raw = self.human_detector.detect(frame)

            if self.tracker is not None:
                raw = self._stabilize(camera_id, frame.timestamp, raw)

            raw_detections.extend(raw)

        resolved = self.identity_resolver.resolve(raw_detections, time)

        self.detection_provider.publish(self.frame_sources.keys(), resolved)

    # =====================================================

    def _stabilize(self, camera_id, timestamp, raw):

        tracked = self.tracker.update(camera_id, timestamp, raw)

        return tuple(
            dataclasses.replace(detection, local_track_id=tracked_human.track_id)
            for detection, tracked_human in zip(raw, tracked)
        )
