from typing import Tuple

from perception.models.human_observation import HumanClassification

from live_camera_pipeline.frame_source import CameraFrame
from live_camera_pipeline.human_detector import HumanDetector, RawHumanDetection

from human_detection.yolo_backend import YOLOInferenceBackend


DEFAULT_PERSON_CLASS_NAME = "person"


class YOLOHumanDetector(HumanDetector):

    # Real Human Detection Pipeline milestone -- the first concrete,
    # non-mock implementation of live_camera_pipeline.human_detector.
    # HumanDetector. Satisfies that interface exactly as it already
    # existed before this milestone (CameraFrame in, Tuple[
    # RawHumanDetection, ...] out) -- zero changes to HumanDetector,
    # RawHumanDetection, IdentityResolver, Detection,
    # LiveCameraPipelineDetectionProvider, CameraManager,
    # MultiCameraFusionEngine, or BuildingState were needed to add this
    # class (Phase 1's own investigation conclusion: RawHumanDetection
    # already carries bounding_box/confidence/local_track_id, and the
    # canonical Detection type deliberately has no bounding-box field
    # at all -- nothing to "pollute").
    #
    # Delegates all actual model inference to an injected
    # YOLOInferenceBackend (yolo_backend.py) -- this class itself never
    # imports ultralytics/torch/cv2, and never cares whether that
    # backend is UltralyticsYOLOBackend (real) or
    # tests.human_detection_fixtures.FakeYOLOBackend (deterministic,
    # offline). It is equally never coupled to RTSP, CameraManager,
    # MultiCameraFusionEngine, BuildingState, LiveOrchestrator, or
    # Command Center -- it only ever sees the one CameraFrame handed to
    # detect(), regardless of whether that frame came from
    # ReplayFrameSource, a future RTSPFrameSource, or a hand-built test
    # frame (Phase 6/9's own "detector must not care where the frame
    # came from" requirement).
    #
    # ===================================================================
    # PERSON DETECTION ONLY (Phase 3) -- the honest capability boundary:
    # ===================================================================
    #
    #   classification_evidence is always HumanClassification.UNKNOWN --
    #   never ADULT/CHILD/ELDERLY/WHEELCHAIR_USER/FIREFIGHTER/
    #   FIRE_WARDEN. A generic person detector has no honest basis for
    #   any of those; UNKNOWN is HumanClassification's own documented
    #   "a source unable to classify a person at all" value, exactly
    #   the case this is.
    #
    #   state_evidence is always None -- HumanState has no UNKNOWN
    #   member (unlike HumanClassification); None is that enum's own
    #   established "no evidence" convention (see
    #   RawHumanDetection.state_evidence's own Optional[HumanState]
    #   type). A single bounding box in one frame cannot honestly
    #   support WALKING/RUNNING/FALLEN/CRAWLING/... none of those are
    #   asserted.
    #
    # ===================================================================
    # local_track_id -- a DETECTOR-LOCAL OBJECT INDEX, not a tracking
    # identity (Phase 4/5, documented carefully as instructed):
    # ===================================================================
    #
    #   Each call to detect() assigns local_track_id = "0", "1", "2", ...
    #   in detection order, purely to give MappingIdentityResolver (or
    #   any IdentityResolver) a distinct (camera_id, local_track_id) key
    #   per person WITHIN THIS ONE FRAME -- without it, every person
    #   detected in the same frame would share local_track_id=None,
    #   collapsing to the same synthetic identity
    #   (f"{camera_id}:" -- see identity_resolver.py's own fallback) and
    #   incorrectly fusing multiple distinct people into one occupant.
    #
    #   This index is NOT stable across frames -- the person who was
    #   index "0" in frame N has no guaranteed relationship to whoever
    #   is index "0" in frame N+1 (detection order can change frame to
    #   frame for reasons as simple as which person YOLO happened to
    #   list first). This is an honest limitation, not a bug: real
    #   single-camera tracking (assigning a STABLE local_track_id across
    #   frames for the same physical person) is explicitly out of this
    #   milestone's scope -- see docs/architecture/human_detection.md's
    #   own "SINGLE-CAMERA TRACKING -- investigate only" section.
    #
    # ===================================================================
    # Never crashes the pipeline (matches every other seam in this
    # codebase's own "catch failures at the boundary, return an honest
    # empty result" discipline -- RTSPFrameSource, RegistryLiveAI
    # InferenceGateway, ReplayCompatibleAdvisoryGateway all do the
    # same): a None/malformed payload_ref, or the backend's own infer()
    # raising, both result in () -- never a fabricated detection, never
    # an unhandled exception reaching LiveCameraPipeline.run_cycle().
    # ===================================================================

    def __init__(
        self,
        backend: YOLOInferenceBackend,
        *,
        person_class_name: str = DEFAULT_PERSON_CLASS_NAME,
    ):

        self._backend = backend
        self._person_class_name = person_class_name

    # =====================================================

    def detect(self, frame: CameraFrame) -> Tuple[RawHumanDetection, ...]:

        if frame.payload_ref is None:
            return ()

        try:
            raw_boxes = self._backend.infer(frame.payload_ref)
        except Exception:
            # A decode/inference failure (a malformed frame, a
            # corrupted model artifact, ...) is an honest "nothing
            # detected this cycle," never a crash.
            return ()

        detections = []
        index = 0

        for box in raw_boxes:

            if box.class_name != self._person_class_name:
                # Non-person classes (chairs, bags, ...) are silently
                # ignored -- this milestone is person detection only.
                continue

            detections.append(
                RawHumanDetection(
                    camera_id=frame.camera_id,
                    local_track_id=str(index),
                    timestamp=frame.timestamp,
                    bounding_box=box.bounding_box,
                    confidence=box.confidence,
                    classification_evidence=HumanClassification.UNKNOWN,
                    state_evidence=None,
                )
            )

            index += 1

        return tuple(detections)
