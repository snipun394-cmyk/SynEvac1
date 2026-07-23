from typing import List, Optional, Tuple

from human_detection.yolo_backend import BoundingBoxDetection, YOLOInferenceBackend


# =====================================================
# Real Human Detection Pipeline milestone -- a fully deterministic,
# offline stand-in for a real YOLOInferenceBackend (Ultralytics/torch),
# lives in tests/, not human_detection/, matching every other test-only
# backend double this codebase already establishes (tests.
# live_camera_pipeline_fixtures.FakeRTSPBackend for FrameDecoderBackend,
# tests.test_building_control.RecordingActionExecutor for
# ActionExecutor). Nothing here touches ultralytics/torch/cv2 or any
# real model -- a test configures exactly what infer() returns via
# plain method calls.
# =====================================================


def person(confidence: float, box=(0.0, 0.0, 10.0, 10.0), class_id: int = 0) -> BoundingBoxDetection:

    return BoundingBoxDetection(class_id=class_id, class_name="person", confidence=confidence, bounding_box=box)


def non_person(class_name: str = "chair", confidence: float = 0.9, box=(0.0, 0.0, 5.0, 5.0)) -> BoundingBoxDetection:

    return BoundingBoxDetection(class_id=99, class_name=class_name, confidence=confidence, bounding_box=box)


class FakeYOLOBackend(YOLOInferenceBackend):

    def __init__(self):

        self._queue: List[Tuple[BoundingBoxDetection, ...]] = []
        self._default: Optional[Tuple[BoundingBoxDetection, ...]] = ()
        self._raise: Optional[Exception] = None
        self.infer_calls: List[object] = []

    # --- test configuration ---------------------------------------

    def queue_result(self, *detections: BoundingBoxDetection) -> None:

        self._queue.append(tuple(detections))

    def set_default_result(self, *detections: BoundingBoxDetection) -> None:

        # Used when a test wants every subsequent infer() call (not
        # just one queued call) to return the same fixed result --
        # e.g. "repeated frames" scenarios.
        self._default = tuple(detections)

    def fail_next_with(self, exception: Exception) -> None:

        self._raise = exception

    # --- YOLOInferenceBackend interface -----------------------------

    def infer(self, image) -> Tuple[BoundingBoxDetection, ...]:

        self.infer_calls.append(image)

        if self._raise is not None:
            exc = self._raise
            self._raise = None
            raise exc

        if self._queue:
            return self._queue.pop(0)

        return self._default
