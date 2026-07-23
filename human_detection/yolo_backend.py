from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union


# =====================================================
# Real Human Detection Pipeline milestone -- the inference seam
# YOLOHumanDetector (yolo_human_detector.py) delegates to. Deliberately
# mirrors live_camera_pipeline.rtsp_backend.FrameDecoderBackend's own
# "smallest interface that makes the real adapter testable without the
# real library" shape: YOLOHumanDetector never imports ultralytics/
# torch/cv2 itself, only this module's own YOLOInferenceBackend
# Protocol-ish ABC -- swapping UltralyticsYOLOBackend for
# tests.human_detection_fixtures.FakeYOLOBackend is the entire test
# strategy for this milestone (Phase 7's own explicit instruction).
#
# This module (and yolo_human_detector.py) is the ONE place in this
# codebase allowed to import cv2/torch/ultralytics -- deliberately kept
# OUT of live_camera_pipeline/ (which tests/test_no_cv_dependencies.py
# mechanically forbids from importing any of them, Phase 1's own
# investigation finding) and out of every other package that guard also
# covers. human_detection/ is not in that guard's TARGETS list and is
# not added to it -- its whole purpose is to hold real computer-vision
# code.
# =====================================================


@dataclass(frozen=True)
class BoundingBoxDetection:

    # One raw object detection from an underlying vision model, BEFORE
    # any human/person filtering -- the smallest honest shape a
    # detector backend can report. class_name is matched against
    # exactly (not class_id), since a real Ultralytics model's own
    # `model.names` mapping is the authoritative source of which
    # integer means "person" for THAT model/weights file -- never
    # assumed to be COCO's own id 0 by convention alone.

    class_id: int
    class_name: str
    confidence: float
    bounding_box: Tuple[float, float, float, float]  # x1, y1, x2, y2, image-pixel space


class YOLOInferenceBackend(ABC):

    # image is whatever CameraFrame.payload_ref already carries by this
    # milestone's own established convention (a decoded frame -- a
    # numpy ndarray, HxWx3) -- this class never decodes video/RTSP
    # itself (that remains live_camera_pipeline.rtsp_backend.
    # FrameDecoderBackend's job, one seam upstream).

    @abstractmethod
    def infer(self, image: Any) -> Tuple[BoundingBoxDetection, ...]:

        # Must raise on a genuine inference failure (a corrupted
        # model/malformed image) rather than silently returning ()  --
        # YOLOHumanDetector.detect() is the one place that failure is
        # caught and converted into an honest empty result, the same
        # "catch at the seam boundary, never at the seam itself"
        # discipline RTSPFrameSource/RegistryLiveAIInferenceGateway
        # both already establish.
        ...


class ModelWeightsNotFoundError(Exception):

    # Raised by UltralyticsYOLOBackend's constructor when the supplied
    # weights path does not exist locally -- the one guard preventing
    # ultralytics.YOLO(...) from ever being given a bare model name
    # (e.g. "yolov8n.pt") that library would otherwise silently
    # download from the network (Phase 2's own explicit "no hidden
    # network downloads" requirement). Never caught internally --
    # surfaces immediately and honestly to whoever constructed this
    # backend with a bad path.

    pass


class UltralyticsYOLOBackend(YOLOInferenceBackend):

    # The one real (non-test) implementation. Requires an EXISTING
    # local weights file -- checked at construction time, before
    # ultralytics is ever given the chance to interpret a bare model
    # name as something to fetch. Loading the actual model (ultralytics.
    # YOLO(path), which reads and deserializes the weights file) is
    # lazy -- deferred to the first infer() call -- so constructing this
    # backend (and therefore constructing a YOLOHumanDetector around it)
    # is cheap and side-effect-free, the same "zero I/O in __init__"
    # discipline RTSPFrameSource already established one milestone ago.
    # Neither constructing this class nor importing this module ever
    # touches the network.

    def __init__(
        self,
        weights_path: Union[str, Path],
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
    ):

        self._weights_path = Path(weights_path)

        if not self._weights_path.exists():

            raise ModelWeightsNotFoundError(
                f"YOLO weights file not found at {self._weights_path} -- "
                f"UltralyticsYOLOBackend never downloads weights automatically; "
                f"supply the path to an existing local .pt file."
            )

        self._device = device
        self._confidence_threshold = confidence_threshold
        self._model: Optional[Any] = None

    # =====================================================

    def _ensure_loaded(self) -> None:

        if self._model is not None:
            return

        # Imported here, not at module scope, so importing this module
        # (and therefore importing human_detection/ at all) never
        # requires ultralytics/torch to be installed unless a real
        # backend is actually constructed and used.
        from ultralytics import YOLO

        self._model = YOLO(str(self._weights_path))

    # =====================================================

    def infer(self, image: Any) -> Tuple[BoundingBoxDetection, ...]:

        self._ensure_loaded()

        results = self._model.predict(
            image, device=self._device, conf=self._confidence_threshold, verbose=False,
        )

        detections = []

        for result in results:

            names = result.names if hasattr(result, "names") else self._model.names

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])

                detections.append(
                    BoundingBoxDetection(
                        class_id=class_id,
                        class_name=names[class_id],
                        confidence=confidence,
                        bounding_box=(x1, y1, x2, y2),
                    )
                )

        return tuple(detections)
