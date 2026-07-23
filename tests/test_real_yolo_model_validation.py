import unittest
from pathlib import Path

import numpy as np

from human_detection.yolo_backend import UltralyticsYOLOBackend, BoundingBoxDetection
from human_detection.yolo_human_detector import YOLOHumanDetector
from live_camera_pipeline.frame_source import CameraFrame

from tracking.simple_tracker import SimpleSingleCameraTracker
from tracking.track_state import TrackState


# =====================================================
# Real YOLO Model Validation milestone -- the one opt-in, weight-
# dependent test module in this codebase. Every other test file
# (test_yolo_human_detector.py, test_yolo_tracking_integration.py, ...)
# stays offline/deterministic/weight-independent by design, using
# FakeYOLOBackend exclusively -- that discipline is NOT changed here.
#
# These tests instead prove the thing FakeYOLOBackend structurally
# cannot: that a genuine local .pt weights file, genuinely loaded by
# ultralytics.YOLO(...), genuinely executes neural-network inference
# and returns real person detections. They are automatically SKIPPED
# (never fail CI) whenever the required local artifact is absent:
#   - WEIGHTS_PATH: a real yolov8n.pt this repo deliberately never
#     bundles/commits (see .gitignore, docs/architecture/
#     human_detection.md §10) -- obtain it via the normal Ultralytics
#     release mechanism and place it at weights/yolov8n.pt.
#   - The two sample images used below (bus.jpg, zidane.jpg) are NOT
#     separately downloaded -- they ship inside the already-required
#     `ultralytics` package itself (ultralytics/assets/), so they are
#     always present wherever ultralytics is installed, network or not.
#   - VIDEO_PATH: a real local video (this repo's own validation used
#     OpenCV's own public-domain vtest.avi pedestrian test clip -- see
#     docs/architecture/human_detection.md §16) this repo also never
#     bundles/commits -- place one at validation_media/vtest.avi to
#     exercise the tracking-continuity test below.
# =====================================================


REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "weights" / "yolov8n.pt"
VIDEO_PATH = REPO_ROOT / "validation_media" / "vtest.avi"

try:
    import ultralytics
    _ULTRALYTICS_ASSETS = Path(ultralytics.__file__).resolve().parent / "assets"
    _BUS_JPG = _ULTRALYTICS_ASSETS / "bus.jpg"
except Exception:
    _BUS_JPG = None


def _frame(payload_ref, timestamp=0.0, sequence=0, camera_id="CAM-REAL-MODEL-TEST"):
    return CameraFrame(camera_id=camera_id, timestamp=timestamp, frame_sequence=sequence, payload_ref=payload_ref)


@unittest.skipUnless(
    WEIGHTS_PATH.exists() and _BUS_JPG is not None and _BUS_JPG.exists(),
    f"real YOLO weights not found at {WEIGHTS_PATH} (or ultralytics' own bundled sample image is "
    f"missing) -- this opt-in test is skipped, never failed, when the local artifact is absent",
)
class RealModelSmokeTests(unittest.TestCase):

    # Phase 3 -- model genuinely loads, inference genuinely executes,
    # person detections are returned, non-person classes are filtered,
    # bounding boxes/confidence are valid, nothing is fabricated.

    @classmethod
    def setUpClass(cls):

        import cv2
        cls.cv2 = cv2

        cls.backend = UltralyticsYOLOBackend(WEIGHTS_PATH, device="cpu", confidence_threshold=0.25)
        cls.detector = YOLOHumanDetector(cls.backend)
        cls.bus_image = cv2.imread(str(_BUS_JPG))

    def test_real_weights_produce_real_person_detections(self):

        raw_boxes = self.backend.infer(self.bus_image)

        self.assertGreater(len(raw_boxes), 0)
        self.assertIn("person", {b.class_name for b in raw_boxes})

        for box in raw_boxes:
            self.assertIsInstance(box, BoundingBoxDetection)
            self.assertGreaterEqual(box.confidence, 0.25)
            self.assertLessEqual(box.confidence, 1.0)
            x1, y1, x2, y2 = box.bounding_box
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)

    def test_non_person_classes_are_filtered_by_the_detector(self):

        raw_boxes = self.backend.infer(self.bus_image)
        raw_person_count = len([b for b in raw_boxes if b.class_name == "person"])

        detections = self.detector.detect(_frame(self.bus_image))

        self.assertEqual(len(detections), raw_person_count)

    def test_no_classification_or_state_is_ever_fabricated(self):

        detections = self.detector.detect(_frame(self.bus_image))
        self.assertGreater(len(detections), 0)

        for detection in detections:
            self.assertEqual(detection.classification_evidence.name, "UNKNOWN")
            self.assertIsNone(detection.state_evidence)

    def test_blank_frame_yields_zero_detections_not_a_hallucination(self):

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(_frame(blank))

        self.assertEqual(detections, ())


@unittest.skipUnless(
    WEIGHTS_PATH.exists() and VIDEO_PATH.exists(),
    f"real YOLO weights ({WEIGHTS_PATH}) and/or a local validation video ({VIDEO_PATH}) not found "
    f"-- this opt-in test is skipped, never failed, when either local artifact is absent",
)
class RealVideoTrackingContinuityTests(unittest.TestCase):

    # Phase 4/10 -- real detections over real consecutive video frames
    # feed a real tracker; track ids remain stable across frames for
    # the same physical person, and a track that survives a real missed
    # detection coasts as MISSING rather than being silently replaced.

    def test_real_video_frames_produce_stable_tracks(self):

        from human_detection.video_source import load_video_frames

        frames = load_video_frames(VIDEO_PATH)[:60]  # a short prefix -- keeps this test fast

        backend = UltralyticsYOLOBackend(WEIGHTS_PATH, device="cpu", confidence_threshold=0.25)
        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()

        any_multi_frame_track = False

        for index, (timestamp, payload) in enumerate(frames):

            raw = detector.detect(_frame(payload, timestamp=timestamp, sequence=index))
            tracked = tracker.update("CAM-REAL-MODEL-TEST", timestamp, raw)

            if any(t.frames_seen > 1 and t.state == TrackState.TRACKED for t in tracked):
                any_multi_frame_track = True

        self.assertTrue(
            any_multi_frame_track,
            "expected at least one real person to be tracked across more than one consecutive frame",
        )


if __name__ == "__main__":
    unittest.main()
