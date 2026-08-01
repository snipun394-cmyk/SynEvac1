import pathlib
import re
import unittest

from human_detection.yolo_backend import ModelWeightsNotFoundError
from human_detection.yolo_human_detector import YOLOHumanDetector

from live_camera_pipeline.identity_resolver import SimulationIdentityResolver

from live_runtime_launcher.human_detector_wiring import build_yolo_human_detector


# =====================================================
# Camera 1 Live Human-Detection Integration milestone -- proves the
# NEW composition seam in isolation: build_yolo_human_detector() pairs
# the real production YOLOHumanDetector/UltralyticsYOLOBackend with
# the real SimulationIdentityResolver, exactly the combination
# docs/architecture/human_detection.md's own real-world validation run
# already used. Uses the REAL local weights file already present in
# this repo (weights/yolov8n.pt) -- construction only, never calls
# .detect()/.infer(), so this never imports/touches ultralytics/torch
# (UltralyticsYOLOBackend._ensure_loaded() is lazy, only triggered by
# infer()) -- no real NVR, no real RTSP, no real credentials, no
# network, no GPU.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL_WEIGHTS_PATH = REPO_ROOT / "weights" / "yolov8n.pt"


class BuildYoloHumanDetectorTests(unittest.TestCase):

    def test_real_weights_file_produces_the_expected_real_classes(self):

        self.assertTrue(REAL_WEIGHTS_PATH.exists(), "expected weights/yolov8n.pt to exist in this repo")

        detector, identity_resolver = build_yolo_human_detector(REAL_WEIGHTS_PATH)

        self.assertIsInstance(detector, YOLOHumanDetector)
        self.assertIsInstance(identity_resolver, SimulationIdentityResolver)

    def test_missing_weights_file_fails_honestly_never_downloads(self):

        with self.assertRaises(ModelWeightsNotFoundError):
            build_yolo_human_detector(REPO_ROOT / "weights" / "does-not-exist.pt")

    def test_device_argument_is_passed_through(self):

        # Construction-only check (never calls infer()) -- proves the
        # device kwarg reaches UltralyticsYOLOBackend rather than being
        # silently dropped.
        detector, _ = build_yolo_human_detector(REAL_WEIGHTS_PATH, device="cpu")

        self.assertEqual(detector._backend._device, "cpu")


class HumanDetectorWiringNeverTouchesCredentialsTests(unittest.TestCase):

    # Requirement 5 (structural): credential values must never become
    # part of detector/pipeline configuration -- proven the same way
    # this codebase already proves similar package-boundary claims
    # (tests/test_no_cv_dependencies.py, tests/test_rtsp_camera_
    # manager_status_integration.py's own dependency-direction check):
    # a regex scan over the source file itself.

    def test_source_never_references_credentials_or_passwords(self):

        path = REPO_ROOT / "live_runtime_launcher" / "human_detector_wiring.py"
        text = path.read_text(encoding="utf-8")

        forbidden = r"(?i)\b(credential|password)\b"

        self.assertIsNone(
            re.search(forbidden, text),
            f"{path} must never reference credentials/passwords -- human detector "
            f"construction has nothing to do with camera authentication",
        )


if __name__ == "__main__":
    unittest.main()
