import pathlib
import re
import unittest


# Camera Calibration & World Coordinate Projection milestone, Phase 11
# -- the same regex-scan-the-source-files architecture guard convention
# this codebase already establishes per package. camera_calibration/
# must depend only on geometry, camera calibration (its own types), and
# building geometry (models/zone.py-shaped data, visibility/geometry.py)
# -- never AI, BuildingState, Advisory, Command Center, a YOLO backend,
# tracking, or behavior_recognition (unlike cross_camera_identity/,
# this package is explicitly forbidden from depending on either of the
# latter two -- it has no notion of time/tracks at all, purely
# per-frame geometry).

FORBIDDEN = (
    r"^\s*(from|import)\s+("
    r"ai_engine|reinforcement_learning|advisory_system|command_center|"
    r"building_state|multi_camera_fusion|camera_manager|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"tracking|behavior_recognition|cross_camera_identity|"
    r"cv2|torch|ultralytics|onvif"
    r")\b"
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CAMERA_CALIBRATION_PACKAGE = REPO_ROOT / "camera_calibration"


class CameraCalibrationArchitectureGuardTests(unittest.TestCase):

    def test_camera_calibration_package_imports_nothing_forbidden(self):

        for path in sorted(CAMERA_CALIBRATION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(FORBIDDEN, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports "
                f"{match.group(0).strip() if match else ''!r} -- camera_calibration/ "
                f"must depend only on geometry, camera calibration, and building geometry "
                f"(Camera Calibration & World Coordinate Projection milestone, Phase 11).",
            )

    def test_camera_calibration_not_nested_inside_another_package(self):

        self.assertTrue(CAMERA_CALIBRATION_PACKAGE.is_dir())
        self.assertEqual(CAMERA_CALIBRATION_PACKAGE.parent, REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
