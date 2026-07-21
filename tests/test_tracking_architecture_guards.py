import pathlib
import re
import unittest


# Single-Camera Tracking Framework milestone, Phase 12 -- the same
# regex-scan-the-source-files architecture guard convention this
# codebase already establishes per package (tests/
# test_no_cv_dependencies.py, tests.test_camera_manager.
# CameraManagerPackageDependencyDirectionTests, tests.
# test_multi_camera_fusion.MultiCameraFusionPackageDependencyDirection
# Tests). tracking/ must depend on nothing but RawHumanDetection,
# geometry, and time -- never AI, BuildingState, Advisory, Command
# Center, MultiCameraFusion, RTSP, or a YOLO backend directly.

FORBIDDEN = (
    r"^\s*(from|import)\s+("
    r"ai_engine|reinforcement_learning|advisory_system|command_center|"
    r"building_state|multi_camera_fusion|camera_manager|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"cv2|torch|ultralytics|onvif"
    r")\b"
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRACKING_PACKAGE = REPO_ROOT / "tracking"


class TrackingArchitectureGuardTests(unittest.TestCase):

    def test_tracking_package_imports_nothing_forbidden(self):

        for path in sorted(TRACKING_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(FORBIDDEN, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports "
                f"{match.group(0).strip() if match else ''!r} -- tracking/ must "
                f"depend only on RawHumanDetection, geometry, and time (Single-Camera "
                f"Tracking Framework milestone, Phase 12).",
            )

    def test_tracking_package_only_depends_on_live_camera_pipeline_human_detector(self):

        # A slightly stronger, positive check: the only cross-package
        # import tracking/ is allowed at all is
        # live_camera_pipeline.human_detector (for RawHumanDetection
        # itself) -- never live_camera_pipeline.frame_source,
        # .identity_resolver, .pipeline, or .detection_provider (those
        # would be a sign this package is reaching beyond its own
        # single-camera-tracking responsibility).

        allowed_live_camera_pipeline_submodule = "live_camera_pipeline.human_detector"
        forbidden_submodule_pattern = (
            r"^\s*(from|import)\s+live_camera_pipeline\.(frame_source|identity_resolver|pipeline|detection_provider)\b"
        )

        for path in sorted(TRACKING_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(forbidden_submodule_pattern, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"tracking/ may only depend on {allowed_live_camera_pipeline_submodule} "
                f"within live_camera_pipeline/.",
            )


if __name__ == "__main__":
    unittest.main()
