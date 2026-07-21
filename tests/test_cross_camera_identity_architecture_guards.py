import pathlib
import re
import unittest


# Cross-Camera Identity Resolution (ReID Framework) milestone, Phase 12
# -- the same regex-scan-the-source-files architecture guard convention
# this codebase already establishes per package (tests/
# test_no_cv_dependencies.py, tests/test_tracking_architecture_guards.py,
# tests/test_behavior_recognition_architecture_guards.py).
# cross_camera_identity/ must depend only on tracking, behavior
# observations, camera topology, and time -- never AI, BuildingState,
# Advisory, Command Center, a YOLO backend, or an RTSP backend.

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
CROSS_CAMERA_IDENTITY_PACKAGE = REPO_ROOT / "cross_camera_identity"


class CrossCameraIdentityArchitectureGuardTests(unittest.TestCase):

    def test_cross_camera_identity_package_imports_nothing_forbidden(self):

        for path in sorted(CROSS_CAMERA_IDENTITY_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(FORBIDDEN, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports "
                f"{match.group(0).strip() if match else ''!r} -- cross_camera_identity/ "
                f"must depend only on tracking, behavior observations, camera topology, "
                f"and time (Cross-Camera Identity Resolution milestone, Phase 12).",
            )

    def test_cross_camera_identity_only_depends_on_expected_tracking_and_behavior_submodules(self):

        # A slightly stronger, positive check: the only cross-package
        # imports allowed at all are tracking.tracked_human,
        # tracking.track_state (mirroring behavior_recognition's own
        # guard) and behavior_recognition.observation -- never
        # tracking.tracker/simple_tracker/cost_functions, and never
        # behavior_recognition.rule_based_recognizer/recognizer/
        # behavior_history/metrics (this package consumes BOTH
        # upstream packages' own OUTPUT TYPES only, never their
        # internal implementation).

        forbidden_submodule_pattern = (
            r"^\s*(from|import)\s+("
            r"tracking\.(tracker|simple_tracker|cost_functions)|"
            r"behavior_recognition\.(rule_based_recognizer|recognizer|behavior_history|metrics)"
            r")\b"
        )

        for path in sorted(CROSS_CAMERA_IDENTITY_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(forbidden_submodule_pattern, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"cross_camera_identity/ may only depend on tracking.tracked_human, "
                f"tracking.track_state, and behavior_recognition.observation.",
            )

    def test_cross_camera_identity_not_nested_inside_tracking_behavior_or_fusion(self):

        # This package must be its own top-level package -- Phase 2's
        # explicit "do NOT place this inside tracking/, behavior_
        # recognition/, multi_camera_fusion/" instruction.

        self.assertTrue(CROSS_CAMERA_IDENTITY_PACKAGE.is_dir())
        self.assertEqual(CROSS_CAMERA_IDENTITY_PACKAGE.parent, REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
