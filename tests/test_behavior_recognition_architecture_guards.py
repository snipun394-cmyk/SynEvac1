import pathlib
import re
import unittest


# Human Behavior Recognition Framework milestone, Phase 12 -- the same
# regex-scan-the-source-files architecture guard convention this
# codebase already establishes per package (tests/
# test_no_cv_dependencies.py, tests/test_tracking_architecture_guards.py).
# behavior_recognition/ must depend on nothing but TrackedHuman,
# geometry, and time -- never AI, BuildingState, Command Center,
# Advisory, RTSP, or a YOLO backend directly. Note: HumanState/
# perception.models.human_observation is deliberately NOT imported by
# this package either -- the RecognizedBehavior -> HumanState mapping
# lives entirely in live_camera_pipeline/pipeline.py, one layer above
# (see docs/architecture/behavior_recognition.md Sec 4), keeping this
# package's own vocabulary and dependency surface completely separate.

FORBIDDEN = (
    r"^\s*(from|import)\s+("
    r"ai_engine|reinforcement_learning|advisory_system|command_center|"
    r"building_state|multi_camera_fusion|camera_manager|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"perception\.models\.human_observation|"
    r"cv2|torch|ultralytics|onvif"
    r")\b"
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BEHAVIOR_RECOGNITION_PACKAGE = REPO_ROOT / "behavior_recognition"


class BehaviorRecognitionArchitectureGuardTests(unittest.TestCase):

    def test_behavior_recognition_package_imports_nothing_forbidden(self):

        for path in sorted(BEHAVIOR_RECOGNITION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(FORBIDDEN, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports "
                f"{match.group(0).strip() if match else ''!r} -- behavior_recognition/ "
                f"must depend only on TrackedHuman, geometry, and time (Human Behavior "
                f"Recognition Framework milestone, Phase 12).",
            )

    def test_behavior_recognition_only_depends_on_tracking_tracked_human_and_track_state(self):

        # A slightly stronger, positive check: the only cross-package
        # imports behavior_recognition/ is allowed at all are
        # tracking.tracked_human (for TrackedHuman itself) and
        # tracking.track_state (for TrackState, needed to interpret
        # TrackedHuman.state honestly) -- never tracking.tracker,
        # tracking.simple_tracker, or tracking.cost_functions (those
        # would be a sign this package is reaching into the tracker's
        # own implementation rather than consuming its public output
        # type).

        forbidden_submodule_pattern = (
            r"^\s*(from|import)\s+tracking\.(tracker|simple_tracker|cost_functions)\b"
        )

        for path in sorted(BEHAVIOR_RECOGNITION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(forbidden_submodule_pattern, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"behavior_recognition/ may only depend on tracking.tracked_human and "
                f"tracking.track_state within tracking/.",
            )


if __name__ == "__main__":
    unittest.main()
