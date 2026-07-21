import pathlib
import re
import unittest


# Live Occupant Digital Twin milestone, Phase 13 -- the same regex-
# scan-the-source-files architecture guard convention this codebase
# already establishes per package. live_occupants/ must depend only on
# Detection, CrossCameraIdentity, Behavior, Geometry, and Time -- never
# AI, Advisory, Command Center, a YOLO backend, or an RTSP backend.
#
# One deliberate, documented exception: live_system.event_bus (NOT the
# rest of live_system/) -- Phase 9's own explicit "use existing event
# infrastructure if present, do not invent another event bus"
# instruction requires it, and event_bus.py itself has zero dependency
# on AI/Advisory/CommandCenter/anything forbidden (verified: it imports
# only dataclasses/enum/typing/uuid).

FORBIDDEN = (
    r"^\s*(from|import)\s+("
    r"ai_engine|reinforcement_learning|advisory_system|command_center|"
    r"building_state|multi_camera_fusion|camera_manager|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"live_system\.(?!event_bus\b)|"
    r"cv2|torch|ultralytics|onvif"
    r")\b"
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LIVE_OCCUPANTS_PACKAGE = REPO_ROOT / "live_occupants"


class LiveOccupantsArchitectureGuardTests(unittest.TestCase):

    def test_live_occupants_package_imports_nothing_forbidden(self):

        for path in sorted(LIVE_OCCUPANTS_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(FORBIDDEN, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports "
                f"{match.group(0).strip() if match else ''!r} -- live_occupants/ must depend "
                f"only on Detection, CrossCameraIdentity, Behavior, Geometry, and Time, plus "
                f"live_system.event_bus specifically (Live Occupant Digital Twin milestone, "
                f"Phase 13).",
            )

    def test_event_bus_module_itself_has_no_forbidden_dependency(self):

        # A defensive re-check of the ONE exception this guard grants:
        # if live_system/event_bus.py were ever changed to import
        # something forbidden, that would silently defeat this guard's
        # whole purpose -- re-verified directly here.

        event_bus_path = REPO_ROOT / "live_system" / "event_bus.py"
        text = event_bus_path.read_text(encoding="utf-8")

        forbidden_anywhere = (
            r"^\s*(from|import)\s+("
            r"ai_engine|reinforcement_learning|advisory_system|command_center|"
            r"building_state|multi_camera_fusion|camera_manager|"
            r"tracking|behavior_recognition|cross_camera_identity|camera_calibration|"
            r"cv2|torch|ultralytics|onvif"
            r")\b"
        )

        self.assertIsNone(re.search(forbidden_anywhere, text, re.MULTILINE))

    def test_live_occupants_not_nested_inside_another_package(self):

        self.assertTrue(LIVE_OCCUPANTS_PACKAGE.is_dir())
        self.assertEqual(LIVE_OCCUPANTS_PACKAGE.parent, REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
