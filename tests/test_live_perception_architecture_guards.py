import pathlib
import re
import unittest


# Live Perception -> BuildingState Integration Bridge milestone, Phase
# 13. Two separate guards:
#
#   1. sensor_fusion/ must remain completely generic -- re-verified
#      here (in addition to tests/test_sensor_fusion_architecture_
#      guards.py, unmodified by this milestone) since this is the
#      milestone that could most easily have tempted a shortcut import
#      from sensor_fusion straight into live_runtime/live_perception
#      internals.
#   2. live_perception/ (the NEW integration layer) MAY import
#      sensor_fusion/building_state/live_occupants/camera_manager/
#      sensor_manager/facp/perception, but must never perform AI
#      inference, make evacuation decisions, execute building controls,
#      or broadcast voice messages.

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SENSOR_FUSION_PACKAGE = REPO_ROOT / "sensor_fusion"
LIVE_PERCEPTION_PACKAGE = REPO_ROOT / "live_perception"


class SensorFusionRemainsGenericTests(unittest.TestCase):

    def test_sensor_fusion_still_imports_nothing_from_this_repository(self):

        project_package_pattern = r"^\s*(from|import)\s+(?!sensor_fusion\b)([a-z_][a-z0-9_]*)\b"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(SENSOR_FUSION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported_name = match.group(2)

                self.assertNotIn(
                    imported_name, project_packages,
                    f"{path.relative_to(REPO_ROOT)} imports project package {imported_name!r} -- "
                    f"sensor_fusion/ must remain completely generic, even after live_perception/ "
                    f"exists to consume it (Live Perception -> BuildingState Integration Bridge "
                    f"milestone, Phase 13).",
                )


_FORBIDDEN_FOR_LIVE_PERCEPTION = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|reinforcement_learning|"
    r"decision_policy|"
    r"advisory_system|command_center|"
    r"voice_evacuation|speaker_manager|"
    r"building_control|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"cv2|torch|ultralytics|onvif"
    r")\b"
)


class LivePerceptionArchitectureGuardTests(unittest.TestCase):

    def test_live_perception_never_imports_ai_advisory_or_execution_capable_modules(self):

        for path in sorted(LIVE_PERCEPTION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_LIVE_PERCEPTION, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} "
                f"-- live_perception/ must never perform AI inference, make evacuation decisions, "
                f"execute building controls, or broadcast voice messages (Live Perception -> "
                f"BuildingState Integration Bridge milestone, Phase 13).",
            )

    def test_live_perception_not_nested_inside_another_package(self):

        self.assertTrue(LIVE_PERCEPTION_PACKAGE.is_dir())
        self.assertEqual(LIVE_PERCEPTION_PACKAGE.parent, REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
