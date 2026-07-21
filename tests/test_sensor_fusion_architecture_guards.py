import pathlib
import re
import unittest


# Sensor Fusion Engine milestone, Phase 12 -- the same regex-scan-the-
# source-files architecture guard convention this codebase already
# establishes per package. sensor_fusion/ must depend only on
# observation providers, geometry, and time -- never AI, Advisory,
# Command Center, a YOLO backend, or an RTSP backend. In practice this
# package needed no external package dependency at all (every concrete
# provider in provider.py is duck-typed against plain objects/mappings,
# never importing perception/facp/live_occupants/hazard/building_state
# directly) -- confirmed by the second, stronger test below.

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
SENSOR_FUSION_PACKAGE = REPO_ROOT / "sensor_fusion"


class SensorFusionArchitectureGuardTests(unittest.TestCase):

    def test_sensor_fusion_package_imports_nothing_forbidden(self):

        for path in sorted(SENSOR_FUSION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(FORBIDDEN, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports "
                f"{match.group(0).strip() if match else ''!r} -- sensor_fusion/ must depend "
                f"only on observation providers, geometry, and time (Sensor Fusion Engine "
                f"milestone, Phase 12).",
            )

    def test_sensor_fusion_imports_no_other_project_package_at_all(self):

        # A stronger, positive check: every concrete provider in
        # provider.py is duck-typed against plain objects/mappings
        # (SmokeDetectorReading/HeatDetectorReading/LiveOccupant/
        # FACPSnapshot are never imported -- a caller adapts its own
        # real objects into plain Observation instances or hands them
        # to set_readings()/set_occupants()/set_snapshot() by
        # attribute access alone), so this package imports NOTHING from
        # the rest of this repository except its own submodules.

        project_package_pattern = r"^\s*(from|import)\s+(?!sensor_fusion\b)([a-z_][a-z0-9_]*)\b"

        # Every top-level package name in the repo root, so a match
        # against a real project package (not a stdlib module) can be
        # identified precisely.
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
                    f"sensor_fusion/ is designed to depend on NOTHING else in this repository.",
                )

    def test_sensor_fusion_not_nested_inside_another_package(self):

        self.assertTrue(SENSOR_FUSION_PACKAGE.is_dir())
        self.assertEqual(SENSOR_FUSION_PACKAGE.parent, REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
