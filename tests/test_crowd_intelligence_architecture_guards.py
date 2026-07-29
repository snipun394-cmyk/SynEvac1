import pathlib
import re
import unittest


# =====================================================
# Live Occupancy, Crowd Density & Congestion Intelligence milestone,
# Phase 18 -- crowd_intelligence/ must never import AI/Advisory/Command
# Center/RL/YOLO/RTSP/Voice Evacuation/Building Control execution, and
# must never itself make an evacuation recommendation, broadcast an
# announcement, execute a control, or change FACP state. It MAY consume
# LiveOccupant models/manager, Building geometry, Navigation geometry,
# and time.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CROWD_INTELLIGENCE_PACKAGE = REPO_ROOT / "crowd_intelligence"


_FORBIDDEN_IMPORTS = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|"
    r"advisory_system|command_center|"
    r"voice_evacuation|speaker_manager|"
    r"building_control|"
    r"facp|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"cv2|torch|ultralytics|onvif"
    r")\b"
)

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"  # FACP mutation verbs
    r"\.broadcast\(|\.announce\(|"  # Voice Evacuation
    r"\.execute_control\(|\.confirm\("  # Building Control
)


class CrowdIntelligenceArchitectureGuardTests(unittest.TestCase):

    def test_crowd_intelligence_never_imports_ai_advisory_or_execution_capable_modules(self):

        for path in sorted(CROWD_INTELLIGENCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_IMPORTS, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"crowd_intelligence/ must never perform AI inference, make evacuation decisions, "
                f"execute building controls, broadcast voice messages, or change FACP state.",
            )

    def test_crowd_intelligence_never_calls_action_execution_verbs(self):

        for path in sorted(CROWD_INTELLIGENCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"crowd_intelligence/ reports state, it never executes an action.",
            )

    def test_crowd_intelligence_not_nested_inside_another_package(self):

        self.assertTrue(CROWD_INTELLIGENCE_PACKAGE.is_dir())
        self.assertEqual(CROWD_INTELLIGENCE_PACKAGE.parent, REPO_ROOT)

    def test_crowd_intelligence_only_depends_on_allowed_project_packages(self):

        # Phase 18's own allow-list: LiveOccupant models/manager, Building
        # geometry (models.*), Navigation geometry (navigation.edge, used
        # only for its read-only Edge view -- crowd_intelligence.capacity),
        # simulator.capacity (pure, stateless capacity FORMULAS -- see
        # crowd_intelligence.capacity's own module docstring for why this
        # is safely reusable, never a MultiAgentSimulation dependency),
        # behavior_recognition (RecognizedBehavior enum only), and time.
        #
        # Observable Asset Perception Framework milestone -- one
        # deliberate addition: observable_assets (a pure value-object +
        # pure-function package, the same "no action-execution
        # capability of any kind" category this allow-list already
        # grants navigation.edge/models/behavior_recognition.observation).
        # CrowdIntelligenceEngine.compute() now takes an
        # observable_assets.models.ObservableAssetSnapshot directly (see
        # crowd_intelligence/engine.py's own docstring for compute()),
        # superseding an earlier plain-dict-only design.
        allowed_prefixes = (
            "live_occupants", "models", "navigation.edge", "simulator.capacity",
            "behavior_recognition.observation", "crowd_intelligence", "observable_assets",
        )

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(CROWD_INTELLIGENCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue  # a stdlib/typing import, not a project package

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, not in crowd_intelligence/'s own "
                    f"documented allow-list (Phase 18).",
                )


if __name__ == "__main__":
    unittest.main()
