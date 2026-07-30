import pathlib
import re
import unittest


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone, Phase 16
# -- mechanically proves stair_flow/ stays perception/intelligence
# EVIDENCE only: it must never import NavigationGraph mutation,
# Recommendation, Guidance, Voice Evacuation, Dynamic Signage dispatch,
# Building Control, predictive-model inference, AI training, or
# simulation behavior, and must never itself execute an action. Mirrors
# tests/test_crowd_intelligence_architecture_guards.py's own exact
# convention.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STAIR_FLOW_PACKAGE = REPO_ROOT / "stair_flow"


_FORBIDDEN_IMPORTS = (
    r"^\s*(from|import)\s+("
    r"navigation\.graph|"
    r"evacuation_recommendation|evacuation_guidance|"
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|predictive_model|predictive_dataset|"
    r"advisory_system|command_center|"
    r"voice_evacuation|speaker_manager|"
    r"dynamic_signage|sign_manager|"
    r"building_control|"
    r"facp|"
    r"simulator|simulation_runtime|ground_truth|"
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


class StairFlowArchitectureGuardTests(unittest.TestCase):

    def test_stair_flow_never_imports_navigation_recommendation_or_execution_capable_modules(self):

        for path in sorted(STAIR_FLOW_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_IMPORTS, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"stair_flow/ must stay perception/intelligence evidence only.",
            )

    def test_stair_flow_never_calls_action_execution_verbs(self):

        for path in sorted(STAIR_FLOW_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"stair_flow/ reports evidence, it never executes an action.",
            )

    def test_stair_flow_never_mutates_a_staircase_or_building(self):

        # A weaker, textual proxy check (this package only ever receives
        # already-constructed Staircase/Building objects and reads their
        # existing attributes/methods) -- no `.add_stair(`, `.add_floor(`,
        # `setattr(` or direct attribute assignment onto a passed-in
        # staircase/building anywhere in this package's source.

        forbidden = re.compile(r"\.add_stair\(|\.add_floor\(|staircase\.\w+\s*=|building\.\w+\s*=")

        for path in sorted(STAIR_FLOW_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = forbidden.search(text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} appears to mutate Staircase/Building state: {match.group(0) if match else ''!r}",
            )

if __name__ == "__main__":
    unittest.main()
