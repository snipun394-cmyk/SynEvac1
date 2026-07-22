import pathlib
import re
import unittest


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase 28
# -- mechanical dependency-direction guards, mirroring
# tests/test_evacuation_recommendation_architecture_guards.py exactly.
#
# evacuation_guidance/ must NOT import YOLO/RTSP/AI training/RL/Command
# Center/voice output providers/Building Control execution/FACP
# mutation code/physical hardware protocols. It may depend on
# navigation/pathfinding, plain recommendation models (evacuation_
# recommendation.models only, never .engine/.ranking), building models,
# and hazard.severity -- kept narrow.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EVACUATION_GUIDANCE_PACKAGE = REPO_ROOT / "evacuation_guidance"
ADVISORY_SYSTEM_PACKAGE = REPO_ROOT / "advisory_system"

_FORBIDDEN_FOR_EVACUATION_GUIDANCE = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|"
    r"advisory_system|command_center|"
    r"voice_evacuation\.provider|speaker_manager\.manager|"
    r"building_control|"
    r"facp|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"cv2|torch|ultralytics|onvif|"
    r"ground_truth|human_decision_engine|simulator|behaviour_profile_resolver|"
    r"crowd_intelligence\.engine|evacuation_progress\.engine|trajectory_intelligence\.engine|"
    r"trajectory_intelligence\.route_progress|emergency_response\.engine|"
    r"evacuation_recommendation\.engine|evacuation_recommendation\.ranking"
    r")\b"
)

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"
    r"\.broadcast\(|\.announce\(|\.send\(|"
    r"\.execute_control\(|\.confirm\(|\.dispatch\("
)


class EvacuationGuidanceArchitectureGuardTests(unittest.TestCase):

    def test_evacuation_guidance_never_imports_forbidden_modules(self):

        for path in sorted(EVACUATION_GUIDANCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_EVACUATION_GUIDANCE, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"evacuation_guidance/ must never depend on AI/Advisory/Command Center/voice output providers/"
                f"building control/FACP execution, sibling live-intelligence engines, or hardware protocols "
                f"(Phase 28).",
            )

    def test_evacuation_guidance_never_calls_action_execution_verbs(self):

        for path in sorted(EVACUATION_GUIDANCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"evacuation_guidance/ plans, it never executes, sends, or broadcasts (Phase 19).",
            )

    def test_evacuation_guidance_not_nested_inside_another_package(self):

        self.assertTrue(EVACUATION_GUIDANCE_PACKAGE.is_dir())
        self.assertEqual(EVACUATION_GUIDANCE_PACKAGE.parent, REPO_ROOT)

    def test_evacuation_guidance_only_depends_on_allowed_project_packages(self):

        # Allow-list: navigation/pathfinding (the Navigation Graph +
        # routing engine), hazard.severity, evacuation_recommendation.models
        # (plain recommendation status/data only, never its engine), and
        # itself. Deliberately does NOT import voice_evacuation/
        # speaker_manager/live_occupants -- every collaborator (speaker_
        # manager instance) arrives duck-typed as a plain compute()
        # parameter, never an import (Phase 28's own "keep dependencies
        # narrow" requirement).
        allowed_prefixes = (
            "navigation", "pathfinding", "hazard.severity", "evacuation_recommendation.models",
            "evacuation_guidance",
        )

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(EVACUATION_GUIDANCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, not in evacuation_guidance/'s own "
                    f"documented allow-list (Phase 28).",
                )


class EvacuationGuidanceNeverModifiesUpstreamTests(unittest.TestCase):

    def test_decision_policy_never_imports_evacuation_guidance(self):

        decision_policy_package = REPO_ROOT / "decision_policy"

        for path in sorted(decision_policy_package.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+evacuation_guidance\b", text, re.MULTILINE)

            self.assertIsNone(match)

    def test_evacuation_recommendation_never_imports_evacuation_guidance(self):

        evacuation_recommendation_package = REPO_ROOT / "evacuation_recommendation"

        for path in sorted(evacuation_recommendation_package.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+evacuation_guidance\b", text, re.MULTILINE)

            self.assertIsNone(match)


class GuidanceEvidenceGuardTests(unittest.TestCase):

    def test_advisory_evidence_never_imports_evacuation_guidance_package(self):

        path = ADVISORY_SYSTEM_PACKAGE / "evacuation_guidance_evidence.py"

        if not path.exists():
            self.skipTest("advisory_system/evacuation_guidance_evidence.py not present")

        text = path.read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"^\s*(from|import)\s+evacuation_guidance\b", text, re.MULTILINE))

    def test_advisory_evidence_never_calls_action_execution_verbs(self):

        path = ADVISORY_SYSTEM_PACKAGE / "evacuation_guidance_evidence.py"

        if not path.exists():
            self.skipTest("advisory_system/evacuation_guidance_evidence.py not present")

        text = path.read_text(encoding="utf-8")
        match = re.search(_FORBIDDEN_ACTION_CALLS, text)

        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
