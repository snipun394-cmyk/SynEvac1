import pathlib
import re
import unittest


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone, Phase 16 --
# mechanical dependency-direction guards, mirroring
# tests/test_trajectory_intelligence_architecture_guards.py exactly.
#
# evacuation_recommendation/ must NOT import AI/RL/Advisory/Command
# Center/Voice Evacuation/Building Control execution/RTSP/YOLO/
# simulation GroundTruth/decision_policy. It must never call an
# execution verb, and it must never MODIFY Decision Policy (i.e. never
# import it at all).
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EVACUATION_RECOMMENDATION_PACKAGE = REPO_ROOT / "evacuation_recommendation"
DECISION_POLICY_PACKAGE = REPO_ROOT / "decision_policy"
ADVISORY_SYSTEM_PACKAGE = REPO_ROOT / "advisory_system"

_FORBIDDEN_FOR_EVACUATION_RECOMMENDATION = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|"
    r"advisory_system|command_center|"
    r"voice_evacuation|speaker_manager|"
    r"building_control|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"cv2|torch|ultralytics|onvif|"
    r"ground_truth|human_decision_engine|simulator|behaviour_profile_resolver|"
    r"crowd_intelligence\.engine|evacuation_progress\.engine|trajectory_intelligence\.engine|"
    r"trajectory_intelligence\.route_progress|emergency_response\.engine"
    r")\b"
)

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"
    r"\.broadcast\(|\.announce\(|"
    r"\.execute_control\(|\.confirm\(|\.dispatch\("
)


class EvacuationRecommendationArchitectureGuardTests(unittest.TestCase):

    def test_evacuation_recommendation_never_imports_forbidden_modules(self):

        for path in sorted(EVACUATION_RECOMMENDATION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_EVACUATION_RECOMMENDATION, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"evacuation_recommendation/ must never depend on AI/Advisory/Command Center/decision_policy/"
                f"simulation-only sources, sibling live-intelligence engines, or execution-capable modules "
                f"(Phase 16).",
            )

    def test_evacuation_recommendation_never_calls_action_execution_verbs(self):

        for path in sorted(EVACUATION_RECOMMENDATION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"evacuation_recommendation/ ranks and explains, it never executes or dispatches.",
            )

    def test_evacuation_recommendation_not_nested_inside_another_package(self):

        self.assertTrue(EVACUATION_RECOMMENDATION_PACKAGE.is_dir())
        self.assertEqual(EVACUATION_RECOMMENDATION_PACKAGE.parent, REPO_ROOT)

    def test_evacuation_recommendation_only_depends_on_allowed_project_packages(self):

        # Allow-list: navigation/pathfinding (the Navigation Graph +
        # routing engine, decision_policy-free), hazard.severity (the
        # shared HazardSeverity enum), and itself. Deliberately does NOT
        # import live_occupants, crowd_intelligence, evacuation_progress,
        # trajectory_intelligence, or emergency_response -- every
        # sibling snapshot this package reads arrives as a plain,
        # untyped/duck-typed object (mirrors emergency_response.engine's
        # own established convention), so this package has no import-
        # level dependency on any of them.
        allowed_prefixes = (
            "navigation", "pathfinding", "hazard.severity", "evacuation_recommendation",
        )

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(EVACUATION_RECOMMENDATION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, not in evacuation_recommendation/'s own "
                    f"documented allow-list (Phase 16).",
                )


class DecisionPolicyNeverImportsEvacuationRecommendationTests(unittest.TestCase):

    def test_decision_policy_never_imports_evacuation_recommendation(self):

        for path in sorted(DECISION_POLICY_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+evacuation_recommendation\b", text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports evacuation_recommendation -- decision_policy must never "
                f"depend on it, and this milestone never modifies decision_policy/ (Phase 16).",
            )

    def test_evacuation_recommendation_never_modifies_decision_policy(self):

        # This milestone's own explicit "Do NOT modify Decision Policy"
        # requirement (Phase 2) -- mechanically confirmed by asserting
        # evacuation_recommendation/ never imports it at all (covered
        # above by the forbidden-modules guard) AND that no file under
        # decision_policy/ was touched by this milestone's own commit
        # is a git-history concern, not a static one -- left to code
        # review; this test only re-asserts the import-direction half.

        for path in sorted(EVACUATION_RECOMMENDATION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+decision_policy\b", text, re.MULTILINE)

            self.assertIsNone(match)


class EvacuationRecommendationEvidenceGuardTests(unittest.TestCase):

    def test_advisory_evidence_never_imports_evacuation_recommendation_package(self):

        path = ADVISORY_SYSTEM_PACKAGE / "evacuation_recommendation_evidence.py"

        if not path.exists():
            self.skipTest("advisory_system/evacuation_recommendation_evidence.py not present")

        text = path.read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"^\s*(from|import)\s+evacuation_recommendation\b", text, re.MULTILINE))

    def test_advisory_evidence_never_calls_action_execution_verbs(self):

        path = ADVISORY_SYSTEM_PACKAGE / "evacuation_recommendation_evidence.py"

        if not path.exists():
            self.skipTest("advisory_system/evacuation_recommendation_evidence.py not present")

        text = path.read_text(encoding="utf-8")
        match = re.search(_FORBIDDEN_ACTION_CALLS, text)

        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
