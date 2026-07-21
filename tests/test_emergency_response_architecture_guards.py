import pathlib
import re
import unittest


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone,
# Phase 25 -- mechanical dependency-direction guards.
#
# emergency_response/ must NOT import AI/RL/Advisory/Command Center/
# Voice Evacuation/Building Control execution/RTSP/YOLO/decision_policy.
# decision_policy must NOT import emergency_response.
# EmergencyResponseIntelligenceEngine must never call an execution verb.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EMERGENCY_RESPONSE_PACKAGE = REPO_ROOT / "emergency_response"
DECISION_POLICY_PACKAGE = REPO_ROOT / "decision_policy"
ADVISORY_SYSTEM_PACKAGE = REPO_ROOT / "advisory_system"

_FORBIDDEN_FOR_EMERGENCY_RESPONSE = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|"
    r"advisory_system|command_center|"
    r"voice_evacuation|speaker_manager|"
    r"building_control|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"cv2|torch|ultralytics|onvif|"
    r"ground_truth|human_decision_engine|simulator|behaviour_profile_resolver"
    r")\b"
)

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"
    r"\.broadcast\(|\.announce\(|"
    r"\.execute_control\(|\.confirm\(|\.dispatch\("
)


class EmergencyResponseArchitectureGuardTests(unittest.TestCase):

    def test_emergency_response_never_imports_forbidden_modules(self):

        for path in sorted(EMERGENCY_RESPONSE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_EMERGENCY_RESPONSE, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"emergency_response/ must never depend on AI/Advisory/Command Center/decision_policy/"
                f"simulation-only sources, or execution-capable modules (Phase 25).",
            )

    def test_emergency_response_never_calls_action_execution_verbs(self):

        for path in sorted(EMERGENCY_RESPONSE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"emergency_response/ ranks and explains, it never executes or dispatches.",
            )

    def test_emergency_response_not_nested_inside_another_package(self):

        self.assertTrue(EMERGENCY_RESPONSE_PACKAGE.is_dir())
        self.assertEqual(EMERGENCY_RESPONSE_PACKAGE.parent, REPO_ROOT)

    def test_emergency_response_only_depends_on_allowed_project_packages(self):

        # Allow-list: live_occupants, behavior_recognition (RecognizedBehavior
        # enum only), perception.models.human_observation (HumanState enum
        # only -- the one deliberately narrow exception documented in
        # engine.py's own module docstring), hazard.severity, crowd_
        # intelligence (reused IntensityLevel), evacuation_progress (reused
        # ZoneClearanceStatus), and itself.
        allowed_prefixes = (
            "live_occupants", "behavior_recognition.observation", "perception.models.human_observation",
            "hazard.severity", "crowd_intelligence.models", "evacuation_progress.models", "emergency_response",
        )

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(EMERGENCY_RESPONSE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, not in emergency_response/'s own "
                    f"documented allow-list (Phase 25).",
                )


class DecisionPolicyNeverImportsEmergencyResponseTests(unittest.TestCase):

    def test_decision_policy_never_imports_emergency_response(self):

        for path in sorted(DECISION_POLICY_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+emergency_response\b", text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports emergency_response -- decision_policy must never "
                f"depend on it (Phase 25).",
            )


class EmergencyResponseEvidenceGuardTests(unittest.TestCase):

    def test_advisory_emergency_response_evidence_never_imports_emergency_response_package(self):

        text = (ADVISORY_SYSTEM_PACKAGE / "emergency_response_evidence.py").read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"^\s*(from|import)\s+emergency_response\b", text, re.MULTILINE))

    def test_advisory_emergency_response_evidence_never_calls_action_execution_verbs(self):

        text = (ADVISORY_SYSTEM_PACKAGE / "emergency_response_evidence.py").read_text(encoding="utf-8")
        match = re.search(_FORBIDDEN_ACTION_CALLS, text)

        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
