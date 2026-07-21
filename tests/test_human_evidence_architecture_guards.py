import pathlib
import re
import unittest


# =====================================================
# Live Human State & Assistance Perception Bridge milestone, Phase 26 --
# mechanical dependency-direction guards.
#
# human_evidence/ must NOT import AI/RL/Advisory/Command Center/Voice
# Evacuation/Building Control execution/RTSP/YOLO/decision_policy, and
# must NOT import emergency_response (it only ever CONSUMES the final
# human evidence, via LiveOccupant, never the other way around) or
# live_occupants (this package is deliberately the lower, standalone
# layer -- live_occupants depends on it, never the reverse).
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HUMAN_EVIDENCE_PACKAGE = REPO_ROOT / "human_evidence"
LIVE_OCCUPANTS_PACKAGE = REPO_ROOT / "live_occupants"
EMERGENCY_RESPONSE_PACKAGE = REPO_ROOT / "emergency_response"

_FORBIDDEN_FOR_HUMAN_EVIDENCE = (
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
    r"emergency_response|live_occupants"
    r")\b"
)

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"
    r"\.broadcast\(|\.announce\(|"
    r"\.execute_control\(|\.confirm\(|\.dispatch\("
)


class HumanEvidenceArchitectureGuardTests(unittest.TestCase):

    def test_human_evidence_never_imports_forbidden_modules(self):

        for path in sorted(HUMAN_EVIDENCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_HUMAN_EVIDENCE, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"human_evidence/ must never depend on AI/Advisory/Command Center/decision_policy/"
                f"simulation-only sources, live_occupants, or emergency_response (Phase 26).",
            )

    def test_human_evidence_never_calls_action_execution_verbs(self):

        for path in sorted(HUMAN_EVIDENCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"human_evidence/ only reconciles evidence, it never executes or dispatches.",
            )

    def test_human_evidence_not_nested_inside_another_package(self):

        self.assertTrue(HUMAN_EVIDENCE_PACKAGE.is_dir())
        self.assertEqual(HUMAN_EVIDENCE_PACKAGE.parent, REPO_ROOT)

    def test_human_evidence_only_depends_on_allowed_project_packages(self):

        # Allow-list: perception.models.human_observation (HumanClassification/
        # HumanState enums only) and itself. Nothing else.
        allowed_prefixes = ("perception.models.human_observation", "human_evidence")

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(HUMAN_EVIDENCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, not in human_evidence/'s own "
                    f"documented allow-list (Phase 26).",
                )


class EmergencyResponseNeverImportsHumanEvidenceTests(unittest.TestCase):

    def test_emergency_response_never_imports_human_evidence_directly(self):

        # Phase 26's own explicit requirement -- emergency_response
        # consumes the already-reconciled evidence via LiveOccupant's
        # own fields, never by importing human_evidence/ itself (no
        # circular or redundant dependency).

        for path in sorted(EMERGENCY_RESPONSE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+human_evidence\b", text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports human_evidence -- emergency_response must consume "
                f"evidence via LiveOccupant only, never generate or import it directly (Phase 26).",
            )


if __name__ == "__main__":
    unittest.main()
