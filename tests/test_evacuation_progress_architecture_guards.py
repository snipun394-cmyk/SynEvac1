import pathlib
import re
import unittest


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone,
# Phase 22 -- mechanical dependency-direction guards.
#
# evacuation_progress/ must NOT import AI/RL/Advisory/Command Center/
# Voice Evacuation/Building Control execution/RTSP/YOLO. decision_policy
# must NOT import evacuation_progress. The crowd-advisory-facing
# evidence files (advisory_system.evacuation_progress_evidence,
# advisory_engine's own evacuation-progress additions) must never
# import evacuation_progress/ itself, and must never call an
# execution verb.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EVACUATION_PROGRESS_PACKAGE = REPO_ROOT / "evacuation_progress"
DECISION_POLICY_PACKAGE = REPO_ROOT / "decision_policy"
ADVISORY_SYSTEM_PACKAGE = REPO_ROOT / "advisory_system"

_FORBIDDEN_FOR_EVACUATION_PROGRESS = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|"
    r"advisory_system|command_center|"
    r"voice_evacuation|speaker_manager|"
    r"building_control|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"cv2|torch|ultralytics|onvif"
    r")\b"
)

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"
    r"\.broadcast\(|\.announce\(|"
    r"\.execute_control\(|\.confirm\("
)


class EvacuationProgressArchitectureGuardTests(unittest.TestCase):

    def test_evacuation_progress_never_imports_forbidden_modules(self):

        for path in sorted(EVACUATION_PROGRESS_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_EVACUATION_PROGRESS, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"evacuation_progress/ must never perform AI inference, make evacuation decisions, "
                f"execute building controls, broadcast voice messages, or depend on decision_policy/"
                f"Advisory/Command Center (Phase 22).",
            )

    def test_evacuation_progress_never_calls_action_execution_verbs(self):

        for path in sorted(EVACUATION_PROGRESS_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"evacuation_progress/ reports state, it never executes an action.",
            )

    def test_evacuation_progress_not_nested_inside_another_package(self):

        self.assertTrue(EVACUATION_PROGRESS_PACKAGE.is_dir())
        self.assertEqual(EVACUATION_PROGRESS_PACKAGE.parent, REPO_ROOT)

    def test_evacuation_progress_only_depends_on_allowed_project_packages(self):

        # Allow-list: live_occupants (models/manager/state/events),
        # crowd_intelligence (reused flow-geometry/trends, never
        # duplicated), live_system.event_bus (the shared pub/sub
        # mechanism), and itself.
        allowed_prefixes = (
            "live_occupants", "crowd_intelligence", "live_system.event_bus", "evacuation_progress",
        )

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(EVACUATION_PROGRESS_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, not in evacuation_progress/'s own "
                    f"documented allow-list (Phase 22).",
                )


class DecisionPolicyNeverImportsEvacuationProgressTests(unittest.TestCase):

    def test_decision_policy_never_imports_evacuation_progress(self):

        for path in sorted(DECISION_POLICY_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+evacuation_progress\b", text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports evacuation_progress -- decision_policy must never "
                f"depend on it (Phase 22).",
            )


class EvacuationProgressEvidenceGuardTests(unittest.TestCase):

    def test_advisory_evacuation_progress_evidence_never_imports_evacuation_progress_package(self):

        # Mirrors advisory_system.crowd_evidence's own "no crowd_
        # intelligence import" discipline -- the reduction from a real
        # EvacuationProgressSnapshot happens in live_system.live_
        # advisory_gateway instead.
        text = (ADVISORY_SYSTEM_PACKAGE / "evacuation_progress_evidence.py").read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"^\s*(from|import)\s+evacuation_progress\b", text, re.MULTILINE))

    def test_advisory_evacuation_progress_evidence_never_calls_action_execution_verbs(self):

        text = (ADVISORY_SYSTEM_PACKAGE / "evacuation_progress_evidence.py").read_text(encoding="utf-8")
        match = re.search(_FORBIDDEN_ACTION_CALLS, text)

        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
