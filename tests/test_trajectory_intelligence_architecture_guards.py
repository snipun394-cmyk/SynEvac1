import pathlib
import re
import unittest


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 30 -- mechanical dependency-direction
# guards, mirroring tests/test_emergency_response_architecture_guards.py
# exactly.
#
# trajectory_intelligence/ must NOT import AI/RL/Advisory/Command
# Center/Voice Evacuation/Building Control execution/RTSP/YOLO/
# simulation GroundTruth/decision_policy. It must never call an
# execution verb. emergency_response/'s own allow-list is extended (not
# replaced) to permit trajectory_intelligence.models -- the same
# pattern crowd_intelligence.models/evacuation_progress.models already
# established there.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAJECTORY_INTELLIGENCE_PACKAGE = REPO_ROOT / "trajectory_intelligence"
DECISION_POLICY_PACKAGE = REPO_ROOT / "decision_policy"
ADVISORY_SYSTEM_PACKAGE = REPO_ROOT / "advisory_system"
EMERGENCY_RESPONSE_PACKAGE = REPO_ROOT / "emergency_response"

_FORBIDDEN_FOR_TRAJECTORY_INTELLIGENCE = (
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


class TrajectoryIntelligenceArchitectureGuardTests(unittest.TestCase):

    def test_trajectory_intelligence_never_imports_forbidden_modules(self):

        for path in sorted(TRAJECTORY_INTELLIGENCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_TRAJECTORY_INTELLIGENCE, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"trajectory_intelligence/ must never depend on AI/Advisory/Command Center/decision_policy/"
                f"simulation-only sources, or execution-capable modules (Phase 30).",
            )

    def test_trajectory_intelligence_never_calls_action_execution_verbs(self):

        for path in sorted(TRAJECTORY_INTELLIGENCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"trajectory_intelligence/ observes and reports, it never executes or dispatches.",
            )

    def test_trajectory_intelligence_not_nested_inside_another_package(self):

        self.assertTrue(TRAJECTORY_INTELLIGENCE_PACKAGE.is_dir())
        self.assertEqual(TRAJECTORY_INTELLIGENCE_PACKAGE.parent, REPO_ROOT)

    def test_trajectory_intelligence_only_depends_on_allowed_project_packages(self):

        # Allow-list: navigation/pathfinding (the Navigation Graph +
        # routing engine, decision_policy-free), hazard.severity (the
        # shared HazardSeverity enum), and itself. Deliberately does NOT
        # import live_occupants -- every LiveOccupant/OccupantHistory
        # field this package reads arrives as a plain, untyped object
        # (the same "typed Any, not imported for its own sake"
        # convention decision_policy.policy.DecisionInputs.ground_truth
        # already established), so this package has no import-level
        # dependency on live_occupants at all.
        allowed_prefixes = (
            "navigation", "pathfinding", "hazard.severity", "trajectory_intelligence",
        )

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(TRAJECTORY_INTELLIGENCE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, not in trajectory_intelligence/'s own "
                    f"documented allow-list (Phase 30).",
                )


class DecisionPolicyNeverImportsTrajectoryIntelligenceTests(unittest.TestCase):

    def test_decision_policy_never_imports_trajectory_intelligence(self):

        for path in sorted(DECISION_POLICY_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+trajectory_intelligence\b", text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports trajectory_intelligence -- decision_policy must never "
                f"depend on it (Phase 30).",
            )


class TrajectoryEvidenceGuardTests(unittest.TestCase):

    def test_advisory_trajectory_evidence_never_imports_trajectory_intelligence_package(self):

        path = ADVISORY_SYSTEM_PACKAGE / "trajectory_evidence.py"

        if not path.exists():
            self.skipTest("advisory_system/trajectory_evidence.py not present")

        text = path.read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"^\s*(from|import)\s+trajectory_intelligence\b", text, re.MULTILINE))

    def test_advisory_trajectory_evidence_never_calls_action_execution_verbs(self):

        path = ADVISORY_SYSTEM_PACKAGE / "trajectory_evidence.py"

        if not path.exists():
            self.skipTest("advisory_system/trajectory_evidence.py not present")

        text = path.read_text(encoding="utf-8")
        match = re.search(_FORBIDDEN_ACTION_CALLS, text)

        self.assertIsNone(match)


class EmergencyResponseNeverImportsTrajectoryIntelligenceTests(unittest.TestCase):

    # Phase 21's own chosen seam: emergency_response.engine.
    # EmergencyResponseIntelligenceEngine.compute() consults its new
    # trajectory_snapshot parameter ONLY through duck-typed attribute
    # access (.occupant(occupant_id).anomaly_severity) -- deliberately
    # untyped, so NO import of trajectory_intelligence was ever needed
    # in emergency_response/, and its own pre-existing allow-list
    # (tests/test_emergency_response_architecture_guards.py) is
    # correctly left unchanged by this milestone.

    def test_emergency_response_engine_imports_no_trajectory_intelligence_module(self):

        text = (EMERGENCY_RESPONSE_PACKAGE / "engine.py").read_text(encoding="utf-8")
        match = re.search(r"^\s*(from|import)\s+trajectory_intelligence\b", text, re.MULTILINE)

        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
