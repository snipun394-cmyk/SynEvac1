import pathlib
import re
import unittest


# =====================================================
# Live Crowd Intelligence -> Operational Advisory Integration milestone,
# Phase 19 -- mechanical dependency-direction guards.
#
#   Decision Policy
#           |
#           v
#       Advisory   <---   Crowd Intelligence
#
# NEVER Crowd Intelligence -> Decision Policy. Crowd-advisory
# integration code (advisory_system.crowd_evidence, live_system.
# live_advisory_gateway's new crowd-facing function) must never import
# voice_evacuation/speaker execution/building_control/hardware
# protocols/RTSP/YOLO/RL -- it contributes evidence to Advisory only.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DECISION_POLICY_PACKAGE = REPO_ROOT / "decision_policy"
ADVISORY_SYSTEM_PACKAGE = REPO_ROOT / "advisory_system"

CROWD_ADVISORY_FILES = (
    ADVISORY_SYSTEM_PACKAGE / "crowd_evidence.py",
    ADVISORY_SYSTEM_PACKAGE / "advisory_engine.py",
    ADVISORY_SYSTEM_PACKAGE / "recommendation_models.py",
    ADVISORY_SYSTEM_PACKAGE / "confidence_engine.py",
)

_FORBIDDEN_FOR_CROWD_ADVISORY = (
    r"^\s*(from|import)\s+("
    r"voice_evacuation|speaker_manager|"
    r"building_control|"
    r"reinforcement_learning|rl_training|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"cv2|torch|ultralytics|onvif"
    r")\b"
)


class CrowdAdvisoryNeverImportsExecutionCapableModulesTests(unittest.TestCase):

    def test_advisory_system_crowd_files_never_import_forbidden_modules(self):

        for path in CROWD_ADVISORY_FILES:

            self.assertTrue(path.is_file(), f"{path} does not exist")

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_CROWD_ADVISORY, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"crowd-advisory integration must never import voice execution, building control "
                f"execution, hardware protocols, RTSP, YOLO, or RL (Phase 19).",
            )

    def test_advisory_system_crowd_evidence_imports_no_crowd_intelligence_module(self):

        # advisory_system.crowd_evidence is deliberately free of any
        # crowd_intelligence import (mirrors advisory_system.ai_evidence's
        # own "advisory_system gains no new package dependency"
        # discipline) -- the reduction from a real CrowdIntelligenceSnapshot
        # happens in live_system.live_advisory_gateway instead.
        text = (ADVISORY_SYSTEM_PACKAGE / "crowd_evidence.py").read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"^\s*(from|import)\s+crowd_intelligence\b", text, re.MULTILINE))

    def test_no_file_in_advisory_system_calls_an_execution_verb(self):

        forbidden_calls = r"\.broadcast\(|\.announce\(|\.execute_control\(|\.confirm\(|\.acknowledge\(|\.silence\(|\.reset\("

        for path in sorted(ADVISORY_SYSTEM_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(forbidden_calls, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"advisory_system reports/recommends, it never executes an action.",
            )


class DecisionPolicyNeverImportsCrowdIntelligenceTests(unittest.TestCase):

    def test_decision_policy_never_imports_crowd_intelligence(self):

        # The one-way dependency direction this milestone must preserve:
        # Decision Policy -> Advisory, Crowd Intelligence -> Advisory,
        # NEVER Crowd Intelligence -> Decision Policy. decision_policy/
        # itself must remain completely unaware crowd_intelligence/
        # exists.

        for path in sorted(DECISION_POLICY_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+crowd_intelligence\b", text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports crowd_intelligence -- decision_policy must never "
                f"depend on crowd_intelligence (Phase 19's own explicit dependency-direction requirement).",
            )

    def test_decision_policy_never_imports_advisory_system_either(self):

        # The existing, pre-milestone direction (Decision Policy computed
        # first, Advisory reads it) -- re-verified here alongside the new
        # crowd_intelligence guard so both halves of the one-way
        # dependency arrow are checked in the same place.

        for path in sorted(DECISION_POLICY_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+advisory_system\b", text, re.MULTILINE)

            self.assertIsNone(match, f"{path.relative_to(REPO_ROOT)} imports advisory_system")


if __name__ == "__main__":
    unittest.main()
