import pathlib
import re
import unittest


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 30 -- mechanical
# dependency-direction guards, mirroring tests/
# test_evacuation_guidance_architecture_guards.py exactly.
#
# dynamic_signage/ must NOT import YOLO/RTSP/AI training/RL/decision_
# policy/Command Center UI/voice output providers/building-control
# providers/FACP mutation/hardware protocols. It may depend on
# evacuation_guidance (its own input), models/building geometry, and
# itself -- kept narrow.
#
# The Planner (dynamic_signage/planner.py) must never import
# DynamicSignageProvider -- controller/provider concerns are strictly
# downstream of planning.
#
# AI/Advisory must never import DynamicSignageController/
# SimulationDynamicSignageProvider directly -- the ONLY path to
# dispatch is command_center.live_operator_action_gateway.
#
# Building Control must never gain a second, competing execution path
# for the same structured signage instruction (Phase 26).
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DYNAMIC_SIGNAGE_PACKAGE = REPO_ROOT / "dynamic_signage"

_FORBIDDEN_FOR_DYNAMIC_SIGNAGE = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|"
    r"advisory_system|command_center|"
    r"voice_evacuation|speaker_manager\.manager|"
    r"building_control|"
    r"facp|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"live_camera_pipeline\.rtsp_frame_source|live_camera_pipeline\.rtsp_backend|"
    r"cv2|torch|ultralytics|onvif|socket\b|serial\b|"
    r"ground_truth|human_decision_engine|simulator|behaviour_profile_resolver|"
    r"crowd_intelligence\.engine|evacuation_progress\.engine|trajectory_intelligence\.engine|"
    r"emergency_response\.engine|evacuation_recommendation\.engine|evacuation_recommendation\.ranking"
    r")\b"
)

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"
    r"\.broadcast\(|\.announce\(|\.send\(|"
    r"\.execute_control\(|\.confirm\(|\.dispatch\("
)


class DynamicSignageArchitectureGuardTests(unittest.TestCase):

    def test_dynamic_signage_never_imports_forbidden_modules(self):

        for path in sorted(DYNAMIC_SIGNAGE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_DYNAMIC_SIGNAGE, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"dynamic_signage/ must never depend on AI/Advisory/Command Center/voice output providers/"
                f"building control/FACP execution, sibling live-intelligence engines, or hardware protocols "
                f"(Phase 30).",
            )

    def test_dynamic_signage_never_calls_action_execution_verbs(self):

        for path in sorted(DYNAMIC_SIGNAGE_PACKAGE.glob("*.py")):

            if path.name in ("controller.py", "provider.py"):
                # These two files ARE the execution/dispatch seam itself
                # (Phase 13/14) -- the guard below applies to the
                # PLANNER only.
                continue

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"dynamic_signage/'s planning code plans, it never executes, sends, or broadcasts.",
            )

    def test_dynamic_signage_not_nested_inside_another_package(self):

        self.assertTrue(DYNAMIC_SIGNAGE_PACKAGE.is_dir())
        self.assertEqual(DYNAMIC_SIGNAGE_PACKAGE.parent, REPO_ROOT)

    def test_dynamic_signage_only_depends_on_allowed_project_packages(self):

        allowed_prefixes = (
            "evacuation_guidance", "models", "dynamic_signage",
        )

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(DYNAMIC_SIGNAGE_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, not in dynamic_signage/'s own "
                    f"documented allow-list (Phase 30).",
                )

    def test_planner_never_imports_provider(self):

        text = (DYNAMIC_SIGNAGE_PACKAGE / "planner.py").read_text(encoding="utf-8")

        self.assertIsNone(
            re.search(r"^\s*(from|import)\s+dynamic_signage\.provider\b", text, re.MULTILINE),
            "dynamic_signage/planner.py must never import dynamic_signage/provider.py -- "
            "the Planner has zero dispatch authority (Phase 30).",
        )

    def test_planner_never_imports_controller(self):

        text = (DYNAMIC_SIGNAGE_PACKAGE / "planner.py").read_text(encoding="utf-8")

        self.assertIsNone(
            re.search(r"^\s*(from|import)\s+dynamic_signage\.controller\b", text, re.MULTILINE),
        )


class AIAdvisoryCannotDispatchSignageTests(unittest.TestCase):

    _FORBIDDEN = (
        r"^\s*(from|import)\s+"
        r"(dynamic_signage\.controller|dynamic_signage\.provider|"
        r"command_center\.live_operator_action_gateway)\b"
    )

    def _assert_file_clean(self, path: pathlib.Path):

        text = path.read_text(encoding="utf-8")

        self.assertIsNone(
            re.search(self._FORBIDDEN, text, re.MULTILINE),
            f"{path.relative_to(REPO_ROOT)} imports a signage dispatch module or the operator action "
            f"gateway directly -- AI/Advisory generation must never be able to reach signage execution, "
            f"only an explicit operator action may (Phase 30).",
        )

    def test_advisory_system_cannot_dispatch_signage(self):

        for path in (REPO_ROOT / "advisory_system").glob("*.py"):
            self._assert_file_clean(path)

    def test_live_ai_gateway_cannot_dispatch_signage(self):

        self._assert_file_clean(REPO_ROOT / "live_system" / "live_ai_gateway.py")

    def test_live_advisory_gateway_cannot_dispatch_signage(self):

        self._assert_file_clean(REPO_ROOT / "live_system" / "live_advisory_gateway.py")

    def test_decision_policy_cannot_dispatch_signage(self):

        for path in (REPO_ROOT / "decision_policy").glob("*.py"):
            self._assert_file_clean(path)

    def test_orchestrator_cannot_dispatch_signage(self):

        self._assert_file_clean(REPO_ROOT / "live_system" / "orchestrator.py")

    def test_facp_cannot_dispatch_signage(self):

        facp_package = REPO_ROOT / "facp"

        for path in facp_package.glob("*.py"):
            self._assert_file_clean(path)


class BuildingControlDuplicateDispatchGuardTests(unittest.TestCase):

    # Phase 26 -- there is historical evidence Advisory generates an
    # "Update Dynamic Exit Signs" BuildingRecommendation. This must
    # NEVER become a second, competing execution path alongside
    # DynamicSignageController -- building_control/advisory_adapter.py
    # must continue to leave it untranslated.

    def test_building_control_never_translates_dynamic_exit_signs(self):

        from building_control.advisory_adapter import _PATTERNS

        for target_type, prefix, system_type, action in _PATTERNS:
            self.assertNotIn("Exit Sign", prefix)
            self.assertNotIn("Dynamic Sign", prefix)

    def test_building_control_never_imports_dynamic_signage(self):

        building_control_package = REPO_ROOT / "building_control"

        for path in building_control_package.glob("*.py"):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+dynamic_signage\b", text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports dynamic_signage -- Building Control must never "
                f"gain a second, competing dispatch path for structured sign instructions (Phase 26).",
            )

    def test_dynamic_signage_controller_never_imports_building_control(self):

        text = (DYNAMIC_SIGNAGE_PACKAGE / "controller.py").read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"^\s*(from|import)\s+building_control\b", text, re.MULTILINE))


if __name__ == "__main__":
    unittest.main()
