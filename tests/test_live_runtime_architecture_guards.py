import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# =====================================================
# Production Live Runtime Composition Root milestone, Phase 9 --
# mechanically verify every execution path still reduces to exactly:
#
#   Human Operator -> LiveOperatorActionGateway -> Existing Controller -> Provider
#
# Same regex-scan-the-source-files convention every other package
# boundary guard in this codebase already uses (tests/
# test_live_operator_action_routing.py::AIAuthorityGuardTests is this
# file's own direct precedent, re-scoped here to additionally cover the
# new live_runtime/ composition layer -- introducing live_runtime must
# never widen who can reach real execution).
# =====================================================


_EXECUTION_CAPABLE_OR_GATEWAY = (
    r"^\s*(from|import)\s+"
    r"(voice_evacuation|speaker_manager|building_control\.controller|"
    r"building_control\.providers|command_center\.live_operator_action_gateway|"
    r"live_runtime)\b"
)


def _assert_package_never_imports_execution_or_gateway(test_case, package_dir_name):

    package_dir = REPO_ROOT / package_dir_name

    if not package_dir.is_dir():
        test_case.skipTest(f"{package_dir_name}/ does not exist")

    for path in package_dir.glob("*.py"):

        text = path.read_text(encoding="utf-8")

        test_case.assertIsNone(
            re.search(_EXECUTION_CAPABLE_OR_GATEWAY, text, re.MULTILINE),
            f"{path.relative_to(REPO_ROOT)} imports an execution-capable module, the operator "
            f"action gateway, or live_runtime directly -- only an explicit operator action, "
            f"routed through LiveOperatorActionGateway, may ever reach real execution.",
        )


class AICannotReachExecutionTests(unittest.TestCase):

    def test_advisory_system_never_imports_execution_gateway_or_live_runtime(self):
        _assert_package_never_imports_execution_or_gateway(self, "advisory_system")

    def test_decision_policy_never_imports_execution_gateway_or_live_runtime(self):
        _assert_package_never_imports_execution_or_gateway(self, "decision_policy")

    def test_ai_registry_never_imports_execution_gateway_or_live_runtime(self):
        _assert_package_never_imports_execution_or_gateway(self, "ai_registry")

    def test_ai_inference_never_imports_execution_gateway_or_live_runtime(self):
        _assert_package_never_imports_execution_or_gateway(self, "ai_inference")

    def test_live_ai_gateway_file_never_imports_execution_gateway_or_live_runtime(self):

        path = REPO_ROOT / "live_system" / "live_ai_gateway.py"
        text = path.read_text(encoding="utf-8")

        self.assertIsNone(re.search(_EXECUTION_CAPABLE_OR_GATEWAY, text, re.MULTILINE))

    def test_live_advisory_gateway_file_never_imports_execution_gateway_or_live_runtime(self):

        path = REPO_ROOT / "live_system" / "live_advisory_gateway.py"
        text = path.read_text(encoding="utf-8")

        self.assertIsNone(re.search(_EXECUTION_CAPABLE_OR_GATEWAY, text, re.MULTILINE))


class FACPCannotReachExecutionTests(unittest.TestCase):

    # New for this milestone -- FACP was never previously checked
    # against the operator-action-gateway/live_runtime vocabulary
    # (those modules did not exist yet when facp/ was built). FACP
    # remains a read-only aggregation/coordination layer; it must never
    # gain a path to voice/control execution now that one exists
    # elsewhere in the codebase.

    def test_facp_never_imports_execution_gateway_or_live_runtime(self):
        _assert_package_never_imports_execution_or_gateway(self, "facp")


class LiveOrchestratorCannotDirectlyCallControllersTests(unittest.TestCase):

    # LiveOrchestrator (live_system/orchestrator.py) is the class that
    # runs the AI inference / advisory generation cycle -- it must never
    # itself hold a VoiceEvacuationController/BuildingControlController
    # reference. tests/test_live_system.py::
    # LiveSystemPackageDependencyDirectionTests already forbids
    # `live_system` from importing voice_evacuation/building_control.
    # controller/.providers package-wide (pre-existing, unmodified);
    # this test re-states that guarantee explicitly for orchestrator.py
    # by name, plus checks the new live_runtime vocabulary too.

    def test_orchestrator_never_imports_execution_gateway_or_controllers(self):

        path = REPO_ROOT / "live_system" / "orchestrator.py"
        text = path.read_text(encoding="utf-8")

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(voice_evacuation|speaker_manager|building_control\.controller|"
            r"building_control\.providers|command_center\.live_operator_action_gateway|"
            r"live_runtime)\b"
        )

        self.assertIsNone(re.search(forbidden, text, re.MULTILINE))

    def test_live_runtime_package_never_reimplements_orchestrator_sequencing(self):

        # LiveRuntime.run_cycle() must remain a pure forward onto
        # LiveOrchestrator.run_cycle() -- never a second, parallel
        # sequencing of perception/AI/advisory stages (Phase 2's own
        # "composition and lifecycle management only" requirement).

        path = REPO_ROOT / "live_runtime" / "runtime.py"
        text = path.read_text(encoding="utf-8")

        self.assertIn("def run_cycle(self, time: float):", text)
        self.assertIn("return self.orchestrator.run_cycle(time)", text)


class CommandCenterPanelsStayCleanTests(unittest.TestCase):

    # Pre-existing tests/test_live_command_center.py::
    # CommandCenterLiveIntegrationGuardTests already proves every Live
    # Command Center file never imports voice_evacuation.controller/
    # voice_evacuation.provider/speaker_manager/building_control.
    # controller/building_control.providers directly -- re-verified
    # here, unmodified, plus a check that no panel needs to import
    # live_runtime either (it only ever needs the already-constructed
    # gateway object handed to it, never the composition layer itself).

    _PANEL_FILES = (
        ("command_center", "building_controls_panel.py"),
        ("command_center", "recommendation_center.py"),
        ("command_center", "dashboard.py"),
        ("command_center", "main_window.py"),
        ("command_center", "data_source.py"),
    )

    def test_panels_never_import_live_runtime(self):

        for package, filename in self._PANEL_FILES:

            text = (REPO_ROOT / package / filename).read_text(encoding="utf-8")

            self.assertNotIn(
                "live_runtime", text,
                f"{package}/{filename} imports live_runtime directly -- Command Center should "
                f"only ever be handed an already-constructed data source/gateway, never compose "
                f"the runtime itself.",
            )

    def test_panels_never_import_execution_capable_modules(self):

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(voice_evacuation\.controller|voice_evacuation\.provider|speaker_manager|"
            r"building_control\.controller|building_control\.providers)\b"
        )

        for package, filename in self._PANEL_FILES:

            text = (REPO_ROOT / package / filename).read_text(encoding="utf-8")

            self.assertIsNone(re.search(forbidden, text, re.MULTILINE))


class GatewayIsTheOnlyExecutionSeamTests(unittest.TestCase):

    def test_live_runtime_factory_never_imports_advisory_or_ai_internals(self):

        # The factory accepts live_ai_gateway/live_advisory_gateway as
        # opaque, already-constructed objects (Phase 6's own "do not let
        # Command Center construct AI models/AdvisoryOrchestrator"
        # requirement, applied equally to this factory) -- it must never
        # need to import advisory_system.orchestrator, ai_registry,
        # ai_inference, ai_training, or decision_policy itself to do so.

        text = (REPO_ROOT / "live_runtime" / "factory.py").read_text(encoding="utf-8")

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(advisory_system\.orchestrator|ai_registry|ai_inference|ai_training|decision_policy)\b"
        )

        self.assertIsNone(re.search(forbidden, text, re.MULTILINE))

    def test_live_runtime_container_holds_no_concrete_class_imports(self):

        # runtime.py is composition/lifecycle only (Phase 2) -- it
        # should not need to import any concrete collaborator class at
        # all, only stdlib typing.

        text = (REPO_ROOT / "live_runtime" / "runtime.py").read_text(encoding="utf-8")

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(camera_manager|sensor_manager|multi_camera_fusion|facp|voice_evacuation|"
            r"building_control|live_system|advisory_system|ai_registry)\b"
        )

        self.assertIsNone(re.search(forbidden, text, re.MULTILINE))


if __name__ == "__main__":
    unittest.main()
