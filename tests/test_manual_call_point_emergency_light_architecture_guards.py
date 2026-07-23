import pathlib
import re
import unittest


# =====================================================
# Manual Call Points & Emergency Lighting milestone, Step 6 -- mechanical
# architecture-boundary guards, mirroring the established pattern in
# tests/test_dynamic_signage_architecture_guards.py and
# tests/test_facp.py::ArchitectureGuardTests.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

MCP_MODEL = REPO_ROOT / "models" / "manual_call_point.py"
EMERGENCY_LIGHT_MODEL = REPO_ROOT / "models" / "emergency_light.py"
EMERGENCY_LIGHT_MANAGER_PACKAGE = REPO_ROOT / "emergency_light_manager"
FACP_PACKAGE = REPO_ROOT / "facp"
FACP_GATEWAY = REPO_ROOT / "live_system" / "facp_gateway.py"

AI_PACKAGES = (
    "ai_decision", "ai_registry", "ai_inference", "ai_training", "ai_explainability",
    "advisory_system", "rl_training",
)


def _text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


class ManualCallPointCannotInvokeExecutionControllersTests(unittest.TestCase):

    def test_manual_call_point_model_never_imports_voice_evacuation(self):

        text = _text(MCP_MODEL)
        self.assertIsNone(re.search(r"^\s*(from|import)\s+voice_evacuation\b", text, re.MULTILINE))

    def test_manual_call_point_model_never_imports_building_control(self):

        text = _text(MCP_MODEL)
        self.assertIsNone(re.search(r"^\s*(from|import)\s+building_control\b", text, re.MULTILINE))

    def test_manual_call_point_model_has_no_broadcast_or_control_execution_calls(self):

        text = _text(MCP_MODEL)
        self.assertIsNone(re.search(r"\.broadcast\(|\.execute_control\(|\.confirm\(|\.dispatch\(", text))

    def test_manual_call_point_model_only_mutates_its_own_intrinsic_state(self):

        # activate()/restore() are the entire mutation surface -- proves
        # by inspection that neither method reaches outside `self`.
        text = _text(MCP_MODEL)

        activate_body = re.search(r"def activate\(self\):\n(.*?)\n\s*# =+", text, re.DOTALL)
        restore_body = re.search(r"def restore\(self\):\n(.*?)\n\s*# =+", text, re.DOTALL)

        self.assertIsNotNone(activate_body)
        self.assertIsNotNone(restore_body)
        self.assertNotIn("VoiceEvacuationController", activate_body.group(1))
        self.assertNotIn("BuildingControlController", activate_body.group(1))
        self.assertNotIn("VoiceEvacuationController", restore_body.group(1))
        self.assertNotIn("BuildingControlController", restore_body.group(1))


class FACPCannotAutomaticallyDispatchTests(unittest.TestCase):

    def test_facp_package_never_imports_voice_evacuation(self):

        for path in sorted(FACP_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+voice_evacuation\b", text, re.MULTILINE),
                f"facp/{path.name} imports voice_evacuation -- FACP must never automatically "
                f"broadcast voice messages.",
            )

    def test_facp_package_never_imports_building_control(self):

        for path in sorted(FACP_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+building_control\b", text, re.MULTILINE),
                f"facp/{path.name} imports building_control -- FACP must never automatically "
                f"execute building controls.",
            )

    def test_facp_gateway_never_imports_voice_or_building_control(self):

        text = _text(FACP_GATEWAY)
        self.assertIsNone(re.search(r"^\s*(from|import)\s+voice_evacuation\b", text, re.MULTILINE))
        self.assertIsNone(re.search(r"^\s*(from|import)\s+building_control\b", text, re.MULTILINE))

    def test_ai_packages_never_import_facp(self):

        for package_name in AI_PACKAGES:

            package_dir = REPO_ROOT / package_name
            if not package_dir.is_dir():
                continue

            for path in sorted(package_dir.glob("*.py")):
                text = _text(path)
                self.assertIsNone(
                    re.search(r"^\s*(from|import)\s+facp\b", text, re.MULTILINE),
                    f"{package_name}/{path.name} imports facp -- AI must never activate, reset, "
                    f"acknowledge, or silence the FACP.",
                )

    # NOTE: a second check scanning AI packages for bare `.acknowledge(`/
    # `.silence(`/`.reset(`/`.manual_alarm(` calls was deliberately left
    # out -- `.reset()` in particular is a generic, widely-used method
    # name (e.g. rl_training/environment.py's own Gym environment API)
    # entirely unrelated to FACP, making a text-level call-site scan
    # unreliable. The import-absence check above already mechanically
    # proves the real guarantee: no AI package imports facp at all, so
    # none can call SimulatedFACP.acknowledge()/silence()/reset()/
    # manual_alarm() regardless of what unrelated methods of the same
    # name exist elsewhere in that package's own code.


class EmergencyLightCannotInfluenceRoutingOrSafetyTests(unittest.TestCase):

    def test_emergency_light_model_never_imports_decision_policy(self):

        text = _text(EMERGENCY_LIGHT_MODEL)
        self.assertIsNone(re.search(r"^\s*(from|import)\s+decision_policy\b", text, re.MULTILINE))

    def test_emergency_light_model_never_imports_pathfinding(self):

        text = _text(EMERGENCY_LIGHT_MODEL)
        self.assertIsNone(re.search(r"^\s*(from|import)\s+pathfinding\b", text, re.MULTILINE))

    def test_emergency_light_manager_never_imports_decision_policy_or_pathfinding(self):

        for path in sorted(EMERGENCY_LIGHT_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(re.search(r"^\s*(from|import)\s+decision_policy\b", text, re.MULTILINE))
            self.assertIsNone(re.search(r"^\s*(from|import)\s+pathfinding\b", text, re.MULTILINE))

    def test_pathfinding_and_decision_policy_never_reference_emergency_light(self):

        for package_dir in (REPO_ROOT / "pathfinding", REPO_ROOT / "decision_policy"):
            for path in sorted(package_dir.glob("*.py")):
                text = _text(path)
                self.assertNotIn("EmergencyLight", text, f"{path} references EmergencyLight.")
                self.assertNotIn("emergency_light", text, f"{path} references emergency_light.")

    def test_availability_enum_has_no_route_safety_concept(self):

        from models.emergency_light import EmergencyLightAvailability

        # AVAILABLE/UNAVAILABLE/FAULT only -- no SAFE/UNSAFE/ROUTE_OK
        # member of any kind.
        for value in EmergencyLightAvailability.ALL:
            self.assertNotIn("SAFE", value)
            self.assertNotIn("ROUTE", value)


class NoHardwareOrVendorProtocolCodeTests(unittest.TestCase):

    _FORBIDDEN_PROTOCOL_IMPORTS = (
        r"^\s*(from|import)\s+(pymodbus|modbus|bacpypes|bacnet|paho|mqtt|opcua|freeopcua|pyserial|serial|socket)\b"
    )

    def test_manual_call_point_and_emergency_light_models_have_no_protocol_imports(self):

        for path in (MCP_MODEL, EMERGENCY_LIGHT_MODEL):
            text = _text(path)
            self.assertIsNone(re.search(self._FORBIDDEN_PROTOCOL_IMPORTS, text, re.MULTILINE | re.IGNORECASE))

    def test_emergency_light_manager_has_no_protocol_imports(self):

        for path in sorted(EMERGENCY_LIGHT_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(re.search(self._FORBIDDEN_PROTOCOL_IMPORTS, text, re.MULTILINE | re.IGNORECASE))

    def test_facp_gateway_has_no_protocol_imports(self):

        text = _text(FACP_GATEWAY)
        self.assertIsNone(re.search(self._FORBIDDEN_PROTOCOL_IMPORTS, text, re.MULTILINE | re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
