import dataclasses
import pathlib
import re
import unittest

from facp.engine import SimulatedFACP


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# =====================================================
# Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
# Phase 15 -- mechanical architecture guards.
# =====================================================


class FACPRemainsLogicalTests(unittest.TestCase):

    def test_simulated_facp_has_no_spatial_fields(self):

        field_names = {f.name for f in dataclasses.fields(SimulatedFACP)} if dataclasses.is_dataclass(SimulatedFACP) else set()

        # SimulatedFACP is a plain class, not a dataclass/BaseObject --
        # confirm it carries no position/floor_id/zone_ids attribute at
        # all, on either the class or a fresh instance.
        facp = SimulatedFACP()

        for spatial_attr in ("position", "floor_id", "zone_ids", "x", "y"):
            self.assertFalse(hasattr(facp, spatial_attr), f"SimulatedFACP unexpectedly has {spatial_attr!r}")

    def test_facp_is_not_a_baseobject_subclass(self):

        from models.base_object import BaseObject

        self.assertFalse(issubclass(SimulatedFACP, BaseObject))

    def test_no_facp_graphics_item_exists(self):

        for path in (REPO_ROOT / "designer" / "items").glob("*.py"):
            self.assertNotIn("facp", path.name.lower())

    def test_no_facp_toolbar_action_exists(self):

        text = (REPO_ROOT / "designer" / "widgets" / "toolbar.py").read_text(encoding="utf-8")
        self.assertNotIn("facp", text.lower())

    def test_no_facp_click_to_place_branch_in_graphics_scene(self):

        text = (REPO_ROOT / "designer" / "scene" / "graphics_scene.py").read_text(encoding="utf-8")
        self.assertNotRegex(text, r'current_tool\s*==\s*["\']facp["\']')

    def test_floor_has_no_facp_list(self):

        from models.floor import Floor

        field_names = {f.name for f in dataclasses.fields(Floor)}
        self.assertNotIn("facps", field_names)
        self.assertNotIn("facp_panels", field_names)


class FACPNeverAutoDispatchesTests(unittest.TestCase):

    _FORBIDDEN_ACTION_CALLS = r"\.broadcast\(|\.announce\(|\.execute_control\(|\.confirm\(|\.dispatch\("

    def test_facp_package_never_calls_execution_verbs(self):

        for path in (REPO_ROOT / "facp").glob("*.py"):

            text = path.read_text(encoding="utf-8")
            match = re.search(self._FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0) if match else ''!r} -- "
                f"FACP must never automatically broadcast or execute a building control.",
            )

    def test_facp_gateway_never_calls_execution_verbs(self):

        text = (REPO_ROOT / "live_system" / "facp_gateway.py").read_text(encoding="utf-8")
        match = re.search(self._FORBIDDEN_ACTION_CALLS, text)

        self.assertIsNone(match)

    def test_facp_package_never_imports_voice_or_building_control(self):

        for path in (REPO_ROOT / "facp").glob("*.py"):

            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*(from|import)\s+(voice_evacuation|building_control)\b", text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"FACP must remain separate from Voice Evacuation/Building Control.",
            )

    def test_facp_gateway_never_imports_voice_or_building_control(self):

        text = (REPO_ROOT / "live_system" / "facp_gateway.py").read_text(encoding="utf-8")
        match = re.search(r"^\s*(from|import)\s+(voice_evacuation|building_control)\b", text, re.MULTILINE)

        self.assertIsNone(match)


class AICannotControlFACPTests(unittest.TestCase):

    _FORBIDDEN = r"^\s*(from|import)\s+facp\b"

    def _assert_clean(self, path: pathlib.Path):

        text = path.read_text(encoding="utf-8")
        match = re.search(self._FORBIDDEN, text, re.MULTILINE)

        self.assertIsNone(
            match,
            f"{path.relative_to(REPO_ROOT)} imports facp directly -- AI/Advisory/Decision Policy "
            f"must never control FACP; only live_system.facp_gateway (composition root) may.",
        )

    def test_ai_inference_never_imports_facp(self):

        package = REPO_ROOT / "ai_inference"
        if package.is_dir():
            for path in package.glob("*.py"):
                self._assert_clean(path)

    def test_ai_registry_never_imports_facp(self):

        package = REPO_ROOT / "ai_registry"
        if package.is_dir():
            for path in package.glob("*.py"):
                self._assert_clean(path)

    def test_decision_policy_never_imports_facp(self):

        package = REPO_ROOT / "decision_policy"
        if package.is_dir():
            for path in package.glob("*.py"):
                self._assert_clean(path)

    def test_advisory_system_never_imports_facp(self):

        package = REPO_ROOT / "advisory_system"
        if package.is_dir():
            for path in package.glob("*.py"):
                self._assert_clean(path)


class ZoneAssignmentIndependentOfCameraTests(unittest.TestCase):

    # Scoped to the NEW zone-assignment code specifically (via
    # inspect.getsource() on each new function/method) rather than the
    # whole file -- property_panel.py/graphics_scene.py both already
    # contain PRE-EXISTING, unrelated Camera coverage-overlay code
    # (_update_camera_visibility_stats, the "Coverage" toolbar toggle)
    # that legitimately references visibility/coverage_polygon and is
    # entirely out of this milestone's scope.

    def test_speaker_zone_checklist_never_references_camera_coverage(self):

        from designer.widgets.property_panel import PropertyPanel

        for method_name in ("_populate_zone_checklist", "_checked_zone_ids", "update_speaker_zones"):

            source = inspect_getsource(PropertyPanel, method_name)
            self.assertNotIn("coverage_polygon", source)
            self.assertNotIn("VisibilityEngine", source)
            self.assertNotIn("visibility", source.lower())

    def test_detector_zone_combo_never_references_camera_coverage(self):

        from designer.widgets.property_panel import PropertyPanel

        for method_name in ("update_smoke_detector_zone", "update_heat_detector_zone", "_populate_zone_combo"):

            source = inspect_getsource(PropertyPanel, method_name)
            self.assertNotIn("coverage_polygon", source)
            self.assertNotIn("VisibilityEngine", source)

    def test_graphics_scene_autoassignment_helper_never_imports_visibility(self):

        from designer.scene.graphics_scene import GraphicsScene

        source = inspect_getsource(GraphicsScene, "_find_unambiguous_zone_at")
        self.assertNotIn("visibility", source.lower())
        self.assertNotIn("Camera", source)


def inspect_getsource(cls, method_name):

    import inspect

    return inspect.getsource(getattr(cls, method_name))


class NoHardwareOrNetworkProtocolTests(unittest.TestCase):

    _FORBIDDEN_IMPORTS = r"^\s*(from|import)\s+(socket|serial|requests|urllib)\b"

    def test_facp_gateway_has_no_network_or_hardware_import(self):

        text = (REPO_ROOT / "live_system" / "facp_gateway.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(self._FORBIDDEN_IMPORTS, text, re.MULTILINE))

    def test_property_panel_has_no_network_or_hardware_import(self):

        text = (REPO_ROOT / "designer" / "widgets" / "property_panel.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(self._FORBIDDEN_IMPORTS, text, re.MULTILINE))


if __name__ == "__main__":
    unittest.main()
