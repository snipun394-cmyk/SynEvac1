import pathlib
import re
import unittest


# =====================================================
# Fire Water Supply & Suppression Infrastructure milestone, Phase 19 --
# mechanical architecture-boundary guards, mirroring tests/
# test_fire_safety_asset_architecture_guards.py's own established
# pattern.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

PUMP_ASSET_MODEL = REPO_ROOT / "models" / "pump_asset.py"
FIRE_PUMP_MODEL = REPO_ROOT / "models" / "fire_pump.py"
JOCKEY_PUMP_MODEL = REPO_ROOT / "models" / "jockey_pump.py"
FIRE_WATER_TANK_MODEL = REPO_ROOT / "models" / "fire_water_tank.py"
FIRE_SERVICE_INLET_MODEL = REPO_ROOT / "models" / "fire_service_inlet.py"
FIRE_WATER_SYSTEM_MODEL = REPO_ROOT / "models" / "fire_water_system.py"
FIRE_WATER_MANAGER_PACKAGE = REPO_ROOT / "fire_water_manager"

NEW_MODEL_FILES = (
    PUMP_ASSET_MODEL, FIRE_PUMP_MODEL, JOCKEY_PUMP_MODEL,
    FIRE_WATER_TANK_MODEL, FIRE_SERVICE_INLET_MODEL, FIRE_WATER_SYSTEM_MODEL,
)

HAZARD_PHYSICS_PACKAGES = ("hazard", "hazard_evolution", "fire_growth", "smoke_propagation")

AI_PACKAGES = (
    "ai_decision", "ai_registry", "ai_inference", "ai_training", "ai_explainability",
    "advisory_system", "rl_training",
)


def _text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _forbidden_import(package_name: str) -> str:
    return rf"^\s*(from|import)\s+{re.escape(package_name)}\b"


class CannotModifyHazardFireGrowthSmokeTests(unittest.TestCase):

    def test_new_models_never_import_hazard_physics_packages(self):

        for path in NEW_MODEL_FILES:
            text = _text(path)
            for package_name in HAZARD_PHYSICS_PACKAGES:
                self.assertIsNone(
                    re.search(_forbidden_import(package_name), text, re.MULTILINE),
                    f"{path.name} imports {package_name} -- fire-water infrastructure must never "
                    f"directly modify hazard/fire-growth/smoke simulation state.",
                )

    def test_fire_water_manager_never_imports_hazard_physics_packages(self):

        for path in sorted(FIRE_WATER_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            for package_name in HAZARD_PHYSICS_PACKAGES:
                self.assertIsNone(
                    re.search(_forbidden_import(package_name), text, re.MULTILINE),
                    f"fire_water_manager/{path.name} imports {package_name}.",
                )

    def test_hazard_physics_packages_never_reference_fire_water_assets(self):

        asset_names = ("FireWaterTank", "FirePump", "JockeyPump", "FireServiceInlet", "PumpAsset")

        for package_name in HAZARD_PHYSICS_PACKAGES:

            package_dir = REPO_ROOT / package_name
            if not package_dir.is_dir():
                continue

            for path in sorted(package_dir.glob("*.py")):
                text = _text(path)
                for name in asset_names:
                    self.assertNotIn(
                        name, text,
                        f"{package_name}/{path.name} references {name} -- hazard/fire-growth/smoke "
                        f"simulation must never be told to react to fire-water infrastructure state.",
                    )


class CannotAutomaticallyExecuteEvacuationOrVoiceTests(unittest.TestCase):

    def test_new_models_never_import_decision_policy(self):

        for path in NEW_MODEL_FILES:
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("decision_policy"), text, re.MULTILINE))

    def test_new_models_never_import_voice_evacuation(self):

        for path in NEW_MODEL_FILES:
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("voice_evacuation"), text, re.MULTILINE))

    def test_fire_water_manager_never_imports_decision_policy_or_voice_evacuation(self):

        for path in sorted(FIRE_WATER_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("decision_policy"), text, re.MULTILINE))
            self.assertIsNone(re.search(_forbidden_import("voice_evacuation"), text, re.MULTILINE))

    def test_new_models_never_import_evacuation_engines(self):

        forbidden_engines = (
            "evacuation_recommendation", "evacuation_guidance", "emergency_response",
            "dynamic_signage", "trajectory_intelligence", "evacuation_progress",
        )

        for path in NEW_MODEL_FILES:
            text = _text(path)
            for package_name in forbidden_engines:
                self.assertIsNone(re.search(_forbidden_import(package_name), text, re.MULTILINE))


class NoFACPOrBuildingControlAuthorityTests(unittest.TestCase):

    def test_new_models_never_import_facp(self):

        for path in NEW_MODEL_FILES:
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("facp"), text, re.MULTILINE))

    def test_fire_water_manager_never_imports_facp(self):

        for path in sorted(FIRE_WATER_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("facp"), text, re.MULTILINE))

    def test_new_models_never_import_building_control(self):

        for path in NEW_MODEL_FILES:
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("building_control"), text, re.MULTILINE))

    def test_fire_water_manager_never_imports_building_control(self):

        for path in sorted(FIRE_WATER_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("building_control"), text, re.MULTILINE))

    def test_facp_and_building_control_never_reference_new_asset_types(self):

        asset_names = ("FireWaterTank", "FirePump", "JockeyPump", "FireServiceInlet")

        for package_name in ("facp", "building_control"):

            package_dir = REPO_ROOT / package_name

            for path in sorted(package_dir.glob("*.py")):
                text = _text(path)
                for name in asset_names:
                    self.assertNotIn(
                        name, text,
                        f"{package_name}/{path.name} references {name} -- no supervisory-device "
                        f"abstraction exists for these assets in FACP, and they are not remotely-"
                        f"commanded BuildingControl actions (Phase 16).",
                    )

    def test_deluge_remains_unrelated_to_fire_water_infrastructure(self):

        from building_control.types import ControlSystemType

        self.assertIn("DELUGE", ControlSystemType.__members__)
        self.assertNotIn("FIRE_PUMP", ControlSystemType.__members__)
        self.assertNotIn("JOCKEY_PUMP", ControlSystemType.__members__)


class NoAIOrRLAuthorityAddedTests(unittest.TestCase):

    def test_ai_packages_never_import_fire_water_manager(self):

        for package_name in AI_PACKAGES:

            package_dir = REPO_ROOT / package_name
            if not package_dir.is_dir():
                continue

            for path in sorted(package_dir.glob("*.py")):
                text = _text(path)
                self.assertIsNone(
                    re.search(_forbidden_import("fire_water_manager"), text, re.MULTILINE),
                    f"{package_name}/{path.name} imports fire_water_manager -- AI/RL must gain no "
                    f"new authority over fire-water infrastructure.",
                )

    def test_new_models_never_import_ai_or_rl_packages(self):

        for path in NEW_MODEL_FILES:
            text = _text(path)
            for package_name in AI_PACKAGES:
                self.assertIsNone(re.search(_forbidden_import(package_name), text, re.MULTILINE))


class NoHydraulicFabricationOrProtocolCodeTests(unittest.TestCase):

    _FORBIDDEN_PROTOCOL_IMPORTS = (
        r"^\s*(from|import)\s+(pymodbus|modbus|bacpypes|bacnet|paho|mqtt|opcua|freeopcua|pyserial|serial|socket)\b"
    )

    def test_new_models_have_no_protocol_imports(self):

        for path in NEW_MODEL_FILES:
            text = _text(path)
            self.assertIsNone(re.search(self._FORBIDDEN_PROTOCOL_IMPORTS, text, re.MULTILINE | re.IGNORECASE))

    def test_fire_water_manager_has_no_protocol_imports(self):

        for path in sorted(FIRE_WATER_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(re.search(self._FORBIDDEN_PROTOCOL_IMPORTS, text, re.MULTILINE | re.IGNORECASE))

    def test_no_hydraulic_calculation_vocabulary_anywhere_in_new_code(self):

        # NOTE: "pump curve"/"NPSH" are deliberately NOT in this list --
        # models/pump_asset.py's own docstring legitimately says "No
        # pump curve, head, NPSH... calculation exists anywhere in this
        # class", an explicit disclaimer, not a fabricated calculation.
        # A bare substring scan cannot distinguish "we do this" from
        # "we explicitly do NOT do this", so those two terms are
        # excluded here the same way a prior milestone excluded a
        # similarly-shaped false positive (`.reset(` on an unrelated
        # Gym API) rather than weakening what the check verifies.
        forbidden_terms = (
            "hazen", "darcy", "k-factor", "kfactor", "cfd",
            "flow rate", "pressure loss", "water hammer",
        )

        all_new_files = list(NEW_MODEL_FILES) + list(FIRE_WATER_MANAGER_PACKAGE.glob("*.py"))

        for path in all_new_files:
            text = _text(path).lower()
            for term in forbidden_terms:
                self.assertNotIn(term, text, f"{path.name} mentions {term!r}.")

    def test_fire_water_system_status_vocabulary_never_claims_hydraulic_adequacy(self):

        # Mechanical proof of Phase 11's own "must NOT support 'required
        # hydraulic demand is satisfied'" instruction -- the status
        # vocabulary itself contains no such claim.
        from fire_water_manager.snapshot import FireWaterSystemStatus

        for value in FireWaterSystemStatus.ALL:
            self.assertNotIn("PRESSURE", value)
            self.assertNotIn("FLOW", value)
            self.assertNotIn("ADEQUATE", value)
            self.assertNotIn("CONFIRMED", value)


if __name__ == "__main__":
    unittest.main()
