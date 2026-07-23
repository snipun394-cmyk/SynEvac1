import pathlib
import re
import unittest


# =====================================================
# Fire Suppression & Water-Based Safety Asset Digital Twin milestone,
# Phase 18 -- mechanical architecture-boundary guards, mirroring
# tests/test_manual_call_point_emergency_light_architecture_guards.py's
# own established pattern.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SPRINKLER_MODEL = REPO_ROOT / "models" / "sprinkler.py"
FIRE_SAFETY_ASSET_MODEL = REPO_ROOT / "models" / "fire_safety_asset.py"
FIRE_EXTINGUISHER_MODEL = REPO_ROOT / "models" / "fire_extinguisher.py"
FIRE_HYDRANT_MODEL = REPO_ROOT / "models" / "fire_hydrant.py"
HOSE_REEL_MODEL = REPO_ROOT / "models" / "hose_reel.py"
FIRE_SAFETY_MANAGER_PACKAGE = REPO_ROOT / "fire_safety_manager"

HAZARD_PHYSICS_PACKAGES = ("hazard", "hazard_evolution", "fire_growth", "smoke_propagation")

AI_PACKAGES = (
    "ai_decision", "ai_registry", "ai_inference", "ai_training", "ai_explainability",
    "advisory_system", "rl_training",
)


def _text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _forbidden_import(package_name: str) -> str:
    return rf"^\s*(from|import)\s+{re.escape(package_name)}\b"


class SprinklerCannotModifyHazardOrFireGrowthTests(unittest.TestCase):

    def test_sprinkler_model_never_imports_hazard_physics_packages(self):

        text = _text(SPRINKLER_MODEL)

        for package_name in HAZARD_PHYSICS_PACKAGES:
            self.assertIsNone(
                re.search(_forbidden_import(package_name), text, re.MULTILINE),
                f"models/sprinkler.py imports {package_name} -- a Sprinkler must never directly "
                f"modify hazard/fire-growth simulation state.",
            )

    def test_fire_safety_manager_never_imports_hazard_physics_packages(self):

        for path in sorted(FIRE_SAFETY_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            for package_name in HAZARD_PHYSICS_PACKAGES:
                self.assertIsNone(
                    re.search(_forbidden_import(package_name), text, re.MULTILINE),
                    f"fire_safety_manager/{path.name} imports {package_name}.",
                )

    def test_hazard_physics_packages_never_reference_sprinkler(self):

        for package_name in HAZARD_PHYSICS_PACKAGES:

            package_dir = REPO_ROOT / package_name
            if not package_dir.is_dir():
                continue

            for path in sorted(package_dir.glob("*.py")):
                text = _text(path)
                self.assertNotIn(
                    "Sprinkler", text,
                    f"{package_name}/{path.name} references Sprinkler -- hazard/fire-growth "
                    f"simulation must never be told to react to sprinkler activation (this "
                    f"milestone's own suppression-physics boundary).",
                )


class NoAssetAutomaticallyExecutesBuildingControlTests(unittest.TestCase):

    def test_sprinkler_model_never_imports_building_control(self):

        text = _text(SPRINKLER_MODEL)
        self.assertIsNone(re.search(_forbidden_import("building_control"), text, re.MULTILINE))

    def test_passive_asset_models_never_import_building_control(self):

        for path in (FIRE_EXTINGUISHER_MODEL, FIRE_HYDRANT_MODEL, HOSE_REEL_MODEL):
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("building_control"), text, re.MULTILINE))

    def test_fire_safety_manager_never_imports_building_control(self):

        for path in sorted(FIRE_SAFETY_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("building_control"), text, re.MULTILINE))

    def test_building_control_never_references_any_of_the_four_new_asset_types(self):

        package_dir = REPO_ROOT / "building_control"

        for path in sorted(package_dir.glob("*.py")):
            text = _text(path)
            for name in ("Sprinkler", "FireExtinguisher", "FireHydrant", "HoseReel"):
                self.assertNotIn(
                    name, text,
                    f"building_control/{path.name} references {name} -- these are physical "
                    f"resources, not remotely-commanded BuildingControl actions (Phase 14).",
                )

    def test_deluge_remains_the_only_water_related_building_control_system(self):

        # Confirms Sprinkler was never merged into Deluge (Phase 14) --
        # DELUGE stays exactly what it already was: a state-only,
        # no-backing-physics remote control system, structurally
        # unrelated to the Sprinkler engineering asset this milestone adds.
        from building_control.types import ControlSystemType

        self.assertIn("DELUGE", ControlSystemType.__members__)
        self.assertNotIn("SPRINKLER", ControlSystemType.__members__)


class NoAssetAutomaticallyModifiesDecisionPolicyOrBroadcastsVoiceTests(unittest.TestCase):

    def test_no_new_model_imports_decision_policy(self):

        for path in (SPRINKLER_MODEL, FIRE_EXTINGUISHER_MODEL, FIRE_HYDRANT_MODEL, HOSE_REEL_MODEL):
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("decision_policy"), text, re.MULTILINE))

    def test_no_new_model_imports_voice_evacuation(self):

        for path in (SPRINKLER_MODEL, FIRE_EXTINGUISHER_MODEL, FIRE_HYDRANT_MODEL, HOSE_REEL_MODEL):
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("voice_evacuation"), text, re.MULTILINE))

    def test_fire_safety_manager_never_imports_decision_policy_or_voice_evacuation(self):

        for path in sorted(FIRE_SAFETY_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(re.search(_forbidden_import("decision_policy"), text, re.MULTILINE))
            self.assertIsNone(re.search(_forbidden_import("voice_evacuation"), text, re.MULTILINE))


class NoAIOrRLAuthorityAddedTests(unittest.TestCase):

    def test_ai_packages_never_import_fire_safety_manager(self):

        for package_name in AI_PACKAGES:

            package_dir = REPO_ROOT / package_name
            if not package_dir.is_dir():
                continue

            for path in sorted(package_dir.glob("*.py")):
                text = _text(path)
                self.assertIsNone(
                    re.search(_forbidden_import("fire_safety_manager"), text, re.MULTILINE),
                    f"{package_name}/{path.name} imports fire_safety_manager -- AI/RL must gain "
                    f"no new authority over fire-suppression assets.",
                )

    def test_new_models_never_import_ai_or_rl_packages(self):

        for path in (SPRINKLER_MODEL, FIRE_EXTINGUISHER_MODEL, FIRE_HYDRANT_MODEL, HOSE_REEL_MODEL):
            text = _text(path)
            for package_name in AI_PACKAGES:
                self.assertIsNone(re.search(_forbidden_import(package_name), text, re.MULTILINE))


class NoHardwareOrNetworkProtocolCodeTests(unittest.TestCase):

    _FORBIDDEN_PROTOCOL_IMPORTS = (
        r"^\s*(from|import)\s+(pymodbus|modbus|bacpypes|bacnet|paho|mqtt|opcua|freeopcua|pyserial|serial|socket)\b"
    )

    def test_new_models_have_no_protocol_imports(self):

        for path in (SPRINKLER_MODEL, FIRE_SAFETY_ASSET_MODEL, FIRE_EXTINGUISHER_MODEL, FIRE_HYDRANT_MODEL, HOSE_REEL_MODEL):
            text = _text(path)
            self.assertIsNone(re.search(self._FORBIDDEN_PROTOCOL_IMPORTS, text, re.MULTILINE | re.IGNORECASE))

    def test_fire_safety_manager_has_no_protocol_imports(self):

        for path in sorted(FIRE_SAFETY_MANAGER_PACKAGE.glob("*.py")):
            text = _text(path)
            self.assertIsNone(re.search(self._FORBIDDEN_PROTOCOL_IMPORTS, text, re.MULTILINE | re.IGNORECASE))

    def test_no_new_model_mentions_hydraulic_calculation_terms(self):

        # Mechanical proof that Phase 19's "do not build" list was
        # respected: no hydraulic/CFD/pump-curve vocabulary anywhere in
        # the new model layer.
        forbidden_terms = (
            "hazen", "darcy", "k-factor", "kfactor", "pump curve", "cfd", "flow rate", "pressure loss",
        )

        for path in (SPRINKLER_MODEL, FIRE_SAFETY_ASSET_MODEL, FIRE_EXTINGUISHER_MODEL, FIRE_HYDRANT_MODEL, HOSE_REEL_MODEL):
            text = _text(path).lower()
            for term in forbidden_terms:
                self.assertNotIn(term, text, f"{path.name} mentions {term!r}.")


if __name__ == "__main__":
    unittest.main()
