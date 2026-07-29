import pathlib
import re
import unittest

from navigation.edge import Edge
from navigation.node import Node

from models.staircase import Staircase
from models.zone import Zone


# =====================================================
# Observable Stair Perception milestone, Phase 26 (updated by the
# Observable Asset Perception Framework milestone: the generic snapshot/
# lookup layer moved from stair_perception/ + camera_calibration.
# stair_lookup into observable_assets/ + camera_calibration.asset_lookup,
# with stair_lookup.py reduced to the genuinely Stair-specific adapter --
# see docs/architecture/observable_asset_perception.md) -- mechanically
# proves the observable-asset perception code reports physical/
# perception evidence only: it never alters decision_policy's safety
# authority, never dispatches Voice/Dynamic Signage, never executes
# Building Control, never mutates hazard/fire/smoke physics, never
# touches predictive-model artifacts or ML training, and never
# redesigns NavigationGraph or turns Stair into a Zone. Mirrors
# tests/test_crowd_intelligence_architecture_guards.py's own
# regex-over-source-files convention exactly.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

OBSERVABLE_ASSETS_PACKAGE = REPO_ROOT / "observable_assets"
ASSET_LOOKUP_MODULE = REPO_ROOT / "camera_calibration" / "asset_lookup.py"
STAIR_LOOKUP_MODULE = REPO_ROOT / "camera_calibration" / "stair_lookup.py"


_FORBIDDEN_IMPORTS = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"predictive_model|predictive_dataset|dataset_builder|"
    r"decision_policy|"
    r"advisory_system|"
    r"voice_evacuation|speaker_manager|"
    r"dynamic_signage|sign_manager|"
    r"building_control|"
    r"facp|"
    r"fire_growth|smoke_propagation|hazard_evolution|tenability|"
    r"human_detection\.yolo_backend|human_detection\.yolo_human_detector|"
    r"cv2|torch|ultralytics|onvif"
    r")\b"
)

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"  # FACP mutation verbs
    r"\.broadcast\(|\.announce\(|"  # Voice Evacuation
    r"\.execute_control\(|\.confirm\(|"  # Building Control
    r"\.fit\(|\.train\("  # ML training
)


def _iter_source_files(*paths):

    for path in paths:

        if path.is_dir():
            yield from sorted(path.glob("*.py"))
        elif path.is_file():
            yield path


_IMPORT_STATEMENT_PATTERN = r"^\s*(?:from|import)\s+([a-z_][a-z0-9_.]*)"


def _imported_top_level_packages(path):

    text = path.read_text(encoding="utf-8")

    return {
        match.group(1).split(".")[0]
        for match in re.finditer(_IMPORT_STATEMENT_PATTERN, text, re.MULTILINE)
    }


def _project_packages():

    return {
        path.name for path in REPO_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }


class NewObservableAssetModulesGuardTests(unittest.TestCase):

    def test_1_never_imports_execution_capable_or_ml_modules(self):

        for path in _iter_source_files(OBSERVABLE_ASSETS_PACKAGE, ASSET_LOOKUP_MODULE, STAIR_LOOKUP_MODULE):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_IMPORTS, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"observable asset perception code must report physical/perception evidence only.",
            )

    def test_2_never_calls_action_execution_or_training_verbs(self):

        for path in _iter_source_files(OBSERVABLE_ASSETS_PACKAGE, ASSET_LOOKUP_MODULE, STAIR_LOOKUP_MODULE):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"this milestone reports state, it never executes an action or trains a model.",
            )

    def test_3_observable_assets_depends_on_nothing_but_itself(self):

        # observable_assets/ is a pure value-object + pure-function
        # package (mirrors facp/building_control/fire_safety_manager's
        # own snapshot-package shape, generalized) -- it needs no project
        # dependency at all beyond itself and the standard library. This
        # is what makes it safe for crowd_intelligence/ (a deliberately
        # narrow-allow-list package, see tests/test_crowd_intelligence_
        # architecture_guards.py) to import directly.
        allowed_prefixes = ("observable_assets",)

        project_packages = _project_packages()

        for path in sorted(OBSERVABLE_ASSETS_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(_IMPORT_STATEMENT_PATTERN, text, re.MULTILINE):

                imported = match.group(1)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, outside observable_assets/'s own "
                    f"deliberately empty project-dependency allow-list.",
                )

    def test_4_asset_lookup_depends_on_nothing_but_itself(self):

        # camera_calibration/asset_lookup.py is the pure GENERIC
        # framework -- it must know NOTHING about Stair, Door, or any
        # other concrete asset type, and needs no project dependency at
        # all: it duck-types against `.id`/`.contains_world_point()`/
        # `.observable_region_for_floor()`, never importing
        # models.staircase or any other concrete asset model.
        project_packages = _project_packages()
        text = ASSET_LOOKUP_MODULE.read_text(encoding="utf-8")

        for match in re.finditer(_IMPORT_STATEMENT_PATTERN, text, re.MULTILINE):

            imported = match.group(1)
            top_level = imported.split(".")[0]

            self.assertNotIn(
                top_level, project_packages,
                f"camera_calibration/asset_lookup.py imports {imported!r} -- the generic framework module "
                f"must have ZERO project-package dependencies (not even camera_calibration itself), "
                f"proving it knows nothing about any concrete asset type.",
            )

    def test_5_stair_lookup_only_depends_on_camera_calibration_itself(self):

        # camera_calibration/stair_lookup.py duck-types against Staircase-
        # shaped objects (`.id`, `.contains_world_point()`,
        # `.observable_region_for_floor()`) without ever importing
        # models.staircase -- the exact same convention camera_
        # calibration.projection.WorldProjector already established for
        # Zone. Its only real project dependency is the generic
        # camera_calibration.asset_lookup framework it registers Stair
        # into.
        project_packages = _project_packages()
        text = STAIR_LOOKUP_MODULE.read_text(encoding="utf-8")

        for match in re.finditer(_IMPORT_STATEMENT_PATTERN, text, re.MULTILINE):

            imported = match.group(1)
            top_level = imported.split(".")[0]

            if top_level not in project_packages:
                continue

            self.assertTrue(
                imported == "camera_calibration" or imported.startswith("camera_calibration."),
                f"camera_calibration/stair_lookup.py imports {imported!r} -- expected duck-typing, no "
                f"models.staircase (or any other project package) dependency.",
            )


class StairRemainsNavigationEdgeNotZoneOrNodeTests(unittest.TestCase):

    # Phase 26's other explicit requirement: prove the milestone did NOT
    # redesign NavigationGraph or turn Stair into a Zone -- structural
    # assertions against the REAL, currently-imported classes (not text
    # search), so a future refactor that violated this would fail here
    # immediately regardless of how the code happened to be reworded.

    def test_6_staircase_is_not_a_zone_subclass(self):

        self.assertFalse(issubclass(Staircase, Zone))
        self.assertNotIn(Zone, Staircase.__mro__)

    def test_7_node_still_has_no_stair_node_type(self):

        # Node.ZONE/Node.OUTSIDE/Node.ASSEMBLY_POINT are the only node
        # types this codebase has ever had -- a Node.STAIR appearing
        # would mean NavigationGraph itself was redesigned to treat
        # Stair as a graph NODE, exactly what the prior audit and both
        # milestones' own briefs explicitly reject.
        self.assertFalse(hasattr(Node, "STAIR"))
        self.assertTrue(hasattr(Node, "ZONE"))

    def test_8_edge_still_has_a_stair_edge_type(self):

        # The inverse check -- Stair must REMAIN a navigation Edge type,
        # completely untouched.
        self.assertTrue(hasattr(Edge, "STAIR"))

    def test_9_staircase_has_no_zone_like_capacity_or_max_occupancy_field(self):

        # Zone.max_occupancy is a genuine Zone-only concept (read by
        # navigation.node.Node.capacity) -- Staircase must not have grown
        # an equivalent field, which would signal a quiet drift toward
        # Zone-like occupancy semantics on the traversal asset itself.
        stair = Staircase(name="S")
        self.assertFalse(hasattr(stair, "max_occupancy"))


class NoCrossPackageLeakageTests(unittest.TestCase):

    # Checks actual `import`/`from ... import` STATEMENTS only (never a
    # blind substring search, which would false-positive on this
    # milestone's own explanatory comments mentioning these package
    # names in prose).

    def test_10_crowd_intelligence_imports_observable_assets_and_nothing_stair_specific(self):

        # Observable Asset Perception Framework milestone, Phase 5 --
        # CrowdIntelligenceEngine.compute() now DELIBERATELY imports
        # observable_assets (a pure value-object package, see
        # tests/test_crowd_intelligence_architecture_guards.py's own
        # updated allow-list) -- this is the intended widening this
        # milestone makes, not a leak. What must NEVER happen: importing
        # camera_calibration (WorldProjector/asset_lookup are not
        # crowd_intelligence's concern) or anything Stair-specific
        # (models.staircase) -- CrowdIntelligenceEngine only ever
        # consumes the already-generic ObservableAssetSnapshot.
        crowd_intelligence_package = REPO_ROOT / "crowd_intelligence"

        any_file_imports_observable_assets = False

        for path in sorted(crowd_intelligence_package.glob("*.py")):

            imported = _imported_top_level_packages(path)

            if "observable_assets" in imported:
                any_file_imports_observable_assets = True

            for forbidden in ("camera_calibration", "stair_perception"):

                self.assertNotIn(
                    forbidden, imported,
                    f"{path.relative_to(REPO_ROOT)} imports {forbidden!r} -- crowd_intelligence/ should only "
                    f"ever consume the already-generic observable_assets.models.ObservableAssetSnapshot.",
                )

        # Confirms the intended integration actually exists somewhere in
        # this package (engine.py) -- not just "never forbidden," but
        # genuinely wired up, per this milestone's own Phase 5.
        self.assertTrue(
            any_file_imports_observable_assets,
            "expected at least one crowd_intelligence/ module (engine.py) to import observable_assets.",
        )

    def test_11_live_occupants_does_not_import_observable_assets_or_camera_calibration(self):

        # live_occupants/ remains a pure occupant-state package -- it
        # only ever receives a plain stair_id string from its caller
        # (live_camera_pipeline), never resolves asset geometry itself,
        # and has no reason to know the generic observable_assets
        # abstraction exists either.
        live_occupants_package = REPO_ROOT / "live_occupants"

        for path in sorted(live_occupants_package.glob("*.py")):

            imported = _imported_top_level_packages(path)

            for forbidden in ("observable_assets", "camera_calibration"):

                self.assertNotIn(
                    forbidden, imported,
                    f"{path.relative_to(REPO_ROOT)} imports {forbidden!r} -- live_occupants/ must stay "
                    f"geometry-agnostic, receiving only plain ids from its caller.",
                )

    def test_12_stair_perception_package_no_longer_exists(self):

        # The Observable Stair Perception milestone's own stair_
        # perception/ package had zero Stair-specific logic in it --
        # this milestone's Phase 1 audit found that, and its Phase 4/6
        # absorbed it into observable_assets/ rather than keeping two
        # parallel, functionally-identical packages (exactly the
        # architectural duplication this milestone exists to prevent).
        self.assertFalse((REPO_ROOT / "stair_perception").exists())


if __name__ == "__main__":
    unittest.main()
