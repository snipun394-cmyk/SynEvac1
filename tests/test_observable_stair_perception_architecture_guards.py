import pathlib
import re
import unittest

from navigation.edge import Edge
from navigation.node import Node

from models.staircase import Staircase
from models.zone import Zone


# =====================================================
# Observable Stair Perception milestone, Phase 26 -- mechanically proves
# the new Stair perception code (stair_perception/, camera_calibration.
# stair_lookup) reports physical/perception evidence only: it never
# alters decision_policy's safety authority, never dispatches Voice/
# Dynamic Signage, never executes Building Control, never mutates
# hazard/fire/smoke physics, never touches predictive-model artifacts or
# ML training, and never redesigns NavigationGraph or turns Stair into a
# Zone. Mirrors tests/test_crowd_intelligence_architecture_guards.py's
# own regex-over-source-files convention exactly.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

STAIR_PERCEPTION_PACKAGE = REPO_ROOT / "stair_perception"
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


class NewStairPerceptionModulesGuardTests(unittest.TestCase):

    def test_1_never_imports_execution_capable_or_ml_modules(self):

        for path in _iter_source_files(STAIR_PERCEPTION_PACKAGE, STAIR_LOOKUP_MODULE):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_IMPORTS, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"observable stair perception code must report physical/perception evidence only.",
            )

    def test_2_never_calls_action_execution_or_training_verbs(self):

        for path in _iter_source_files(STAIR_PERCEPTION_PACKAGE, STAIR_LOOKUP_MODULE):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"this milestone reports state, it never executes an action or trains a model.",
            )

    def test_3_stair_perception_only_depends_on_allowed_project_packages(self):

        # stair_perception/ is a pure value-object + pure-function
        # package (mirrors facp/building_control/fire_safety_manager's
        # own snapshot-package shape) -- it needs no project dependency
        # at all beyond itself and the standard library.
        allowed_prefixes = ("stair_perception",)

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in sorted(STAIR_PERCEPTION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in allowed_prefixes),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, outside stair_perception/'s own "
                    f"deliberately empty project-dependency allow-list.",
                )

    def test_4_stair_lookup_only_depends_on_camera_calibration_itself(self):

        # camera_calibration/stair_lookup.py duck-types against Staircase-
        # shaped objects (`.id`, `.contains_world_point()`,
        # `.observable_region_for_floor()`) without ever importing
        # models.staircase -- the exact same convention camera_
        # calibration.projection.WorldProjector already established for
        # Zone.
        text = STAIR_LOOKUP_MODULE.read_text(encoding="utf-8")

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"
        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for match in re.finditer(project_package_pattern, text, re.MULTILINE):

            imported = match.group(2)
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

    def test_5_staircase_is_not_a_zone_subclass(self):

        self.assertFalse(issubclass(Staircase, Zone))
        self.assertNotIn(Zone, Staircase.__mro__)

    def test_6_node_still_has_no_stair_node_type(self):

        # Node.ZONE/Node.OUTSIDE/Node.ASSEMBLY_POINT are the only node
        # types this codebase has ever had -- a Node.STAIR appearing
        # would mean NavigationGraph itself was redesigned to treat
        # Stair as a graph NODE, exactly what the prior audit and this
        # milestone's own brief both explicitly reject.
        self.assertFalse(hasattr(Node, "STAIR"))
        self.assertTrue(hasattr(Node, "ZONE"))

    def test_7_edge_still_has_a_stair_edge_type(self):

        # The inverse check -- Stair must REMAIN a navigation Edge type,
        # completely untouched.
        self.assertTrue(hasattr(Edge, "STAIR"))

    def test_8_staircase_has_no_zone_like_capacity_or_max_occupancy_field(self):

        # Zone.max_occupancy is a genuine Zone-only concept (read by
        # navigation.node.Node.capacity) -- Staircase must not have grown
        # an equivalent field, which would signal a quiet drift toward
        # Zone-like occupancy semantics on the traversal asset itself.
        stair = Staircase(name="S")
        self.assertFalse(hasattr(stair, "max_occupancy"))


_IMPORT_STATEMENT_PATTERN = r"^\s*(?:from|import)\s+([a-z_][a-z0-9_.]*)"


def _imported_top_level_packages(path):

    text = path.read_text(encoding="utf-8")

    return {
        match.group(1).split(".")[0]
        for match in re.finditer(_IMPORT_STATEMENT_PATTERN, text, re.MULTILINE)
    }


class NoCrossPackageLeakageTests(unittest.TestCase):

    # Checks actual `import`/`from ... import` STATEMENTS only (never a
    # blind substring search, which would false-positive on this
    # milestone's own explanatory comments mentioning these package
    # names in prose -- e.g. crowd_intelligence/engine.py's own
    # docstring explains WHY it deliberately does not import
    # stair_perception, which of course mentions the string).

    def test_9_crowd_intelligence_does_not_import_stair_perception(self):

        # By design (see crowd_intelligence/engine.py's own docstring for
        # compute()): CrowdIntelligenceEngine.observed_stair_occupancy is
        # a plain Dict[str, Optional[int]] parameter, never a
        # stair_perception.models.StairOccupancySnapshot import -- keeps
        # crowd_intelligence/'s own existing, separately-guarded
        # (tests/test_crowd_intelligence_architecture_guards.py) import
        # allow-list untouched by this milestone.
        crowd_intelligence_package = REPO_ROOT / "crowd_intelligence"

        for path in sorted(crowd_intelligence_package.glob("*.py")):

            self.assertNotIn(
                "stair_perception", _imported_top_level_packages(path),
                f"{path.relative_to(REPO_ROOT)} imports stair_perception -- crowd_intelligence/ should "
                f"only ever receive a plain, pre-reduced Dict[str, Optional[int]], never this package's own type.",
            )

    def test_10_live_occupants_does_not_import_stair_perception_or_camera_calibration(self):

        # live_occupants/ remains a pure occupant-state package -- it
        # only ever receives a plain stair_id string from its caller
        # (live_camera_pipeline), never resolves stair geometry itself.
        live_occupants_package = REPO_ROOT / "live_occupants"

        for path in sorted(live_occupants_package.glob("*.py")):

            imported = _imported_top_level_packages(path)

            for forbidden in ("stair_perception", "camera_calibration"):

                self.assertNotIn(
                    forbidden, imported,
                    f"{path.relative_to(REPO_ROOT)} imports {forbidden!r} -- live_occupants/ must stay "
                    f"geometry-agnostic, receiving only plain ids from its caller.",
                )


if __name__ == "__main__":
    unittest.main()
