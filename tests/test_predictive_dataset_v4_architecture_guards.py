import ast
import unittest
from pathlib import Path

from predictive_dataset.target_generator_v2 import MIN_PERSISTENCE_SECONDS, TARGET_VERSION_V2


# =====================================================
# Predictive Dataset V4 milestone, Phase 10/26/30 -- (1) Target V2
# freeze reconfirmation: this milestone must not have changed target
# semantics to improve class balance or anything else, and (2)
# leakage-boundary architecture guards for the 4 NEW modules
# (graph_context_v4.py, simulation_extractor_v4.py,
# live_extractor_v4.py, schema_v4.py), mirroring
# tests/test_predictive_dataset_target_v2_architecture_guards.py's own
# established AST-import-scanning discipline rather than inventing a
# second one.
# =====================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIVE_DATASET_DIR = REPO_ROOT / "predictive_dataset"


def _imported_module_names(file_path: Path):

    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    names = set()

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)

        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)

    return names


class TargetV2FreezeReconfirmationTests(unittest.TestCase):
    """Phase 10 -- Target V2 remains frozen. This milestone changed
    Dataset V4's FEATURES and TOPOLOGY FAMILIES, never Target V2 itself."""

    def test_target_version_string_unchanged(self):
        self.assertEqual(TARGET_VERSION_V2, "v2-persistent-demand-service-imbalance")

    def test_persistence_floor_unchanged(self):
        """MUST still be 3.0s -- lowering it to manufacture more
        positives for the new families would be exactly the "change the
        target to improve class balance" move Phase 10 explicitly
        forbids."""

        self.assertEqual(MIN_PERSISTENCE_SECONDS, 3.0)

    def test_campaign_runner_v4_uses_target_v2_not_v1(self):
        """The new campaign runner must call the SAME target_generator_v2
        module every V3+ campaign uses -- never target_generator.py (V1)."""

        imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "campaign_runner_v4.py")
        self.assertIn("predictive_dataset.target_generator_v2", imports)
        self.assertNotIn("predictive_dataset.target_generator", imports)


class V4ModuleLeakageBoundaryGuardTests(unittest.TestCase):
    """None of the 4 new V4 extraction-side modules may import
    target_generator_v2 or target_semantics_analysis -- exactly the same
    rule every prior extractor (V1/V2.1) already had to follow."""

    NEW_MODULES = ("graph_context_v4.py", "simulation_extractor_v4.py", "live_extractor_v4.py", "schema_v4.py")

    def test_no_new_module_imports_target_generator_v2(self):

        for module_file in self.NEW_MODULES:
            with self.subTest(module=module_file):
                imports = _imported_module_names(PREDICTIVE_DATASET_DIR / module_file)
                self.assertFalse(any("target_generator_v2" in name for name in imports))

    def test_no_new_module_imports_target_semantics_analysis(self):

        for module_file in self.NEW_MODULES:
            with self.subTest(module=module_file):
                imports = _imported_module_names(PREDICTIVE_DATASET_DIR / module_file)
                self.assertFalse(any("target_semantics_analysis" in name for name in imports))

    def test_graph_context_v4_has_zero_occupancy_or_time_dependence(self):
        """Static-analysis proxy for Phase 1's own "zero occupancy, fire,
        time, or scenario-outcome dependence" leakage claim: the module
        must never reference movement_result/occupancy_snapshot/
        crowd_snapshot -- it only ever takes a `building`."""

        source = (PREDICTIVE_DATASET_DIR / "graph_context_v4.py").read_text(encoding="utf-8")
        for forbidden in ("movement_result", "occupancy_snapshot", "crowd_snapshot", "evacuation_snapshot"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_v4_schema_module_does_not_import_topologies_v4(self):
        """schema_v4.py must stay pure feature-metadata -- it must never
        import a specific topology/campaign module (that would be a
        layering violation: feature schema should not depend on which
        buildings exist)."""

        imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "schema_v4.py")
        self.assertFalse(any("topologies_v4" in name or "topologies_v3" in name for name in imports))


if __name__ == "__main__":
    unittest.main()
