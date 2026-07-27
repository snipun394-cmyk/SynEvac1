import ast
import unittest
from pathlib import Path


# =====================================================
# Localized Predictive Model V2.2 milestone, Phase 5 -- architecture
# guards for the V2.1 experimental modules, extending tests/
# test_predictive_dataset_architecture_guards.py's own AST-based import
# scan to the new files rather than re-litigating the leakage boundary
# concept. Four separate rules, each answering one of this milestone's
# own explicit "add architecture guards preventing..." bullets:
#
#   1. simulation ground truth never enters live extraction
#      (live_extractor_v2_1.py must never import predictive_dataset.
#      simulation_extractor, simulator.*, or target_generator).
#   2. future timestamps never enter prediction features (both v2_1
#      extractors must never import target_generator).
#   3. live-only packages never enter simulation core (simulation_
#      extractor_v2_1.py must never import live_occupants.*,
#      crowd_intelligence.*, or evacuation_progress.*).
#   4. predictive model / dataset code never enters the deterministic
#      intelligence engines (crowd_intelligence/, evacuation_progress/
#      must never import predictive_dataset or predictive_model --
#      the ML layer depends on the deterministic layer, never the
#      reverse).
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


def _package_imports(package_dir: Path):
    """Union of every import name across every .py file directly in
    package_dir (non-recursive -- matches how crowd_intelligence/
    evacuation_progress are each a single flat package)."""

    all_imports = set()
    for py_file in package_dir.glob("*.py"):
        all_imports |= _imported_module_names(py_file)
    return all_imports


class SimulationExtractorV21GuardTests(unittest.TestCase):

    def setUp(self):
        self.imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "simulation_extractor_v2_1.py")

    def test_never_imports_target_generator(self):
        self.assertFalse(any("target_generator" in name for name in self.imports))

    def test_never_imports_live_only_packages(self):
        for forbidden in ("live_occupants", "crowd_intelligence", "evacuation_progress"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(name == forbidden or name.startswith(forbidden + ".") for name in self.imports))


class LiveExtractorV21GuardTests(unittest.TestCase):

    def setUp(self):
        self.imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "live_extractor_v2_1.py")

    def test_never_imports_target_generator(self):
        self.assertFalse(any("target_generator" in name for name in self.imports))

    def test_never_imports_simulator_execution_engine(self):
        # simulator.* is the discrete-event SIMULATION engine -- live
        # code must never depend on it (predictive_dataset.
        # simulation_extractor_v2_1 is fine to import for its purely
        # structural, occupancy-independent build_alternative_route_
        # counts() helper, which itself never imports simulator either).
        self.assertFalse(any(name == "simulator" or name.startswith("simulator.") for name in self.imports))

    def test_never_constructs_a_live_occupant_manager(self):
        # live_extractor.py's own documented rule ("this module has no
        # LiveOccupantManager reference of its own and must not
        # construct one") extended to the v2_1 module -- occupants are
        # always passed in by the caller.
        self.assertFalse(any("live_occupants.manager" in name for name in self.imports))


class DeterministicEngineGuardTests(unittest.TestCase):
    """The ML/dataset layer (predictive_dataset, predictive_model)
    depends on the deterministic intelligence layer (crowd_intelligence,
    evacuation_progress) -- never the reverse. If either engine ever
    imported predictive_dataset/predictive_model, a live deployment
    could not run its own deterministic congestion/evacuation-progress
    computations without also loading ML training code."""

    def test_crowd_intelligence_never_imports_predictive_layers(self):

        imports = _package_imports(REPO_ROOT / "crowd_intelligence")
        self.assertFalse(any(name.startswith("predictive_dataset") or name.startswith("predictive_model") for name in imports))

    def test_evacuation_progress_never_imports_predictive_layers(self):

        imports = _package_imports(REPO_ROOT / "evacuation_progress")
        self.assertFalse(any(name.startswith("predictive_dataset") or name.startswith("predictive_model") for name in imports))


if __name__ == "__main__":
    unittest.main()
