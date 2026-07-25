import ast
import unittest
from pathlib import Path


# =====================================================
# Phase 9/16 -- mechanical guard that the feature/target leakage
# boundary stays a CODE boundary, not just a documented convention:
# predictive_dataset.simulation_extractor and predictive_dataset.
# live_extractor (both live-deployable, both must stay ignorant of
# future/outcome information) must never import predictive_dataset.
# target_generator (the one module in this package allowed to inspect
# what happens after `time`). Mirrors this codebase's own established
# "*_architecture_guards" test convention (e.g.
# tests/test_crowd_intelligence_architecture_guards.py,
# tests/test_evacuation_recommendation_architecture_guards.py).
# =====================================================

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "predictive_dataset"


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


class LeakageBoundaryGuardTests(unittest.TestCase):

    def test_simulation_extractor_never_imports_target_generator(self):

        imports = _imported_module_names(PACKAGE_DIR / "simulation_extractor.py")

        self.assertFalse(any("target_generator" in name for name in imports))

    def test_live_extractor_never_imports_target_generator(self):

        imports = _imported_module_names(PACKAGE_DIR / "live_extractor.py")

        self.assertFalse(any("target_generator" in name for name in imports))

    def test_schema_never_imports_target_generator(self):

        imports = _imported_module_names(PACKAGE_DIR / "schema.py")

        self.assertFalse(any("target_generator" in name for name in imports))


if __name__ == "__main__":
    unittest.main()
