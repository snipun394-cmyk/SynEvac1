import ast
import unittest
from pathlib import Path

import calibration_studio


CALIBRATION_STUDIO_DIR = Path(calibration_studio.__file__).parent

# Names that would signal a reimplemented statistics/comparison/
# recommendation primitive if defined *inside* calibration_studio --
# this milestone's own explicit "No duplicated statistics. No
# duplicated comparison logic. No duplicated recommendation logic."
_PROHIBITED_DEFINED_NAMES = frozenset({
    "paired_comparison", "confidence_interval", "effect_size_cohens_d",
    "compare_arms", "one_way_anova", "feature_sensitivity", "distribution_shift",
    "_compare", "_paired_non_none", "recommend", "Verdict", "MetricComparison",
    "MetricVerdict", "AdoptionRecommendation",
})

# Importing either directly would mean calibration_studio is doing its
# own statistics/array-crunching rather than delegating -- a real,
# checkable signal, not just a naming convention.
_PROHIBITED_IMPORT_MODULES = frozenset({"scipy", "numpy"})


def _parse(path: Path) -> ast.Module:

    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _all_calibration_studio_source_files():

    return sorted(CALIBRATION_STUDIO_DIR.glob("*.py"))


class NoDuplicatedStatisticsOrComparisonLogicTests(unittest.TestCase):

    def test_no_calibration_studio_module_defines_a_prohibited_name(self):

        offenders = []

        for path in _all_calibration_studio_source_files():

            tree = _parse(path)

            for node in ast.walk(tree):

                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):

                    if node.name in _PROHIBITED_DEFINED_NAMES:
                        offenders.append(f"{path.name}::{node.name}")

        self.assertEqual(offenders, [], f"Reimplemented comparison/statistics primitive(s) found: {offenders}")

    def test_no_calibration_studio_module_imports_scipy_or_numpy_directly(self):

        offenders = []

        for path in _all_calibration_studio_source_files():

            tree = _parse(path)

            for node in ast.walk(tree):

                module_names = []

                if isinstance(node, ast.Import):
                    module_names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module_names = [node.module.split(".")[0]]

                for name in module_names:
                    if name in _PROHIBITED_IMPORT_MODULES:
                        offenders.append(f"{path.name}: imports {name!r}")

        self.assertEqual(
            offenders, [],
            f"calibration_studio imports a numeric/statistics library directly -- statistics must "
            f"come from research_framework.statistics via calibration_benchmark only: {offenders}",
        )

    def test_studio_module_imports_run_calibration_benchmark_from_calibration_benchmark(self):

        tree = _parse(CALIBRATION_STUDIO_DIR / "studio.py")

        imported_names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "calibration_benchmark":
                imported_names.update(alias.name for alias in node.names)

        self.assertIn(
            "run_calibration_benchmark", imported_names,
            "studio.py must import the real calibration_benchmark.run_calibration_benchmark, "
            "not reimplement any part of what it does.",
        )

    def test_run_calibration_benchmark_is_called_exactly_once_in_studio_py(self):

        # One single delegation point (_execute()) -- both public
        # runner methods route through it, never call into
        # calibration_benchmark's internals a second, different way.
        # AST-based (not a substring count): studio.py's own comments
        # and docstrings mention "run_calibration_benchmark()" several
        # times in prose, which a naive string count would also catch.
        tree = _parse(CALIBRATION_STUDIO_DIR / "studio.py")

        call_sites = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_calibration_benchmark"
        ]

        self.assertEqual(len(call_sites), 1)

    def test_calibration_studio_never_imports_calibration_benchmark_recommendation_module(self):

        # recommend()/Verdict/AdoptionRecommendation are already
        # produced *inside* calibration_benchmark.run_calibration_benchmark()
        # -- calibration_studio.studio must consume the resulting
        # CalibrationBenchmarkResult, never call calibration_benchmark.
        # recommendation itself a second time.
        for path in _all_calibration_studio_source_files():

            tree = _parse(path)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "calibration_benchmark.recommendation":
                    self.fail(f"{path.name} imports calibration_benchmark.recommendation directly")

    def test_calibration_studio_never_imports_research_framework_statistics_directly(self):

        for path in _all_calibration_studio_source_files():

            tree = _parse(path)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "research_framework.statistics":
                    self.fail(f"{path.name} imports research_framework.statistics directly")


if __name__ == "__main__":
    unittest.main()
