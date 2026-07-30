import ast
import re
import unittest
from pathlib import Path


# =====================================================
# Stair Predictive-Feature Live Parity milestone, Phase 14 -- mechanically
# proves the dependency direction this milestone's wiring must respect:
#
#   Perception / Crowd Intelligence  ->  Feature extraction
#
# never the reverse. Extends tests/test_predictive_dataset_v2_1_
# architecture_guards.py's own AST-based import-scan convention (never
# re-litigating the leakage-boundary concept) to the two new facts this
# milestone introduces: (1) stair_flow/ itself must never import
# predictive_dataset/predictive_model -- perception evidence must not
# know feature extraction exists; (2) no model inference, Recommendation,
# or Guidance package enters live_extractor_v2_1.py/live_extractor_v4.py
# as a RESULT of this milestone's wiring.
# =====================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
STAIR_FLOW_DIR = REPO_ROOT / "stair_flow"
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

    all_imports = set()
    for py_file in package_dir.glob("*.py"):
        all_imports |= _imported_module_names(py_file)
    return all_imports


class DependencyDirectionTests(unittest.TestCase):
    """Perception / Crowd Intelligence -> Feature extraction, never the
    reverse."""

    def test_stair_flow_never_imports_predictive_layers(self):

        imports = _package_imports(STAIR_FLOW_DIR)
        self.assertFalse(any(name.startswith("predictive_dataset") or name.startswith("predictive_model") for name in imports))

    def test_stair_flow_never_imports_ai_or_recommendation_layers(self):

        imports = _package_imports(STAIR_FLOW_DIR)
        forbidden = ("ai_registry", "ai_inference", "ai_training", "evacuation_recommendation", "evacuation_guidance")
        for name in imports:
            for prefix in forbidden:
                self.assertFalse(
                    name == prefix or name.startswith(prefix + "."),
                    f"stair_flow/ imports {name!r} -- perception evidence must never depend on the feature/ML layer.",
                )

    def test_predictive_dataset_live_extractors_may_import_stair_flow(self):

        # The allowed direction, confirmed present (not a guard against
        # absence -- a positive sanity check that the wiring actually
        # exists where expected).
        imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "live_extractor_v2_1.py")
        self.assertTrue(any(name == "stair_flow" or name.startswith("stair_flow.") for name in imports))


class NoModelInferenceOrDecisionCapabilityTests(unittest.TestCase):
    """This milestone makes a feature CORRECTLY AVAILABLE -- it must
    never trigger inference, change Recommendation, or change Guidance."""

    _FORBIDDEN_IMPORTS = (
        r"^\s*(from|import)\s+("
        r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
        r"decision_policy|"
        r"evacuation_recommendation|evacuation_guidance|"
        r"advisory_system|command_center|"
        r"voice_evacuation|speaker_manager|"
        r"dynamic_signage|sign_manager|"
        r"building_control|"
        r"facp"
        r")\b"
    )

    _FORBIDDEN_ACTION_CALLS = (
        r"\.predict\(|\.infer\(|\.run_inference\(|"
        r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"
        r"\.broadcast\(|\.announce\(|"
        r"\.execute_control\(|\.confirm\("
    )

    def test_live_extractors_never_import_decision_or_ml_execution_packages(self):

        for filename in ("live_extractor_v2_1.py", "live_extractor_v4.py"):

            path = PREDICTIVE_DATASET_DIR / filename
            text = path.read_text(encoding="utf-8")
            match = re.search(self._FORBIDDEN_IMPORTS, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"live feature extraction must stay correctness-only, never trigger a decision/ML action.",
            )

    def test_live_extractors_never_call_action_or_inference_verbs(self):

        for filename in ("live_extractor_v2_1.py", "live_extractor_v4.py"):

            path = PREDICTIVE_DATASET_DIR / filename
            text = path.read_text(encoding="utf-8")
            match = re.search(self._FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"this milestone makes a feature available, it never triggers inference or an action.",
            )

    def test_stair_flow_never_calls_action_or_inference_verbs(self):

        for path in sorted(STAIR_FLOW_DIR.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(self._FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r}",
            )


class SimulationUnchangedTests(unittest.TestCase):
    """simulation_extractor_v2_1.py (the frozen simulation-side feature
    definition) must be byte-for-byte behaviorally untouched -- this
    milestone only builds the LIVE counterpart, never touches SIM."""

    def test_simulation_extractor_v2_1_never_imports_stair_flow_or_live_packages(self):

        imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "simulation_extractor_v2_1.py")
        forbidden = ("stair_flow", "live_occupants", "crowd_intelligence", "evacuation_progress")

        for name in imports:
            for prefix in forbidden:
                self.assertFalse(
                    name == prefix or name.startswith(prefix + "."),
                    f"simulation_extractor_v2_1.py imports {name!r} -- the frozen SIM feature definition "
                    f"must never depend on live-only packages.",
                )


if __name__ == "__main__":
    unittest.main()
