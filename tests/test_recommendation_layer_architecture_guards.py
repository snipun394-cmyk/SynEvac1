import pathlib
import re
import unittest


# =====================================================
# The Recommendation Layer milestone -- mechanical dependency-direction
# guards, mirroring tests/test_evacuation_recommendation_architecture_
# guards.py exactly.
#
# recommendation_layer/ must NOT import AI/RL/Advisory-engine/Command
# Center/Voice Evacuation/Building Control execution/decision_policy,
# or any sibling package's own .engine/.orchestrator internals -- it
# may only read their plain output models. It must never call an
# execution verb. And the frozen providers (evacuation_recommendation/
# evacuation_guidance/advisory_system/emergency_response/crowd_
# intelligence) must never import recommendation_layer back --
# mechanical proof they stay frozen and unaware.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RECOMMENDATION_LAYER_PACKAGE = REPO_ROOT / "recommendation_layer"

_PROVIDER_PACKAGES = ("evacuation_recommendation", "evacuation_guidance", "advisory_system", "emergency_response", "crowd_intelligence")

_FORBIDDEN_FOR_RECOMMENDATION_LAYER = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|"
    r"command_center|voice_evacuation|speaker_manager|building_control|live_runtime|"
    r"evacuation_recommendation\.(engine|ranking|scoring)|"
    r"evacuation_guidance\.(engine|route_planner|instruction_builder|message_planner)|"
    r"advisory_system\.(orchestrator|advisory_engine|confidence_engine|explanation_engine)|"
    r"emergency_response\.engine|crowd_intelligence\.engine|evacuation_progress\.engine|trajectory_intelligence\.engine|"
    r"cv2|torch|ultralytics|onvif|ground_truth|simulator|human_decision_engine|behaviour_profile_resolver"
    r")\b"
)

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"
    r"\.broadcast\(|\.announce\(|"
    r"\.execute_control\(|\.confirm\(|\.dispatch\(|\.execute\(|\.apply\("
)

_ALLOWED_PROJECT_PACKAGE_PREFIXES = (
    "evacuation_recommendation.models", "evacuation_guidance.models", "emergency_response.models",
    "crowd_intelligence.models", "advisory_system.recommendation_models", "hazard.severity",
    "recommendation_layer",
)


def _all_recommendation_layer_files():

    return sorted(RECOMMENDATION_LAYER_PACKAGE.glob("**/*.py"))


class RecommendationLayerArchitectureGuardTests(unittest.TestCase):

    def test_recommendation_layer_never_imports_forbidden_modules(self):

        for path in _all_recommendation_layer_files():

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_RECOMMENDATION_LAYER, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"recommendation_layer/ must never depend on AI/decision_policy/Command Center/execution "
                f"modules, or any provider's own .engine/.orchestrator internals.",
            )

    def test_recommendation_layer_never_calls_action_execution_verbs(self):

        for path in _all_recommendation_layer_files():

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"recommendation_layer/ only ever recommends, it never executes or dispatches.",
            )

    def test_recommendation_layer_not_nested_inside_another_package(self):

        self.assertTrue(RECOMMENDATION_LAYER_PACKAGE.is_dir())
        self.assertEqual(RECOMMENDATION_LAYER_PACKAGE.parent, REPO_ROOT)

    def test_recommendation_layer_only_depends_on_allowed_project_packages(self):

        project_package_pattern = r"^\s*(from|import)\s+([a-z_][a-z0-9_.]*)"

        project_packages = {
            path.name for path in REPO_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }

        for path in _all_recommendation_layer_files():

            text = path.read_text(encoding="utf-8")

            for match in re.finditer(project_package_pattern, text, re.MULTILINE):

                imported = match.group(2)
                top_level = imported.split(".")[0]

                if top_level not in project_packages:
                    continue

                self.assertTrue(
                    any(imported == prefix or imported.startswith(prefix + ".") for prefix in _ALLOWED_PROJECT_PACKAGE_PREFIXES),
                    f"{path.relative_to(REPO_ROOT)} imports {imported!r}, not in recommendation_layer/'s own "
                    f"documented allow-list.",
                )


class ProvidersNeverImportRecommendationLayerTests(unittest.TestCase):

    def test_providers_never_import_recommendation_layer_back(self):

        for package_name in _PROVIDER_PACKAGES:

            package_path = REPO_ROOT / package_name

            if not package_path.is_dir():
                continue

            for path in sorted(package_path.glob("**/*.py")):

                text = path.read_text(encoding="utf-8")
                match = re.search(r"^\s*(from|import)\s+recommendation_layer\b", text, re.MULTILINE)

                self.assertIsNone(
                    match,
                    f"{path.relative_to(REPO_ROOT)} imports recommendation_layer -- {package_name}/ must stay "
                    f"frozen and unaware this layer exists (one-directional dependency only).",
                )


if __name__ == "__main__":
    unittest.main()
