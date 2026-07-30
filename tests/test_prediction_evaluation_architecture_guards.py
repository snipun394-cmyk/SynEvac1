import pathlib
import re
import unittest


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 10 --
# mechanically proves evaluation is completely passive: prediction_
# evaluation/ must never import Recommendation, Guidance, Simulation,
# Live Runtime, Voice, Signage, Building Control, or any operational UI
# package, and none of THOSE packages may import prediction_evaluation
# either -- a two-way guarantee that evaluation can only ever be
# invoked AFTER the fact, by a standalone caller (a script or a test),
# never as part of any live/simulated cycle.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PREDICTION_EVALUATION_PACKAGE = REPO_ROOT / "prediction_evaluation"

_OPERATIONAL_PACKAGES = (
    "evacuation_recommendation", "evacuation_guidance",
    "simulator", "simulation_runtime",
    "live_system", "live_runtime", "live_runtime_launcher",
    "voice_evacuation", "speaker_manager",
    "dynamic_signage", "sign_manager",
    "building_control",
    "facp",
    "command_center",
    "ai_registry", "ai_inference", "ai_training",
)

_FORBIDDEN_IMPORTS = r"^\s*(from|import)\s+(" + "|".join(_OPERATIONAL_PACKAGES) + r")\b"

_FORBIDDEN_ACTION_CALLS = (
    r"\.evaluate\(|\.acknowledge\(|\.silence\(|\.reset\(|"  # FACP mutation verbs
    r"\.broadcast\(|\.announce\(|"  # Voice Evacuation
    r"\.execute_control\(|\.confirm\(|"  # Building Control
    r"\.predict_bottleneck_occurrence\(|\.predict_evacuation_time\("  # model inference
)


class PredictionEvaluationArchitectureGuardTests(unittest.TestCase):

    def test_prediction_evaluation_never_imports_operational_packages(self):

        for path in sorted(PREDICTION_EVALUATION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_IMPORTS, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"prediction_evaluation/ must stay a purely offline, post-hoc evaluation package.",
            )

    def test_prediction_evaluation_never_calls_action_or_inference_verbs(self):

        for path in sorted(PREDICTION_EVALUATION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_ACTION_CALLS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"prediction_evaluation/ evaluates already-produced predictions, it never runs inference "
                f"or executes an action.",
            )

    def test_prediction_evaluation_not_nested_inside_another_package(self):

        self.assertTrue(PREDICTION_EVALUATION_PACKAGE.is_dir())
        self.assertEqual(PREDICTION_EVALUATION_PACKAGE.parent, REPO_ROOT)

    def test_no_operational_package_imports_prediction_evaluation(self):

        # The reverse direction -- Recommendation/Guidance/Simulation/
        # Live Runtime/Voice/Signage/Building Control/Command Center must
        # never reach INTO the evaluation framework either (there is no
        # legitimate reason for an operational decision to consult a
        # post-hoc evaluation report).

        forbidden_pattern = r"^\s*(from|import)\s+prediction_evaluation\b"

        for package_name in _OPERATIONAL_PACKAGES:

            package_dir = REPO_ROOT / package_name

            if not package_dir.is_dir():
                continue

            for path in sorted(package_dir.rglob("*.py")):

                if "__pycache__" in path.parts:
                    continue

                text = path.read_text(encoding="utf-8")
                match = re.search(forbidden_pattern, text, re.MULTILINE)

                self.assertIsNone(
                    match,
                    f"{path.relative_to(REPO_ROOT)} imports prediction_evaluation -- an operational "
                    f"package must never depend on the evaluation framework.",
                )


if __name__ == "__main__":
    unittest.main()
