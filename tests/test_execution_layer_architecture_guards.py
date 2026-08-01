import pathlib
import re
import unittest


# =====================================================
# Execution Layer V1 milestone -- mechanical dependency-direction
# guards, mirroring tests/test_recommendation_layer_architecture_
# guards.py exactly.
#
# execution_layer/ must NOT import AI/decision_policy/hardware-protocol
# modules, and must never call a provider's own execution verb
# (.execute(/.apply(/.send(/.notify() itself for the three PRE-EXISTING
# categories (Voice/BuildingControl/Signage) -- those calls belong
# exclusively to the real controllers, which execution_layer only ever
# READS from. The AI/Advisory/DecisionPolicy/LiveOrchestrator side of
# the platform must never be able to reach execution_layer/
# warden_notification/the controllers directly either -- extends the
# existing AIAuthorityGuardTests scan (tests/test_live_operator_
# action_routing.py) without modifying that file.
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXECUTION_LAYER_PACKAGE = REPO_ROOT / "execution_layer"

_AI_ADVISORY_PACKAGES = (
    REPO_ROOT / "advisory_system",
    REPO_ROOT / "decision_policy",
)
_AI_ADVISORY_FILES = (
    REPO_ROOT / "live_system" / "live_ai_gateway.py",
    REPO_ROOT / "live_system" / "live_advisory_gateway.py",
    REPO_ROOT / "live_system" / "orchestrator.py",
)

_FORBIDDEN_FOR_EXECUTION_LAYER = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|advisory_system\.orchestrator|advisory_system\.advisory_engine|"
    r"modbus|bacnet|mqtt|paho|pymodbus|bacpypes|opcua|socket|serial|"
    r"cv2|torch|ultralytics"
    r")\b"
)

# The three PRE-EXISTING controllers' own execution verbs -- read-only
# adapters for these categories must never call them. warden_adapter is
# exempt from .notify( only insofar as it never calls it either (it
# only constructs a WardenNotificationRequest for the gateway to
# submit() -- see warden_adapter.py's own docstring).
_FORBIDDEN_EXECUTION_VERBS = r"\.execute\(|\.apply\(|\.send\(|\.notify\(|\.broadcast\(|\.dispatch\("


def _all_execution_layer_files():

    return sorted(EXECUTION_LAYER_PACKAGE.glob("**/*.py"))


class ExecutionLayerArchitectureGuardTests(unittest.TestCase):

    def test_execution_layer_never_imports_forbidden_modules(self):

        for path in _all_execution_layer_files():

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_EXECUTION_LAYER, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"execution_layer/ must never depend on AI/decision_policy/hardware-protocol modules.",
            )

    def test_execution_layer_adapters_never_call_execution_verbs(self):

        # This is the mechanical proof of "orchestration, not
        # replacement": every adapter in this package only ever READS
        # controller.history()/all_requests()/broadcast_log -- it never
        # calls a provider's own execute()/apply()/send()/notify(), and
        # never calls submit()/approve()/reject()/dispatch() on a
        # controller either (those stay exclusively operator-triggered,
        # routed through command_center.live_operator_action_gateway).

        for path in _all_execution_layer_files():

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_EXECUTION_VERBS, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0).strip() if match else ''!r} -- "
                f"execution_layer/ only ever reads already-recorded controller state, it never executes.",
            )

    def test_execution_layer_not_nested_inside_another_package(self):

        self.assertTrue(EXECUTION_LAYER_PACKAGE.is_dir())
        self.assertEqual(EXECUTION_LAYER_PACKAGE.parent, REPO_ROOT)


class AIAuthorityCannotReachExecutionLayerTests(unittest.TestCase):

    def test_ai_and_advisory_packages_never_import_execution_layer_or_controllers(self):

        forbidden_pattern = (
            r"^\s*(from|import)\s+("
            r"execution_layer|warden_notification|"
            r"voice_evacuation\.controller|building_control\.controller|dynamic_signage\.controller|"
            r"command_center\.live_operator_action_gateway"
            r")\b"
        )

        for package_dir in _AI_ADVISORY_PACKAGES:

            if not package_dir.is_dir():
                continue

            for path in sorted(package_dir.glob("**/*.py")):

                text = path.read_text(encoding="utf-8")
                match = re.search(forbidden_pattern, text, re.MULTILINE)

                self.assertIsNone(
                    match,
                    f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                    f"the AI/Advisory side of the platform must never reach execution authority directly.",
                )

        for file_path in _AI_ADVISORY_FILES:

            if not file_path.is_file():
                continue

            text = file_path.read_text(encoding="utf-8")
            match = re.search(forbidden_pattern, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{file_path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"the AI/Advisory side of the platform must never reach execution authority directly.",
            )


if __name__ == "__main__":
    unittest.main()
