import unittest

from dynamic_signage.models import SignageInstruction
from dynamic_signage.provider import DynamicSignageProvider, SignageApplyResult, SimulationDynamicSignageProvider


class FailingDynamicSignageProvider(DynamicSignageProvider):

    is_simulation_only = True

    def apply(self, instruction):
        return SignageApplyResult(confirmed=False, message="hardware unreachable")


class SimulationProviderTests(unittest.TestCase):

    def test_apply_records_current_and_history(self):

        provider = SimulationDynamicSignageProvider()
        instruction = SignageInstruction(sign_id="SIGN-1", indication="STRAIGHT", signage_revision=1)

        result = provider.apply(instruction)

        self.assertTrue(result.confirmed)
        self.assertEqual(provider.current_indication("SIGN-1"), instruction)
        self.assertEqual(provider.applied_instructions(), (instruction,))

    def test_no_network_or_hardware_access(self):

        # Structural guard: the simulation provider does nothing beyond
        # in-memory bookkeeping -- confirmed by inspecting its own
        # module for the absence of any networking/hardware import
        # statement (not merely the substring, which would also flag
        # innocuous prose like "mirrors building_control.requests").
        import inspect
        import re

        import dynamic_signage.provider as provider_module

        source = inspect.getsource(provider_module)

        match = re.search(
            r"^\s*(from|import)\s+(socket|serial|requests|urllib|cv2)\b", source, re.MULTILINE,
        )
        self.assertIsNone(match)

    def test_current_indication_unknown_sign_is_none(self):

        provider = SimulationDynamicSignageProvider()

        self.assertIsNone(provider.current_indication("SIGN-UNKNOWN"))

    def test_is_simulation_only_flag(self):

        self.assertTrue(SimulationDynamicSignageProvider.is_simulation_only)
        self.assertFalse(DynamicSignageProvider.is_simulation_only)


class FailingProviderTests(unittest.TestCase):

    def test_provider_failure_reported_honestly(self):

        provider = FailingDynamicSignageProvider()
        instruction = SignageInstruction(sign_id="SIGN-1", indication="STRAIGHT", signage_revision=1)

        result = provider.apply(instruction)

        self.assertFalse(result.confirmed)
        self.assertEqual(result.message, "hardware unreachable")


if __name__ == "__main__":
    unittest.main()
