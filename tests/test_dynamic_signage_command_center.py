import unittest

from command_center.live_operator_action_gateway import (
    OperatorActionUnavailable,
    PROVIDER_CAPABILITY_NO_PROVIDER,
    PROVIDER_CAPABILITY_SIMULATION,
    LiveOperatorActionGateway,
)

from dynamic_signage.controller import DynamicSignageController, SignageRequestStatus
from dynamic_signage.models import DynamicSignageSnapshot, SignageInstruction, SignageStatus
from dynamic_signage.provider import SimulationDynamicSignageProvider

from live_system.event_bus import EventBus, EventType


def _active_instruction(sign_id="SIGN-1", revision=1):

    return SignageInstruction(
        sign_id=sign_id, zone_id="z1", indication="STRAIGHT", status=SignageStatus.ACTIVE,
        signage_revision=revision,
    )


def _unavailable_instruction(sign_id="SIGN-2"):

    return SignageInstruction(sign_id=sign_id, indication="UNAVAILABLE", status=SignageStatus.UNAVAILABLE)


class SignageOperatorWorkflowTests(unittest.TestCase):

    def test_no_provider_reports_honest_capability(self):

        gateway = LiveOperatorActionGateway()

        self.assertEqual(gateway.signage_capability, PROVIDER_CAPABILITY_NO_PROVIDER)

    def test_no_provider_raises_on_approve(self):

        gateway = LiveOperatorActionGateway()

        with self.assertRaises(OperatorActionUnavailable):
            gateway.approve_signage_instruction(_active_instruction(), 0.0)

    def test_ingest_only_submits_active_instructions(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())
        gateway = LiveOperatorActionGateway(signage_controller=controller)

        snapshot = DynamicSignageSnapshot(
            timestamp=0.0,
            instructions={"SIGN-1": _active_instruction(), "SIGN-2": _unavailable_instruction()},
            conflicts={},
        )

        submitted = gateway.ingest_signage_instructions(snapshot)

        self.assertEqual({i.sign_id for i in submitted}, {"SIGN-1"})
        self.assertEqual(len(controller.pending_instructions()), 1)

    def test_approve_dispatches_and_publishes_event(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())
        event_bus = EventBus()
        gateway = LiveOperatorActionGateway(signage_controller=controller, event_bus=event_bus)

        instruction = _active_instruction()
        gateway.ingest_signage_instructions(
            DynamicSignageSnapshot(timestamp=0.0, instructions={"SIGN-1": instruction}, conflicts={}),
        )

        gateway.approve_signage_instruction(instruction, 1.0)

        self.assertEqual(controller.provider.current_indication("SIGN-1").sign_id, "SIGN-1")
        self.assertEqual(len(event_bus.history_of(EventType.SIGNAGE_INSTRUCTION_APPROVED)), 1)

    def test_reject_never_dispatches_and_publishes_event(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())
        event_bus = EventBus()
        gateway = LiveOperatorActionGateway(signage_controller=controller, event_bus=event_bus)

        instruction = _active_instruction()
        gateway.ingest_signage_instructions(
            DynamicSignageSnapshot(timestamp=0.0, instructions={"SIGN-1": instruction}, conflicts={}),
        )

        gateway.reject_signage_instruction(instruction, 1.0)

        self.assertIsNone(controller.provider.current_indication("SIGN-1"))
        self.assertEqual(len(event_bus.history_of(EventType.SIGNAGE_INSTRUCTION_REJECTED)), 1)

    def test_capability_reports_simulation(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())
        gateway = LiveOperatorActionGateway(signage_controller=controller)

        self.assertEqual(gateway.signage_capability, PROVIDER_CAPABILITY_SIMULATION)

    def test_no_controller_pending_and_history_are_empty(self):

        gateway = LiveOperatorActionGateway()

        self.assertEqual(gateway.pending_signage_instructions(), ())
        self.assertEqual(gateway.all_signage_instructions(), ())
        self.assertEqual(gateway.signage_history(), ())


if __name__ == "__main__":
    unittest.main()
