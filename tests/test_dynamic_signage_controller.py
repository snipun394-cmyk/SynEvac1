import unittest

from dynamic_signage.controller import DynamicSignageController, SignageRequestStatus
from dynamic_signage.models import SignageInstruction
from dynamic_signage.provider import SimulationDynamicSignageProvider

from tests.test_dynamic_signage_provider import FailingDynamicSignageProvider


def _instruction(sign_id="SIGN-1", revision=1, indication="STRAIGHT", target="DOOR-1", timestamp=0.0):

    return SignageInstruction(
        sign_id=sign_id, indication=indication, target_asset_id=target,
        signage_revision=revision, timestamp=timestamp, status="ACTIVE",
    )


class SubmitApproveRejectTests(unittest.TestCase):

    def test_submit_starts_pending(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())
        instruction = _instruction()

        controller.submit(instruction, 0.0)

        self.assertEqual(controller.status_of("SIGN-1", 1), SignageRequestStatus.PENDING_APPROVAL)
        self.assertIn(instruction, controller.pending_instructions())

    def test_no_automatic_approval(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())
        controller.submit(_instruction(), 0.0)

        # Nothing beyond submit() was called -- the provider must still
        # show no current indication for this sign.
        self.assertIsNone(controller.provider.current_indication("SIGN-1"))

    def test_approve_dispatches_to_provider(self):

        provider = SimulationDynamicSignageProvider()
        controller = DynamicSignageController(provider)
        controller.submit(_instruction(), 0.0)

        result = controller.approve("SIGN-1", 1, 1.0)

        self.assertEqual(controller.status_of("SIGN-1", 1), SignageRequestStatus.CONFIRMED)
        self.assertEqual(provider.current_indication("SIGN-1"), result)

    def test_reject_never_dispatches(self):

        provider = SimulationDynamicSignageProvider()
        controller = DynamicSignageController(provider)
        controller.submit(_instruction(), 0.0)

        controller.reject("SIGN-1", 1, 1.0)

        self.assertEqual(controller.status_of("SIGN-1", 1), SignageRequestStatus.REJECTED)
        self.assertIsNone(provider.current_indication("SIGN-1"))

    def test_cannot_approve_twice(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())
        controller.submit(_instruction(), 0.0)
        controller.approve("SIGN-1", 1, 1.0)

        with self.assertRaises(ValueError):
            controller.approve("SIGN-1", 1, 2.0)

    def test_provider_failure_recorded_as_failed(self):

        controller = DynamicSignageController(FailingDynamicSignageProvider())
        controller.submit(_instruction(), 0.0)

        controller.approve("SIGN-1", 1, 1.0)

        self.assertEqual(controller.status_of("SIGN-1", 1), SignageRequestStatus.FAILED)


class SupersessionTests(unittest.TestCase):

    def test_new_revision_supersedes_old_pending(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())

        controller.submit(_instruction(revision=1), 0.0)
        controller.submit(_instruction(revision=2, indication="RIGHT"), 1.0)

        self.assertEqual(controller.status_of("SIGN-1", 1), SignageRequestStatus.SUPERSEDED)
        self.assertEqual(controller.status_of("SIGN-1", 2), SignageRequestStatus.PENDING_APPROVAL)

    def test_applied_instruction_is_never_superseded_retroactively(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())

        controller.submit(_instruction(revision=1), 0.0)
        controller.approve("SIGN-1", 1, 1.0)

        controller.submit(_instruction(revision=2, indication="RIGHT"), 2.0)

        # Revision 1 was already CONFIRMED -- it must stay CONFIRMED in
        # history, never rewritten to SUPERSEDED.
        self.assertEqual(controller.status_of("SIGN-1", 1), SignageRequestStatus.CONFIRMED)
        self.assertEqual(controller.status_of("SIGN-1", 2), SignageRequestStatus.PENDING_APPROVAL)

    def test_history_retains_every_transition(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())

        controller.submit(_instruction(revision=1), 0.0)
        controller.approve("SIGN-1", 1, 1.0)

        history = controller.history()

        statuses = [event.to_status for event in history]
        self.assertIn(SignageRequestStatus.PENDING_APPROVAL, statuses)
        self.assertIn(SignageRequestStatus.APPROVED, statuses)
        self.assertIn(SignageRequestStatus.DISPATCHED, statuses)
        self.assertIn(SignageRequestStatus.CONFIRMED, statuses)

    def test_all_instructions_include_every_revision(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())

        controller.submit(_instruction(revision=1), 0.0)
        controller.submit(_instruction(revision=2, indication="RIGHT"), 1.0)

        all_ids = {(i.sign_id, i.signage_revision) for i in controller.all_instructions()}
        self.assertEqual(all_ids, {("SIGN-1", 1), ("SIGN-1", 2)})

    def test_duplicate_submission_is_a_no_op(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())

        controller.submit(_instruction(revision=1), 0.0)
        controller.submit(_instruction(revision=1), 5.0)

        self.assertEqual(len(controller.all_instructions()), 1)


if __name__ == "__main__":
    unittest.main()
