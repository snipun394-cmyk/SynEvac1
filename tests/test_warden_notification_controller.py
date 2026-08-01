import unittest

from warden_notification.controller import WardenNotificationController
from warden_notification.provider import SimulationWardenNotificationProvider, WardenNotificationProvider
from warden_notification.requests import WardenNotificationRequest, WardenNotificationResult
from warden_notification.types import WardenNotificationStatus


class _FailingProvider(WardenNotificationProvider):

    is_simulation_only = True

    def notify(self, instruction):

        return WardenNotificationResult(instruction_id=instruction.instruction_id, confirmed=False, message="provider reported failure")


def make_request(zone_id="zone-1", reason="CRITICAL response priority", source_recommendation_id=None):

    return WardenNotificationRequest(zone_id=zone_id, reason=reason, source_recommendation_id=source_recommendation_id)


class RequestLifecycleTests(unittest.TestCase):

    def setUp(self):

        self.provider = SimulationWardenNotificationProvider()
        self.controller = WardenNotificationController(self.provider)

    def test_request_remains_pending_without_approval(self):

        request = self.controller.submit(make_request())

        self.assertEqual(self.controller.status_of(request.request_id), WardenNotificationStatus.PENDING_APPROVAL)

    def test_approved_request_dispatches_and_confirms(self):

        request = self.controller.submit(make_request())
        self.controller.approve(request.request_id)

        self.assertEqual(self.controller.status_of(request.request_id), WardenNotificationStatus.CONFIRMED)

    def test_rejected_request_never_dispatches(self):

        request = self.controller.submit(make_request())
        self.controller.reject(request.request_id)

        self.assertEqual(self.controller.status_of(request.request_id), WardenNotificationStatus.REJECTED)

    def test_cancelled_request_never_dispatches(self):

        request = self.controller.submit(make_request())
        self.controller.cancel(request.request_id)

        self.assertEqual(self.controller.status_of(request.request_id), WardenNotificationStatus.CANCELLED)

    def test_provider_failure_does_not_become_confirmed(self):

        controller = WardenNotificationController(_FailingProvider())

        request = controller.submit(make_request())
        controller.approve(request.request_id)

        self.assertEqual(controller.status_of(request.request_id), WardenNotificationStatus.FAILED)

    def test_approve_requires_pending_approval_status(self):

        request = self.controller.submit(make_request())
        self.controller.approve(request.request_id)

        with self.assertRaises(ValueError):
            self.controller.approve(request.request_id)

    def test_reject_requires_pending_approval_status(self):

        request = self.controller.submit(make_request())
        self.controller.reject(request.request_id)

        with self.assertRaises(ValueError):
            self.controller.reject(request.request_id)

    def test_duplicate_pending_request_is_deduplicated(self):

        first = self.controller.submit(make_request(source_recommendation_id="rec-1"))
        second = self.controller.submit(make_request(source_recommendation_id="rec-1"))

        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(len(self.controller.all_requests()), 1)

    def test_different_zone_is_not_deduplicated(self):

        self.controller.submit(make_request(zone_id="zone-1", source_recommendation_id="rec-1"))
        self.controller.submit(make_request(zone_id="zone-2", source_recommendation_id="rec-1"))

        self.assertEqual(len(self.controller.all_requests()), 2)

    def test_pending_requests_ordering_is_deterministic(self):

        first = self.controller.submit(make_request(zone_id="zone-a"))
        second = self.controller.submit(make_request(zone_id="zone-b"))

        pending = self.controller.pending_requests()

        self.assertEqual(pending, (first, second))

    def test_history_is_append_only_and_complete(self):

        request = self.controller.submit(make_request())
        self.controller.approve(request.request_id)

        events = self.controller.history()

        self.assertEqual(len(events), 4)
        self.assertEqual(events[0].to_status, WardenNotificationStatus.PENDING_APPROVAL)
        self.assertEqual(events[1].to_status, WardenNotificationStatus.APPROVED)
        self.assertEqual(events[2].to_status, WardenNotificationStatus.DISPATCHED)
        self.assertEqual(events[3].to_status, WardenNotificationStatus.CONFIRMED)

    def test_provider_property_exposes_injected_provider(self):

        self.assertIs(self.controller.provider, self.provider)


if __name__ == "__main__":
    unittest.main()
