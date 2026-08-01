import unittest

from building_control.requests import ControlRequest
from building_control.types import ControlAction, ControlSystemType, RequestSource

from execution_layer.adapters import building_control_adapter
from execution_layer.models import ExecutionCategory, ExecutionStatus, RecommendationIdProvenance

from tests.execution_layer_fixtures import make_control_controller


class BuildingControlAdapterTests(unittest.TestCase):

    def test_all_four_timestamps_derived_from_history(self):

        controller = make_control_controller()
        request = ControlRequest(
            system_type=ControlSystemType.DOOR, target_id="door-1", requested_action=ControlAction.CLOSE,
            reason="hazard", source=RequestSource.OPERATOR,
        )
        controller.submit(request)
        controller.approve(request.request_id)

        requests = building_control_adapter.build_execution_requests(controller)

        self.assertEqual(len(requests), 1)
        execution_request = requests[0]

        self.assertEqual(execution_request.category, ExecutionCategory.BUILDING_CONTROL)
        self.assertIsNotNone(execution_request.created_at)
        self.assertIsNotNone(execution_request.approved_at)
        self.assertIsNotNone(execution_request.dispatched_at)
        self.assertIsNotNone(execution_request.completed_at)
        # No real action_executor supplied -- Door/Exit always FAILED
        # in practice, per the Execution Layer V1 architectural review.
        self.assertEqual(execution_request.status, ExecutionStatus.FAILED)
        self.assertFalse(execution_request.result_confirmed)

    def test_pending_request_has_created_at_only(self):

        controller = make_control_controller()
        request = ControlRequest(
            system_type=ControlSystemType.DOOR, target_id="door-1", requested_action=ControlAction.OPEN,
            reason="hazard", source=RequestSource.OPERATOR,
        )
        controller.submit(request)

        requests = building_control_adapter.build_execution_requests(controller)

        self.assertIsNotNone(requests[0].created_at)
        self.assertIsNone(requests[0].approved_at)
        self.assertIsNone(requests[0].dispatched_at)
        self.assertIsNone(requests[0].completed_at)
        self.assertEqual(requests[0].status, ExecutionStatus.PENDING_APPROVAL)

    def test_state_only_system_confirms_correctly(self):

        controller = make_control_controller()
        request = ControlRequest(
            system_type=ControlSystemType.SMOKE_EXHAUST, target_id="z1", requested_action=ControlAction.ACTIVATE,
            reason="hazard", source=RequestSource.OPERATOR,
        )
        controller.submit(request)
        controller.approve(request.request_id)

        requests = building_control_adapter.build_execution_requests(controller)

        self.assertEqual(requests[0].status, ExecutionStatus.CONFIRMED)
        self.assertTrue(requests[0].result_confirmed)

    def test_source_recommendation_id_from_advisory_adapter_is_tagged_correctly(self):

        controller = make_control_controller()
        request = ControlRequest(
            system_type=ControlSystemType.DOOR, target_id="door-1", requested_action=ControlAction.CLOSE,
            reason="hazard", source=RequestSource.ADVISORY_ADAPTER, source_recommendation_id="rec-deadbeef00000000",
        )
        controller.submit(request)

        requests = building_control_adapter.build_execution_requests(controller)

        self.assertEqual(requests[0].originating_recommendation_id, "rec-deadbeef00000000")
        self.assertEqual(requests[0].recommendation_id_provenance, RecommendationIdProvenance.ADVISORY_SYSTEM)

    def test_none_controller_produces_no_requests(self):

        self.assertEqual(building_control_adapter.build_execution_requests(None), ())


if __name__ == "__main__":
    unittest.main()
