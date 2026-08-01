import unittest

from dynamic_signage.models import SignageInstruction

from execution_layer.adapters import signage_adapter
from execution_layer.models import ExecutionCategory, ExecutionStatus, RecommendationIdProvenance

from tests.execution_layer_fixtures import make_signage_controller


class SignageAdapterTests(unittest.TestCase):

    def test_approved_instruction_confirms_with_all_timestamps(self):

        controller = make_signage_controller()
        instruction = SignageInstruction(sign_id="sign-1", zone_id="z1", signage_revision=1, timestamp=1.0)
        controller.submit(instruction, 1.0)
        controller.approve("sign-1", 1, 2.0)

        requests = signage_adapter.build_execution_requests(controller)

        self.assertEqual(len(requests), 1)
        request = requests[0]

        self.assertEqual(request.category, ExecutionCategory.DYNAMIC_SIGNAGE)
        self.assertEqual(request.status, ExecutionStatus.CONFIRMED)
        self.assertEqual(request.created_at, 1.0)
        self.assertEqual(request.approved_at, 2.0)
        self.assertEqual(request.dispatched_at, 2.0)
        self.assertEqual(request.completed_at, 2.0)

    def test_recommendation_id_provenance_is_always_unavailable(self):

        # SignageInstruction carries no source_recommendation_id field
        # at all today -- confirmed during the Execution Layer V1
        # architectural review.
        controller = make_signage_controller()
        instruction = SignageInstruction(sign_id="sign-1", zone_id="z1", signage_revision=1, timestamp=1.0)
        controller.submit(instruction, 1.0)

        requests = signage_adapter.build_execution_requests(controller)

        self.assertIsNone(requests[0].originating_recommendation_id)
        self.assertEqual(requests[0].recommendation_id_provenance, RecommendationIdProvenance.UNAVAILABLE)

    def test_execution_request_id_is_the_sign_revision_key(self):

        controller = make_signage_controller()
        instruction = SignageInstruction(sign_id="sign-1", zone_id="z1", signage_revision=3, timestamp=1.0)
        controller.submit(instruction, 1.0)

        requests = signage_adapter.build_execution_requests(controller)

        self.assertEqual(requests[0].execution_request_id, "sign-1::3")

    def test_none_controller_produces_no_requests(self):

        self.assertEqual(signage_adapter.build_execution_requests(None), ())


if __name__ == "__main__":
    unittest.main()
