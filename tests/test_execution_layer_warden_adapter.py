import unittest

from recommendation_layer.models import Recommendation, RecommendationSet, RecommendationType

from execution_layer.adapters import warden_adapter
from execution_layer.models import ExecutionCategory, ExecutionStatus, RecommendationIdProvenance

from tests.execution_layer_fixtures import make_warden_controller


def make_recommendation(type_=RecommendationType.WARDEN_DISPATCH, recommendation_id="rec-1", zone_id="zone-1"):

    return Recommendation(
        recommendation_id=recommendation_id, type=type_, affected_zones=(zone_id,) if zone_id else (),
        recommended_action="Dispatch a warden", confidence=0.75,
    )


class WardenAdapterTranslateTests(unittest.TestCase):

    def test_translate_warden_dispatch_carries_real_recommendation_id(self):

        recommendation = make_recommendation()

        request = warden_adapter.translate(recommendation)

        self.assertIsNotNone(request)
        self.assertEqual(request.source_recommendation_id, "rec-1")
        self.assertEqual(request.zone_id, "zone-1")

    def test_translate_non_warden_type_returns_none(self):

        recommendation = make_recommendation(type_=RecommendationType.OCCUPANT_ROUTING)

        self.assertIsNone(warden_adapter.translate(recommendation))

    def test_translate_recommendation_set_filters_by_type(self):

        recommendation_set = RecommendationSet(timestamp=1.0, recommendations=(
            make_recommendation(type_=RecommendationType.WARDEN_DISPATCH, recommendation_id="rec-a"),
            make_recommendation(type_=RecommendationType.OCCUPANT_ROUTING, recommendation_id="rec-b"),
        ))

        requests = warden_adapter.translate_recommendation_set(recommendation_set)

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].source_recommendation_id, "rec-a")

    def test_translate_recommendation_set_none_produces_no_requests(self):

        self.assertEqual(warden_adapter.translate_recommendation_set(None), ())


class WardenAdapterReadTests(unittest.TestCase):

    def test_confirmed_request_has_real_recommendation_id_and_all_timestamps(self):

        controller = make_warden_controller()
        recommendation = make_recommendation()
        request = warden_adapter.translate(recommendation)
        controller.submit(request)
        controller.approve(request.request_id)

        execution_requests = warden_adapter.build_execution_requests(controller)

        self.assertEqual(len(execution_requests), 1)
        execution_request = execution_requests[0]

        self.assertEqual(execution_request.category, ExecutionCategory.WARDEN_NOTIFICATION)
        self.assertEqual(execution_request.status, ExecutionStatus.CONFIRMED)
        self.assertEqual(execution_request.originating_recommendation_id, "rec-1")
        self.assertEqual(execution_request.recommendation_id_provenance, RecommendationIdProvenance.RECOMMENDATION_LAYER)
        self.assertIsNotNone(execution_request.created_at)
        self.assertIsNotNone(execution_request.approved_at)
        self.assertIsNotNone(execution_request.dispatched_at)
        self.assertIsNotNone(execution_request.completed_at)

    def test_none_controller_produces_no_requests(self):

        self.assertEqual(warden_adapter.build_execution_requests(None), ())


if __name__ == "__main__":
    unittest.main()
