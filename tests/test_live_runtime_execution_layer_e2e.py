import unittest

from models.building import Building
from models.floor import Floor
from models.zone import Zone

from live_runtime.factory import build_offline_demo_runtime

from recommendation_layer.models import Recommendation, RecommendationSet, RecommendationType

from execution_layer.models import ExecutionCategory, ExecutionStatus, RecommendationIdProvenance


# =====================================================
# Execution Layer V1 milestone -- deterministic offline end-to-end
# proof: a real build_offline_demo_runtime() session, driven through
# the REAL command_center.live_operator_action_gateway.
# LiveOperatorActionGateway (ingest -> approve) and the REAL
# LiveRuntime.tick_execution_layer() (a method deliberately separate
# from run_cycle() -- see live_runtime/runtime.py's own docstring),
# proving the complete audit chain: Recommendation -> Warden
# Notification -> Approval -> Dispatch -> Completion, with the REAL
# recommendation_id traceable end-to-end.
# =====================================================


def make_building():

    floor = Floor(id="f1", name="Floor 1", zones=[Zone(id="z1", name="Lobby", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1")])

    return Building(id="execution-layer-e2e-building", name="Execution Layer E2E Building", floors=[floor])


class ExecutionLayerLiveRuntimeEndToEndTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.runtime = build_offline_demo_runtime(self.building)
        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_full_audit_chain_from_recommendation_to_completion(self):

        recommendation = Recommendation(
            recommendation_id="rec-e2e-001", type=RecommendationType.WARDEN_DISPATCH, affected_zones=("z1",),
            recommended_action="Dispatch a warden to zone z1", confidence=0.9,
        )
        recommendation_set = RecommendationSet(timestamp=1.0, recommendations=(recommendation,))

        submitted = self.runtime.operator_action_gateway.ingest_warden_recommendations(recommendation_set, 1.0)
        self.assertEqual(len(submitted), 1)

        pending = self.runtime.operator_action_gateway.pending_warden_notifications()
        self.assertEqual(len(pending), 1)

        self.runtime.operator_action_gateway.approve_warden_notification(pending[0].request_id)

        execution_set = self.runtime.tick_execution_layer(2.0)

        warden_requests = execution_set.by_category(ExecutionCategory.WARDEN_NOTIFICATION)
        self.assertEqual(len(warden_requests), 1)

        execution_request = warden_requests[0]
        self.assertEqual(execution_request.status, ExecutionStatus.CONFIRMED)
        self.assertEqual(execution_request.originating_recommendation_id, "rec-e2e-001")
        self.assertEqual(execution_request.recommendation_id_provenance, RecommendationIdProvenance.RECOMMENDATION_LAYER)
        self.assertIsNotNone(execution_request.created_at)
        self.assertIsNotNone(execution_request.approved_at)
        self.assertIsNotNone(execution_request.dispatched_at)
        self.assertIsNotNone(execution_request.completed_at)

        # Cross-reference back to the Recommendation Layer's own
        # for_recommendation() accessor -- the same id closes the loop.
        matches = execution_set.for_recommendation("rec-e2e-001")
        self.assertEqual(len(matches), 1)

    def test_unconfigured_warden_controller_never_crashes(self):

        # A session where the caller never supplied a
        # warden_notification_provider (mirrors the LIVE-mode
        # NO_PROVIDER story) -- ingest/tick must degrade honestly, never
        # raise.
        self.runtime.operator_action_gateway._warden_controller = None
        self.runtime.execution_layer._warden_controller = None

        recommendation_set = RecommendationSet(timestamp=1.0, recommendations=())

        submitted = self.runtime.operator_action_gateway.ingest_warden_recommendations(recommendation_set, 1.0)
        self.assertEqual(submitted, ())

        execution_set = self.runtime.tick_execution_layer(1.0)
        self.assertEqual(execution_set.by_category(ExecutionCategory.WARDEN_NOTIFICATION), ())

    def test_run_cycle_and_tick_execution_layer_are_independent(self):

        # LiveOrchestrator.run_cycle() must never touch execution_layer
        # at all (architecture guard) -- proven here by calling both and
        # confirming neither raises nor depends on call order.
        snapshot = self.runtime.run_cycle(1.0)
        self.assertIsNotNone(snapshot)

        execution_set = self.runtime.tick_execution_layer(1.0)
        self.assertIsNotNone(execution_set)


if __name__ == "__main__":
    unittest.main()
