import unittest

from live_system.event_bus import EventBus, EventType
from live_system.orchestrator import LiveOrchestrator
from live_system.evacuation_guidance_gateway import EngineEvacuationGuidanceGateway
from live_system.evacuation_signage_gateway import EngineEvacuationSignageGateway

from evacuation_guidance.engine import EvacuationGuidanceEngine

from dynamic_signage.planner import DynamicSignagePlanner

from tests.dynamic_signage_fixtures import make_recommendation_snapshot, make_sign, make_signage_building, make_signage_graph


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 27 -- test matrix
# items 53-56: signage events emitted once on actual change, no event
# spam for unchanged state, conflict transition events emitted once.
# =====================================================


class FakeRecommendationGateway:

    def __init__(self):
        self._snapshot = None

    def set(self, snapshot):
        self._snapshot = snapshot

    def compute(self, *args, **kwargs):
        return self._snapshot


def _make_orchestrator(building, graph, signs):

    guidance_engine = EvacuationGuidanceEngine(building, graph)
    signage_planner = DynamicSignagePlanner(building)

    event_bus = EventBus()

    orchestrator = LiveOrchestrator(
        event_bus=event_bus,
        evacuation_recommendation_gateway=FakeRecommendationGateway(),
        evacuation_guidance_gateway=EngineEvacuationGuidanceGateway(guidance_engine),
        evacuation_signage_gateway=EngineEvacuationSignageGateway(signage_planner, sign_manager=None),
        interval_seconds=1.0,
    )

    orchestrator.evacuation_recommendation_gateway.set(
        make_recommendation_snapshot("z2", "f1", "EXIT-1", timestamp=0.0),
    )

    orchestrator.start()

    return orchestrator, event_bus


class SignageEventTests(unittest.TestCase):

    def test_plan_updated_emitted_every_cycle_signage_configured(self):

        building = make_signage_building()
        graph = make_signage_graph(building)
        sign = make_sign("SIGN-1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0)

        orchestrator, event_bus = _make_orchestrator(building, graph, [sign])

        orchestrator.run_cycle(0.0)
        orchestrator.run_cycle(1.0)

        self.assertEqual(len(event_bus.history_of(EventType.SIGNAGE_PLAN_UPDATED)), 2)

    def test_no_instruction_changed_event_when_unchanged(self):

        building = make_signage_building()
        graph = make_signage_graph(building)
        sign = make_sign("SIGN-1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0)

        orchestrator, event_bus = _make_orchestrator(building, graph, [sign])

        orchestrator.run_cycle(0.0)
        orchestrator.run_cycle(1.0)
        orchestrator.run_cycle(2.0)

        # First cycle: UNAVAILABLE -> ACTIVE is a "recovered" transition,
        # not a "changed" one; every cycle after that is identical, so
        # SIGNAGE_INSTRUCTION_CHANGED must never fire at all here.
        self.assertEqual(len(event_bus.history_of(EventType.SIGNAGE_INSTRUCTION_CHANGED)), 0)

    def test_recovered_then_changed_on_genuine_transitions(self):

        building = make_signage_building()
        graph = make_signage_graph(building)

        signage_planner_probe = DynamicSignagePlanner(building)
        sign = make_sign("SIGN-1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0)

        guidance_engine = EvacuationGuidanceEngine(building, graph)
        event_bus = EventBus()

        recommendation_gateway = FakeRecommendationGateway()

        orchestrator = LiveOrchestrator(
            event_bus=event_bus,
            evacuation_recommendation_gateway=recommendation_gateway,
            evacuation_guidance_gateway=EngineEvacuationGuidanceGateway(guidance_engine),
            evacuation_signage_gateway=EngineEvacuationSignageGateway(signage_planner_probe, sign_manager=None),
            interval_seconds=1.0,
        )
        orchestrator.start()

        def compute(time, guidance_snapshot=None, signs=None):
            return signage_planner_probe.compute(time, guidance_snapshot, [sign])

        orchestrator.evacuation_signage_gateway.compute = compute

        # Cycle 0 -- no recommendation at all yet: the sign's own
        # instruction is honestly UNAVAILABLE.
        recommendation_gateway.set(None)
        orchestrator.run_cycle(0.0)

        # Cycle 1 -- a real recommendation now exists: ACTIVE, a genuine
        # UNAVAILABLE -> ACTIVE recovery.
        recommendation_gateway.set(make_recommendation_snapshot("z2", "f1", "EXIT-1", timestamp=1.0))
        orchestrator.run_cycle(1.0)

        # Cycle 2 -- same recommendation, but the sign's own orientation
        # changed: still ACTIVE, but a genuinely different indication.
        sign.orientation = 90.0
        orchestrator.run_cycle(2.0)

        self.assertEqual(len(event_bus.history_of(EventType.SIGNAGE_RECOVERED)), 1)
        self.assertEqual(len(event_bus.history_of(EventType.SIGNAGE_INSTRUCTION_CHANGED)), 1)


if __name__ == "__main__":
    unittest.main()
