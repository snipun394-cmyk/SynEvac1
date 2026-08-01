import unittest

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from live_runtime.factory import build_live_runtime

from recommendation_layer.models import RecommendationType


# =====================================================
# The Recommendation Layer milestone -- deterministic offline
# end-to-end proof that the new stage runs inside a REAL run_cycle()
# (via build_live_runtime()'s own default-constructed engines, exactly
# the codebase's established "engine-shaped, always wired" convention)
# without a camera/YOLO pipeline -- an occupant is seeded directly via
# LiveOccupantManager.update(), the same minimal-seeding precedent
# tests/test_evacuation_recommendation.py's own engine-level unit tests
# already establish, one layer up (a real live cycle, not just the
# engine in isolation).
# =====================================================


def make_building():

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Lobby", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1")],
        doors=[],
        exits=[Exit(id="EXIT-1", name="Lobby Exit", floor_id="f1", zone_id="z1")],
    )

    return Building(id="recommendation-layer-e2e-building", name="Recommendation Layer E2E Building", floors=[floor])


class RecommendationLayerLiveRuntimeEndToEndTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.runtime = build_live_runtime(self.building)
        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_recommendation_set_populates_and_evolves_across_cycles(self):

        self.runtime.live_occupant_manager.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0,
        )

        self.runtime.run_cycle(0.0)

        first_set = self.runtime.orchestrator.latest_recommendation_set
        self.assertIsNotNone(first_set)

        routing = first_set.by_type(RecommendationType.OCCUPANT_ROUTING)
        self.assertEqual(len(routing), 1)
        self.assertEqual(routing[0].affected_zones, ("z1",))
        first_id = routing[0].recommendation_id

        self.runtime.run_cycle(1.0)

        second_set = self.runtime.orchestrator.latest_recommendation_set
        second_routing = second_set.by_type(RecommendationType.OCCUPANT_ROUTING)

        self.assertEqual(len(second_routing), 1)
        self.assertEqual(second_routing[0].recommendation_id, first_id)

    def test_unconfigured_gateway_never_crashes_the_cycle(self):

        self.runtime.orchestrator.recommendation_layer_gateway = None

        snapshot = self.runtime.run_cycle(0.0)

        self.assertIsNone(snapshot.recommendation_set)


if __name__ == "__main__":
    unittest.main()
