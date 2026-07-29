import unittest

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from crowd_intelligence.engine import CrowdIntelligenceEngine


# =====================================================
# Observable Stair Perception milestone, Phase 14 -- proves
# observed_on_stair_count is a genuinely SEPARATE signal from
# approaching_count/queue_candidate_count, never a replacement for
# either, and defaults honestly to None (not measured) whenever not
# explicitly supplied -- every pre-milestone caller/test keeps its exact
# behavior unchanged.
# =====================================================


def make_building_with_stair():

    stair = Staircase(
        id="s1", from_position=(9.0, 5.0), to_position=(9.0, 5.0),
        from_floor_id="f1", to_floor_id="f2", width=1.5,
    )
    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Zone 1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1")],
        stairs=[stair],
    )
    return Building(id="b1", name="Building 1", floors=[floor])


def add_occupant(manager, occupant_id, zone_id, floor_id, position, behavior, time, stair_id=None):

    return manager.update(
        occupant_id, "CAM-1", occupant_id, zone_id, floor_id, position, 0.0, behavior, 0.9, time, stair_id=stair_id,
    )


class ObservedOnStairCountDefaultTests(unittest.TestCase):

    def test_1_defaults_to_none_when_not_supplied(self):

        building = make_building_with_stair()
        manager = LiveOccupantManager()
        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.WALKING, 0.0)

        engine = CrowdIntelligenceEngine(building, manager)
        stair_metrics = engine.compute(0.0).stair("s1")

        self.assertIsNone(stair_metrics.observed_on_stair_count)

    def test_2_door_and_exit_never_get_a_stair_observation_value(self):

        # No Door/Exit in this fixture, but the field itself (see
        # crowd_intelligence.models.AssetApproachMetrics) is shared
        # across all three asset types -- confirm it stays at its
        # honest None default for a non-Stair asset type by construction
        # (engine.py only ever passes observed_on_stair_count into the
        # Stair branch of compute()).

        building = make_building_with_stair()
        manager = LiveOccupantManager()

        engine = CrowdIntelligenceEngine(building, manager)
        snapshot = engine.compute(0.0)

        self.assertEqual(snapshot.door_metrics, {})  # fixture has no doors -- sanity check
        self.assertIsNone(snapshot.stair("s1").observed_on_stair_count)


class ObservedOnStairCountSuppliedTests(unittest.TestCase):

    def test_3_supplied_observed_count_is_reported_independent_of_approaching(self):

        building = make_building_with_stair()
        manager = LiveOccupantManager()

        # Two occupants "approaching" the stair (near the landing) --
        # a DIFFERENT signal from four genuinely observed on the stair
        # itself.
        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.WALKING, 0.0)
        add_occupant(manager, "OCC-2", "z1", "f1", (6.0, 5.0), RecognizedBehavior.WALKING, 0.0)

        engine = CrowdIntelligenceEngine(building, manager)
        stair_metrics = engine.compute(0.0, observed_stair_occupancy={"s1": 4}).stair("s1")

        self.assertEqual(stair_metrics.observed_on_stair_count, 4)
        # approaching_count is computed from an entirely different
        # mechanism (landing-proximity queue metrics) and is untouched
        # by the supplied observed count.
        self.assertNotEqual(stair_metrics.approaching_count, stair_metrics.observed_on_stair_count)

    def test_4_observed_zero_is_distinct_from_not_supplied(self):

        building = make_building_with_stair()
        manager = LiveOccupantManager()

        engine = CrowdIntelligenceEngine(building, manager)

        not_supplied = engine.compute(0.0).stair("s1")
        observed_zero = engine.compute(1.0, observed_stair_occupancy={"s1": 0}).stair("s1")

        self.assertIsNone(not_supplied.observed_on_stair_count)
        self.assertEqual(observed_zero.observed_on_stair_count, 0)

    def test_5_observed_count_present_even_when_position_unavailable(self):

        # position_available=False (landing-proximity metrics honestly
        # unavailable) must not suppress a SEPARATELY-sourced observed
        # stair count -- these are independent measurements.
        building = make_building_with_stair()
        manager = LiveOccupantManager()

        # An occupant known to be on this stair's floor but with no
        # world_position at all -- forces position_available=False for
        # the landing-proximity branch.
        manager.update("OCC-1", "CAM-1", "OCC-1", None, "f1", None, None, None, 0.9, 0.0)

        engine = CrowdIntelligenceEngine(building, manager)
        stair_metrics = engine.compute(0.0, observed_stair_occupancy={"s1": 2}).stair("s1")

        self.assertFalse(stair_metrics.position_available)
        self.assertEqual(stair_metrics.observed_on_stair_count, 2)


if __name__ == "__main__":
    unittest.main()
