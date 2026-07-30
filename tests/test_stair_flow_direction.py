import unittest

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase

from live_occupants.manager import LiveOccupantManager

from stair_flow.compute import compute_stair_flow_snapshot
from stair_flow.direction import derive_direction
from stair_flow.models import TrafficDirection


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone -- Phase 7
# deterministic, offline unit tests. Direction is grounded in
# models.building.Building.floor_elevation()'s own genuine vertical
# ordering (accumulated Floor.height via display_order), never assumed
# from from_floor_id/to_floor_id naming and never derived from screen-
# space/image-Y movement.
# =====================================================


def make_three_floor_building():

    building = Building(name="Test Building")

    ground = Floor(name="Ground", display_order=0, height=3.0)
    first = Floor(name="First", display_order=1, height=4.0)
    second = Floor(name="Second", display_order=2, height=3.0)

    building.add_floor(ground)
    building.add_floor(first)
    building.add_floor(second)

    return building, ground, first, second


class DirectDerivationTests(unittest.TestCase):

    def test_1_entering_from_lower_floor_is_up(self):

        building, ground, first, _second = make_three_floor_building()
        stair = Staircase(from_floor_id=ground.id, to_floor_id=first.id)

        self.assertEqual(derive_direction(building, stair, ground.id), TrafficDirection.UP)

    def test_2_entering_from_higher_floor_is_down(self):

        building, ground, first, _second = make_three_floor_building()
        stair = Staircase(from_floor_id=ground.id, to_floor_id=first.id)

        self.assertEqual(derive_direction(building, stair, first.id), TrafficDirection.DOWN)

    def test_3_from_floor_id_does_not_mean_bottom(self):

        # A Staircase authored "backwards" (from_floor_id is the HIGHER
        # floor) must still resolve correctly from actual elevation, not
        # field naming.
        building, ground, first, _second = make_three_floor_building()
        stair = Staircase(from_floor_id=first.id, to_floor_id=ground.id)

        self.assertEqual(derive_direction(building, stair, first.id), TrafficDirection.DOWN)
        self.assertEqual(derive_direction(building, stair, ground.id), TrafficDirection.UP)

    def test_4_multi_floor_building_second_to_first_is_down(self):

        building, _ground, first, second = make_three_floor_building()
        stair = Staircase(from_floor_id=first.id, to_floor_id=second.id)

        self.assertEqual(derive_direction(building, stair, second.id), TrafficDirection.DOWN)
        self.assertEqual(derive_direction(building, stair, first.id), TrafficDirection.UP)

    def test_5_unknown_when_entered_floor_id_is_none(self):

        building, ground, first, _second = make_three_floor_building()
        stair = Staircase(from_floor_id=ground.id, to_floor_id=first.id)

        self.assertEqual(derive_direction(building, stair, None), TrafficDirection.UNKNOWN)

    def test_6_unknown_when_entered_floor_is_neither_end(self):

        building, ground, first, second = make_three_floor_building()
        stair = Staircase(from_floor_id=ground.id, to_floor_id=first.id)

        self.assertEqual(derive_direction(building, stair, second.id), TrafficDirection.UNKNOWN)

    def test_7_unknown_when_ends_have_equal_elevation(self):

        building = Building(name="Degenerate Building")
        a = Floor(name="A", display_order=0, height=3.0)
        building.add_floor(a)

        stair = Staircase(from_floor_id=a.id, to_floor_id=a.id)  # both ends the same floor

        self.assertEqual(derive_direction(building, stair, a.id), TrafficDirection.UNKNOWN)

    def test_8_unknown_when_floor_not_found_in_building(self):

        building, ground, first, _second = make_three_floor_building()
        stair = Staircase(from_floor_id=ground.id, to_floor_id="NEVER-ADDED-FLOOR")

        self.assertEqual(derive_direction(building, stair, ground.id), TrafficDirection.UNKNOWN)


class EndToEndDirectionViaSnapshotTests(unittest.TestCase):

    def test_9_upward_traversal_counted_in_snapshot(self):

        building, ground, first, _second = make_three_floor_building()
        stair = Staircase(from_floor_id=ground.id, to_floor_id=first.id)
        stair.id = "S1"
        ground.add_stair(stair)

        manager = LiveOccupantManager()
        manager.update("OCC-1", "CAM-1", "T1", "ZONE-G", ground.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-1", "CAM-1", "T1", None, ground.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=1.0)
        metrics = snapshot.for_stair("S1")

        self.assertEqual(metrics.entries, 1)
        self.assertEqual(metrics.upward_count, 1)
        self.assertEqual(metrics.downward_count, 0)
        self.assertEqual(metrics.unknown_direction_count, 0)

    def test_10_downward_traversal_counted_in_snapshot(self):

        building, ground, first, _second = make_three_floor_building()
        stair = Staircase(from_floor_id=ground.id, to_floor_id=first.id)
        stair.id = "S1"
        ground.add_stair(stair)

        manager = LiveOccupantManager()
        manager.update("OCC-2", "CAM-1", "T1", "ZONE-1", first.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-2", "CAM-1", "T1", None, first.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=1.0)
        metrics = snapshot.for_stair("S1")

        self.assertEqual(metrics.entries, 1)
        self.assertEqual(metrics.upward_count, 0)
        self.assertEqual(metrics.downward_count, 1)

    def test_11_direction_counts_always_sum_to_entries(self):

        building, ground, first, second = make_three_floor_building()
        stair = Staircase(from_floor_id=ground.id, to_floor_id=first.id)
        stair.id = "S1"
        ground.add_stair(stair)

        manager = LiveOccupantManager()

        # OCC-3 has a genuine floor entry, OCC-4 has NO position sample
        # at the entry instant (world_position=None) -> UNKNOWN direction.
        manager.update("OCC-3", "CAM-1", "T1", "ZONE-G", ground.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-3", "CAM-1", "T1", None, ground.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        manager.update("OCC-4", "CAM-1", "T2", "ZONE-G", ground.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-4", "CAM-1", "T2", None, None, None, None, None, 0.9, 1.0, stair_id="S1",
        )

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=1.0)
        metrics = snapshot.for_stair("S1")

        self.assertEqual(metrics.entries, 2)
        self.assertEqual(
            metrics.upward_count + metrics.downward_count + metrics.unknown_direction_count, metrics.entries,
        )
        self.assertEqual(metrics.unknown_direction_count, 1)

    def test_12_exit_events_never_carry_a_direction(self):

        building, ground, first, _second = make_three_floor_building()
        stair = Staircase(from_floor_id=ground.id, to_floor_id=first.id)
        stair.id = "S1"
        ground.add_stair(stair)

        manager = LiveOccupantManager()
        manager.update("OCC-5", "CAM-1", "T1", "ZONE-G", ground.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-5", "CAM-1", "T1", None, ground.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        manager.update("OCC-5", "CAM-1", "T1", "ZONE-1", first.id, (3.0, 3.0), 0.5, None, 0.9, 2.0)

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=2.0)

        exit_events = [e for e in snapshot.events_for_stair("S1") if e.stair_id == "S1"]
        exit_only = [e for e in exit_events if e.direction != TrafficDirection.UNKNOWN]

        # The exit event itself must never carry a derived direction --
        # only the paired entry event (already asserted UP above) does.
        for event in exit_events:
            if event.timestamp == 2.0:
                self.assertEqual(event.direction, TrafficDirection.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
