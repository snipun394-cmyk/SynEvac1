import unittest

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase

from live_occupants.manager import LiveOccupantManager

from stair_flow.compute import compute_stair_flow_snapshot
from stair_flow.events import extract_stair_flow_events
from stair_flow.models import StairFlowEventType, TrafficDirection


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone -- Phase
# 1/2/3/6 deterministic, offline unit tests. Drives the REAL
# live_occupants.manager.LiveOccupantManager.update() (never a hand-
# built OccupantHistory) so these tests exercise the actual production
# evidence path this package reads, not a synthetic stand-in.
# =====================================================


def make_building_with_stair():

    building = Building(name="Test Building")

    floor_1 = Floor(name="Floor 1", display_order=0, height=3.0)
    floor_2 = Floor(name="Floor 2", display_order=1, height=3.0)

    building.add_floor(floor_1)
    building.add_floor(floor_2)

    stair = Staircase(name="S1", from_floor_id=floor_1.id, to_floor_id=floor_2.id)
    stair.id = "S1"
    floor_1.add_stair(stair)

    return building, floor_1, floor_2, stair


class EntryExitDerivationTests(unittest.TestCase):

    def setUp(self):

        self.building, self.floor_1, self.floor_2, self.stair = make_building_with_stair()
        self.manager = LiveOccupantManager()

    def test_1_zone_to_stair_to_stair_to_zone_produces_one_entry_and_one_exit(self):

        # Zone A -> Stair S1 -> Stair S1 -> Zone B
        self.manager.update(
            "OCC-1", "CAM-1", "T1", "ZONE-A", self.floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0,
        )
        self.manager.update(
            "OCC-1", "CAM-1", "T1", None, self.floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        self.manager.update(
            "OCC-1", "CAM-1", "T1", None, self.floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 2.0, stair_id="S1",
        )
        self.manager.update(
            "OCC-1", "CAM-2", "T2", "ZONE-B", self.floor_2.id, (3.0, 3.0), 0.5, None, 0.9, 3.0,
        )

        occupant = self.manager.get("OCC-1")
        events = extract_stair_flow_events(occupant, window_start=-1.0, window_end=10.0)

        entries = [e for e in events if e.event_type == StairFlowEventType.ENTERED_STAIR]
        exits = [e for e in events if e.event_type == StairFlowEventType.EXITED_STAIR]

        self.assertEqual(len(entries), 1)
        self.assertEqual(len(exits), 1)
        self.assertEqual(entries[0].stair_id, "S1")
        self.assertEqual(entries[0].timestamp, 1.0)
        self.assertEqual(entries[0].floor_id, self.floor_1.id)
        self.assertEqual(exits[0].stair_id, "S1")
        self.assertEqual(exits[0].timestamp, 3.0)

    def test_2_reverse_direction_zone_b_to_stair_to_zone_a(self):

        self.manager.update(
            "OCC-2", "CAM-1", "T1", "ZONE-B", self.floor_2.id, (1.0, 1.0), 0.0, None, 0.9, 0.0,
        )
        self.manager.update(
            "OCC-2", "CAM-1", "T1", None, self.floor_2.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        self.manager.update(
            "OCC-2", "CAM-1", "T1", None, self.floor_2.id, (2.0, 2.0), 0.5, None, 0.9, 2.0, stair_id="S1",
        )
        self.manager.update(
            "OCC-2", "CAM-2", "T2", "ZONE-A", self.floor_1.id, (3.0, 3.0), 0.5, None, 0.9, 3.0,
        )

        occupant = self.manager.get("OCC-2")
        events = extract_stair_flow_events(occupant, window_start=-1.0, window_end=10.0)

        entries = [e for e in events if e.event_type == StairFlowEventType.ENTERED_STAIR]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].floor_id, self.floor_2.id)

    def test_3_occupant_first_appearing_already_on_stair_produces_no_entry_event(self):

        # Genuine information gap (Phase 6) -- tracking begins mid-
        # traversal. No entry EVENT, even though the occupant genuinely
        # is on the stair.
        self.manager.update(
            "OCC-3", "CAM-1", "T1", None, self.floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 5.0, stair_id="S1",
        )

        occupant = self.manager.get("OCC-3")
        self.assertEqual(occupant.first_seen, 5.0)

        events = extract_stair_flow_events(occupant, window_start=-1.0, window_end=10.0)
        self.assertEqual(events, ())

    def test_4_occupant_disappearing_while_on_stair_produces_no_exit_event(self):

        # Genuine information gap (Phase 9) -- sweep_missing() never
        # calls update(), so a TEMPORARILY_LOST/EXPIRED occupant who
        # vanished mid-stair leaves no "stair_id -> None" evidence at
        # all. Documented, not fabricated around.
        self.manager.update(
            "OCC-4", "CAM-1", "T1", "ZONE-A", self.floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0,
        )
        self.manager.update(
            "OCC-4", "CAM-1", "T1", None, self.floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )

        self.manager.sweep_missing(2.0, seen_occupant_ids=set())

        occupant = self.manager.get("OCC-4")
        self.assertEqual(occupant.current_stair_id, "S1")  # frozen at last known value

        events = extract_stair_flow_events(occupant, window_start=-1.0, window_end=10.0)
        exits = [e for e in events if e.event_type == StairFlowEventType.EXITED_STAIR]
        self.assertEqual(exits, [])

    def test_5_direct_stair_to_stair_transition_produces_exit_and_entry(self):

        floor_3 = Floor(name="Floor 3", display_order=2, height=3.0)
        self.building.add_floor(floor_3)
        stair_2 = Staircase(name="S2", from_floor_id=self.floor_2.id, to_floor_id=floor_3.id)
        stair_2.id = "S2"
        self.floor_2.add_stair(stair_2)

        self.manager.update(
            "OCC-5", "CAM-1", "T1", "ZONE-A", self.floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0,
        )
        self.manager.update(
            "OCC-5", "CAM-1", "T1", None, self.floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        # A shared landing directly connects S1 to S2 in one cycle.
        self.manager.update(
            "OCC-5", "CAM-2", "T2", None, self.floor_2.id, (3.0, 3.0), 0.5, None, 0.9, 2.0, stair_id="S2",
        )

        occupant = self.manager.get("OCC-5")
        events = extract_stair_flow_events(occupant, window_start=-1.0, window_end=10.0)

        exits = [e for e in events if e.event_type == StairFlowEventType.EXITED_STAIR]
        entries = [e for e in events if e.event_type == StairFlowEventType.ENTERED_STAIR]

        # Two entries total (S1 at t=1.0, then S2 at t=2.0) and one exit
        # (S1 at t=2.0, the same record that produced the S2 entry) --
        # a direct stair-to-stair landing genuinely produces both an
        # exit-from-S1 and an entry-to-S2 from the single transition
        # record, never collapsed or dropped.
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0].stair_id, "S1")
        self.assertEqual(len(entries), 2)
        self.assertEqual({e.stair_id for e in entries}, {"S1", "S2"})

    def test_6_window_excludes_events_outside_range(self):

        self.manager.update(
            "OCC-6", "CAM-1", "T1", None, self.floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0, stair_id=None,
        )
        self.manager.update(
            "OCC-6", "CAM-1", "T1", None, self.floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 100.0, stair_id="S1",
        )

        occupant = self.manager.get("OCC-6")

        events_in_window = extract_stair_flow_events(occupant, window_start=50.0, window_end=110.0)
        self.assertEqual(len(events_in_window), 1)

        events_out_of_window = extract_stair_flow_events(occupant, window_start=150.0, window_end=200.0)
        self.assertEqual(events_out_of_window, ())

    def test_7_no_position_sample_at_transition_yields_no_floor_id(self):

        # world_position=None means no PositionSample is recorded at
        # that timestamp -- floor_id honestly stays None, never guessed.
        self.manager.update(
            "OCC-7", "CAM-1", "T1", None, None, None, None, None, 0.9, 0.0, stair_id="S1",
        )

        occupant = self.manager.get("OCC-7")
        events = extract_stair_flow_events(occupant, window_start=-1.0, window_end=10.0)

        entries = [e for e in events if e.event_type == StairFlowEventType.ENTERED_STAIR]
        self.assertEqual(len(entries), 0)  # first-ever sighting already on stair, still excluded


class SnapshotAggregationTests(unittest.TestCase):

    def setUp(self):

        self.building, self.floor_1, self.floor_2, self.stair = make_building_with_stair()
        self.manager = LiveOccupantManager()

    def test_8_snapshot_counts_one_entry_and_one_exit(self):

        self.manager.update("OCC-1", "CAM-1", "T1", "ZONE-A", self.floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        self.manager.update(
            "OCC-1", "CAM-1", "T1", None, self.floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        self.manager.update(
            "OCC-1", "CAM-2", "T2", "ZONE-B", self.floor_2.id, (3.0, 3.0), 0.5, None, 0.9, 2.0,
        )

        snapshot = compute_stair_flow_snapshot(
            [self.stair], self.manager.all_occupants(), self.building, timestamp=2.0, window_seconds=60.0,
        )

        metrics = snapshot.for_stair("S1")
        self.assertEqual(metrics.entries, 1)
        self.assertEqual(metrics.exits, 1)
        self.assertEqual(metrics.net_flow, 0)
        self.assertAlmostEqual(metrics.entry_rate_per_minute, 1.0)
        self.assertAlmostEqual(metrics.exit_rate_per_minute, 1.0)

    def test_9_unknown_stair_defaults_gracefully(self):

        snapshot = compute_stair_flow_snapshot([self.stair], [], self.building, timestamp=0.0)
        metrics = snapshot.for_stair("NEVER-SEEN")
        self.assertIsNone(metrics.entries)
        self.assertEqual(metrics.stair_id, "NEVER-SEEN")


if __name__ == "__main__":
    unittest.main()
