import unittest

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase

from live_occupants.manager import LiveOccupantManager

from stair_flow.compute import compute_stair_flow_snapshot
from stair_flow.models import StairFlowEventType, TrafficDirection


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone -- Phase
# 3/8. Proves that two cameras genuinely covering the same physical
# Stair, both observing the SAME canonical occupant traversing it in the
# same cycle, produce exactly ONE entry event, ONE exit event, and ONE
# occupancy count -- never doubled.
#
# Mirrors live_camera_pipeline.pipeline.LiveCameraPipeline.run_cycle()'s
# own real call pattern EXACTLY: within one cycle (one `time` value),
# live_occupant_manager.update() is called once PER CAMERA THAT SAW this
# occupant this cycle, each with its OWN camera_id/track_id but the SAME
# already-resolved global occupant_id (cross-camera identity resolution
# already happened upstream -- this test does not re-implement it, it
# reuses LiveOccupantManager's own real idempotent-same-cycle-update
# behavior, exactly as documented on live_occupants.occupant.LiveOccupant.
# current_stair_id and traced in docs/architecture/
# stair_flow_intelligence.md Sec "Transition identity").
# =====================================================


def make_building_with_stair():

    building = Building(name="Multi-Camera Test Building")

    floor_1 = Floor(name="Floor 1", display_order=0, height=3.0)
    floor_2 = Floor(name="Floor 2", display_order=1, height=3.0)

    building.add_floor(floor_1)
    building.add_floor(floor_2)

    stair = Staircase(name="S1", from_floor_id=floor_1.id, to_floor_id=floor_2.id)
    stair.id = "S1"
    floor_1.add_stair(stair)

    return building, floor_1, floor_2, stair


class MultiCameraDedupTests(unittest.TestCase):

    def test_two_cameras_covering_one_traversal_produce_exactly_one_of_each(self):

        building, floor_1, floor_2, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        # Cycle 0 -- occupant in Zone A, seen by CAM-A only.
        manager.update(
            "TRACK-17", "CAM-A", "T-A1", "ZONE-A", floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0,
        )

        # Cycle 1 -- occupant enters STAIR-1. BOTH CAM-A and CAM-B
        # independently detect them this SAME cycle (their own
        # WorldProjector.project() calls both resolve stair_id="S1"),
        # exactly the pipeline.run_cycle() loop's own zip(resolved,
        # pending_occupant_updates) pattern -- update() called TWICE,
        # same occupant_id, same timestamp.
        manager.update(
            "TRACK-17", "CAM-A", "T-A1", None, floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        manager.update(
            "TRACK-17", "CAM-B", "T-B1", None, floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )

        # Cycle 2 -- still on the stair, both cameras confirm again.
        manager.update(
            "TRACK-17", "CAM-A", "T-A1", None, floor_1.id, (2.5, 2.5), 0.5, None, 0.9, 2.0, stair_id="S1",
        )
        manager.update(
            "TRACK-17", "CAM-B", "T-B1", None, floor_1.id, (2.5, 2.5), 0.5, None, 0.9, 2.0, stair_id="S1",
        )

        occupancy_mid_traversal = manager.canonical_occupancy(2.0)
        self.assertEqual(occupancy_mid_traversal.stair_count("S1"), 1)

        # Cycle 3 -- occupant exits onto Zone B, again both cameras agree.
        manager.update(
            "TRACK-17", "CAM-A", "T-A1", "ZONE-B", floor_2.id, (3.0, 3.0), 0.5, None, 0.9, 3.0,
        )
        manager.update(
            "TRACK-17", "CAM-B", "T-B1", "ZONE-B", floor_2.id, (3.0, 3.0), 0.5, None, 0.9, 3.0,
        )

        self.assertEqual(len(manager.all_occupants()), 1)  # one canonical occupant throughout

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=3.0)
        metrics = snapshot.for_stair("S1")

        self.assertEqual(metrics.entries, 1)
        self.assertEqual(metrics.exits, 1)
        self.assertEqual(metrics.net_flow, 0)
        self.assertEqual(metrics.upward_count, 1)
        self.assertEqual(metrics.downward_count, 0)

        entry_events = [e for e in snapshot.events if e.event_type == StairFlowEventType.ENTERED_STAIR]
        exit_events = [e for e in snapshot.events if e.event_type == StairFlowEventType.EXITED_STAIR]

        self.assertEqual(len(entry_events), 1)
        self.assertEqual(len(exit_events), 1)
        self.assertEqual(entry_events[0].direction, TrafficDirection.UP)

    def test_three_cameras_all_observing_the_same_entry_still_count_once(self):

        building, floor_1, floor_2, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        manager.update("TRACK-9", "CAM-A", "T-A", "ZONE-A", floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)

        for camera_id, track_id in (("CAM-A", "T-A"), ("CAM-B", "T-B"), ("CAM-C", "T-C")):
            manager.update(
                "TRACK-9", camera_id, track_id, None, floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
            )

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=1.0)
        self.assertEqual(snapshot.for_stair("S1").entries, 1)


if __name__ == "__main__":
    unittest.main()
