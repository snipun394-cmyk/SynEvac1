import unittest

from behavior_recognition.observation import RecognizedBehavior

from live_system.event_bus import EventBus, EventType

from live_occupants.manager import LiveOccupantManager
from live_occupants.state import OccupantStatus


# =====================================================
# Live Occupant Digital Twin milestone, Phase 10 -- deterministic,
# offline unit tests. No randomness anywhere in this file.
# =====================================================


class FakeExit:

    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point


class SingleOccupantTests(unittest.TestCase):

    def test_1_creating_an_occupant_reports_new_then_active(self):

        manager = LiveOccupantManager()

        first = manager.update(
            "OCC-1", camera_id="CAM-A", track_id="CAM-A-T1", zone_id="zone-1", floor_id="floor-1",
            world_position=(1.0, 1.0), world_velocity=0.0, behavior=RecognizedBehavior.STATIONARY,
            confidence=0.9, timestamp=0.0,
        )
        self.assertEqual(first.status, OccupantStatus.NEW)
        self.assertEqual(first.first_seen, 0.0)
        self.assertEqual(first.last_seen, 0.0)

        second = manager.update(
            "OCC-1", camera_id="CAM-A", track_id="CAM-A-T1", zone_id="zone-1", floor_id="floor-1",
            world_position=(1.1, 1.0), world_velocity=0.1, behavior=RecognizedBehavior.WALKING,
            confidence=0.9, timestamp=1.0,
        )
        self.assertEqual(second.status, OccupantStatus.ACTIVE)
        self.assertEqual(second.first_seen, 0.0)  # unchanged
        self.assertEqual(second.last_seen, 1.0)


class MultipleOccupantTests(unittest.TestCase):

    def test_2_multiple_occupants_tracked_independently(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)
        manager.update("OCC-2", "CAM-B", "T1", "zone-2", "floor-1", (10.0, 10.0), None, None, 0.9, 0.0)

        self.assertEqual(len(manager), 2)
        self.assertIsNotNone(manager.get("OCC-1"))
        self.assertIsNotNone(manager.get("OCC-2"))
        self.assertEqual(manager.get("OCC-1").current_zone_id, "zone-1")
        self.assertEqual(manager.get("OCC-2").current_zone_id, "zone-2")


class CameraTransitionTests(unittest.TestCase):

    def test_3_camera_change_recorded_in_history_and_publishes_event(self):

        bus = EventBus()
        manager = LiveOccupantManager(event_bus=bus)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)
        updated = manager.update("OCC-1", "CAM-B", "T2", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 1.0)

        self.assertEqual(updated.current_camera_id, "CAM-B")
        self.assertEqual(len(updated.history.camera_transitions), 2)  # None->CAM-A, CAM-A->CAM-B
        self.assertEqual(updated.history.camera_transitions[-1].from_camera_id, "CAM-A")
        self.assertEqual(updated.history.camera_transitions[-1].to_camera_id, "CAM-B")

        camera_events = bus.history_of(EventType.OCCUPANT_CAMERA_CHANGED)
        self.assertEqual(len(camera_events), 1)
        self.assertEqual(camera_events[0].payload.to_camera_id, "CAM-B")


class ZoneTransitionTests(unittest.TestCase):

    def test_4_zone_change_recorded_in_history_and_publishes_event_and_updates_index(self):

        bus = EventBus()
        manager = LiveOccupantManager(event_bus=bus)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)
        manager.update("OCC-1", "CAM-A", "T1", "zone-2", "floor-1", (5.0, 0.0), None, None, 0.9, 1.0)

        self.assertEqual(manager.occupants_in_zone("zone-1"), ())
        self.assertEqual(len(manager.occupants_in_zone("zone-2")), 1)

        zone_events = bus.history_of(EventType.OCCUPANT_ZONE_CHANGED)
        self.assertEqual(len(zone_events), 1)
        self.assertEqual(zone_events[0].payload.from_zone_id, "zone-1")
        self.assertEqual(zone_events[0].payload.to_zone_id, "zone-2")


class BehaviorChangeTests(unittest.TestCase):

    def test_5_behavior_change_recorded_and_published_and_indexed(self):

        bus = EventBus()
        manager = LiveOccupantManager(event_bus=bus)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (1.0, 0.0), 1.0, RecognizedBehavior.WALKING, 0.9, 1.0)

        self.assertEqual(len(manager.occupants_with_behavior(RecognizedBehavior.STATIONARY)), 0)
        self.assertEqual(len(manager.occupants_with_behavior(RecognizedBehavior.WALKING)), 1)

        behavior_events = bus.history_of(EventType.OCCUPANT_BEHAVIOR_CHANGED)
        self.assertEqual(len(behavior_events), 1)
        self.assertEqual(behavior_events[0].payload.from_behavior, RecognizedBehavior.STATIONARY)
        self.assertEqual(behavior_events[0].payload.to_behavior, RecognizedBehavior.WALKING)


class TemporaryDisappearanceAndReappearanceTests(unittest.TestCase):

    def test_6_missing_occupant_becomes_temporarily_lost(self):

        manager = LiveOccupantManager(expire_after_seconds=100.0)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)

        manager.sweep_missing(timestamp=1.0, seen_occupant_ids=set())

        self.assertEqual(manager.get("OCC-1").status, OccupantStatus.TEMPORARILY_LOST)
        self.assertEqual(len(manager), 1)  # not removed

    def test_7_reappearance_returns_to_active(self):

        manager = LiveOccupantManager(expire_after_seconds=100.0)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)
        manager.sweep_missing(timestamp=1.0, seen_occupant_ids=set())
        self.assertEqual(manager.get("OCC-1").status, OccupantStatus.TEMPORARILY_LOST)

        reappeared = manager.update("OCC-1", "CAM-B", "T5", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 2.0)

        self.assertEqual(reappeared.status, OccupantStatus.ACTIVE)

    def test_near_exit_missing_occupant_becomes_exited(self):

        bus = EventBus()
        exits = [FakeExit(start_point=(0.0, 0.0), end_point=(2.0, 0.0))]
        manager = LiveOccupantManager(event_bus=bus, exits=exits, exit_proximity_threshold=1.0, expire_after_seconds=100.0)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (1.0, 0.5), None, None, 0.9, 0.0)  # near the exit segment

        manager.sweep_missing(timestamp=1.0, seen_occupant_ids=set())

        self.assertEqual(manager.get("OCC-1").status, OccupantStatus.EXITED)

        exited_events = bus.history_of(EventType.OCCUPANT_EXITED)
        self.assertEqual(len(exited_events), 1)

    def test_far_from_exit_missing_occupant_becomes_temporarily_lost_not_exited(self):

        exits = [FakeExit(start_point=(0.0, 0.0), end_point=(2.0, 0.0))]
        manager = LiveOccupantManager(exits=exits, exit_proximity_threshold=1.0, expire_after_seconds=100.0)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (500.0, 500.0), None, None, 0.9, 0.0)

        manager.sweep_missing(timestamp=1.0, seen_occupant_ids=set())

        self.assertEqual(manager.get("OCC-1").status, OccupantStatus.TEMPORARILY_LOST)


class ExpirationTests(unittest.TestCase):

    def test_8_expiration_removes_the_occupant_and_publishes_event(self):

        bus = EventBus()
        manager = LiveOccupantManager(event_bus=bus, expire_after_seconds=5.0)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)

        manager.sweep_missing(timestamp=10.0, seen_occupant_ids=set())  # well past 5s timeout

        self.assertIsNone(manager.get("OCC-1"))
        self.assertEqual(len(manager), 0)

        expired_events = bus.history_of(EventType.OCCUPANT_EXPIRED)
        self.assertEqual(len(expired_events), 1)
        self.assertEqual(expired_events[0].payload.occupant.occupant_id, "OCC-1")

    def test_8_expiration_is_a_strict_boundary(self):

        manager = LiveOccupantManager(expire_after_seconds=5.0)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)

        manager.sweep_missing(timestamp=5.0, seen_occupant_ids=set())
        self.assertEqual(manager.get("OCC-1").status, OccupantStatus.TEMPORARILY_LOST)  # exactly at boundary -- not yet

        manager.sweep_missing(timestamp=5.0001, seen_occupant_ids=set())
        self.assertIsNone(manager.get("OCC-1"))  # just past -- expired


class LookupTests(unittest.TestCase):

    def test_9_lookup_by_occupant_id_is_direct_and_returns_none_for_unknown(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)

        self.assertIsNotNone(manager.get("OCC-1"))
        self.assertIsNone(manager.get("OCC-UNKNOWN"))


class HistoryTrimmingTests(unittest.TestCase):

    def test_10_history_never_grows_past_configured_length(self):

        manager = LiveOccupantManager(history_length=3)

        occupant = None
        for i in range(10):
            occupant = manager.update(
                "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (float(i), 0.0), float(i), None, 0.9, float(i),
            )

        self.assertEqual(len(occupant.history.position_samples), 3)
        self.assertEqual(len(occupant.history.velocity_samples), 3)
        # Only the 3 most recent samples survive.
        self.assertEqual(occupant.history.position_samples[0].timestamp, 7.0)
        self.assertEqual(occupant.history.position_samples[-1].timestamp, 9.0)


class QueryTests(unittest.TestCase):

    def test_11_query_by_zone_floor_behavior_and_camera(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        manager.update("OCC-2", "CAM-A", "T2", "zone-1", "floor-1", (1.0, 0.0), None, RecognizedBehavior.RUNNING, 0.9, 0.0)
        manager.update("OCC-3", "CAM-B", "T1", "zone-2", "floor-2", (2.0, 0.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        self.assertEqual({o.occupant_id for o in manager.occupants_in_zone("zone-1")}, {"OCC-1", "OCC-2"})
        self.assertEqual({o.occupant_id for o in manager.occupants_on_floor("floor-2")}, {"OCC-3"})
        self.assertEqual({o.occupant_id for o in manager.occupants_with_behavior(RecognizedBehavior.WALKING)}, {"OCC-1", "OCC-3"})
        self.assertEqual({o.occupant_id for o in manager.occupants_on_camera("CAM-A")}, {"OCC-1", "OCC-2"})

    def test_11_active_occupants_excludes_lost_and_exited(self):

        manager = LiveOccupantManager(expire_after_seconds=100.0)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)
        manager.update("OCC-2", "CAM-A", "T2", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)

        manager.sweep_missing(timestamp=1.0, seen_occupant_ids={"OCC-1"})  # OCC-2 goes missing

        active_ids = {o.occupant_id for o in manager.active_occupants()}
        self.assertEqual(active_ids, {"OCC-1"})
        self.assertEqual(len(manager.all_occupants()), 2)  # both still tracked


class EventOrderingTests(unittest.TestCase):

    def test_12_update_event_precedes_field_specific_events_in_fixed_order(self):

        bus = EventBus()
        manager = LiveOccupantManager(event_bus=bus)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.update("OCC-1", "CAM-B", "T2", "zone-2", "floor-1", (1.0, 0.0), None, RecognizedBehavior.WALKING, 0.9, 1.0)

        event_types = [event.event_type for event in bus.history]

        # Creation cycle: just OCCUPANT_CREATED.
        # Update cycle: OCCUPANT_UPDATED, then CAMERA_CHANGED, then
        # ZONE_CHANGED, then BEHAVIOR_CHANGED, in that fixed order.
        self.assertEqual(
            event_types,
            [
                EventType.OCCUPANT_CREATED,
                EventType.OCCUPANT_UPDATED,
                EventType.OCCUPANT_CAMERA_CHANGED,
                EventType.OCCUPANT_ZONE_CHANGED,
                EventType.OCCUPANT_BEHAVIOR_CHANGED,
            ],
        )

    def test_12_no_event_bus_configured_never_raises(self):

        manager = LiveOccupantManager(event_bus=None)

        # Must not raise despite no bus being configured.
        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)


class ManagerCleanupTests(unittest.TestCase):

    def test_13_expired_occupants_leave_no_leaked_index_entries(self):

        manager = LiveOccupantManager(expire_after_seconds=1.0)

        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        manager.sweep_missing(timestamp=10.0, seen_occupant_ids=set())

        self.assertEqual(manager.occupants_in_zone("zone-1"), ())
        self.assertEqual(manager.occupants_on_camera("CAM-A"), ())
        self.assertEqual(manager.occupants_with_behavior(RecognizedBehavior.WALKING), ())
        self.assertEqual(len(manager), 0)


class SnapshotConsistencyTests(unittest.TestCase):

    def test_14_returned_occupant_is_immutable_and_independent_of_later_updates(self):

        manager = LiveOccupantManager()

        snapshot = manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0)

        manager.update("OCC-1", "CAM-B", "T2", "zone-2", "floor-1", (5.0, 5.0), None, None, 0.9, 1.0)

        # The earlier snapshot must not have mutated in place.
        self.assertEqual(snapshot.current_camera_id, "CAM-A")
        self.assertEqual(snapshot.current_zone_id, "zone-1")

        with self.assertRaises(Exception):
            snapshot.current_camera_id = "CAM-C"  # frozen dataclass -- must reject mutation


if __name__ == "__main__":
    unittest.main()
