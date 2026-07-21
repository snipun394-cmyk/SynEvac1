import unittest

from behavior_recognition.observation import RecognizedBehavior

from sensor_fusion.observation import ObservationKind

from live_perception.building_state_adapter import BuildingStateInputAdapter
from live_perception.coordinator import LivePerceptionFusionCoordinator
from live_perception.providers import (
    LiveFACPObservationProvider, LiveHeatObservationProvider,
    LiveOccupantObservationProvider, LiveSmokeObservationProvider,
)


# =====================================================
# Live Perception -> BuildingState Integration Bridge milestone --
# deterministic, offline unit tests for the production providers/
# adapter/coordinator, isolated from the full LiveRuntime. No
# randomness anywhere in this file.
# =====================================================


class FakeOccupant:

    def __init__(self, occupant_id, current_zone_id, behavior, confidence):
        self.occupant_id = occupant_id
        self.current_zone_id = current_zone_id
        self.behavior = behavior
        self.confidence = confidence


class FakeOccupantManager:

    def __init__(self, occupants):
        self._occupants = tuple(occupants)

    def active_occupants(self):
        return self._occupants


class FakeSensorStatus:

    def __init__(self, sensor_id, zone_ids):
        self.sensor_id = sensor_id
        self.zone_ids = tuple(zone_ids)


class FakeSensorManager:

    def __init__(self, statuses):
        self._statuses = tuple(statuses)

    def all_statuses(self):
        return self._statuses


class FakeReading:

    def __init__(self, detector_id, alarm_active, confidence=0.9, timestamp=0.0):
        self.detector_id = detector_id
        self.alarm_active = alarm_active
        self.confidence = confidence
        self.timestamp = timestamp


class FakeFACPSnapshot:

    def __init__(self, active_alarm_source_ids, timestamp=0.0):
        self.active_alarm_source_ids = tuple(active_alarm_source_ids)
        self.timestamp = timestamp


class LiveOccupantObservationProviderTests(unittest.TestCase):

    def test_produces_occupancy_and_behavior_observations(self):

        occupants = [
            FakeOccupant("OCC-1", "zone-1", RecognizedBehavior.WALKING, 0.9),
            FakeOccupant("OCC-2", "zone-1", RecognizedBehavior.STATIONARY, 0.8),
            FakeOccupant("OCC-3", "zone-2", None, 0.7),  # no known behavior yet
        ]
        provider = LiveOccupantObservationProvider(FakeOccupantManager(occupants))

        observations = provider.collect(time=0.0)

        occupancy = [o for o in observations if o.kind == ObservationKind.OCCUPANCY]
        behavior = [o for o in observations if o.kind == ObservationKind.BEHAVIOR]

        self.assertEqual({o.location: o.measurement for o in occupancy}, {"zone-1": 2.0, "zone-2": 1.0})
        self.assertEqual(len(behavior), 2)  # OCC-3 has no behavior -- honestly excluded

    def test_occupant_with_no_zone_is_excluded_never_guessed(self):

        occupants = [FakeOccupant("OCC-1", None, RecognizedBehavior.WALKING, 0.9)]
        provider = LiveOccupantObservationProvider(FakeOccupantManager(occupants))

        observations = provider.collect(time=0.0)

        self.assertEqual(observations, ())


class LiveDetectorObservationProviderTests(unittest.TestCase):

    def test_smoke_reading_resolves_zone_via_sensor_manager(self):

        sensor_manager = FakeSensorManager([FakeSensorStatus("SD-1", ["zone-1"])])
        provider = LiveSmokeObservationProvider(sensor_manager, reading_provider=lambda t: [FakeReading("SD-1", True)])

        observations = provider.collect(time=0.0)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].kind, ObservationKind.SMOKE)
        self.assertEqual(observations[0].location, "zone-1")
        self.assertTrue(observations[0].measurement)

    def test_no_reading_provider_configured_produces_nothing(self):

        sensor_manager = FakeSensorManager([])
        provider = LiveHeatObservationProvider(sensor_manager, reading_provider=None)

        self.assertEqual(provider.collect(time=0.0), ())

    def test_multi_zone_detector_contributes_to_every_assigned_zone(self):

        sensor_manager = FakeSensorManager([FakeSensorStatus("HD-1", ["zone-1", "zone-2"])])
        provider = LiveHeatObservationProvider(sensor_manager, reading_provider=lambda t: [FakeReading("HD-1", True)])

        observations = provider.collect(time=0.0)

        self.assertEqual({o.location for o in observations}, {"zone-1", "zone-2"})

    def test_unresolvable_detector_zone_is_honestly_dropped(self):

        sensor_manager = FakeSensorManager([])  # detector not registered at all
        provider = LiveSmokeObservationProvider(sensor_manager, reading_provider=lambda t: [FakeReading("SD-UNKNOWN", True)])

        self.assertEqual(provider.collect(time=0.0), ())


class LiveFACPObservationProviderTests(unittest.TestCase):

    def test_active_alarm_source_becomes_an_alarm_observation(self):

        sensor_manager = FakeSensorManager([FakeSensorStatus("SD-1", ["zone-1"])])
        provider = LiveFACPObservationProvider(sensor_manager, snapshot_provider=lambda t: FakeFACPSnapshot(["SD-1"]))

        observations = provider.collect(time=0.0)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].kind, ObservationKind.ALARM)
        self.assertEqual(observations[0].location, "zone-1")

    def test_no_snapshot_provider_produces_nothing(self):

        provider = LiveFACPObservationProvider(FakeSensorManager([]), snapshot_provider=None)

        self.assertEqual(provider.collect(time=0.0), ())

    def test_snapshot_provider_returning_none_produces_nothing(self):

        provider = LiveFACPObservationProvider(FakeSensorManager([]), snapshot_provider=lambda t: None)

        self.assertEqual(provider.collect(time=0.0), ())


class BuildingStateInputAdapterTests(unittest.TestCase):

    def test_missing_smoke_never_becomes_zero_smoke(self):

        adapter = BuildingStateInputAdapter()

        hazard_snapshot = adapter.to_hazard_snapshot((), timestamp=0.0)

        self.assertEqual(dict(hazard_snapshot.node_states), {})  # no entry at all, not a zeroed-out one

    def test_missing_occupancy_never_becomes_zero_occupants(self):

        adapter = BuildingStateInputAdapter()

        occupancy_snapshot = adapter.to_occupancy_snapshot((), timestamp=0.0)

        self.assertEqual(dict(occupancy_snapshot.observations), {})
        self.assertIsNone(occupancy_snapshot.observation_at("zone-1").occupant_count)

    def test_alarming_smoke_produces_a_conservative_hazard_score(self):

        from sensor_fusion.observation import FusedObservation

        adapter = BuildingStateInputAdapter()

        fused = (
            FusedObservation(
                kind=ObservationKind.SMOKE, location="zone-1", timestamp=0.0,
                measurement=True, confidence=0.9, contributing_sources=("smoke-SD-1",),
            ),
        )

        hazard_snapshot = adapter.to_hazard_snapshot(fused, timestamp=0.0)

        self.assertAlmostEqual(hazard_snapshot.node_states["zone-1"].hazard_score, 0.9)


class LivePerceptionFusionCoordinatorTests(unittest.TestCase):

    def test_collect_is_memoized_per_timestamp(self):

        call_count = {"n": 0}

        class CountingProvider:
            def collect(self, time):
                call_count["n"] += 1
                return ()

        coordinator = LivePerceptionFusionCoordinator(providers=[CountingProvider()])

        coordinator.hazard_snapshot_provider(0.0)
        coordinator.occupancy_snapshot_provider(0.0)  # SAME time -- must reuse the cached fusion

        self.assertEqual(call_count["n"], 1)

        coordinator.collect(1.0)  # a NEW time -- must re-run

        self.assertEqual(call_count["n"], 2)

    def test_end_to_end_occupancy_and_smoke_through_the_coordinator(self):

        occupants = [FakeOccupant("OCC-1", "zone-1", RecognizedBehavior.WALKING, 0.9)]
        occupant_provider = LiveOccupantObservationProvider(FakeOccupantManager(occupants))

        sensor_manager = FakeSensorManager([FakeSensorStatus("SD-1", ["zone-1"])])
        smoke_provider = LiveSmokeObservationProvider(sensor_manager, reading_provider=lambda t: [FakeReading("SD-1", True)])

        coordinator = LivePerceptionFusionCoordinator(providers=[occupant_provider, smoke_provider])

        snapshot = coordinator.collect(0.0)

        self.assertEqual(snapshot.occupancy_snapshot.observation_at("zone-1").occupant_count, 1.0)
        self.assertGreater(snapshot.hazard_snapshot.node_states["zone-1"].hazard_score, 0.0)


if __name__ == "__main__":
    unittest.main()
