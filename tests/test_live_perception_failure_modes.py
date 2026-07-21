import unittest

from behavior_recognition.observation import RecognizedBehavior

from sensor_fusion.engine import SensorFusionEngine
from sensor_fusion.provider import ObservationProvider

from live_occupants.manager import LiveOccupantManager

from live_perception.coordinator import LivePerceptionFusionCoordinator
from live_perception.providers import (
    LiveFACPObservationProvider, LiveHeatObservationProvider,
    LiveOccupantObservationProvider, LiveSmokeObservationProvider,
)


# =====================================================
# Live Perception -> BuildingState Integration Bridge milestone, Phase
# 10 -- failure/degradation behavior. The runtime must never crash, and
# must never fabricate normality (missing data stays missing, never
# silently becomes a healthy/zero/NORMAL reading).
# =====================================================


class FakeOccupant:

    def __init__(self, occupant_id, current_zone_id, behavior, confidence):
        self.occupant_id = occupant_id
        self.current_zone_id = current_zone_id
        self.behavior = behavior
        self.confidence = confidence


class FakeOccupantManager:

    def __init__(self, occupants=()):
        self._occupants = tuple(occupants)

    def active_occupants(self):
        return self._occupants


class FakeSensorStatus:

    def __init__(self, sensor_id, zone_ids):
        self.sensor_id = sensor_id
        self.zone_ids = tuple(zone_ids)


class FakeSensorManager:

    def __init__(self, statuses=()):
        self._statuses = tuple(statuses)

    def all_statuses(self):
        return self._statuses


class FakeReading:

    def __init__(self, detector_id, alarm_active, confidence=0.9, timestamp=0.0):
        self.detector_id = detector_id
        self.alarm_active = alarm_active
        self.confidence = confidence
        self.timestamp = timestamp


class RaisingProvider(ObservationProvider):

    def collect(self, time):
        raise RuntimeError("simulated sensor_fusion provider failure")


class CameraOfflineTests(unittest.TestCase):

    def test_camera_with_zero_occupants_produces_no_occupancy_never_zero(self):

        provider = LiveOccupantObservationProvider(LiveOccupantManager())  # never updated -- "camera offline"

        coordinator = LivePerceptionFusionCoordinator(providers=[provider])
        snapshot = coordinator.collect(time=0.0)

        self.assertEqual(dict(snapshot.occupancy_snapshot.observations), {})
        self.assertIsNone(snapshot.occupancy_snapshot.observation_at("zone-1").occupant_count)

    def test_all_cameras_offline_pipeline_produces_a_valid_empty_snapshot(self):

        coordinator = LivePerceptionFusionCoordinator(providers=[LiveOccupantObservationProvider(LiveOccupantManager())])

        snapshot = coordinator.collect(time=0.0)

        self.assertEqual(snapshot.fused_observations, ())
        self.assertEqual(dict(snapshot.hazard_snapshot.node_states), {})
        self.assertEqual(dict(snapshot.occupancy_snapshot.observations), {})


class DetectorUnavailableTests(unittest.TestCase):

    def test_one_detector_unavailable_others_still_fuse(self):

        sensor_manager = FakeSensorManager([FakeSensorStatus("SD-1", ["zone-1"])])

        # HeatObservationProvider has NO reading_provider configured --
        # "one detector unavailable" -- must not block SmokeObservationProvider.
        smoke_provider = LiveSmokeObservationProvider(sensor_manager, reading_provider=lambda t: [FakeReading("SD-1", True)])
        heat_provider = LiveHeatObservationProvider(sensor_manager, reading_provider=None)

        coordinator = LivePerceptionFusionCoordinator(providers=[smoke_provider, heat_provider])
        snapshot = coordinator.collect(time=0.0)

        self.assertIn("zone-1", snapshot.hazard_snapshot.node_states)
        self.assertGreater(snapshot.hazard_snapshot.node_states["zone-1"].hazard_score, 0.0)

    def test_all_detectors_unavailable_produces_no_hazard_never_fabricated_clear(self):

        sensor_manager = FakeSensorManager([])
        smoke_provider = LiveSmokeObservationProvider(sensor_manager, reading_provider=None)
        heat_provider = LiveHeatObservationProvider(sensor_manager, reading_provider=None)

        coordinator = LivePerceptionFusionCoordinator(providers=[smoke_provider, heat_provider])
        snapshot = coordinator.collect(time=0.0)

        # No entry at all -- never a HazardNodeState(hazard_score=0.0)
        # fabricated for a zone with genuinely zero evidence.
        self.assertEqual(dict(snapshot.hazard_snapshot.node_states), {})


class FACPUnavailableTests(unittest.TestCase):

    def test_no_snapshot_provider_produces_no_alarm_observations(self):

        provider = LiveFACPObservationProvider(FakeSensorManager([]), snapshot_provider=None)

        coordinator = LivePerceptionFusionCoordinator(providers=[provider])
        snapshot = coordinator.collect(time=0.0)

        self.assertEqual(snapshot.fused_observations, ())


class SensorFusionProviderExceptionTests(unittest.TestCase):

    def test_a_raising_provider_never_crashes_collection_or_fusion(self):

        good_provider = LiveOccupantObservationProvider(
            LiveOccupantManager()
        )

        engine = SensorFusionEngine()

        # RaisingProvider must not prevent collection from the good one.
        observations = engine.collect([RaisingProvider(), good_provider], time=0.0)
        fused = engine.fuse(observations, time=0.0)  # must not raise

        self.assertEqual(fused, ())  # good_provider had nothing to report either, honestly

    def test_coordinator_survives_a_raising_provider(self):

        coordinator = LivePerceptionFusionCoordinator(providers=[RaisingProvider()])

        snapshot = coordinator.collect(time=0.0)  # must not raise

        self.assertEqual(snapshot.fused_observations, ())


class StaleObservationTests(unittest.TestCase):

    def test_stale_camera_observation_confidence_decays_but_is_not_dropped(self):

        occupant = FakeOccupant("OCC-1", "zone-1", RecognizedBehavior.STATIONARY, 0.9)
        provider = LiveOccupantObservationProvider(FakeOccupantManager([occupant]))

        engine = SensorFusionEngine(staleness_half_life_seconds=10.0)
        coordinator = LivePerceptionFusionCoordinator(providers=[provider], engine=engine)

        # The provider itself always reports "now" as the observation's
        # own timestamp (see LiveOccupantObservationProvider.collect()),
        # so staleness only manifests when a caller fuses against a
        # LATER wall-clock time than collection actually happened --
        # simulated directly against the engine here.
        from sensor_fusion.observation import Observation, ObservationKind

        stale_observation = Observation(
            source="camera", kind=ObservationKind.OCCUPANCY, location="zone-1",
            timestamp=0.0, confidence=0.9, measurement=1.0,
        )

        fresh = engine.fuse([stale_observation], time=0.0)[0]
        stale = engine.fuse([stale_observation], time=10.0)[0]

        self.assertAlmostEqual(fresh.confidence, 0.9)
        self.assertAlmostEqual(stale.confidence, 0.45)
        self.assertEqual(stale.measurement, 1.0)  # still reported, never dropped just for being stale

    def test_stale_detector_observation_confidence_decays_but_is_not_dropped(self):

        sensor_manager = FakeSensorManager([FakeSensorStatus("SD-1", ["zone-1"])])
        provider = LiveSmokeObservationProvider(sensor_manager, reading_provider=lambda t: [FakeReading("SD-1", True, timestamp=0.0)])

        engine = SensorFusionEngine(staleness_half_life_seconds=10.0)
        coordinator = LivePerceptionFusionCoordinator(providers=[provider], engine=engine)

        snapshot = coordinator.collect(time=10.0)  # fusing 10s after the reading's own timestamp

        fused_smoke = next(f for f in snapshot.fused_observations if f.kind.name == "SMOKE")
        self.assertAlmostEqual(fused_smoke.confidence, 0.45)
        self.assertTrue(fused_smoke.measurement)


class PartialBuildingStateTests(unittest.TestCase):

    def test_no_occupant_observations_but_smoke_present_still_produces_partial_snapshot(self):

        sensor_manager = FakeSensorManager([FakeSensorStatus("SD-1", ["zone-1"])])
        smoke_provider = LiveSmokeObservationProvider(sensor_manager, reading_provider=lambda t: [FakeReading("SD-1", True)])
        occupant_provider = LiveOccupantObservationProvider(LiveOccupantManager())  # empty

        coordinator = LivePerceptionFusionCoordinator(providers=[occupant_provider, smoke_provider])
        snapshot = coordinator.collect(time=0.0)

        self.assertGreater(snapshot.hazard_snapshot.node_states["zone-1"].hazard_score, 0.0)
        self.assertEqual(dict(snapshot.occupancy_snapshot.observations), {})

    def test_camera_available_but_no_detections_produces_no_occupancy(self):

        provider = LiveOccupantObservationProvider(FakeOccupantManager([]))  # camera online, genuinely nobody there

        coordinator = LivePerceptionFusionCoordinator(providers=[provider])
        snapshot = coordinator.collect(time=0.0)

        self.assertEqual(dict(snapshot.occupancy_snapshot.observations), {})


class ConflictingObservationTests(unittest.TestCase):

    def test_conflicting_manual_and_sensor_smoke_observations_are_flagged_not_hidden(self):

        from sensor_fusion.observation import Observation, ObservationKind
        from sensor_fusion.provider import ManualObservationProvider

        sensor_manager = FakeSensorManager([FakeSensorStatus("SD-1", ["zone-1"])])
        smoke_provider = LiveSmokeObservationProvider(sensor_manager, reading_provider=lambda t: [FakeReading("SD-1", False)])

        manual_provider = ManualObservationProvider()
        manual_provider.report(ObservationKind.SMOKE, "zone-1", True, 0.6, 0.0)

        # A manual report disagrees with SD-1's own silent reading --
        # fuse directly (bypassing LiveSmokeObservationProvider's own
        # "smoke-" source prefix requirement) to combine both cleanly.
        engine = SensorFusionEngine()
        coordinator = LivePerceptionFusionCoordinator(providers=[smoke_provider, manual_provider], engine=engine)

        snapshot = coordinator.collect(time=0.0)

        fused_smoke = next(f for f in snapshot.fused_observations if f.kind.name == "SMOKE")

        self.assertTrue(fused_smoke.conflict)
        self.assertTrue(fused_smoke.measurement)  # any alarming source wins -- worst-case, life-safety


if __name__ == "__main__":
    unittest.main()
