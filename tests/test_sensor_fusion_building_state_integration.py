import unittest

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from building_state.estimator import BuildingStateEstimator

from sensor_fusion.engine import SensorFusionEngine
from sensor_fusion.observation import FusedObservation, ObservationKind
from sensor_fusion.provider import CameraObservationProvider, HeatObservationProvider, SmokeObservationProvider


# =====================================================
# Sensor Fusion Engine milestone, Phase 8 -- proves LiveOccupantManager
# -> SensorFusionEngine -> BuildingStateEstimator, WITHOUT modifying
# BuildingStateEstimator/BuildingState at all (Phase 8's own "keep
# BuildingState unchanged if possible" -- honored literally: zero
# changes to either type this milestone). The bridge converting
# FusedObservations into HazardSnapshot/OccupancySnapshot lives HERE,
# in this integration test, deliberately OUTSIDE sensor_fusion/ itself
# (that package must depend only on observation providers/geometry/
# time -- see tests/test_sensor_fusion_architecture_guards.py) --
# BuildingStateEstimator.estimate()'s own existing parameters are
# exactly what this bridge targets, proving fused observations can
# replace "independently querying each provider and hand-assembling
# HazardSnapshot/OccupancySnapshot" (designer/building_state_debug_
# runner.py's own pattern, Phase 1's investigation finding) without
# requiring that call site -- or any other existing caller -- to change.
# =====================================================


class FakeOccupant:

    def __init__(self, occupant_id, current_zone_id, behavior, confidence):
        self.occupant_id = occupant_id
        self.current_zone_id = current_zone_id
        self.behavior = behavior
        self.confidence = confidence


def _fused_to_hazard_snapshot(fused_observations, timestamp) -> HazardSnapshot:

    # A small, explicit, additive translation -- never a second Ground
    # Truth computation, purely reshaping already-fused values (the
    # same "pure re-shaping of already-computed values" discipline
    # designer.building_state_debug_runner.BuildingStateDebugRunner.
    # _reconstruct_hazard_snapshot() already establishes for its own,
    # differently-sourced inputs).

    node_states = {}

    for fused in fused_observations:

        if fused.kind not in (ObservationKind.SMOKE, ObservationKind.HEAT):
            continue

        existing = node_states.get(fused.location, HazardNodeState())
        alarming = bool(fused.measurement)

        node_states[fused.location] = HazardNodeState(
            smoke_level=1.0 if (fused.kind == ObservationKind.SMOKE and alarming) else existing.smoke_level,
            temperature=100.0 if (fused.kind == ObservationKind.HEAT and alarming) else existing.temperature,
            visibility=existing.visibility,
            hazard_score=max(existing.hazard_score, fused.confidence if alarming else 0.0),
        )

    return HazardSnapshot(timestamp=timestamp, node_states=node_states)


def _fused_to_occupancy_snapshot(fused_observations, timestamp) -> OccupancySnapshot:

    observations = {
        fused.location: OccupancyObservation(occupant_count=fused.measurement, confidence=fused.confidence)
        for fused in fused_observations
        if fused.kind == ObservationKind.OCCUPANCY
    }

    return OccupancySnapshot(timestamp=timestamp, observations=observations)


class SensorFusionFeedsBuildingStateEstimatorTests(unittest.TestCase):

    def test_fused_occupancy_and_smoke_reach_buildingstate_via_unmodified_estimator(self):

        from behavior_recognition.observation import RecognizedBehavior

        occupants = [
            FakeOccupant("OCC-1", "zone-1", RecognizedBehavior.WALKING, 0.9),
            FakeOccupant("OCC-2", "zone-1", RecognizedBehavior.WALKING, 0.8),
        ]

        camera_provider = CameraObservationProvider()
        camera_provider.set_occupants(occupants)

        smoke_provider = SmokeObservationProvider(zone_by_detector_id={"SD-1": "zone-1"})

        class FakeSmokeReading:
            detector_id = "SD-1"
            timestamp = 0.0
            alarm_active = True
            confidence = 0.95

        smoke_provider.set_readings([FakeSmokeReading()])

        heat_provider = HeatObservationProvider(zone_by_detector_id={})
        heat_provider.set_readings([])

        engine = SensorFusionEngine()
        observations = engine.collect([camera_provider, smoke_provider, heat_provider], time=0.0)
        fused = engine.fuse(observations, time=0.0)

        hazard_snapshot = _fused_to_hazard_snapshot(fused, timestamp=0.0)
        occupancy_snapshot = _fused_to_occupancy_snapshot(fused, timestamp=0.0)

        # BuildingStateEstimator itself is completely unmodified --
        # same call shape as designer/building_state_debug_runner.py's
        # own existing usage.
        estimator = BuildingStateEstimator()
        state = estimator.estimate(
            0.0, hazard_snapshot=hazard_snapshot, occupancy_snapshot=occupancy_snapshot,
        )

        self.assertEqual(state.zone_occupancy.observation_at("zone-1").occupant_count, 2.0)
        self.assertEqual(state.hazard_summary.zone_severities["zone-1"].name, "CRITICAL")

    def test_missing_sensor_data_never_fabricates_a_hazard_reading(self):

        engine = SensorFusionEngine()
        fused = engine.fuse((), time=0.0)  # no observations collected at all this cycle

        hazard_snapshot = _fused_to_hazard_snapshot(fused, timestamp=0.0)
        occupancy_snapshot = _fused_to_occupancy_snapshot(fused, timestamp=0.0)

        estimator = BuildingStateEstimator()
        state = estimator.estimate(0.0, hazard_snapshot=hazard_snapshot, occupancy_snapshot=occupancy_snapshot)

        self.assertEqual(state.hazard_summary.zone_severities, {})
        self.assertEqual(dict(state.zone_occupancy.observations), {})


if __name__ == "__main__":
    unittest.main()
