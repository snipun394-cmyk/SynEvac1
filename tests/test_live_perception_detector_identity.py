import unittest

from models.building import Building
from models.detector import Detector
from models.floor import Floor
from models.heat_detector import HeatDetector
from models.smoke_detector import SmokeDetector
from models.zone import Zone

from sensor_manager.manager import SensorManager

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport

from live_perception.building_state_adapter import BuildingStateInputAdapter
from live_perception.coordinator import LivePerceptionFusionCoordinator
from live_perception.providers import LiveFACPObservationProvider, LiveHeatObservationProvider, LiveSmokeObservationProvider


# =====================================================
# Live Perception -> BuildingState Integration Bridge milestone, Phase
# 9 -- proves ONE detector identity survives every stage, for BOTH a
# canonical SmokeDetector/HeatDetector asset AND a legacy generic
# Detector adapted via models.detector_migration.adapt_legacy_detector()
# (called automatically inside sensor_manager.manager.SensorManager.
# discover_sensors() -- never reimplemented or duplicated here, per
# this milestone's own explicit "do NOT introduce another detector
# migration layer" instruction):
#
#   Detector ID -> SensorManager -> Perception reading -> SensorFusion
#   observation -> FACP condition/source -> BuildingState
# =====================================================


class FakeReading:

    def __init__(self, detector_id, alarm_active, confidence=0.9, timestamp=0.0):
        self.detector_id = detector_id
        self.alarm_active = alarm_active
        self.confidence = confidence
        self.timestamp = timestamp


def make_building():

    floor = Floor(
        id="floor-1", name="Ground Floor",
        zones=[Zone(id="zone-1", name="Zone 1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="floor-1")],
        smoke_detectors=[
            SmokeDetector(id="CANONICAL-SD-1", name="Canonical Smoke", floor_id="floor-1", zone_ids=("zone-1",)),
        ],
        heat_detectors=[
            HeatDetector(id="CANONICAL-HD-1", name="Canonical Heat", floor_id="floor-1", zone_ids=("zone-1",)),
        ],
        detectors=[
            # A LEGACY generic Detector, positioned inside zone-1's own
            # bounding box so adapt_legacy_detector()'s own
            # Zone.contains() geometry test resolves zone_ids correctly.
            Detector(id="LEGACY-SD-1", name="Legacy Smoke", floor_id="floor-1", position=(5.0, 5.0), detector_type="Smoke"),
        ],
    )

    return Building(id="detector-identity-building", name="Detector Identity Building", floors=[floor])


class DetectorIdentityConsistencyTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        self.sensor_manager = SensorManager()
        self.sensor_manager.discover_sensors(self.building)

    def test_canonical_and_legacy_detectors_both_resolve_the_same_sensor_id_and_zone(self):

        statuses_by_id = {status.sensor_id: status for status in self.sensor_manager.all_statuses()}

        self.assertIn("CANONICAL-SD-1", statuses_by_id)
        self.assertIn("LEGACY-SD-1", statuses_by_id)

        # The legacy detector's own id is PRESERVED exactly (never
        # re-derived/renamed) through SensorManager's automatic
        # adaptation, and its zone_ids is honestly resolved from real
        # geometry, not fabricated.
        self.assertEqual(statuses_by_id["LEGACY-SD-1"].zone_ids, ("zone-1",))
        self.assertEqual(statuses_by_id["CANONICAL-SD-1"].zone_ids, ("zone-1",))

    def test_both_detector_kinds_produce_smoke_observations_under_their_own_preserved_id(self):

        provider = LiveSmokeObservationProvider(
            self.sensor_manager,
            reading_provider=lambda t: [
                FakeReading("CANONICAL-SD-1", alarm_active=True),
                FakeReading("LEGACY-SD-1", alarm_active=True),
            ],
        )

        observations = provider.collect(time=0.0)

        sources = {o.source for o in observations}
        self.assertIn("smoke-CANONICAL-SD-1", sources)
        self.assertIn("smoke-LEGACY-SD-1", sources)
        self.assertTrue(all(o.location == "zone-1" for o in observations))

    def test_detector_identity_survives_through_fusion_into_buildingstate_hazard(self):

        smoke_provider = LiveSmokeObservationProvider(
            self.sensor_manager,
            reading_provider=lambda t: [
                FakeReading("CANONICAL-SD-1", alarm_active=True),
                FakeReading("LEGACY-SD-1", alarm_active=False),
            ],
        )
        heat_provider = LiveHeatObservationProvider(
            self.sensor_manager, reading_provider=lambda t: [FakeReading("CANONICAL-HD-1", alarm_active=False)],
        )

        coordinator = LivePerceptionFusionCoordinator(providers=[smoke_provider, heat_provider])
        snapshot = coordinator.collect(time=0.0)

        # Both detectors agree the zone is the SAME location -- fused
        # into ONE SMOKE FusedObservation for zone-1, contributed to by
        # BOTH preserved detector ids.
        smoke_fused = next(f for f in snapshot.fused_observations if f.kind.name == "SMOKE")
        self.assertEqual(set(smoke_fused.contributing_sources), {"smoke-CANONICAL-SD-1", "smoke-LEGACY-SD-1"})

        # ANY alarming source wins (worst-case) -- the fused hazard
        # picture reflects the real, still-identity-traceable evidence.
        self.assertTrue(smoke_fused.measurement)
        self.assertGreater(snapshot.hazard_snapshot.node_states["zone-1"].hazard_score, 0.0)

    def test_facp_condition_reports_and_snapshot_preserve_the_same_detector_id(self):

        smoke_status = next(s for s in self.sensor_manager.all_statuses() if s.sensor_id == "CANONICAL-SD-1")
        reading = FakeReading("CANONICAL-SD-1", alarm_active=True)

        condition_report = DetectorConditionReport.from_status_and_reading(smoke_status, reading)
        self.assertEqual(condition_report.asset_id, "CANONICAL-SD-1")

        facp = SimulatedFACP(panel_id="FACP-1")
        facp.evaluate({"CANONICAL-SD-1": condition_report}, time=0.0)
        facp_snapshot = facp.current_snapshot(time=0.0)

        self.assertIn("CANONICAL-SD-1", facp_snapshot.active_alarm_source_ids)

        facp_provider = LiveFACPObservationProvider(self.sensor_manager, snapshot_provider=lambda t: facp_snapshot)
        observations = facp_provider.collect(time=0.0)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].location, "zone-1")


if __name__ == "__main__":
    unittest.main()
