import unittest

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from crowd_intelligence.engine import CrowdIntelligenceEngine

from observable_assets.models import AssetObservation, ObservableAssetSnapshot, ObservationStatus

from stair_flow.models import StairFlowMetrics


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone -- Phase
# 10. Proves stair_flow_metrics is reached through the EXISTING
# CrowdIntelligenceEngine.compute()/CrowdIntelligenceSnapshot, never a
# second, independently-scheduled engine -- and that Door/Exit metrics,
# stair_metrics (occupancy), and every other pre-existing field are
# completely unaffected.
# =====================================================


def make_building_with_stair():

    building = Building(name="Integration Test Building")

    floor_1 = Floor(name="Floor 1", display_order=0, height=3.0)
    floor_2 = Floor(name="Floor 2", display_order=1, height=3.0)

    zone_a = Zone(name="Zone A", x=0.0, y=0.0, width=10.0, height=10.0, floor_id=floor_1.id)
    floor_1.zones.append(zone_a)

    building.add_floor(floor_1)
    building.add_floor(floor_2)

    stair = Staircase(name="S1", from_floor_id=floor_1.id, to_floor_id=floor_2.id)
    stair.id = "S1"
    floor_1.add_stair(stair)

    return building, floor_1, floor_2, zone_a, stair


class CrowdIntelligenceStairFlowIntegrationTests(unittest.TestCase):

    def test_1_snapshot_carries_stair_flow_metrics_keyed_by_stair_id(self):

        building, floor_1, floor_2, zone_a, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-1", "T1", zone_a.id, floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-1", "CAM-1", "T1", None, floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )

        engine = CrowdIntelligenceEngine(building, manager)
        snapshot = engine.compute(1.0)

        self.assertIn("S1", snapshot.stair_flow_metrics)
        self.assertIsInstance(snapshot.stair_flow(stair_id="S1"), StairFlowMetrics)
        self.assertEqual(snapshot.stair_flow("S1").entries, 1)
        self.assertEqual(snapshot.stair_flow("S1").upward_count, 1)

    def test_2_unknown_stair_id_via_accessor_returns_none(self):

        building, floor_1, floor_2, zone_a, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        engine = CrowdIntelligenceEngine(building, manager)
        snapshot = engine.compute(0.0)

        self.assertIsNone(snapshot.stair_flow("NEVER-EXISTED"))

    def test_3_observed_occupant_count_reused_not_recomputed(self):

        building, floor_1, floor_2, zone_a, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        manager.update("OCC-2", "CAM-1", "T1", None, floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0, stair_id="S1")

        observable_assets = ObservableAssetSnapshot(
            timestamp=0.0,
            observations={
                "S1": AssetObservation(
                    asset_id="S1", asset_type="Stair", status=ObservationStatus.OBSERVED,
                    occupant_ids=("OCC-2",),
                ),
            },
        )

        engine = CrowdIntelligenceEngine(building, manager)
        snapshot = engine.compute(0.0, observable_assets=observable_assets)

        # SAME number on both the pre-existing occupancy metric and the
        # new flow metric -- proof of reuse, not duplicate computation.
        self.assertEqual(snapshot.stair("S1").observed_occupant_count, 1)
        self.assertEqual(snapshot.stair_flow("S1").observed_occupant_count, 1)

    def test_4_pre_existing_fields_unaffected(self):

        building, floor_1, floor_2, zone_a, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        manager.update("OCC-3", "CAM-1", "T1", zone_a.id, floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)

        engine = CrowdIntelligenceEngine(building, manager)
        snapshot = engine.compute(0.0)

        self.assertIn(zone_a.id, snapshot.zone_metrics)
        self.assertIn("S1", snapshot.stair_metrics)
        self.assertEqual(snapshot.door_metrics, {})
        self.assertEqual(snapshot.exit_metrics, {})

    def test_5_stair_flow_window_seconds_is_configurable(self):

        building, floor_1, floor_2, zone_a, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        engine = CrowdIntelligenceEngine(building, manager, stair_flow_window_seconds=120.0)
        snapshot = engine.compute(0.0)

        self.assertEqual(snapshot.stair_flow_metrics["S1"].window_seconds, 120.0)


if __name__ == "__main__":
    unittest.main()
