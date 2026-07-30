import unittest

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase

from live_occupants.manager import LiveOccupantManager

from observable_assets.models import AssetObservation, ObservableAssetSnapshot, ObservationStatus

from stair_flow.compute import compute_stair_flow_snapshot


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone -- Phase 9.
# Proves this package never converts unavailable evidence into a
# fabricated zero-flow reading -- UNKNOWN and observed-zero must stay
# distinct throughout.
# =====================================================


def make_building_with_stair():

    building = Building(name="Failure Modes Test Building")

    floor_1 = Floor(name="Floor 1", display_order=0, height=3.0)
    floor_2 = Floor(name="Floor 2", display_order=1, height=3.0)

    building.add_floor(floor_1)
    building.add_floor(floor_2)

    stair = Staircase(name="S1", from_floor_id=floor_1.id, to_floor_id=floor_2.id)
    stair.id = "S1"
    floor_1.add_stair(stair)

    return building, floor_1, floor_2, stair


class ObservationFailureModeTests(unittest.TestCase):

    def test_1_no_camera_coverage_at_all_yields_unknown_not_zero(self):

        building, _f1, _f2, stair = make_building_with_stair()

        snapshot = compute_stair_flow_snapshot([stair], [], building, timestamp=100.0, observable_assets=None)
        metrics = snapshot.for_stair("S1")

        self.assertEqual(metrics.status, ObservationStatus.UNKNOWN)
        self.assertIsNone(metrics.entries)
        self.assertIsNone(metrics.exits)
        self.assertIsNone(metrics.entry_rate_per_minute)
        self.assertIsNone(metrics.net_flow)
        self.assertIsNone(metrics.observed_occupant_count)

    def test_2_observable_snapshot_reports_unknown_status(self):

        building, _f1, _f2, stair = make_building_with_stair()

        observable_assets = ObservableAssetSnapshot(
            timestamp=100.0,
            observations={"S1": AssetObservation(asset_id="S1", asset_type="Stair", status=ObservationStatus.UNKNOWN)},
        )

        snapshot = compute_stair_flow_snapshot(
            [stair], [], building, timestamp=100.0, observable_assets=observable_assets,
        )
        metrics = snapshot.for_stair("S1")

        self.assertEqual(metrics.status, ObservationStatus.UNKNOWN)
        self.assertIsNone(metrics.entries)

    def test_3_genuinely_observed_zero_is_distinct_from_unknown(self):

        building, _f1, _f2, stair = make_building_with_stair()

        observable_assets = ObservableAssetSnapshot(
            timestamp=100.0,
            observations={
                "S1": AssetObservation(
                    asset_id="S1", asset_type="Stair", status=ObservationStatus.OBSERVED, occupant_ids=(),
                ),
            },
        )

        snapshot = compute_stair_flow_snapshot(
            [stair], [], building, timestamp=100.0, observable_assets=observable_assets,
        )
        metrics = snapshot.for_stair("S1")

        self.assertEqual(metrics.status, ObservationStatus.OBSERVED)
        # A real, confirmed zero -- not None.
        self.assertEqual(metrics.entries, 0)
        self.assertEqual(metrics.exits, 0)
        self.assertEqual(metrics.net_flow, 0)
        self.assertEqual(metrics.entry_rate_per_minute, 0.0)
        self.assertEqual(metrics.observed_occupant_count, 0)

    def test_4_evidence_in_window_reported_even_if_currently_unobserved(self):

        # A camera that covered the stair a moment ago, and has since
        # failed/gone uncalibrated, must not erase the genuine evidence
        # it already produced.
        building, floor_1, floor_2, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-1", "T1", "ZONE-A", floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-1", "CAM-1", "T1", None, floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )

        observable_assets = ObservableAssetSnapshot(
            timestamp=5.0,
            observations={"S1": AssetObservation(asset_id="S1", asset_type="Stair", status=ObservationStatus.UNKNOWN)},
        )

        snapshot = compute_stair_flow_snapshot(
            [stair], manager.all_occupants(), building, timestamp=5.0, window_seconds=60.0,
            observable_assets=observable_assets,
        )
        metrics = snapshot.for_stair("S1")

        self.assertEqual(metrics.entries, 1)  # real evidence, reported despite current UNKNOWN status

    def test_5_temporarily_lost_occupant_still_contributes_recent_evidence(self):

        building, floor_1, floor_2, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        manager.update("OCC-2", "CAM-1", "T1", "ZONE-A", floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-2", "CAM-1", "T1", None, floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        manager.sweep_missing(2.0, seen_occupant_ids=set())  # occupant now TEMPORARILY_LOST

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=2.0)
        metrics = snapshot.for_stair("S1")

        self.assertEqual(metrics.entries, 1)

    def test_6_expired_occupant_evidence_is_genuinely_lost(self):

        # Documented limitation (Phase 9/17): once an occupant's global
        # identity fully expires, their history -- including any recent
        # stair transition -- is removed from LiveOccupantManager
        # entirely, and can no longer contribute to a flow count even if
        # the transition happened within the configured window.
        building, floor_1, floor_2, stair = make_building_with_stair()
        manager = LiveOccupantManager(expire_after_seconds=1.0)

        manager.update("OCC-3", "CAM-1", "T1", "ZONE-A", floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-3", "CAM-1", "T1", None, floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        manager.sweep_missing(10.0, seen_occupant_ids=set())  # far past expire_after_seconds -> EXPIRED and removed

        self.assertEqual(len(manager.all_occupants()), 0)

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=10.0)
        metrics = snapshot.for_stair("S1")

        self.assertIsNone(metrics.entries)  # honestly UNKNOWN, not a fabricated 0

    def test_7_ambiguous_localization_never_produces_a_stair_transition(self):

        # WorldProjection.stair_id is None when asset_localization_ambiguous
        # is True (never an arbitrary pick) -- current_stair_id simply
        # never becomes non-None from an ambiguous match, so no entry
        # event can ever be derived from one. Verified here at the
        # LiveOccupant/history layer this package actually reads.
        building, floor_1, floor_2, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        manager.update("OCC-4", "CAM-1", "T1", "ZONE-A", floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0, stair_id=None)

        occupant = manager.get("OCC-4")
        self.assertIsNone(occupant.current_stair_id)

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=0.0)
        self.assertIsNone(snapshot.for_stair("S1").entries)

    def test_8_one_camera_fails_while_another_covering_the_same_stair_remains_healthy(self):

        # Simulated at the evidence layer: CAM-A's own detections stop
        # arriving (no update() calls reference it again), but CAM-B
        # keeps confirming the SAME occupant on the SAME stair -- the
        # occupant's current_stair_id, and therefore this package's own
        # evidence, is entirely unaffected by CAM-A's failure.
        building, floor_1, floor_2, stair = make_building_with_stair()
        manager = LiveOccupantManager()

        manager.update("OCC-5", "CAM-A", "T-A", "ZONE-A", floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        manager.update(
            "OCC-5", "CAM-A", "T-A", None, floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        manager.update(
            "OCC-5", "CAM-B", "T-B", None, floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id="S1",
        )
        # CAM-A goes dark from here on; only CAM-B keeps reporting.
        manager.update(
            "OCC-5", "CAM-B", "T-B", None, floor_1.id, (2.2, 2.2), 0.5, None, 0.9, 2.0, stair_id="S1",
        )

        snapshot = compute_stair_flow_snapshot([stair], manager.all_occupants(), building, timestamp=2.0)
        self.assertEqual(snapshot.for_stair("S1").entries, 1)


if __name__ == "__main__":
    unittest.main()
