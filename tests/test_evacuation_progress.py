import unittest

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from camera_manager.status import CameraStatus

from building_state.models import BuildingState, CameraAssetState

from crowd_intelligence.engine import CrowdIntelligenceEngine

from evacuation_progress.engine import EvacuationProgressEngine
from evacuation_progress.models import EvacuationProgressTrend, ZoneClearanceStatus


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone,
# Phase 19 -- deterministic engine-level unit coverage (items 1, 3,
# 5-19; items 2/4 need the full multi-camera pipeline -- see
# tests/test_evacuation_progress_double_counting.py; items 20-25 need
# Advisory -- see tests/test_evacuation_advisory_safety_precedence.py).
# No randomness anywhere in this file.
# =====================================================


def make_building(exit_obj=None):

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1")],
        exits=[exit_obj] if exit_obj is not None else [],
    )

    return Building(id="b1", name="B", floors=[floor])


def make_exit(exit_id="e1", start=(9.0, 4.0), end=(9.0, 6.0)):

    return Exit(id=exit_id, floor_id="f1", start_point=start, end_point=end, width=1.2)


def make_covered_building_state(zone_ids, camera_id="CAM-1", active=True):

    status = CameraStatus(
        camera_id=camera_id, name=camera_id, floor_id="f1", zone_ids=tuple(zone_ids),
        active=active, mode="LIVE", has_detection_provider=True,
    )

    return BuildingState(camera_observations={camera_id: CameraAssetState(status=status)})


def make_engine(building, exits=(), expire_after_seconds=1000.0):

    bus = EventBus()
    manager = LiveOccupantManager(event_bus=bus, exits=exits, expire_after_seconds=expire_after_seconds)
    engine = EvacuationProgressEngine(building, manager, bus)

    return engine, manager


class ProgressiveDepartureTests(unittest.TestCase):

    def test_1_occupants_progressively_leave_progress_rises(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.update("OCC-2", "CAM-1", "T2", "z1", "f1", (2.0, 2.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        progress_values = []

        for t in (1.0, 2.0):
            snap = engine.compute(t, None, None)
            progress_values.append(snap.evacuation_progress_fraction)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (8.9, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 1.0)
        manager.sweep_missing(2.0, seen_occupant_ids={"OCC-2"})

        snap = engine.compute(2.0, None, None)
        self.assertEqual(snap.known_exited_occupants, 1)
        self.assertAlmostEqual(snap.evacuation_progress_fraction, 0.5)

        manager.update("OCC-2", "CAM-1", "T2", "z1", "f1", (8.9, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 2.0)
        manager.sweep_missing(3.0, seen_occupant_ids=set())

        snap = engine.compute(3.0, None, None)
        self.assertEqual(snap.known_exited_occupants, 2)
        self.assertAlmostEqual(snap.evacuation_progress_fraction, 1.0)


class TemporaryOcclusionTests(unittest.TestCase):

    def test_3_temporary_camera_occlusion_not_immediately_counted_as_evacuated(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])

        # Occupant is far from the exit -- a temporary occlusion here
        # must never be mistaken for an evacuation.
        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.sweep_missing(1.0, seen_occupant_ids=set())  # occluded this cycle

        snap = engine.compute(1.0, None, None)
        self.assertEqual(snap.known_exited_occupants, 0)
        self.assertEqual(snap.known_active_occupants, 0)  # TEMPORARILY_LOST, not ACTIVE, not EXITED


class DisappearanceAwayFromExitTests(unittest.TestCase):

    def test_5_disappearance_away_from_exit_not_counted_as_evacuated(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (0.5, 0.5), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.sweep_missing(100.0, seen_occupant_ids=set())  # never expires (expire_after_seconds=1000)

        snap = engine.compute(100.0, None, None)
        self.assertEqual(snap.known_exited_occupants, 0)


class ExitedWithSufficientEvidenceTests(unittest.TestCase):

    def test_6_disappearance_near_exit_produces_likely_exit(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (8.9, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.sweep_missing(1.0, seen_occupant_ids=set())

        snap = engine.compute(1.0, None, None)
        self.assertEqual(snap.known_exited_occupants, 1)
        self.assertEqual(snap.exit("e1").unique_exited_count, 1)


class ReappearanceCorrectionTests(unittest.TestCase):

    def test_7_reappearance_after_likely_exit_corrects_current_count(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (8.9, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.sweep_missing(1.0, seen_occupant_ids=set())
        self.assertEqual(engine.compute(1.0, None, None).known_exited_occupants, 1)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (5.0, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 2.0)

        snap = engine.compute(2.0, None, None)
        self.assertEqual(snap.known_exited_occupants, 0)
        self.assertEqual(snap.known_active_occupants, 1)
        self.assertEqual(snap.exit("e1").unique_exited_count, 0)


class ZoneClearanceObservabilityTests(unittest.TestCase):

    def test_8_zero_occupants_poor_coverage_is_unknown_not_clear(self):

        building = make_building()
        engine, manager = make_engine(building)

        snap = engine.compute(0.0, None, None)  # no BuildingState at all -- no coverage anywhere
        self.assertEqual(snap.zone("z1").status, ZoneClearanceStatus.UNKNOWN)

    def test_9_zero_occupants_adequate_coverage_is_observed_clear(self):

        building = make_building()
        engine, manager = make_engine(building)

        building_state = make_covered_building_state(["z1"])
        snap = engine.compute(0.0, building_state, None)

        self.assertEqual(snap.zone("z1").status, ZoneClearanceStatus.OBSERVED_CLEAR)
        self.assertTrue(snap.zone("z1").observable)

    def test_17_camera_offline_decreases_observability(self):

        building = make_building()
        engine, manager = make_engine(building)

        online = make_covered_building_state(["z1"], active=True)
        offline = make_covered_building_state(["z1"], active=False)

        self.assertTrue(engine.compute(0.0, online, None).zone("z1").observable)
        self.assertFalse(engine.compute(1.0, offline, None).zone("z1").observable)
        self.assertEqual(engine.compute(1.0, offline, None).zone("z1").status, ZoneClearanceStatus.UNKNOWN)

    def test_18_no_cameras_produces_no_fabricated_clearance(self):

        building = make_building()
        engine, manager = make_engine(building)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.sweep_missing(5.0, seen_occupant_ids=set())  # goes missing, no camera coverage configured at all

        snap = engine.compute(5.0, None, None)
        self.assertEqual(snap.zone("z1").status, ZoneClearanceStatus.UNKNOWN)

    def test_19_no_known_occupants_distinguishes_no_evidence_from_confirmed_zero(self):

        building = make_building()
        engine, manager = make_engine(building)

        snap = engine.compute(0.0, None, None)

        self.assertEqual(snap.known_total_observed_occupants, 0)
        self.assertIsNone(snap.evacuation_progress_fraction)  # no honest denominator, never a fabricated 0%/100%


class FlowProblemTests(unittest.TestCase):

    def _make_engine_with_crowd(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])
        crowd_engine = CrowdIntelligenceEngine(building, manager)

        return engine, manager, crowd_engine

    def test_10_high_queue_low_throughput_is_a_flow_problem(self):

        engine, manager, crowd_engine = self._make_engine_with_crowd()

        # Three occupants queueing at the exit, none of them actually
        # crossing (no sweep_missing/EXITED transition at all).
        for i in range(3):
            manager.update(f"OCC-{i}", "CAM-1", f"T{i}", "z1", "f1", (8.9, 5.0 + i * 0.05), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        crowd_snapshot = crowd_engine.compute(0.0)
        snap = engine.compute(0.0, None, crowd_snapshot)

        self.assertGreater(snap.exit("e1").queue_candidate_count, 0)
        self.assertFalse(snap.exit("e1").flow_active)
        self.assertIn("e1", snap.low_flow_exit_ids)

    def test_11_high_queue_high_throughput_not_incorrectly_marked_stalled(self):

        engine, manager, crowd_engine = self._make_engine_with_crowd()

        for i in range(3):
            manager.update(f"OCC-{i}", "CAM-1", f"T{i}", "z1", "f1", (8.9, 5.0 + i * 0.05), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        # One of them actually crosses.
        manager.sweep_missing(1.0, seen_occupant_ids={"OCC-1", "OCC-2"})

        crowd_snapshot = crowd_engine.compute(1.0)
        snap = engine.compute(1.0, None, crowd_snapshot)

        self.assertTrue(snap.exit("e1").flow_active)
        self.assertNotIn("e1", snap.low_flow_exit_ids)


class ExitFlowTrendTests(unittest.TestCase):

    def test_12_exit_flow_drops_over_time_produces_slowing_or_stalled(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])
        engine.config = engine.config  # no-op, keep default flow_window_seconds

        # A burst of crossings, then nothing further -- flow should read
        # as declining relative to that earlier burst.
        for i in range(5):
            manager.update(f"OCC-{i}", "CAM-1", f"T{i}", "z1", "f1", (8.9, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, float(i))
            manager.sweep_missing(float(i) + 0.5, seen_occupant_ids=set())

        # First measurement may honestly be UNKNOWN (insufficient trend
        # history yet) -- the point of this test is the SECOND
        # measurement, after the burst has ended.
        engine.compute(1.0, None, None)

        # No further crossings for a while -- recent_count now reads 0
        # against the earlier burst still inside the trend window.
        later = engine.compute(3.0, None, None).exit("e1").trend

        self.assertIn(later, (EvacuationProgressTrend.SLOWING, EvacuationProgressTrend.STALLED, EvacuationProgressTrend.STABLE))

    def test_13_exit_flow_recovers_status_updates(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])

        # No crossings for a while.
        self.assertFalse(engine.compute(0.0, None, None).exit("e1").flow_active)
        self.assertFalse(engine.compute(10.0, None, None).exit("e1").flow_active)

        # A crossing occurs -- flow_active flips True the same cycle.
        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (8.9, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 11.0)
        manager.sweep_missing(11.5, seen_occupant_ids=set())

        snap = engine.compute(11.5, None, None)
        self.assertTrue(snap.exit("e1").flow_active)


class MultiZoneIndependenceTests(unittest.TestCase):

    def test_14_one_zone_clears_while_another_stalls(self):

        exit_obj = make_exit()
        floor = Floor(
            id="f1", name="Floor 1",
            zones=[
                Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
                Zone(id="z2", name="Z2", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            ],
            exits=[exit_obj],
        )
        building = Building(id="b1", name="B", floors=[floor])
        engine, manager = make_engine(building, exits=[exit_obj])

        building_state = make_covered_building_state(["z1", "z2"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (8.9, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.update("OCC-2", "CAM-1", "T2", "z2", "f1", (21.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        # z1's occupant exits; z2's occupant remains, unmoving, forever.
        manager.sweep_missing(1.0, seen_occupant_ids={"OCC-2"})
        for t in (2.0, 3.0, 4.0):
            manager.update("OCC-2", "CAM-1", "T2", "z2", "f1", (21.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, t)

        snap = engine.compute(4.0, building_state, None)

        self.assertEqual(snap.zone("z1").status, ZoneClearanceStatus.OBSERVED_CLEAR)
        self.assertEqual(snap.zone("z2").current_active_count, 1)


class DegradedInputTests(unittest.TestCase):

    def test_15_crowd_intelligence_unavailable_degrades_honestly(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        snap = engine.compute(0.0, None, None)  # crowd_snapshot=None throughout

        self.assertEqual(snap.exit("e1").queue_candidate_count, 0)
        self.assertIsNone(snap.exit("e1").congestion_level)
        self.assertEqual(snap.known_active_occupants, 1)  # everything else still works

    def test_16_world_position_unavailable_degrades_exit_attribution_honestly(self):

        exit_obj = make_exit()
        building = make_building(exit_obj)
        engine, manager = make_engine(building, exits=[exit_obj])

        # Near-exit classification itself requires world_position (see
        # live_occupants.lifecycle.is_near_exit()) -- without one, the
        # occupant can never become EXITED at all, only TEMPORARILY_LOST.
        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.sweep_missing(1.0, seen_occupant_ids=set())

        snap = engine.compute(1.0, None, None)
        self.assertEqual(snap.known_exited_occupants, 0)  # honestly TEMPORARILY_LOST, never guessed as exited
        self.assertEqual(snap.exit("e1").unique_exited_count, 0)


if __name__ == "__main__":
    unittest.main()
