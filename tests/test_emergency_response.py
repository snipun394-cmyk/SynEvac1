import unittest

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanState

from models.building import Building
from models.floor import Floor
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from camera_manager.status import CameraStatus

from building_state.models import BuildingState, CameraAssetState, DetectorAssetState, HazardSummary

from hazard.severity import HazardSeverity

from crowd_intelligence.engine import CrowdIntelligenceEngine

from evacuation_progress.engine import EvacuationProgressEngine

from emergency_response.engine import EmergencyResponseIntelligenceEngine
from emergency_response.models import ResponsePriorityLevel, ResponseReason


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone,
# Phase 22 -- deterministic engine-level unit coverage (items 1, 3-15,
# 17-24; item 16 needs the full multi-camera pipeline -- see
# tests/test_emergency_response_double_counting.py; items 25-30 need
# Advisory -- see tests/test_emergency_response_advisory_safety_precedence.py).
# No randomness anywhere in this file.
# =====================================================


def make_building():

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1")],
    )

    return Building(id="b1", name="B", floors=[floor])


def make_covered_building_state(zone_ids, camera_id="CAM-1", active=True, hazard_severity=None):

    status = CameraStatus(
        camera_id=camera_id, name=camera_id, floor_id="f1", zone_ids=tuple(zone_ids),
        active=active, mode="LIVE", has_detection_provider=True,
    )

    hazard_summary = HazardSummary()
    if hazard_severity is not None:
        hazard_summary = HazardSummary(zone_severities={zid: hazard_severity for zid in zone_ids})

    return BuildingState(camera_observations={camera_id: CameraAssetState(status=status)}, hazard_summary=hazard_summary)


def make_engines(building):

    bus = EventBus()
    manager = LiveOccupantManager(event_bus=bus, exits=[], expire_after_seconds=1000.0)
    crowd_engine = CrowdIntelligenceEngine(building, manager)
    progress_engine = EvacuationProgressEngine(building, manager, bus)
    response_engine = EmergencyResponseIntelligenceEngine(building, manager)

    return manager, crowd_engine, progress_engine, response_engine


class OccupiedLowHazardTests(unittest.TestCase):

    def test_1_occupied_low_hazard_clearing_normally_is_moderate_or_lower(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)
        building_state = make_covered_building_state(["z1"])

        snap = response_engine.compute(0.0, building_state, crowd_engine.compute(0.0), progress_engine.compute(0.0, building_state, None))

        self.assertIn(snap.zone("z1").priority_level, (ResponsePriorityLevel.LOW, ResponsePriorityLevel.MODERATE))


class StalledIncreasesTests(unittest.TestCase):

    def test_2_stalled_evacuation_increases_priority(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        baseline = response_engine.compute(0.0, building_state, None, None).zone("z1").priority_score

        # Manually simulate a STALLED evacuation_progress reading.
        from evacuation_progress.models import EvacuationProgressSnapshot, ZoneClearance, ZoneClearanceStatus

        stalled_progress = EvacuationProgressSnapshot(
            timestamp=1.0,
            zones={"z1": ZoneClearance(zone_id="z1", baseline_observed_count=1, current_active_count=1, status=ZoneClearanceStatus.STALLED, observable=True)},
        )

        stalled_score = response_engine.compute(1.0, building_state, None, stalled_progress).zone("z1").priority_score

        self.assertGreater(stalled_score, baseline)

    def test_3_stalled_plus_high_hazard_is_even_higher(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        from evacuation_progress.models import EvacuationProgressSnapshot, ZoneClearance, ZoneClearanceStatus

        stalled_progress = EvacuationProgressSnapshot(
            timestamp=0.0,
            zones={"z1": ZoneClearance(zone_id="z1", baseline_observed_count=1, current_active_count=1, status=ZoneClearanceStatus.STALLED, observable=True)},
        )

        low_hazard_state = make_covered_building_state(["z1"], hazard_severity=HazardSeverity.LOW)
        high_hazard_state = make_covered_building_state(["z1"], hazard_severity=HazardSeverity.CRITICAL)

        low_score = response_engine.compute(0.0, low_hazard_state, None, stalled_progress).zone("z1").priority_score
        high_score = response_engine.compute(0.0, high_hazard_state, None, stalled_progress).zone("z1").priority_score

        self.assertGreater(high_score, low_score)


class AssistanceSignalTests(unittest.TestCase):

    def test_4_possible_assistance_signal_increases_priority(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)
        walking_score = response_engine.compute(0.0, building_state, None, None).zone("z1").priority_score

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 1.0)
        fallen_score = response_engine.compute(1.0, building_state, None, None).zone("z1").priority_score

        self.assertGreater(fallen_score, walking_score)

    def test_5_possibly_fallen_produces_possible_never_confirmed(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 0.0)

        zone = response_engine.compute(0.0, building_state, None, None).zone("z1")

        self.assertEqual(zone.possible_assistance_count, 1)
        self.assertEqual(zone.confirmed_assistance_count, 0)
        self.assertIn(ResponseReason.POSSIBLE_ASSISTANCE_REQUIRED, zone.reason_codes)
        self.assertNotIn(ResponseReason.CONFIRMED_ASSISTANCE_REQUIRED, zone.reason_codes)

    def test_6_confirmed_human_state_fallen_is_stronger_than_possibly_fallen(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 0.0)
        possible_score = response_engine.compute(0.0, building_state, None, None).zone("z1").priority_score

        confirmed_score = response_engine.compute(
            0.0, building_state, None, None, human_state_by_occupant_id={"OCC-1": HumanState.FALLEN},
        ).zone("z1").priority_score

        self.assertGreater(confirmed_score, possible_score)

        zone = response_engine.compute(
            0.0, building_state, None, None, human_state_by_occupant_id={"OCC-1": HumanState.FALLEN},
        ).zone("z1")
        self.assertEqual(zone.confirmed_assistance_count, 1)
        self.assertIn(ResponseReason.CONFIRMED_ASSISTANCE_REQUIRED, zone.reason_codes)
        self.assertNotIn(ResponseReason.POSSIBLE_ASSISTANCE_REQUIRED, zone.reason_codes)


class ObservedClearTests(unittest.TestCase):

    def test_7_observed_clear_good_coverage_reduces_priority(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])

        from evacuation_progress.models import EvacuationProgressSnapshot, ZoneClearance, ZoneClearanceStatus

        clear_progress = EvacuationProgressSnapshot(
            timestamp=0.0,
            zones={"z1": ZoneClearance(zone_id="z1", baseline_observed_count=1, current_active_count=0, status=ZoneClearanceStatus.OBSERVED_CLEAR, observable=True)},
        )

        zone = response_engine.compute(0.0, building_state, None, clear_progress).zone("z1")

        self.assertEqual(zone.priority_level, ResponsePriorityLevel.LOW)
        self.assertIn(ResponseReason.OBSERVED_CLEAR, zone.reason_codes)

    def test_8_zero_occupants_poor_coverage_not_treated_as_safely_clear(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])

        from evacuation_progress.models import EvacuationProgressSnapshot, ZoneClearance, ZoneClearanceStatus

        unknown_progress = EvacuationProgressSnapshot(
            timestamp=0.0,
            zones={"z1": ZoneClearance(zone_id="z1", baseline_observed_count=1, current_active_count=0, status=ZoneClearanceStatus.UNKNOWN, observable=False)},
        )

        zone = response_engine.compute(0.0, building_state, None, unknown_progress).zone("z1")

        self.assertNotEqual(zone.priority_level, ResponsePriorityLevel.LOW)
        self.assertIn(ResponseReason.UNCERTAIN_OCCUPANCY, zone.reason_codes)

    def test_9_camera_offline_unresolved_clearance_produces_uncertainty_reason(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"], active=False)

        from evacuation_progress.models import EvacuationProgressSnapshot, ZoneClearance, ZoneClearanceStatus

        unknown_progress = EvacuationProgressSnapshot(
            timestamp=0.0,
            zones={"z1": ZoneClearance(zone_id="z1", baseline_observed_count=1, current_active_count=0, status=ZoneClearanceStatus.UNKNOWN, observable=False)},
        )

        zone = response_engine.compute(0.0, building_state, None, unknown_progress).zone("z1")
        self.assertIn(ResponseReason.UNCERTAIN_OCCUPANCY, zone.reason_codes)

    def test_10_no_cameras_produces_no_fabricated_clear_zones(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)

        zone = response_engine.compute(0.0, None, None, None).zone("z1")
        self.assertNotIn(ResponseReason.OBSERVED_CLEAR, zone.reason_codes)


class DeterministicOrderingTests(unittest.TestCase):

    def test_11_multiple_zones_deterministic_priority_ordering(self):

        floor = Floor(
            id="f1", name="Floor 1",
            zones=[
                Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
                Zone(id="z2", name="Z2", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            ],
        )
        building = Building(id="b1", name="B", floors=[floor])
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1", "z2"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 0.0)
        manager.update("OCC-2", "CAM-1", "T2", "z2", "f1", (21.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

        results = [
            response_engine.compute(0.0, building_state, None, None).response_priority_order
            for _ in range(5)
        ]

        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(results[0][0], "z1")  # the fallen-occupant zone ranks first

    def test_12_tied_priority_score_deterministic_tie_break(self):

        floor = Floor(
            id="f1", name="Floor 1",
            zones=[
                Zone(id="z2", name="Z2", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
                Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            ],
        )
        building = Building(id="b1", name="B", floors=[floor])
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1", "z2"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)
        manager.update("OCC-2", "CAM-1", "T2", "z2", "f1", (21.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

        order = response_engine.compute(0.0, building_state, None, None).response_priority_order

        # Identical evidence -> identical score -> alphabetical zone_id tie-break.
        self.assertEqual(order, ("z1", "z2"))


class TemporalChangeTests(unittest.TestCase):

    def test_13_zone_escalates_from_moderate_to_critical(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)
        before = response_engine.compute(0.0, building_state, None, None).zone("z1").priority_level

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 1.0)
        critical_state = make_covered_building_state(["z1"], hazard_severity=HazardSeverity.CRITICAL)

        from evacuation_progress.models import EvacuationProgressSnapshot, ZoneClearance, ZoneClearanceStatus
        stalled_progress = EvacuationProgressSnapshot(
            timestamp=1.0,
            zones={"z1": ZoneClearance(zone_id="z1", baseline_observed_count=1, current_active_count=1, status=ZoneClearanceStatus.STALLED, observable=True)},
        )

        after = response_engine.compute(1.0, critical_state, None, stalled_progress).zone("z1").priority_level

        self.assertNotEqual(before, ResponsePriorityLevel.CRITICAL)
        self.assertEqual(after, ResponsePriorityLevel.CRITICAL)


class CrowdDegradationTests(unittest.TestCase):

    def test_18_high_congestion_plus_stalled_is_represented(self):

        # A small zone (2 m^2) so 5 stationary occupants genuinely
        # register as HIGH+ density (2.5 people/m^2) -- crowd_
        # intelligence.models.DensityThresholds' own default high_at is
        # 2.0 people/m^2.
        floor = Floor(id="f1", name="Floor 1", zones=[Zone(id="z1", name="Z1", x=0.0, y=0.0, width=2.0, height=1.0, floor_id="f1")])
        building = Building(id="b1", name="B", floors=[floor])
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)

        for i in range(5):
            manager.update(f"OCC-{i}", "CAM-1", f"T{i}", "z1", "f1", (0.1 * i, 0.5), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        crowd_snapshot = crowd_engine.compute(0.0)

        from evacuation_progress.models import EvacuationProgressSnapshot, ZoneClearance, ZoneClearanceStatus
        stalled_progress = EvacuationProgressSnapshot(
            timestamp=0.0,
            zones={"z1": ZoneClearance(zone_id="z1", baseline_observed_count=5, current_active_count=5, status=ZoneClearanceStatus.STALLED, observable=True)},
        )

        zone = response_engine.compute(0.0, None, crowd_snapshot, stalled_progress).zone("z1")
        self.assertIn(ResponseReason.HIGH_CONGESTION_RESTRICTING_EVACUATION, zone.reason_codes)

    def test_19_high_congestion_excellent_throughput_not_incorrectly_stalled(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)

        for i in range(5):
            manager.update(f"OCC-{i}", "CAM-1", f"T{i}", "z1", "f1", (1.0 + i, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        crowd_snapshot = crowd_engine.compute(0.0)

        # Clearing normally (not stalled) despite high density.
        from evacuation_progress.models import EvacuationProgressSnapshot, ZoneClearance, ZoneClearanceStatus
        clearing_progress = EvacuationProgressSnapshot(
            timestamp=0.0,
            zones={"z1": ZoneClearance(zone_id="z1", baseline_observed_count=10, current_active_count=5, status=ZoneClearanceStatus.CLEARING, observable=True)},
        )

        zone = response_engine.compute(0.0, None, crowd_snapshot, clearing_progress).zone("z1")
        self.assertNotIn(ResponseReason.HIGH_CONGESTION_RESTRICTING_EVACUATION, zone.reason_codes)


class DegradedInputTests(unittest.TestCase):

    def test_21_crowd_unavailable_response_engine_degrades_honestly(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

        zone = response_engine.compute(0.0, building_state, None, None).zone("z1")
        self.assertEqual(zone.known_occupant_count, 1)
        self.assertNotIn(ResponseReason.HIGH_CONGESTION_RESTRICTING_EVACUATION, zone.reason_codes)

    def test_22_evacuation_progress_unavailable_response_engine_degrades_honestly(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

        zone = response_engine.compute(0.0, building_state, None, None).zone("z1")
        self.assertFalse(zone.evacuation_stalled)
        self.assertIsNone(zone.clearance_status)

    def test_23_hazard_unavailable_never_fabricates_safe_conditions(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

        zone = response_engine.compute(0.0, None, None, None).zone("z1")
        self.assertIsNone(zone.hazard_severity)
        self.assertNotIn(ResponseReason.HAZARD_PRESENT, zone.reason_codes)

    def test_24_facp_unavailable_response_engine_still_operates(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)
        building_state = make_covered_building_state(["z1"])  # no facp_status at all

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

        zone = response_engine.compute(0.0, building_state, None, None).zone("z1")
        self.assertEqual(zone.known_occupant_count, 1)
        self.assertNotIn(ResponseReason.FACP_ALARM_ACTIVE, zone.reason_codes)


class NoEvidenceTests(unittest.TestCase):

    def test_20_no_evidence_at_all_is_unknown_priority(self):

        building = make_building()
        manager, crowd_engine, progress_engine, response_engine = make_engines(building)

        zone = response_engine.compute(0.0, None, None, None).zone("z1")
        self.assertEqual(zone.priority_level, ResponsePriorityLevel.UNKNOWN)
        self.assertIsNone(zone.priority_score)


if __name__ == "__main__":
    unittest.main()
