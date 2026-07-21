import unittest

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanClassification, HumanState

from models.building import Building
from models.floor import Floor
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from emergency_response.engine import EmergencyResponseIntelligenceEngine
from emergency_response.models import ResponsePriorityLevel, ResponseReason


# =====================================================
# Live Human State & Assistance Perception Bridge milestone, Phase 23 --
# covers required test-matrix items 12, 13, 14, 31-36: the live-sourced
# LiveOccupant.human_state/human_classification path now reaching
# EmergencyResponseIntelligenceEngine, distinct from the pre-existing
# human_state_by_occupant_id caller override (already covered by
# tests/test_emergency_response.py::AssistanceSignalTests). No
# randomness anywhere in this file.
# =====================================================


def make_building():

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1")],
    )

    return Building(id="b1", name="B", floors=[floor])


def make_engine(building):

    bus = EventBus()
    manager = LiveOccupantManager(event_bus=bus, exits=[], expire_after_seconds=1000.0)
    engine = EmergencyResponseIntelligenceEngine(building, manager)

    return manager, engine


class LiveSourcedAssistanceTests(unittest.TestCase):

    def test_12_possibly_fallen_never_becomes_fallen(self):

        building = make_building()
        manager, engine = make_engine(building)

        manager.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 0.0,
        )

        occupant = manager.get("OCC-1")
        self.assertIsNone(occupant.human_state)

    def test_13_possibly_fallen_produces_only_possible_assistance_evidence(self):

        building = make_building()
        manager, engine = make_engine(building)

        manager.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 0.0,
        )

        snap = engine.compute(0.0, None, None, None)
        zone = snap.zone("z1")

        self.assertEqual(zone.possible_assistance_count, 1)
        self.assertEqual(zone.confirmed_assistance_count, 0)
        self.assertIn(ResponseReason.POSSIBLE_ASSISTANCE_REQUIRED, zone.reason_codes)
        self.assertNotIn(ResponseReason.CONFIRMED_ASSISTANCE_REQUIRED, zone.reason_codes)

    def test_14_live_sourced_fallen_produces_stronger_confirmed_assistance(self):

        building = make_building()
        manager, engine = make_engine(building)

        manager.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0,
            state_evidence=HumanState.FALLEN, state_confidence=0.9,
        )

        snap = engine.compute(0.0, None, None, None)
        zone = snap.zone("z1")

        self.assertEqual(zone.confirmed_assistance_count, 1)
        self.assertIn(ResponseReason.CONFIRMED_ASSISTANCE_REQUIRED, zone.reason_codes)

    def test_31_confirmed_fallen_evidence_increases_priority_score(self):

        building = make_building()

        manager_a, engine_a = make_engine(building)
        manager_a.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)
        baseline_score = engine_a.compute(0.0, None, None, None).zone("z1").priority_score

        manager_b, engine_b = make_engine(building)
        manager_b.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0,
            state_evidence=HumanState.FALLEN, state_confidence=0.9,
        )
        fallen_score = engine_b.compute(0.0, None, None, None).zone("z1").priority_score

        self.assertGreater(fallen_score, baseline_score)

    def test_32_possible_fall_heuristic_has_weaker_effect_than_confirmed_fallen(self):

        building = make_building()

        manager_a, engine_a = make_engine(building)
        manager_a.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 0.0,
        )
        possible_score = engine_a.compute(0.0, None, None, None).zone("z1").priority_score

        manager_b, engine_b = make_engine(building)
        manager_b.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0,
            state_evidence=HumanState.FALLEN, state_confidence=0.9,
        )
        confirmed_score = engine_b.compute(0.0, None, None, None).zone("z1").priority_score

        self.assertGreater(confirmed_score, possible_score)

    def test_33_being_assisted_is_distinguishable_from_unassisted_fallen(self):

        building = make_building()

        manager_a, engine_a = make_engine(building)
        manager_a.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0,
            state_evidence=HumanState.FALLEN, state_confidence=0.9,
        )
        fallen_zone = engine_a.compute(0.0, None, None, None).zone("z1")

        manager_b, engine_b = make_engine(building)
        manager_b.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0,
            state_evidence=HumanState.BEING_ASSISTED, state_confidence=0.9,
        )
        assisted_zone = engine_b.compute(0.0, None, None, None).zone("z1")

        self.assertEqual(fallen_zone.confirmed_assistance_count, 1)
        self.assertEqual(fallen_zone.being_assisted_count, 0)
        self.assertIn(ResponseReason.CONFIRMED_ASSISTANCE_REQUIRED, fallen_zone.reason_codes)

        self.assertEqual(assisted_zone.confirmed_assistance_count, 0)
        self.assertEqual(assisted_zone.being_assisted_count, 1)
        self.assertIn(ResponseReason.ASSISTANCE_IN_PROGRESS, assisted_zone.reason_codes)

        self.assertNotEqual(fallen_zone.priority_score, assisted_zone.priority_score)


class NoFabricationTests(unittest.TestCase):

    def test_34_poor_camera_coverage_does_not_fabricate_human_classification(self):

        building = make_building()
        manager, engine = make_engine(building)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

        occupant = manager.get("OCC-1")
        self.assertEqual(occupant.human_classification, HumanClassification.UNKNOWN)

    def test_35_camera_offline_does_not_fabricate_human_state(self):

        building = make_building()
        manager, engine = make_engine(building)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

        occupant = manager.get("OCC-1")
        self.assertIsNone(occupant.human_state)

    def test_36_no_hazard_or_advisory_gateway_configured_evidence_still_processes(self):

        building = make_building()
        manager, engine = make_engine(building)

        manager.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0,
            state_evidence=HumanState.FALLEN, state_confidence=0.9,
        )

        # building_state/crowd_snapshot/evacuation_progress_snapshot all
        # None -- human evidence must still be processed honestly.
        snap = engine.compute(0.0, None, None, None)

        self.assertEqual(snap.zone("z1").confirmed_assistance_count, 1)


class VulnerableClassificationConservatismTests(unittest.TestCase):

    def test_vulnerable_classification_contributes_small_awareness_weight_only(self):

        building = make_building()

        manager_a, engine_a = make_engine(building)
        manager_a.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)
        baseline_score = engine_a.compute(0.0, None, None, None).zone("z1").priority_score

        manager_b, engine_b = make_engine(building)
        manager_b.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0,
            classification_evidence=HumanClassification.CHILD, classification_confidence=0.9,
        )
        child_zone = engine_b.compute(0.0, None, None, None).zone("z1")

        # Raises awareness slightly -- never approaches confirmed/
        # possible assistance's own magnitude (Phase 12's conservatism).
        self.assertGreater(child_zone.priority_score, baseline_score)
        self.assertLess(child_zone.priority_score - baseline_score, engine_b.weights.possible_assistance_weight)
        self.assertIn(ResponseReason.VULNERABLE_PERSON_OBSERVED, child_zone.reason_codes)
        self.assertEqual(child_zone.confirmed_assistance_count, 0)
        self.assertEqual(child_zone.possible_assistance_count, 0)

    def test_firefighter_classification_never_treated_as_vulnerable(self):

        building = make_building()
        manager, engine = make_engine(building)

        manager.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0,
            classification_evidence=HumanClassification.FIREFIGHTER, classification_confidence=0.9,
        )

        zone = engine.compute(0.0, None, None, None).zone("z1")
        self.assertFalse(zone.vulnerable_person_observed)
        self.assertNotIn(ResponseReason.VULNERABLE_PERSON_OBSERVED, zone.reason_codes)


if __name__ == "__main__":
    unittest.main()
