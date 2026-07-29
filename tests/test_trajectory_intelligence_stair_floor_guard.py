import unittest

from behavior_recognition.observation import RecognizedBehavior

from live_occupants.manager import LiveOccupantManager

from trajectory_intelligence.models import TrajectoryConfig
from trajectory_intelligence.trajectory import compute_movement_facts


# =====================================================
# Observable Stair Perception milestone, Phase 15 -- deterministic unit
# tests proving trajectory_intelligence.trajectory.compute_movement_facts()
# no longer produces a spurious cross-floor geometry artifact for a
# CONFIRMED floor change (Zone A -> Stair S1 -> Zone B), while leaving
# every same-floor / floor-unknown case exactly as it always computed.
# No Navigation Graph or BuildingState constructed anywhere in this file
# -- this module's own documented boundary stays intact.
# =====================================================


class SameFloorUnaffectedTests(unittest.TestCase):

    def test_1_same_floor_movement_computed_exactly_as_before(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (0.0, 0.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (3.0, 4.0), None, RecognizedBehavior.WALKING, 0.9, 2.0)

        occupant = manager.get("OCC-1")
        facts = compute_movement_facts(occupant, TrajectoryConfig())

        self.assertAlmostEqual(facts["distance_travelled"], 5.0)
        self.assertAlmostEqual(facts["net_displacement"], 5.0)
        self.assertAlmostEqual(facts["current_speed"], 2.5)
        self.assertIsNotNone(facts["movement_direction"])


class ConfirmedFloorChangeGuardTests(unittest.TestCase):

    def test_2_confirmed_floor_change_suppresses_spurious_distance_and_direction(self):

        manager = LiveOccupantManager()

        # Zone A on floor-1 -> (via Stair S1) -> Zone B on floor-2. Two
        # position samples straddling the crossing, in TWO UNRELATED
        # floor-local coordinate spaces (same magnitude coordinates
        # deliberately reused across floors -- if this were computed as
        # a raw Euclidean jump it would read as "teleported 3m").
        manager.update("OCC-1", "CAM-1", "T1", "z1", "floor-1", (0.0, 0.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        manager.update("OCC-1", "CAM-2", "T1", "z2", "floor-2", (3.0, 4.0), None, RecognizedBehavior.WALKING, 0.9, 2.0)

        occupant = manager.get("OCC-1")
        facts = compute_movement_facts(occupant, TrajectoryConfig())

        # The one pair straddles a confirmed floor change -- contributes
        # nothing to distance_travelled, never a fabricated 5.0m jump.
        self.assertEqual(facts["distance_travelled"], 0.0)
        self.assertIsNone(facts["net_displacement"])
        self.assertIsNone(facts["movement_direction"])
        # current_speed's own fallback (raw distance / dt) is likewise
        # suppressed -- honestly None rather than a fabricated 2.5 m/s.
        self.assertIsNone(facts["current_speed"])

        # trajectory_duration is a pure time delta, not a spatial
        # quantity across the two coordinate spaces -- still honestly
        # computable and left untouched.
        self.assertAlmostEqual(facts["trajectory_duration"], 2.0)

    def test_3_third_sample_back_on_a_known_floor_resumes_normal_computation(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "floor-1", (0.0, 0.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        manager.update("OCC-1", "CAM-2", "T1", "z2", "floor-2", (3.0, 4.0), None, RecognizedBehavior.WALKING, 0.9, 2.0)
        manager.update("OCC-1", "CAM-2", "T1", "z2", "floor-2", (6.0, 8.0), None, RecognizedBehavior.WALKING, 0.9, 4.0)

        occupant = manager.get("OCC-1")
        facts = compute_movement_facts(occupant, TrajectoryConfig())

        # Only the floor-1 -> floor-2 pair is excluded; the floor-2 ->
        # floor-2 pair (both samples confirmed same floor) still
        # contributes its real 5.0m normally.
        self.assertAlmostEqual(facts["distance_travelled"], 5.0)
        # The LAST pair (samples[-2] -> samples[-1]) is same-floor, so
        # movement_direction/current_speed's fallback are both honestly
        # computable again.
        self.assertIsNotNone(facts["movement_direction"])
        self.assertAlmostEqual(facts["current_speed"], 2.5)


class UnknownFloorContextPreservesPriorBehaviorTests(unittest.TestCase):

    def test_4_missing_floor_context_never_suppresses_legitimate_data(self):

        manager = LiveOccupantManager()

        # floor_id=None on every update -- floor context genuinely
        # unavailable, not confirmed same or confirmed different. Must
        # NOT be treated as a floor crossing (that would regress
        # legitimate movement data for every pre-milestone caller that
        # never threads floor_id at all).
        manager.update("OCC-1", "CAM-1", "T1", "z1", None, (0.0, 0.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        manager.update("OCC-1", "CAM-1", "T1", "z1", None, (3.0, 4.0), None, RecognizedBehavior.WALKING, 0.9, 2.0)

        occupant = manager.get("OCC-1")
        facts = compute_movement_facts(occupant, TrajectoryConfig())

        self.assertAlmostEqual(facts["distance_travelled"], 5.0)
        self.assertAlmostEqual(facts["net_displacement"], 5.0)
        self.assertIsNotNone(facts["movement_direction"])


if __name__ == "__main__":
    unittest.main()
