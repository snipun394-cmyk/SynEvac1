import unittest

from behavior_recognition.observation import RecognizedBehavior

from models.staircase import Staircase, StairObservableRegion
from models.zone import Zone

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from live_occupants.manager import LiveOccupantManager
from live_occupants.state import OccupantStatus

from building_state.estimator import BuildingStateEstimator
from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from stair_perception.facts import compute_stair_occupancy_snapshot
from stair_perception.models import StairObservationStatus


# =====================================================
# Observable Stair Perception milestone -- deterministic, offline unit
# tests proving the chain: calibrated projection -> spatial stair lookup
# -> LiveOccupant.current_stair_id -> canonical occupancy -> Stair
# observation snapshot -> BuildingState. Mirrors tests/test_camera_
# calibration.py's own hand-verifiable trigonometry convention: pitch=
# 45deg, mount_height=3.0 -> straight-ahead ground point is exactly
# height/tan(pitch) = 3.0 meters away.
# =====================================================


def make_calibration(camera_id="CAM-1", floor_id="floor-1", position=(0.0, 0.0)):

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=position, mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0, roll_degrees=0.0)

    return CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)


CENTER_PIXEL_BOX = (315.0, 200.0, 325.0, 240.0)  # projects to (3.0, 0.0) on the camera's own floor


def make_stair(stair_id="STAIR-1", floor_id="floor-1", center=(3.0, 0.0), size=4.0):

    stair = Staircase(name="Stair", from_floor_id=floor_id, to_floor_id="floor-2", from_position=center, to_position=(100.0, 100.0))
    stair.id = stair_id
    stair.from_observable_region = StairObservableRegion(center_x=center[0], center_y=center[1], width=size, depth=size)
    return stair


class WorldProjectorStairFieldTests(unittest.TestCase):

    def test_1_stair_id_populated_when_point_falls_inside_region(self):

        stair = make_stair()
        projector = WorldProjector(
            calibrations={"CAM-1": make_calibration()}, zones_by_floor={},
            stairs_by_floor={"floor-1": (stair,)},
        )

        result = projector.project("CAM-1", CENTER_PIXEL_BOX, 0.9)

        self.assertEqual(result.stair_id, "STAIR-1")
        self.assertFalse(result.stair_localization_ambiguous)

    def test_2_stair_id_none_when_no_stairs_configured(self):

        projector = WorldProjector(calibrations={"CAM-1": make_calibration()}, zones_by_floor={})

        result = projector.project("CAM-1", CENTER_PIXEL_BOX, 0.9)

        self.assertIsNone(result.stair_id)
        self.assertFalse(result.stair_localization_ambiguous)

    def test_3_zone_and_stair_are_independent_never_conflated(self):

        # A Zone polygon and a Stair observable region genuinely overlap
        # at the same point -- both fields must be populated
        # independently, neither one silently overwriting the other.
        zone = Zone(name="Stair Lobby", x=0.0, y=-5.0, width=10.0, height=10.0, floor_id="floor-1")
        zone.id = "ZONE-1"

        stair = make_stair()
        projector = WorldProjector(
            calibrations={"CAM-1": make_calibration()}, zones_by_floor={"floor-1": (zone,)},
            stairs_by_floor={"floor-1": (stair,)},
        )

        result = projector.project("CAM-1", CENTER_PIXEL_BOX, 0.9)

        self.assertEqual(result.zone_id, "ZONE-1")
        self.assertEqual(result.stair_id, "STAIR-1")

    def test_4_ambiguous_overlap_never_arbitrarily_resolved(self):

        stair_a = make_stair(stair_id="STAIR-1", center=(3.0, 0.0))
        stair_b = make_stair(stair_id="STAIR-2", center=(3.0, 0.0))

        projector = WorldProjector(
            calibrations={"CAM-1": make_calibration()}, zones_by_floor={},
            stairs_by_floor={"floor-1": (stair_a, stair_b)},
        )

        result = projector.project("CAM-1", CENTER_PIXEL_BOX, 0.9)

        self.assertIsNone(result.stair_id)
        self.assertTrue(result.stair_localization_ambiguous)

    def test_5_calibrated_floor_ids_reflects_construction(self):

        projector = WorldProjector(
            calibrations={"CAM-1": make_calibration(floor_id="floor-1"), "CAM-2": make_calibration(camera_id="CAM-2", floor_id="floor-2")},
            zones_by_floor={},
        )

        self.assertEqual(projector.calibrated_floor_ids(), frozenset({"floor-1", "floor-2"}))


class LiveOccupantCurrentStairIdTests(unittest.TestCase):

    def test_6_occupant_gains_current_stair_id(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-1", "T1", None, "floor-1", (3.0, 0.0), 0.0,
            RecognizedBehavior.STATIONARY, 0.9, 0.0, stair_id="STAIR-1",
        )

        self.assertEqual(occupant.current_stair_id, "STAIR-1")

    def test_7_stair_id_coexists_independently_with_zone_id(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-1", "T1", "ZONE-1", "floor-1", (3.0, 0.0), 0.0,
            RecognizedBehavior.STATIONARY, 0.9, 0.0, stair_id="STAIR-1",
        )

        self.assertEqual(occupant.current_zone_id, "ZONE-1")
        self.assertEqual(occupant.current_stair_id, "STAIR-1")

    def test_8_leaving_the_stair_clears_current_stair_id_next_update(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-1", "T1", None, "floor-1", (3.0, 0.0), 0.0, None, 0.9, 0.0, stair_id="STAIR-1")
        left = manager.update("OCC-1", "CAM-1", "T1", "ZONE-1", "floor-1", (30.0, 30.0), 0.0, None, 0.9, 1.0, stair_id=None)

        self.assertIsNone(left.current_stair_id)
        self.assertEqual(left.current_zone_id, "ZONE-1")

    def test_9_stair_transition_recorded_in_history(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-1", "T1", None, "floor-1", (3.0, 0.0), 0.0, None, 0.9, 0.0, stair_id="STAIR-1")
        updated = manager.update("OCC-1", "CAM-1", "T1", None, "floor-1", (3.0, 0.0), 0.0, None, 0.9, 1.0, stair_id=None)

        transitions = updated.history.stair_transitions
        self.assertEqual(len(transitions), 2)  # None->STAIR-1 (entry), STAIR-1->None (exit)
        self.assertIsNone(transitions[0].from_stair_id)
        self.assertEqual(transitions[0].to_stair_id, "STAIR-1")
        self.assertEqual(transitions[1].from_stair_id, "STAIR-1")
        self.assertIsNone(transitions[1].to_stair_id)

    def test_10_occupants_on_stair_query_and_index(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-1", "T1", None, "floor-1", (3.0, 0.0), 0.0, None, 0.9, 0.0, stair_id="STAIR-1")
        manager.update("OCC-2", "CAM-1", "T2", None, "floor-1", (3.0, 0.0), 0.0, None, 0.9, 0.0, stair_id="STAIR-1")
        manager.update("OCC-3", "CAM-1", "T3", "ZONE-1", "floor-1", (30.0, 30.0), 0.0, None, 0.9, 0.0, stair_id=None)

        on_stair = manager.occupants_on_stair("STAIR-1")

        self.assertEqual({o.occupant_id for o in on_stair}, {"OCC-1", "OCC-2"})

    def test_11_temporary_detection_loss_preserves_current_stair_id(self):

        manager = LiveOccupantManager(expire_after_seconds=30.0)

        manager.update("OCC-1", "CAM-1", "T1", None, "floor-1", (3.0, 0.0), 0.0, None, 0.9, 0.0, stair_id="STAIR-1")

        # One frame missed entirely -- sweep_missing() never calls
        # update(), so current_stair_id must stay frozen, not silently
        # cleared, while status honestly reflects the loss.
        manager.sweep_missing(1.0, seen_occupant_ids=set())

        occupant = manager.get("OCC-1")
        self.assertEqual(occupant.status, OccupantStatus.TEMPORARILY_LOST)
        self.assertEqual(occupant.current_stair_id, "STAIR-1")

        # Not counted in active occupancy while temporarily lost --
        # same existing convention current_zone_id already has.
        self.assertNotIn("OCC-1", manager.canonical_occupancy(1.0).occupant_ids_by_stair.get("STAIR-1", ()))

        # Seen again within expire_after_seconds -- recovers cleanly,
        # never fabricating an "entered/exited" pair for the gap.
        recovered = manager.update("OCC-1", "CAM-1", "T1", None, "floor-1", (3.0, 0.0), 0.0, None, 0.9, 2.0, stair_id="STAIR-1")
        self.assertEqual(recovered.status, OccupantStatus.ACTIVE)
        self.assertEqual(recovered.current_stair_id, "STAIR-1")
        self.assertIn("OCC-1", manager.canonical_occupancy(2.0).occupant_ids_by_stair.get("STAIR-1", ()))


class OccupancyFactsStairGroupingTests(unittest.TestCase):

    def test_12_occupant_ids_by_stair_computed_and_no_double_count(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-1", "T1", "ZONE-1", "floor-1", (3.0, 0.0), 0.0, None, 0.9, 0.0, stair_id="STAIR-1")
        manager.update("OCC-2", "CAM-1", "T2", None, "floor-1", (3.0, 0.0), 0.0, None, 0.9, 0.0, stair_id="STAIR-1")

        facts = manager.canonical_occupancy(0.0)

        self.assertEqual(set(facts.occupant_ids_by_stair["STAIR-1"]), {"OCC-1", "OCC-2"})
        # Total headcount is keyed by occupant_id membership, not by
        # summing zone+stair groupings -- OCC-1 legitimately appears in
        # BOTH occupant_ids_by_zone and occupant_ids_by_stair without
        # inflating total_observed_occupant_ids.
        self.assertEqual(sorted(facts.total_observed_occupant_ids), ["OCC-1", "OCC-2"])
        self.assertEqual(facts.stair_count("STAIR-1"), 2)

    def test_13_not_on_a_stair_never_counted_as_unlocalized(self):

        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-1", "T1", "ZONE-1", "floor-1", (0.0, 0.0), 0.0, None, 0.9, 0.0, stair_id=None)

        facts = manager.canonical_occupancy(0.0)

        self.assertEqual(facts.unlocalized_occupant_ids, ())
        self.assertEqual(facts.occupant_ids_by_stair, {})


class StairOccupancySnapshotTests(unittest.TestCase):

    def test_14_covered_stair_reports_observed_status(self):

        snapshot = compute_stair_occupancy_snapshot(
            stair_ids=["STAIR-1"], occupant_ids_by_stair={"STAIR-1": ("OCC-1", "OCC-2")},
            covered_stair_ids=frozenset({"STAIR-1"}), timestamp=0.0,
        )

        observation = snapshot.observation_for("STAIR-1")
        self.assertEqual(observation.status, StairObservationStatus.OBSERVED)
        self.assertEqual(observation.occupant_count, 2)

    def test_15_covered_stair_with_zero_occupants_is_observed_zero_not_unknown(self):

        snapshot = compute_stair_occupancy_snapshot(
            stair_ids=["STAIR-1"], occupant_ids_by_stair={}, covered_stair_ids=frozenset({"STAIR-1"}), timestamp=0.0,
        )

        observation = snapshot.observation_for("STAIR-1")
        self.assertEqual(observation.status, StairObservationStatus.OBSERVED)
        self.assertEqual(observation.occupant_count, 0)

    def test_16_uncovered_stair_is_unknown_never_a_fabricated_zero(self):

        snapshot = compute_stair_occupancy_snapshot(
            stair_ids=["STAIR-2"], occupant_ids_by_stair={}, covered_stair_ids=frozenset(), timestamp=0.0,
        )

        observation = snapshot.observation_for("STAIR-2")
        self.assertEqual(observation.status, StairObservationStatus.UNKNOWN)
        self.assertEqual(observation.occupant_count, 0)
        self.assertEqual(observation.occupant_ids, ())

    def test_17_entirely_absent_stair_id_also_defaults_unknown(self):

        snapshot = compute_stair_occupancy_snapshot(
            stair_ids=[], occupant_ids_by_stair={}, covered_stair_ids=frozenset(), timestamp=0.0,
        )

        observation = snapshot.observation_for("NEVER-SEEN")
        self.assertEqual(observation.status, StairObservationStatus.UNKNOWN)


class BuildingStateStairOccupancyPassthroughTests(unittest.TestCase):

    def test_18_none_when_not_configured_never_fabricated(self):

        estimator = BuildingStateEstimator()

        state = estimator.estimate(
            0.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
        )

        self.assertIsNone(state.stair_occupancy)

    def test_19_pure_passthrough_when_supplied(self):

        estimator = BuildingStateEstimator()

        stair_snapshot = compute_stair_occupancy_snapshot(
            stair_ids=["STAIR-1"], occupant_ids_by_stair={"STAIR-1": ("OCC-1",)},
            covered_stair_ids=frozenset({"STAIR-1"}), timestamp=0.0,
        )

        state = estimator.estimate(
            0.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
            stair_occupancy_snapshot=stair_snapshot,
        )

        self.assertIs(state.stair_occupancy, stair_snapshot)
        self.assertEqual(state.stair_occupancy.observation_for("STAIR-1").occupant_count, 1)

        # zone_occupancy stays exactly what it always was -- Stair
        # occupancy is a genuinely separate sibling, never folded in.
        self.assertIsInstance(state.zone_occupancy, OccupancySnapshot)


if __name__ == "__main__":
    unittest.main()
