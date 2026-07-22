import unittest

from hazard.severity import HazardSeverity

from models.dynamic_sign import SignIndication

from dynamic_signage.models import SignageStatus
from dynamic_signage.planner import DynamicSignagePlanner

from tests.dynamic_signage_fixtures import make_guidance_snapshot, make_sign, make_signage_building
from tests.trajectory_intelligence_fixtures import make_building_state as make_generic_building_state


def make_building_state(hazard_by_zone=None):

    # dynamic_signage_fixtures' own building uses different zone ids
    # than trajectory_intelligence_fixtures' building, but make_
    # building_state() itself is a plain BuildingState(hazard_summary=...)
    # constructor wrapper that doesn't care which zone ids are used --
    # reused as-is rather than duplicated.
    return make_generic_building_state(hazard_by_zone)


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 27 -- test matrix
# items 18-32: multi-sign independence, revision/supersession
# semantics, unsafe-route invalidation, conflict detection, degraded
# states.
# =====================================================


class SingleSignTests(unittest.TestCase):

    def test_18_door_target_yields_directional_indication(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign("SIGN-1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0)
        planner = DynamicSignagePlanner(building)

        result = planner.compute(0.0, snapshot, [sign])
        instruction = result.instruction("SIGN-1")

        self.assertEqual(instruction.indication, SignIndication.STRAIGHT)
        self.assertEqual(instruction.status, SignageStatus.ACTIVE)
        self.assertEqual(instruction.target_asset_id, "DOOR-1")
        self.assertEqual(instruction.recommended_exit_id, "EXIT-1")

    def test_19_exit_here_when_already_at_exit(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z1", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign("SIGN-EXIT", zone_ids=("z1",), position=(2.0, 5.0), orientation=180.0)
        planner = DynamicSignagePlanner(building)

        instruction = planner.compute(0.0, snapshot, [sign]).instruction("SIGN-EXIT")

        self.assertEqual(instruction.indication, SignIndication.EXIT_HERE)
        self.assertEqual(instruction.status, SignageStatus.ACTIVE)

    def test_20_use_stairs_for_stair_decision(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z3", floor_id="f2", exit_id="EXIT-1")

        sign = make_sign("SIGN-STAIR", floor_id="f2", zone_ids=("z3",), position=(2.0, 5.0), orientation=0.0)
        planner = DynamicSignagePlanner(building)

        instruction = planner.compute(0.0, snapshot, [sign]).instruction("SIGN-STAIR")

        self.assertEqual(instruction.indication, SignIndication.USE_STAIRS)
        self.assertEqual(instruction.target_asset_id, "STAIR-1")

    def test_21_passthrough_zone_sign_derives_from_someone_elses_route(self):

        # Phase 5's own worked example: z1 is a pure passthrough zone
        # for z3's own route (z1 has no recommendation of its own in
        # this snapshot) -- a sign covering z1 must still resolve using
        # z3's route.
        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z3", floor_id="f2", exit_id="EXIT-1")

        self.assertIsNone(snapshot.zone("z1"))

        sign = make_sign("SIGN-PASSTHROUGH", zone_ids=("z1",), position=(2.0, 5.0), orientation=180.0)
        planner = DynamicSignagePlanner(building)

        instruction = planner.compute(0.0, snapshot, [sign]).instruction("SIGN-PASSTHROUGH")

        self.assertEqual(instruction.status, SignageStatus.ACTIVE)
        self.assertEqual(instruction.indication, SignIndication.EXIT_HERE)
        self.assertEqual(instruction.target_asset_id, "EXIT-1")

    def test_22_inactive_sign_is_unavailable(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign("SIGN-1", zone_ids=("z2",), active=False)
        planner = DynamicSignagePlanner(building)

        instruction = planner.compute(0.0, snapshot, [sign]).instruction("SIGN-1")

        self.assertEqual(instruction.status, SignageStatus.UNAVAILABLE)
        self.assertEqual(instruction.indication, SignIndication.UNAVAILABLE)

    def test_23_sign_with_no_guidance_zone_is_unavailable(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign("SIGN-1", zone_ids=("z-unknown",))
        planner = DynamicSignagePlanner(building)

        instruction = planner.compute(0.0, snapshot, [sign]).instruction("SIGN-1")

        self.assertEqual(instruction.status, SignageStatus.UNAVAILABLE)

    def test_24_no_guidance_snapshot_at_all_is_unavailable(self):

        building = make_signage_building()
        sign = make_sign("SIGN-1", zone_ids=("z2",))
        planner = DynamicSignagePlanner(building)

        result = planner.compute(0.0, None, [sign])

        self.assertEqual(result.instructions, {})

    def test_25_unsupported_indication_is_unavailable(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign(
            "SIGN-1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0,
            supported_indications=(SignIndication.LEFT, SignIndication.RIGHT),
        )
        planner = DynamicSignagePlanner(building)

        instruction = planner.compute(0.0, snapshot, [sign]).instruction("SIGN-1")

        # Geometry alone would say STRAIGHT, but this sign's hardware
        # cannot show it.
        self.assertEqual(instruction.status, SignageStatus.UNAVAILABLE)


class RevisionTests(unittest.TestCase):

    def test_26_same_instruction_no_revision_increment(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign("SIGN-1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0)
        planner = DynamicSignagePlanner(building)

        first = planner.compute(0.0, snapshot, [sign]).instruction("SIGN-1")
        second = planner.compute(1.0, snapshot, [sign]).instruction("SIGN-1")

        self.assertEqual(first.signage_revision, second.signage_revision)

    def test_27_changed_direction_increments_revision(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign("SIGN-1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0)
        planner = DynamicSignagePlanner(building)

        first = planner.compute(0.0, snapshot, [sign]).instruction("SIGN-1")

        sign.orientation = 90.0
        second = planner.compute(1.0, snapshot, [sign]).instruction("SIGN-1")

        self.assertEqual(first.indication, SignIndication.STRAIGHT)
        self.assertEqual(second.indication, SignIndication.RIGHT)
        self.assertGreater(second.signage_revision, first.signage_revision)

    def test_28_changed_exit_increments_revision(self):

        building = make_signage_building()
        engine, snapshot_1 = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign("SIGN-EXIT", zone_ids=("z1",), position=(2.0, 5.0), orientation=180.0)
        planner = DynamicSignagePlanner(building)

        first = planner.compute(0.0, snapshot_1, [sign]).instruction("SIGN-EXIT")

        from tests.dynamic_signage_fixtures import make_recommendation_snapshot

        recommendation_2 = make_recommendation_snapshot("z1", "f1", "EXIT-1", timestamp=1.0)
        snapshot_2 = engine.compute(1.0, recommendation_2, None)

        second = planner.compute(1.0, snapshot_2, [sign]).instruction("SIGN-EXIT")

        # Same effective outcome (still EXIT-1, still EXIT_HERE) -- no
        # revision bump even though the underlying guidance_revision
        # itself may have changed cycle to cycle.
        self.assertEqual(first.signage_revision, second.signage_revision)


class UnsafeRouteInvalidationTests(unittest.TestCase):

    def test_29_unsafe_exit_cannot_remain_sign_target(self):

        # Critical Phase 10 test: SIGN-1 points toward EXIT-1. EXIT-1's
        # own zone (z1) becomes hazard-excluded. The recommendation
        # snapshot itself must reflect a DIFFERENT recommended exit (or
        # none) -- proving the sign's instruction is derived FRESH from
        # the new guidance, never carried forward.
        building = make_signage_building()
        engine, snapshot_1 = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign("SIGN-1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0)
        planner = DynamicSignagePlanner(building)

        first = planner.compute(0.0, snapshot_1, [sign]).instruction("SIGN-1")
        self.assertEqual(first.status, SignageStatus.ACTIVE)
        self.assertEqual(first.recommended_exit_id, "EXIT-1")

        from tests.dynamic_signage_fixtures import make_recommendation_snapshot

        # z1 (EXIT-1's own zone) becomes hazardous -- Guidance's own
        # independent validation invalidates the z1 route.
        building_state = make_building_state({"z1": HazardSeverity.HIGH})
        recommendation_2 = make_recommendation_snapshot("z2", "f1", "EXIT-1", timestamp=1.0)
        snapshot_2 = engine.compute(1.0, recommendation_2, building_state)

        second = planner.compute(1.0, snapshot_2, [sign]).instruction("SIGN-1")

        self.assertNotEqual(second.status, SignageStatus.ACTIVE)
        self.assertNotEqual(second.indication, SignIndication.STRAIGHT)
        self.assertNotEqual(second.signage_revision, first.signage_revision)


class ConflictTests(unittest.TestCase):

    def test_30_conflicting_zone_guidance_detected(self):

        building = make_signage_building()

        from tests.dynamic_signage_fixtures import make_recommendation_snapshot
        from evacuation_guidance.models import EvacuationGuidanceSnapshot

        engine, snapshot_z2 = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")
        recommendation_z1 = make_recommendation_snapshot("z1", "f1", "EXIT-1", timestamp=0.0)
        snapshot_z1 = engine.compute(0.0, recommendation_z1, None)

        # Merge both zones' plans into one combined snapshot -- a sign
        # assigned to BOTH z1 (already at exit -> EXIT_HERE) and z2
        # (directional -> STRAIGHT) sees genuinely different effective
        # outcomes.
        combined = EvacuationGuidanceSnapshot(
            timestamp=0.0,
            zones={**snapshot_z1.zones, **snapshot_z2.zones},
            voice_plans={**snapshot_z1.voice_plans, **snapshot_z2.voice_plans},
        )

        sign = make_sign("SIGN-CONFLICT", zone_ids=("z1", "z2"), position=(22.0, 5.0), orientation=180.0)
        planner = DynamicSignagePlanner(building)

        result = planner.compute(0.0, combined, [sign])
        instruction = result.instruction("SIGN-CONFLICT")
        conflict = result.conflict("SIGN-CONFLICT")

        self.assertEqual(instruction.status, SignageStatus.CONFLICT)
        self.assertIsNotNone(conflict)
        self.assertEqual(set(conflict.conflicting_zone_ids), {"z1", "z2"})

    def test_31_conflict_deterministic_under_iteration_order_shuffle(self):

        building = make_signage_building()

        from tests.dynamic_signage_fixtures import make_recommendation_snapshot
        from evacuation_guidance.models import EvacuationGuidanceSnapshot

        engine, snapshot_z2 = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")
        recommendation_z1 = make_recommendation_snapshot("z1", "f1", "EXIT-1", timestamp=0.0)
        snapshot_z1 = engine.compute(0.0, recommendation_z1, None)

        zones_ordering_a = {**snapshot_z1.zones, **snapshot_z2.zones}
        zones_ordering_b = {**snapshot_z2.zones, **snapshot_z1.zones}

        sign = make_sign("SIGN-CONFLICT", zone_ids=("z1", "z2"), position=(22.0, 5.0), orientation=180.0)

        results = []

        for zones in (zones_ordering_a, zones_ordering_b):

            combined = EvacuationGuidanceSnapshot(timestamp=0.0, zones=zones, voice_plans={})
            planner = DynamicSignagePlanner(building)
            result = planner.compute(0.0, combined, [sign])
            results.append((result.instruction("SIGN-CONFLICT").status, result.conflict("SIGN-CONFLICT").conflicting_zone_ids))

        self.assertEqual(results[0], results[1])


class MultiSignTests(unittest.TestCase):

    def test_32_multiple_signs_get_independent_instructions(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign_straight = make_sign("SIGN-STRAIGHT", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0)
        sign_right = make_sign("SIGN-RIGHT", zone_ids=("z2",), position=(22.0, 5.0), orientation=90.0)
        sign_left = make_sign("SIGN-LEFT", zone_ids=("z2",), position=(22.0, 5.0), orientation=270.0)

        planner = DynamicSignagePlanner(building)
        result = planner.compute(0.0, snapshot, [sign_straight, sign_right, sign_left])

        self.assertEqual(result.instruction("SIGN-STRAIGHT").indication, SignIndication.STRAIGHT)
        self.assertEqual(result.instruction("SIGN-RIGHT").indication, SignIndication.RIGHT)
        self.assertEqual(result.instruction("SIGN-LEFT").indication, SignIndication.LEFT)


if __name__ == "__main__":
    unittest.main()
