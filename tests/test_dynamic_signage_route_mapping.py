import unittest

from evacuation_guidance.models import NavigationStepType

from dynamic_signage.models import TargetAssetType
from dynamic_signage.route_mapping import meaningful_steps, next_meaningful_step_for_zone, resolve_target

from tests.dynamic_signage_fixtures import make_guidance_snapshot, make_signage_building


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 27 -- test matrix
# items 13-17: guidance route maps to the correct first meaningful
# target, Door/Stair/Exit targets supported, multi-floor guidance.
# =====================================================


class RouteMappingTests(unittest.TestCase):

    def test_1_door_target_for_originating_zone(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")
        plan = snapshot.zone("z2")

        entry = next_meaningful_step_for_zone(plan, "z2")

        self.assertIsNotNone(entry)
        self.assertEqual(entry.step.step_type, NavigationStepType.PASS_THROUGH_DOOR)
        self.assertEqual(entry.step.door_id, "DOOR-1")

        target = resolve_target(building, entry)

        self.assertEqual(target.asset_id, "DOOR-1")
        self.assertEqual(target.asset_type, TargetAssetType.DOOR)
        self.assertEqual(target.position, (15.0, 5.0))

    def test_2_exit_target_for_terminal_zone(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")
        plan = snapshot.zone("z2")

        entry = next_meaningful_step_for_zone(plan, "z1")

        self.assertEqual(entry.step.step_type, NavigationStepType.CONTINUE_TO_EXIT)

        target = resolve_target(building, entry)

        self.assertEqual(target.asset_id, "EXIT-1")
        self.assertEqual(target.asset_type, TargetAssetType.EXIT)
        self.assertEqual(target.position, (0.0, 5.0))

    def test_3_stair_target_multi_floor(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z3", floor_id="f2", exit_id="EXIT-1")
        plan = snapshot.zone("z3")

        entry = next_meaningful_step_for_zone(plan, "z3")

        self.assertEqual(entry.step.step_type, NavigationStepType.USE_STAIR)
        self.assertEqual(entry.step.stair_id, "STAIR-1")

        target = resolve_target(building, entry)

        self.assertEqual(target.asset_id, "STAIR-1")
        self.assertEqual(target.asset_type, TargetAssetType.STAIR)
        # from_position (5, 5) is the f2 landing -- the departure floor.
        self.assertEqual(target.position, (5.0, 5.0))
        self.assertEqual(target.floor_id, "f2")

    def test_4_full_multi_floor_route_has_two_meaningful_steps(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z3", floor_id="f2", exit_id="EXIT-1")
        plan = snapshot.zone("z3")

        entries = meaningful_steps(plan)

        step_types = [e.step.step_type for e in entries]
        self.assertEqual(step_types, [NavigationStepType.USE_STAIR, NavigationStepType.CONTINUE_TO_EXIT])
        self.assertEqual(entries[0].departure_zone_id, "z3")
        self.assertEqual(entries[1].departure_zone_id, "z1")

    def test_5_no_meaningful_step_for_zone_not_on_route(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")
        plan = snapshot.zone("z2")

        self.assertIsNone(next_meaningful_step_for_zone(plan, "z3"))

    def test_6_missing_target_asset_returns_none(self):

        building = make_signage_building()
        _, snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")
        plan = snapshot.zone("z2")

        entry = next_meaningful_step_for_zone(plan, "z2")

        # Remove the door from the building AFTER the plan was computed
        # -- resolve_target() must honestly report None, never fabricate
        # a position.
        building.floors[0].doors = []

        self.assertIsNone(resolve_target(building, entry))


if __name__ == "__main__":
    unittest.main()
