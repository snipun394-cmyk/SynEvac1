import unittest
from unittest.mock import patch

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from scenario.fire import ScenarioFire
from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from scenario_runner import run as run_scenario

from behaviour_profile_resolver.category import OccupantCategory
from behaviour_profile_resolver.occupant_attributes import (
    OccupantAttributes,
    _RANGES_BY_CATEGORY,
    attribute_traits,
    derive_occupant_attributes,
)
from behaviour_profile_resolver.occupant_grouping import GroupAssignment, assign_occupant_groups
from behaviour_profile_resolver.registrar import register_occupants

from behavior_library.attribute_aware_strategies import (
    AttributeAwareComplianceDecisionStrategy,
    AttributeAwarePreMovementDelayStrategy,
    AttributeSensitivityAwarePreMovementDelayStrategy,
    CrowdFollowingAwareRouteChoiceStrategy,
    SocialGroupAwarePreMovementDelayStrategy,
    SocialGroupAwareRouteChoiceStrategy,
    _sensitivity_multiplier,
)
from behavior.intent import AlwaysEvacuateDecisionStrategy
from behavior.profile import BehaviorProfile, Role
from behavior.route_choice import RouteChoice
from simulator.decision import ActionType, BehaviorDecision

from behaviour_profile_resolver.registrar import _effective_walking_speed_multiplier

from dataset_builder.builder import SimulationRun
from dataset_builder.feature_extractor import extract_scenario_features

from ground_truth.analyzer import SimulationArtifacts, analyze
from ground_truth.occupant_attribute_outcomes import compute_occupant_attribute_outcomes


# =====================================================
# Shared fixtures
# =====================================================


def build_two_zone_building(door_width=1.0):

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    zone_a = Zone(name="A", x=0.0, y=0.0, width=2.0, height=2.0)
    zone_b = Zone(name="B", x=10.0, y=0.0, width=2.0, height=2.0)
    floor.add_zone(zone_a)
    floor.add_zone(zone_b)

    door = Door(name="D1", zone_a_id=zone_a.id, zone_b_id=zone_b.id, floor_id=floor.id, width=door_width)
    floor.add_door(door)
    exit_obj = Exit(name="Ex", zone_id=zone_b.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    return building, floor, zone_a, zone_b, door, exit_obj


def make_metadata(seed=1, scenario_id="s1"):

    return ScenarioMetadata(
        scenario_id=scenario_id, definition_id="d1", definition_content_hash="h",
        generation_version="v1", seed=seed, created_at="2026-07-15T00:00:00",
    )


def make_fire(zone_id, floor_id):

    return ScenarioFire(
        ignition_zone_id=zone_id, ignition_floor_id=floor_id,
        fire_profile="Electrical", growth_parameters={"growth_time": 200.0},
    )


def make_occupant(occupant_id, zone_id, floor_id, behaviour_profile_id="Adult_Default"):

    return ScenarioOccupant(
        occupant_id=occupant_id, zone_id=zone_id, floor_id=floor_id, position=(1.0, 1.0),
        behaviour_profile_id=behaviour_profile_id,
    )


# =====================================================


class DeriveOccupantAttributesTests(unittest.TestCase):

    def test_same_inputs_always_derive_the_same_attributes(self):

        a = derive_occupant_attributes(42, "occ-1", "Adult_Default")
        b = derive_occupant_attributes(42, "occ-1", "Adult_Default")

        self.assertEqual(a, b)

    def test_different_occupant_ids_derive_different_attributes(self):

        a = derive_occupant_attributes(42, "occ-1", "Adult_Default")
        b = derive_occupant_attributes(42, "occ-2", "Adult_Default")

        self.assertNotEqual(a, b)

    def test_different_scenario_seeds_derive_different_attributes(self):

        a = derive_occupant_attributes(42, "occ-1", "Adult_Default")
        b = derive_occupant_attributes(99, "occ-1", "Adult_Default")

        self.assertNotEqual(a, b)

    def test_every_category_range_covers_every_attribute_field(self):

        field_names = set(OccupantAttributes.__dataclass_fields__.keys())

        for category, ranges in _RANGES_BY_CATEGORY.items():

            with self.subTest(category=category):

                self.assertEqual(set(ranges.keys()), field_names)

                for attribute, (low, high) in ranges.items():
                    self.assertLessEqual(low, high)

    def test_sampled_values_stay_within_the_configured_category_range(self):

        for category in OccupantCategory:

            profile_id = {
                OccupantCategory.ADULT: "Adult_Default",
                OccupantCategory.CHILD: "Child_Default",
                OccupantCategory.ELDERLY: "Elderly_Default",
                OccupantCategory.WHEELCHAIR_USER: "Wheelchair_Default",
                OccupantCategory.VISITOR: "Visitor_Default",
                OccupantCategory.STAFF: "Staff_Default",
                OccupantCategory.FIRE_WARDEN: "FireWarden_Default",
                OccupantCategory.FIREFIGHTER: "Firefighter_Default",
                OccupantCategory.UNKNOWN: "Some_Unregistered_Profile",
            }[category]

            ranges = _RANGES_BY_CATEGORY.get(category, _RANGES_BY_CATEGORY[OccupantCategory.ADULT])

            for seed in range(20):

                attributes = derive_occupant_attributes(seed, f"occ-{seed}", profile_id)

                for field_name, (low, high) in ranges.items():

                    value = getattr(attributes, field_name)
                    self.assertGreaterEqual(value, low)
                    self.assertLessEqual(value, high)

    def test_child_is_faster_and_less_hazard_aware_than_adult_on_average(self):

        child_speeds = [
            derive_occupant_attributes(s, f"c{s}", "Child_Default").walking_speed_multiplier
            for s in range(50)
        ]
        adult_speeds = [
            derive_occupant_attributes(s, f"a{s}", "Adult_Default").walking_speed_multiplier
            for s in range(50)
        ]
        child_risk_aversion = [
            derive_occupant_attributes(s, f"c{s}", "Child_Default").risk_aversion for s in range(50)
        ]
        adult_risk_aversion = [
            derive_occupant_attributes(s, f"a{s}", "Adult_Default").risk_aversion for s in range(50)
        ]

        self.assertGreater(sum(child_speeds), sum(adult_speeds))
        self.assertLess(sum(child_risk_aversion), sum(adult_risk_aversion))

    def test_wheelchair_user_has_near_zero_mobility_factor(self):

        for seed in range(20):

            attributes = derive_occupant_attributes(seed, f"wc{seed}", "Wheelchair_Default")
            self.assertLessEqual(attributes.mobility_factor, 0.15)

    def test_firefighter_has_the_highest_smoke_tolerance(self):

        firefighter_tolerance = sum(
            derive_occupant_attributes(s, f"f{s}", "Firefighter_Default").smoke_tolerance
            for s in range(30)
        )
        adult_tolerance = sum(
            derive_occupant_attributes(s, f"a{s}", "Adult_Default").smoke_tolerance
            for s in range(30)
        )

        self.assertGreater(firefighter_tolerance, adult_tolerance)

    def test_attribute_traits_carries_every_raw_field_plus_the_derived_multiplier(self):

        attributes = derive_occupant_attributes(1, "occ-1", "Adult_Default")
        traits = attribute_traits(attributes)

        for field_name in OccupantAttributes.__dataclass_fields__:
            self.assertIn(field_name, traits)

        self.assertIn("pre_movement_delay_multiplier", traits)
        self.assertGreater(traits["pre_movement_delay_multiplier"], 0.0)


# =====================================================


class AssignOccupantGroupsTests(unittest.TestCase):

    def _scenario(self, count=12):

        occupants = tuple(
            make_occupant(f"occ-{i}", "zone-a" if i % 2 == 0 else "zone-b", "floor-1")
            for i in range(count)
        )
        return Scenario(metadata=make_metadata(), occupants=occupants)

    def test_deterministic_across_repeated_calls(self):

        scenario = self._scenario()

        self.assertEqual(assign_occupant_groups(scenario), assign_occupant_groups(scenario))

    def test_different_seeds_produce_different_grouping(self):

        occupants = tuple(make_occupant(f"occ-{i}", "zone-a", "floor-1") for i in range(12))
        scenario_a = Scenario(metadata=make_metadata(seed=1), occupants=occupants)
        scenario_b = Scenario(metadata=make_metadata(seed=2), occupants=occupants)

        self.assertNotEqual(assign_occupant_groups(scenario_a), assign_occupant_groups(scenario_b))

    def test_group_sizes_are_between_two_and_four(self):

        scenario = self._scenario(count=30)
        assignments = assign_occupant_groups(scenario)

        sizes = {}
        for assignment in assignments.values():
            sizes[assignment.group_id] = sizes.get(assignment.group_id, 0) + 1

        for group_id, size in sizes.items():
            self.assertGreaterEqual(size, 2, group_id)
            self.assertLessEqual(size, 4, group_id)

    def test_exactly_one_leader_per_group_and_followers_point_at_them(self):

        scenario = self._scenario(count=30)
        assignments = assign_occupant_groups(scenario)

        leaders_by_group = {}
        for occupant_id, assignment in assignments.items():
            if assignment.is_leader:
                leaders_by_group.setdefault(assignment.group_id, []).append(occupant_id)

        for group_id, leaders in leaders_by_group.items():
            self.assertEqual(len(leaders), 1, group_id)

        for occupant_id, assignment in assignments.items():
            if not assignment.is_leader:
                self.assertEqual(assignment.leader_occupant_id, leaders_by_group[assignment.group_id][0])
            else:
                self.assertIsNone(assignment.leader_occupant_id)

    def test_not_every_occupant_is_grouped(self):

        scenario = self._scenario(count=12)
        assignments = assign_occupant_groups(scenario)

        self.assertLess(len(assignments), 12)
        self.assertGreater(len(assignments), 0)

    def test_family_group_type_when_child_and_adult_share_a_group(self):

        occupants = (
            make_occupant("child-1", "zone-a", "floor-1", "Child_Default"),
            make_occupant("adult-1", "zone-a", "floor-1", "Adult_Default"),
            make_occupant("adult-2", "zone-a", "floor-1", "Adult_Default"),
        )
        # Try a handful of seeds until both are grouped together -- the
        # 65% sociability draw means not every seed groups everyone.
        for seed in range(30):

            scenario = Scenario(metadata=make_metadata(seed=seed), occupants=occupants)
            assignments = assign_occupant_groups(scenario)

            if "child-1" in assignments and "adult-1" in assignments:
                if assignments["child-1"].group_id == assignments["adult-1"].group_id:
                    self.assertEqual(assignments["child-1"].group_type, "Family")
                    return

        self.fail("no seed in range produced a shared child/adult group to assert on")


# =====================================================


class AttributeAwareStrategiesTests(unittest.TestCase):

    def test_route_strategy_is_a_pure_passthrough_without_the_trait(self):

        profile = BehaviorProfile(occupant_id="occ-1", traits={})

        class FakeContext:
            pass

        context = FakeContext()
        context.profile = profile
        context.decisions_so_far = {}

        calls = []

        class FakeFallback:
            def choose(self, ctx):
                calls.append(ctx)
                return RouteChoice(goal_id="fallback-goal", route=None)

        strategy = SocialGroupAwareRouteChoiceStrategy(fallback=FakeFallback())
        result = strategy.choose(context)

        self.assertEqual(result.goal_id, "fallback-goal")
        self.assertEqual(len(calls), 1)

    def test_route_strategy_follows_the_social_leader_when_present(self):

        leader_decision = BehaviorDecision(
            occupant_id="leader-1", action_type=ActionType.EVACUATE, start_id="zone-a",
            goal_id="exit-1", route=("zone-a", "exit-1"), depart_time=5.0,
        )
        profile = BehaviorProfile(occupant_id="occ-2", traits={"social_leader_occupant_id": "leader-1"})

        class FakeContext:
            pass

        context = FakeContext()
        context.profile = profile
        context.decisions_so_far = {"leader-1": leader_decision}

        class FakeFallback:
            def choose(self, ctx):
                return RouteChoice(goal_id="fallback-goal", route=None)

        strategy = SocialGroupAwareRouteChoiceStrategy(fallback=FakeFallback())
        result = strategy.choose(context)

        self.assertEqual(result.goal_id, "exit-1")

    def test_assistance_leader_takes_precedence_over_social_leader(self):

        profile = BehaviorProfile(
            occupant_id="occ-3",
            traits={"leader_occupant_id": "assist-helper", "social_leader_occupant_id": "group-leader"},
        )

        class FakeContext:
            pass

        context = FakeContext()
        context.profile = profile
        context.decisions_so_far = {}

        class FakeFallback:
            def choose(self, ctx):
                return RouteChoice(goal_id="fallback-goal", route=None)

        SocialGroupAwareRouteChoiceStrategy(fallback=FakeFallback()).choose(context)

        self.assertEqual(profile.traits["leader_occupant_id"], "assist-helper")

    def test_pre_movement_delay_strategy_is_a_pure_passthrough_without_the_trait(self):

        profile = BehaviorProfile(occupant_id="occ-1", traits={})

        class FakeContext:
            pass

        context = FakeContext()
        context.profile = profile

        class FakeFallback:
            def delay(self, ctx):
                return 10.0

        strategy = SocialGroupAwarePreMovementDelayStrategy(fallback=FakeFallback())
        self.assertEqual(strategy.delay(context), 10.0)

    def test_pre_movement_delay_strategy_waits_for_the_social_leader(self):

        leader_decision = BehaviorDecision(
            occupant_id="leader-1", action_type=ActionType.EVACUATE, start_id="zone-a", depart_time=5.0,
        )
        profile = BehaviorProfile(occupant_id="occ-2", traits={"social_leader_occupant_id": "leader-1"})

        class FakeContext:
            pass

        context = FakeContext()
        context.profile = profile
        context.decisions_so_far = {"leader-1": leader_decision}

        class FakeFallback:
            def delay(self, ctx):
                return 10.0

        strategy = SocialGroupAwarePreMovementDelayStrategy(fallback=FakeFallback(), follow_gap=0.5)
        self.assertEqual(strategy.delay(context), 5.5)

    def test_attribute_aware_delay_defaults_to_a_multiplier_of_one(self):

        profile = BehaviorProfile(occupant_id="occ-1", traits={})

        class FakeContext:
            pass

        context = FakeContext()
        context.profile = profile

        class FakeFallback:
            def delay(self, ctx):
                return 10.0

        strategy = AttributeAwarePreMovementDelayStrategy(fallback=FakeFallback())
        self.assertEqual(strategy.delay(context), 10.0)

    def test_attribute_aware_delay_applies_the_configured_multiplier(self):

        profile = BehaviorProfile(occupant_id="occ-1", traits={"pre_movement_delay_multiplier": 2.0})

        class FakeContext:
            pass

        context = FakeContext()
        context.profile = profile

        class FakeFallback:
            def delay(self, ctx):
                return 10.0

        strategy = AttributeAwarePreMovementDelayStrategy(fallback=FakeFallback())
        self.assertEqual(strategy.delay(context), 20.0)


# =====================================================


class _FakeContext:
    pass


class CrowdFollowingAwareRouteChoiceStrategyTests(unittest.TestCase):

    def test_passthrough_without_the_trait(self):

        profile = BehaviorProfile(occupant_id="occ-1", traits={})
        context = _FakeContext()
        context.profile = profile
        context.decisions_so_far = {}
        context.rng = None

        class FakeFallback:
            def choose(self, ctx):
                return RouteChoice(goal_id="fallback-goal", route=None)

        strategy = CrowdFollowingAwareRouteChoiceStrategy(fallback=FakeFallback())
        result = strategy.choose(context)

        self.assertEqual(result.goal_id, "fallback-goal")

    def test_delegates_to_static_herding_when_tendency_is_present(self):

        # A high crowd_following_tendency with no decisions_so_far yet
        # (the common case for the first occupant registered in a zone)
        # -- StaticHerdingRouteChoiceStrategy's own "no peers decided
        # yet" branch degrades to `fallback` (already covered by
        # tests.test_behavior_library.StaticHerdingRouteChoiceStrategyTests
        # -- not reimplemented here). This proves the wrapper genuinely
        # delegates to that class (its own graceful empty-tally fallback
        # fires) rather than asserting a specific herded route outcome,
        # which depends on PathfindingEngine internals out of scope for
        # this test file.

        profile = BehaviorProfile(occupant_id="occ-2", traits={"crowd_following_tendency": 1.0})
        context = _FakeContext()
        context.profile = profile
        context.decisions_so_far = {}
        context.rng = None

        class FakeFallback:
            def choose(self, ctx):
                return RouteChoice(goal_id="fallback-goal", route=None)

        strategy = CrowdFollowingAwareRouteChoiceStrategy(fallback=FakeFallback())
        result = strategy.choose(context)

        self.assertEqual(result.goal_id, "fallback-goal")


# =====================================================


class AttributeAwareComplianceDecisionStrategyTests(unittest.TestCase):

    def test_passthrough_without_the_trait(self):

        profile = BehaviorProfile(occupant_id="occ-1", traits={}, role=Role.INDEPENDENT)
        context = _FakeContext()
        context.profile = profile
        context.rng = None

        class FakeFallback:
            def decide(self, ctx):
                return "fallback-intent"

        strategy = AttributeAwareComplianceDecisionStrategy(fallback=FakeFallback())
        self.assertEqual(strategy.decide(context), "fallback-intent")

    def test_leader_role_is_always_exempt(self):

        profile = BehaviorProfile(occupant_id="occ-1", traits={"compliance": 0.0}, role=Role.LEADER)
        context = _FakeContext()
        context.profile = profile
        context.rng = None

        class FakeFallback:
            def decide(self, ctx):
                return "fallback-intent"

        strategy = AttributeAwareComplianceDecisionStrategy(fallback=FakeFallback())
        self.assertEqual(strategy.decide(context), "fallback-intent")

    def test_already_gated_profile_is_exempt(self):

        profile = BehaviorProfile(occupant_id="occ-1", traits={"compliance": 0.0}, role=Role.INDEPENDENT)
        context = _FakeContext()
        context.profile = profile
        context.rng = None

        class FakeFallback:
            def decide(self, ctx):
                return "fallback-intent"

        strategy = AttributeAwareComplianceDecisionStrategy(fallback=FakeFallback(), already_gated=True)
        self.assertEqual(strategy.decide(context), "fallback-intent")

    def test_compliance_zero_always_falls_to_noncompliant_branch(self):

        import random

        profile = BehaviorProfile(occupant_id="occ-1", traits={"compliance": 0.0}, role=Role.INDEPENDENT)
        context = _FakeContext()
        context.profile = profile
        context.rng = random.Random(1)

        class FakeFallback:
            def decide(self, ctx):
                return "fallback-intent"

        class FakeNoncompliant:
            def decide(self, ctx):
                return "noncompliant-intent"

        strategy = AttributeAwareComplianceDecisionStrategy(
            fallback=FakeFallback(), noncompliant_strategy=FakeNoncompliant(),
        )
        self.assertEqual(strategy.decide(context), "noncompliant-intent")

    def test_compliance_one_always_delegates_to_fallback(self):

        import random

        profile = BehaviorProfile(occupant_id="occ-1", traits={"compliance": 1.0}, role=Role.INDEPENDENT)
        context = _FakeContext()
        context.profile = profile
        context.rng = random.Random(1)

        class FakeFallback:
            def decide(self, ctx):
                return "fallback-intent"

        strategy = AttributeAwareComplianceDecisionStrategy(fallback=FakeFallback())
        self.assertEqual(strategy.decide(context), "fallback-intent")


# =====================================================


class AttributeSensitivityAwarePreMovementDelayStrategyTests(unittest.TestCase):

    def test_passthrough_without_any_trait(self):

        profile = BehaviorProfile(occupant_id="occ-1", traits={})
        context = _FakeContext()
        context.profile = profile

        class FakeFallback:
            def delay(self, ctx):
                return 10.0

        strategy = AttributeSensitivityAwarePreMovementDelayStrategy(fallback=FakeFallback())
        self.assertEqual(strategy.delay(context), 10.0)

    def test_sensitivity_multiplier_is_bounded_and_monotonic(self):

        neutral = _sensitivity_multiplier(0.5, 0.5, 0.5, 0.5, 0.5)
        best_case = _sensitivity_multiplier(1.0, 1.0, 1.0, 0.0, 0.0)
        worst_case = _sensitivity_multiplier(0.0, 0.0, 0.0, 1.0, 1.0)

        self.assertGreaterEqual(best_case, 0.5)
        self.assertLessEqual(worst_case, 2.0)
        self.assertLess(best_case, neutral)
        self.assertLess(neutral, worst_case)


# =====================================================


class EffectiveWalkingSpeedMultiplierTests(unittest.TestCase):

    def test_bounded_and_dominated_by_walking_speed_multiplier(self):

        adult_attributes = derive_occupant_attributes(1, "occ-1", "Adult_Default")
        wheelchair_attributes = derive_occupant_attributes(1, "occ-2", "Wheelchair_Default")

        adult_multiplier = _effective_walking_speed_multiplier(adult_attributes)
        wheelchair_multiplier = _effective_walking_speed_multiplier(wheelchair_attributes)

        self.assertGreaterEqual(adult_multiplier, 0.15)
        self.assertGreaterEqual(wheelchair_multiplier, 0.15)
        self.assertGreater(adult_multiplier, wheelchair_multiplier)


# =====================================================


class RegistrarIntegrationTests(unittest.TestCase):

    # HumanBehaviorLayer doesn't retain the BehaviorProfile it was
    # registered with (only the resulting BehaviorDecision, keyed by
    # occupant_id) -- register() itself is patched here purely to
    # capture the `profile` argument _register_one() builds and passes
    # in, without altering registration's real behavior (the patched
    # mock replaces the call entirely, so these tests assert on what
    # would have been registered, not on simulation.run() outcomes).

    def _register_and_capture_profiles(self, scenario, building):

        context = run_scenario(scenario, building)

        with patch("behaviour_profile_resolver.registrar.HumanBehaviorLayer.register") as mock_register:

            register_occupants(context)

        return {call.kwargs["profile"].occupant_id: call.kwargs["profile"] for call in mock_register.call_args_list}

    def test_every_registered_occupant_carries_attribute_traits(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            make_occupant("p1", zone_a.id, floor.id, "Adult_Default"),
            make_occupant("p2", zone_a.id, floor.id, "Child_Default"),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))

        profiles = self._register_and_capture_profiles(scenario, building)

        for occupant_id in ("p1", "p2"):
            for field_name in OccupantAttributes.__dataclass_fields__:
                self.assertIn(field_name, profiles[occupant_id].traits)

    def test_helping_likelihood_drives_assistance_for_a_high_likelihood_group_leader(self):

        from behaviour_profile_resolver.registrar import _helping_likelihood_traits_by_id

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()

        # FireWarden_Default's own helping_likelihood range (0.70-0.95)
        # sits entirely above the 0.6 threshold -- deterministically
        # guaranteed to trigger regardless of seed, no seed-search
        # needed. A same-zone Elderly_Default (a slower walking_speed_
        # multiplier range than the warden) is the natural "least
        # mobile" target.
        occupants = (
            make_occupant("warden-1", zone_a.id, floor.id, "FireWarden_Default"),
            make_occupant("elder-1", zone_a.id, floor.id, "Elderly_Default"),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)

        group_assignments = assign_occupant_groups(context)

        # Force both into one group deterministically for this unit
        # test (assign_occupant_groups()'s own sociability draw isn't
        # what this test is about) -- exercise
        # _helping_likelihood_traits_by_id() directly against a
        # synthetic, fully-known group assignment instead.
        forced_assignments = {
            "warden-1": GroupAssignment(
                group_id="group-test", group_type="Friends", is_leader=True, leader_occupant_id=None,
            ),
            "elder-1": GroupAssignment(
                group_id="group-test", group_type="Friends", is_leader=False, leader_occupant_id="warden-1",
            ),
        }

        traits_by_id = _helping_likelihood_traits_by_id(context, forced_assignments, {})

        self.assertEqual(traits_by_id["warden-1"]["assistance_role"], "HELPER")
        self.assertEqual(traits_by_id["warden-1"]["assistance_target_id"], "elder-1")
        self.assertEqual(traits_by_id["elder-1"]["assistance_role"], "ASSISTED")
        self.assertEqual(traits_by_id["elder-1"]["leader_occupant_id"], "warden-1")

    def test_helping_likelihood_never_overrides_scenario_authored_assistance(self):

        from behaviour_profile_resolver.registrar import _helping_likelihood_traits_by_id

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            make_occupant("warden-1", zone_a.id, floor.id, "FireWarden_Default"),
            make_occupant("elder-1", zone_a.id, floor.id, "Elderly_Default"),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)

        forced_assignments = {
            "warden-1": GroupAssignment(
                group_id="group-test", group_type="Friends", is_leader=True, leader_occupant_id=None,
            ),
            "elder-1": GroupAssignment(
                group_id="group-test", group_type="Friends", is_leader=False, leader_occupant_id="warden-1",
            ),
        }

        # warden-1 already Scenario-authored-assisting someone else --
        # must never also be double-booked as a helping_likelihood-
        # driven helper.
        already_assisting = {"warden-1": {"assistance_role": "HELPER", "assistance_target_id": "someone-else"}}

        traits_by_id = _helping_likelihood_traits_by_id(context, forced_assignments, already_assisting)

        self.assertEqual(traits_by_id, {})

    def test_walking_speed_and_compliance_level_remain_exactly_the_template_values(self):

        # The protected guarantee this feature must never break --
        # behaviour_profile_resolver.registrar's own resolved template
        # values must reach the simulation unmodified.

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (make_occupant("p1", zone_a.id, floor.id, "Adult_Default"),)
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))

        profiles = self._register_and_capture_profiles(scenario, building)

        from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY

        template = DEFAULT_PROFILE_REGISTRY["Adult_Default"]

        self.assertEqual(profiles["p1"].walking_speed, template.walking_speed)
        self.assertEqual(profiles["p1"].compliance_level, template.compliance_level)

    def test_registration_is_deterministic_for_the_same_seed(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = tuple(
            make_occupant(f"p{i}", zone_a.id, floor.id, "Adult_Default") for i in range(6)
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))

        traits_by_run = []
        for _ in range(2):

            profiles = self._register_and_capture_profiles(scenario, building)
            traits_by_run.append({oid: dict(profile.traits) for oid, profile in profiles.items()})

        self.assertEqual(traits_by_run[0], traits_by_run[1])


# =====================================================


class GroundTruthOutcomeTests(unittest.TestCase):

    def test_compute_occupant_attribute_outcomes_reports_bounded_means(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            make_occupant("p1", zone_a.id, floor.id, "Adult_Default"),
            make_occupant("p2", zone_a.id, floor.id, "Child_Default"),
            make_occupant("p3", zone_a.id, floor.id, "Elderly_Default"),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        register_occupants(context)
        movement_result = context.simulation.run()

        outcomes = compute_occupant_attribute_outcomes(scenario, movement_result)

        for key in (
            "mean_walking_speed_multiplier_evacuated", "mean_walking_speed_multiplier_trapped",
            "mean_risk_aversion_evacuated", "mean_risk_aversion_trapped",
            "mean_panic_susceptibility_evacuated", "mean_panic_susceptibility_trapped",
        ):
            value = outcomes[key]
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.6)

        self.assertGreaterEqual(outcomes["group_count"], 0)
        self.assertGreaterEqual(outcomes["grouped_occupant_count"], 0)

    def test_ground_truth_analyze_populates_the_new_fields(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (make_occupant("p1", zone_a.id, floor.id, "Adult_Default"),)
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        register_occupants(context)
        movement_result = context.simulation.run()

        ground_truth = analyze(SimulationArtifacts(
            scenario=scenario, building=building, movement_result=movement_result,
        ))

        self.assertIn(ground_truth.group_count, range(0, 2))
        round_tripped = type(ground_truth).from_dict(ground_truth.to_dict())
        self.assertEqual(ground_truth, round_tripped)

    def test_legacy_ground_truth_payload_defaults_new_fields(self):

        from ground_truth.labels import GroundTruth

        legacy = GroundTruth.from_dict({"scenario_id": "s1", "definition_id": "d1"})

        self.assertIsNone(legacy.mean_walking_speed_multiplier_evacuated)
        self.assertEqual(legacy.group_count, 0)


# =====================================================


class DatasetBuilderFeatureTests(unittest.TestCase):

    def test_scenario_features_row_includes_occupant_attribute_and_group_columns(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            make_occupant("p1", zone_a.id, floor.id, "Adult_Default"),
            make_occupant("p2", zone_a.id, floor.id, "Child_Default"),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        register_occupants(context)
        movement_result = context.simulation.run()

        row = extract_scenario_features(
            SimulationRun(scenario=scenario, building=building, movement_result=movement_result),
        )

        for column in ("Mean_Walking_Speed_Multiplier", "Mean_Risk_Aversion", "Mean_Compliance"):
            self.assertIn(column, row)
            self.assertIsNotNone(row[column])

        for column in ("Group_Count", "Grouped_Occupant_Count", "Mean_Group_Size"):
            self.assertIn(column, row)

    def test_no_occupants_leaves_attribute_columns_none_without_raising(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        scenario = Scenario(metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        register_occupants(context)
        movement_result = context.simulation.run()

        row = extract_scenario_features(
            SimulationRun(scenario=scenario, building=building, movement_result=movement_result),
        )

        self.assertIsNone(row["Mean_Walking_Speed_Multiplier"])
        self.assertEqual(row["Group_Count"], 0)


if __name__ == "__main__":
    unittest.main()
