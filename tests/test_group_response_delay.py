import random
import unittest

from behavior.context import DecisionContext
from behavior.pre_movement import NoPreMovementDelay
from behavior.profile import BehaviorProfile
from behavior.route_choice import RouteChoice

from behavior_library.attribute_aware_strategies import (
    SocialGroupAwarePreMovementDelayStrategy,
    SocialGroupAwareRouteChoiceStrategy,
)

from behaviour_profile_resolver.occupant_grouping import GroupAssignment
from behaviour_profile_resolver.registrar import _occupant_group_traits_by_id

from simulator.decision import ActionType, BehaviorDecision


# Group Collective Response Delay milestone -- Bode, Holl, Mehner &
# Seyfried (2015), PLOS ONE, DOI 10.1371/journal.pone.0121227. Tests
# exercise SocialGroupAwarePreMovementDelayStrategy and
# _occupant_group_traits_by_id() directly (the only two places
# modified), the same lightweight DecisionContext-construction pattern
# already used throughout this test suite for pre-movement strategies
# -- no engine/graph is needed since none of the strategies under test
# read either. No production default is touched anywhere: registrar.py's
# own _GROUP_RESPONSE_DELAY_S stays 0.0, only ever overridden locally
# below via direct constructor arguments.


class _FixedDelay:

    # A minimal deterministic PreMovementDelayStrategy test double --
    # avoids relying on ProbabilisticPreMovementDelay's own lognormal
    # draw for tests that need an exact, non-zero baseline.

    def __init__(self, value):
        self.value = value

    def delay(self, context):
        return self.value


def make_context(profile, decisions_so_far=None):

    return DecisionContext(
        graph=None, engine=None, profile=profile, start_id="zone-1",
        decisions_so_far=decisions_so_far or {}, rng=random.Random(1),
    )


def make_profile(occupant_id, traits=None):

    return BehaviorProfile(occupant_id=occupant_id, traits=dict(traits or {}))


def make_leader_decision(depart_time):

    return BehaviorDecision(
        occupant_id="leader", action_type=ActionType.EVACUATE, start_id="zone-1",
        goal_id="Outside", depart_time=depart_time,
    )


# =====================================================
# Phase 1 wiring -- _occupant_group_traits_by_id()
# =====================================================


class TraitPopulationTests(unittest.TestCase):

    def test_leader_now_receives_a_group_id_trait(self):

        assignments = {
            "leader": GroupAssignment(group_id="g1", group_type="Friends", is_leader=True, leader_occupant_id=None),
        }

        traits_by_id = _occupant_group_traits_by_id(assignments)

        self.assertEqual(traits_by_id["leader"], {"group_id": "g1", "group_type": "Friends"})
        self.assertNotIn("social_leader_occupant_id", traits_by_id["leader"])

    def test_follower_trait_shape_is_unchanged(self):

        assignments = {
            "follower": GroupAssignment(
                group_id="g1", group_type="Friends", is_leader=False, leader_occupant_id="leader",
            ),
        }

        traits_by_id = _occupant_group_traits_by_id(assignments)

        self.assertEqual(
            traits_by_id["follower"],
            {"social_leader_occupant_id": "leader", "group_id": "g1", "group_type": "Friends"},
        )

    def test_ungrouped_occupant_gets_no_entry(self):

        # Unchanged: assign_occupant_groups() never produces an entry
        # for an occupant who isn't in any group; _occupant_group_
        # traits_by_id() only ever iterates the assignments it is given.
        self.assertEqual(_occupant_group_traits_by_id({}), {})


# =====================================================
# Phase 4 -- Backward Compatibility (group_response_delay=0.0)
# =====================================================


class BackwardCompatibilityTests(unittest.TestCase):

    def test_solo_occupant_unaffected(self):

        strategy = SocialGroupAwarePreMovementDelayStrategy(fallback=NoPreMovementDelay())
        context = make_context(make_profile("solo"))

        self.assertEqual(strategy.delay(context), 0.0)

    def test_group_leader_unaffected_when_group_response_delay_is_zero(self):

        strategy = SocialGroupAwarePreMovementDelayStrategy(fallback=NoPreMovementDelay(), group_response_delay=0.0)
        context = make_context(make_profile("leader", traits={"group_id": "g1"}))

        self.assertEqual(strategy.delay(context), 0.0)

    def test_follower_behavior_is_byte_identical_to_before_this_milestone(self):

        strategy = SocialGroupAwarePreMovementDelayStrategy(
            fallback=NoPreMovementDelay(), follow_gap=2.0, group_response_delay=0.0,
        )
        profile = make_profile("follower", traits={"social_leader_occupant_id": "leader", "group_id": "g1"})
        context = make_context(profile, {"leader": make_leader_decision(depart_time=5.0)})

        self.assertEqual(strategy.delay(context), 7.0)

    def test_route_choice_is_unaffected_by_the_new_leader_group_id_trait(self):

        # SocialGroupAwareRouteChoiceStrategy reads "social_leader_
        # occupant_id" only -- the new "group_id" trait a leader now
        # carries is invisible to it, so a leader's own route choice
        # is exactly as before: delegated to fallback, untouched.

        class _FakeRouteChoiceStrategy:
            def choose(self, context):
                return RouteChoice(goal_id="Outside", route=None)

        strategy = SocialGroupAwareRouteChoiceStrategy(fallback=_FakeRouteChoiceStrategy())
        profile = make_profile("leader", traits={"group_id": "g1"})

        context = DecisionContext(graph=None, engine=None, profile=profile, start_id="zone-1")
        choice = strategy.choose(context)

        self.assertEqual(choice.goal_id, "Outside")

    def test_walking_speed_and_stair_speed_are_never_touched_by_the_new_trait_or_delay(self):

        profile = make_profile("leader", traits={"group_id": "g1"})
        profile.walking_speed = 1.2

        strategy = SocialGroupAwarePreMovementDelayStrategy(fallback=NoPreMovementDelay(), group_response_delay=0.67)
        strategy.delay(make_context(profile))

        self.assertEqual(profile.walking_speed, 1.2)
        self.assertIsNone(profile.stair_speed)


# =====================================================
# Phase 12 -- Targeted Mechanism Tests
# =====================================================


class GroupResponseDelayMechanismTests(unittest.TestCase):

    def test_solo_occupant_receives_no_group_delay(self):

        strategy = SocialGroupAwarePreMovementDelayStrategy(fallback=NoPreMovementDelay(), group_response_delay=0.67)
        context = make_context(make_profile("solo"))

        self.assertEqual(strategy.delay(context), 0.0)

    def test_group_leader_receives_the_group_delay(self):

        strategy = SocialGroupAwarePreMovementDelayStrategy(fallback=NoPreMovementDelay(), group_response_delay=0.67)
        context = make_context(make_profile("leader", traits={"group_id": "g1"}))

        self.assertAlmostEqual(strategy.delay(context), 0.67)

    def test_group_follower_inherits_the_group_delay_via_the_leaders_own_depart_time(self):

        strategy = SocialGroupAwarePreMovementDelayStrategy(
            fallback=NoPreMovementDelay(), follow_gap=1.0, group_response_delay=0.67,
        )
        profile = make_profile("follower", traits={"social_leader_occupant_id": "leader", "group_id": "g1"})
        # The leader's own already-resolved depart_time already has
        # +0.67 baked in (computed by this same strategy for the
        # leader, one step earlier in registration order).
        context = make_context(profile, {"leader": make_leader_decision(depart_time=0.67)})

        self.assertAlmostEqual(strategy.delay(context), 0.67 + 1.0)

    def test_follow_gap_and_group_response_delay_are_independent_and_additive_not_compounded(self):

        gap_only = SocialGroupAwarePreMovementDelayStrategy(
            fallback=NoPreMovementDelay(), follow_gap=1.0, group_response_delay=0.0,
        )
        both = SocialGroupAwarePreMovementDelayStrategy(
            fallback=NoPreMovementDelay(), follow_gap=1.0, group_response_delay=0.67,
        )

        follower_gap_only = make_profile("follower", traits={"social_leader_occupant_id": "leader", "group_id": "g1"})
        follower_both = make_profile("follower", traits={"social_leader_occupant_id": "leader", "group_id": "g1"})

        delay_gap_only = gap_only.delay(make_context(follower_gap_only, {"leader": make_leader_decision(0.0)}))
        delay_both = both.delay(make_context(follower_both, {"leader": make_leader_decision(0.67)}))

        # The follower's own delay grows by EXACTLY group_response_delay,
        # never doubled and never interacting with follow_gap's own value.
        self.assertAlmostEqual(delay_both - delay_gap_only, 0.67)

    def test_group_delay_composes_additively_with_a_nonzero_baseline(self):

        strategy = SocialGroupAwarePreMovementDelayStrategy(fallback=_FixedDelay(10.0), group_response_delay=0.5)
        context = make_context(make_profile("leader", traits={"group_id": "g1"}))

        self.assertAlmostEqual(strategy.delay(context), 10.5)
