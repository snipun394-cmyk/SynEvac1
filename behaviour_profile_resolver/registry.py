from types import MappingProxyType

from behavior.intent import AlwaysEvacuateDecisionStrategy
from behavior.pre_movement import NoPreMovementDelay
from behavior.profile import Role
from behavior.route_choice import ShortestRouteChoiceStrategy

from behavior_library.decision_strategies import (
    AlwaysWaitDecisionStrategy,
    BasicHelpingDecisionStrategy,
    ComplianceDecisionStrategy,
)
from behavior_library.pre_movement_strategies import ProbabilisticPreMovementDelay

from behaviour_profile_resolver.template import BehaviorProfileTemplate


# A default, illustrative registry -- reuses existing behavior/
# behavior_library strategies exclusively (no new DecisionStrategy/
# RouteChoiceStrategy/PreMovementDelayStrategy is defined anywhere in
# this package). The specific parameter choices below (walking speeds,
# compliance levels, median delays) are a starting, documented
# simplification, not a validated life-safety model -- the same
# honesty convention TSquaredFireGrowthCurve's growth_time default and
# HazardSeverity's score cutoffs already use elsewhere in this
# codebase. Named after exactly the behaviour_profile_id worked
# examples already used throughout the Scenario Engine architecture
# documents (Adult_Default, Child_Default, Visitor_Default,
# Staff_Default, FireWarden_Default, Wheelchair_Default), plus
# Elderly_Default (Human Population & Assisted Evacuation Modeling
# Phase 1's own required minimum profile set) -- no other new profile
# name is invented here.
#
# This is a *default*, not the only possible registry --
# register_occupants() (registrar.py) accepts any
# Mapping[str, BehaviorProfileTemplate] in its place.

DEFAULT_PROFILE_REGISTRY = MappingProxyType(
    {
        "Adult_Default": BehaviorProfileTemplate(
            walking_speed=1.2,
            compliance_level=0.9,
            role=Role.INDEPENDENT,
            decision_strategy=ComplianceDecisionStrategy(),
            route_choice_strategy=ShortestRouteChoiceStrategy(),
            pre_movement_strategy=ProbabilisticPreMovementDelay(median_delay=30.0),
        ),
        "Child_Default": BehaviorProfileTemplate(
            walking_speed=0.9,
            compliance_level=0.5,
            role=Role.FOLLOWER,
            decision_strategy=ComplianceDecisionStrategy(
                noncompliant_strategy=AlwaysWaitDecisionStrategy(),
            ),
            route_choice_strategy=ShortestRouteChoiceStrategy(),
            pre_movement_strategy=ProbabilisticPreMovementDelay(median_delay=45.0),
        ),
        "Elderly_Default": BehaviorProfileTemplate(
            # Slower gait and a longer pre-movement delay than Adult
            # (mirrors the real, well-documented mobility/reaction-time
            # differences this platform's other profiles already
            # encode as a difference of degree, not of decision logic)
            # -- still INDEPENDENT/ComplianceDecisionStrategy, since an
            # elderly occupant is not assumed to need assistance by
            # default; see behavior_library.assistance_strategies for
            # the opt-in mechanism that models an occupant actually
            # being escorted.
            walking_speed=0.75,
            compliance_level=0.85,
            role=Role.INDEPENDENT,
            decision_strategy=ComplianceDecisionStrategy(),
            route_choice_strategy=ShortestRouteChoiceStrategy(),
            pre_movement_strategy=ProbabilisticPreMovementDelay(median_delay=50.0, spread=0.6),
        ),
        "Wheelchair_Default": BehaviorProfileTemplate(
            walking_speed=0.7,
            compliance_level=0.9,
            role=Role.INDEPENDENT,
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
            route_choice_strategy=ShortestRouteChoiceStrategy(),
            pre_movement_strategy=ProbabilisticPreMovementDelay(median_delay=60.0, spread=0.7),
        ),
        "Staff_Default": BehaviorProfileTemplate(
            walking_speed=1.3,
            compliance_level=1.0,
            role=Role.LEADER,
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
            route_choice_strategy=ShortestRouteChoiceStrategy(),
            pre_movement_strategy=NoPreMovementDelay(),
        ),
        "FireWarden_Default": BehaviorProfileTemplate(
            walking_speed=1.3,
            compliance_level=1.0,
            role=Role.LEADER,
            decision_strategy=BasicHelpingDecisionStrategy(),
            route_choice_strategy=ShortestRouteChoiceStrategy(),
            pre_movement_strategy=NoPreMovementDelay(),
        ),
        "Visitor_Default": BehaviorProfileTemplate(
            walking_speed=1.1,
            compliance_level=0.6,
            role=Role.INDEPENDENT,
            decision_strategy=ComplianceDecisionStrategy(),
            route_choice_strategy=ShortestRouteChoiceStrategy(),
            pre_movement_strategy=ProbabilisticPreMovementDelay(median_delay=40.0),
        ),
    }
)
