import hashlib
import random
from typing import Optional

from behavior.context import DecisionContext
from behavior.intent import ActionIntent, DecisionStrategy
from behavior.orchestrator import HumanBehaviorLayer
from behavior.route_choice import RouteChoice, RouteChoiceStrategy

from simulator.decision import ActionType, BehaviorDecision
from simulator.multi_agent_result import MultiAgentSimulationResult

from hazard.snapshot import HazardSnapshot


# =====================================================
# Deterministic per-replan seeding.
#
# The "same algorithm, independently implemented per package"
# precedent behaviour_profile_resolver/registrar.py already
# establishes for its own occupant seeding (sha256 of a stable key,
# truncated to an int) -- restated here rather than imported, scoped
# to this package's own replanning events. A replan-scoped seed
# (rather than reusing the occupant's original registration seed
# verbatim) means a second, third, ... replan for the same occupant
# doesn't silently replay the exact same first-ever random draw --
# while still being fully deterministic given an identical action
# sequence at identical times.
# =====================================================


def derive_replan_seed(scenario_seed: int, occupant_id: str, replan_index: int) -> int:

    key = f"{scenario_seed}|simulation_interactive|{occupant_id}|{replan_index}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big")


# =====================================================


def replan_occupant(
    behavior_layer: HumanBehaviorLayer,
    registration,
    occupant_id: str,
    start_id: str,
    time: float,
    hazard_snapshot: Optional[HazardSnapshot],
    prior_result: Optional[MultiAgentSimulationResult],
    rng: random.Random,
) -> str:

    # Re-runs the same Decision Stage -> Navigation Stage -> Pre-
    # Movement Delay Stage pipeline HumanBehaviorLayer.register()
    # already runs, reusing the exact same DecisionContext/
    # BehaviorDecision types and the exact same strategy *instances*
    # originally assigned to this occupant. Not a call to register()
    # itself: that method's signature never threads hazard_snapshot/
    # prior_result through to DecisionContext, and always treats
    # start_id as a registration-time constant -- both of which a
    # mid-run replan needs to vary. behavior/orchestrator.py stays
    # unmodified; this is a second, independent orchestration of the
    # same public pieces, not a change to the first one.

    context = DecisionContext(
        graph=behavior_layer.graph,
        engine=behavior_layer.engine,
        profile=registration.profile,
        start_id=start_id,
        decisions_so_far=dict(behavior_layer._decisions_so_far),
        prior_result=prior_result,
        hazard_snapshot=hazard_snapshot,
        rng=rng,
    )

    intent = registration.decision_strategy.decide(context)

    if intent.requires_movement:

        route_choice = registration.route_choice_strategy.choose(context)
        delay = registration.pre_movement_strategy.delay(context)

        # Same route_unavailable disambiguation
        # HumanBehaviorLayer.register() applies -- movement was
        # required but no route to any exit exists is a structural
        # disconnection, not a choice to stay put, and must not be
        # conflated with a genuine WAIT/IGNORE by submit_decision().
        # See docs/validation/technical_report.md §6.
        decision = BehaviorDecision(
            occupant_id=occupant_id,
            action_type=intent.action_type,
            start_id=start_id,
            goal_id=route_choice.goal_id,
            route=route_choice.route,
            walking_speed=registration.profile.walking_speed,
            depart_time=time + delay,
            metadata=intent.metadata,
            route_unavailable=(route_choice.goal_id is None and route_choice.route is None),
        )

    else:

        decision = BehaviorDecision(
            occupant_id=occupant_id,
            action_type=intent.action_type,
            start_id=start_id,
            metadata=intent.metadata,
        )

    # Keeps herding/leader RouteChoiceStrategy implementations
    # (StaticHerdingRouteChoiceStrategy, FollowLeaderRouteChoiceStrategy,
    # LoadBalancingRouteChoiceStrategy) consistent for whichever
    # occupant is resolved next -- the same bookkeeping register()
    # itself performs internally.
    behavior_layer._decisions_so_far[occupant_id] = decision

    return behavior_layer.simulation.submit_decision(decision)


# =====================================================
# Recommendation/broadcast-aware strategy wrappers.
#
# DecisionContext (behavior/context.py) has no "recommendation" field
# and is off-limits to modify. External recommendations/broadcasts
# are instead written into BehaviorProfile.traits -- the existing,
# explicitly open-ended extension bag ("safe... a BehaviorProfile has
# no rebuild-from-source-of-truth lifecycle", behavior/profile.py) --
# and read here by composition over whatever base strategy an
# occupant's profile template already resolved to. Neither wrapper
# lives in behavior_library/, which stays untouched; both implement
# the exact same RouteChoiceStrategy/DecisionStrategy interfaces those
# strategies already implement, so a base strategy can be swapped in
# or out without either wrapper caring what it is.
# =====================================================


class RecommendationAwareRouteChoiceStrategy(RouteChoiceStrategy):

    def __init__(self, base_strategy: RouteChoiceStrategy):

        self._base = base_strategy

    def choose(self, context) -> RouteChoice:

        recommended_edge_id = (
            context.profile.traits.get("recommended_exit_edge_id")
            or context.profile.traits.get("recommended_stair_edge_id")
        )

        if recommended_edge_id is None:
            return self._base.choose(context)

        baseline = context.engine.nearest_exit(context.start_id)

        if baseline is None:
            return self._base.choose(context)

        candidates = context.engine.alternative_paths(
            context.start_id, baseline.goal.id, k=5,
        )

        for candidate in candidates:

            if any(edge.id == recommended_edge_id for edge in candidate.edges):
                return RouteChoice(goal_id=candidate.goal.id, route=candidate)

        # Recommended exit/stair unreachable from here (closed, or no
        # alternative path passes through it) -- fall back rather than
        # fail; the occupant still gets a route.
        return self._base.choose(context)


# =====================================================


class BroadcastAwareDecisionStrategy(DecisionStrategy):

    def __init__(self, base_strategy: DecisionStrategy):

        self._base = base_strategy

    def decide(self, context) -> ActionIntent:

        override = context.profile.traits.get("broadcast_override")

        if override == "SHELTER_IN_PLACE":
            return ActionIntent(
                occupant_id=context.profile.occupant_id,
                action_type=ActionType.WAIT,
                requires_movement=False,
            )

        if override == "EVACUATE_IMMEDIATELY":
            return ActionIntent(
                occupant_id=context.profile.occupant_id,
                action_type=ActionType.EVACUATE,
                requires_movement=True,
            )

        return self._base.decide(context)
