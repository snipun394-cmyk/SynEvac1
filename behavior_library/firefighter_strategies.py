from pathfinding.route import Route

from simulator.decision import ActionType

from behavior.intent import ActionIntent, DecisionStrategy
from behavior.route_choice import RouteChoice, RouteChoiceStrategy, ShortestRouteChoiceStrategy


# =====================================================
# Human Population & Assisted Evacuation Modeling Phase 4 -- Firefighter
# Modeling. Deliberately its own module, sharing no DecisionStrategy/
# RouteChoiceStrategy class and no trait-key vocabulary with
# behavior_library.assistance_strategies (civilian behavior) --
# per that phase's own "keep firefighter logic separate from civilian
# behavior logic" rule. The only things shared with the civilian
# module are the primitive, domain-neutral building blocks every
# strategy in this whole library already shares (ShortestRouteChoiceStrategy,
# Route, ActionIntent/DecisionStrategy themselves) -- never a firefighter-
# specific mechanism reused by, or reusing, an assistance-specific one.
#
# Firefighters are not evacuating -- Phase 4's own framing ("their
# objective is rescue and incident response rather than self-
# evacuation"). A rescue is modeled as one combined Route: from the
# firefighter's own entry point, through the rescue target's zone, on
# to the nearest exit -- the rescued occupant is then registered
# (behaviour_profile_resolver.firefighter_registrar, alongside
# behavior_library.assistance_strategies.AssistanceAwareRouteChoiceStrategy,
# entirely reused as-is, not duplicated here) to ride the tail of that
# same combined route once the firefighter's own decision is resolved.
# A firefighter with no rescue assignment searches instead -- Phase 4's
# "searching rooms," modeled honestly as a documented simplification
# (patrol toward the building's own nearest exit from their entry
# point, the only "which area to cover" signal available without a
# real search algorithm -- this platform's own established "not a
# validated life-safety model, an illustrative default" convention,
# restated here for exactly the same reason).
# =====================================================

TRAIT_RESCUE_TARGET_OCCUPANT_ID = "rescue_target_occupant_id"
TRAIT_RESCUE_TARGET_ZONE_ID = "rescue_target_zone_id"
TRAIT_BASE_WALKING_SPEED = "firefighter_base_walking_speed"

# Applied to the firefighter's own walking_speed while actively
# carrying a rescued occupant out -- a documented, illustrative
# simplification (same honesty convention behavior_library.
# assistance_strategies's own multipliers already carry), covering the
# full combined route (search-to-target *and* target-to-exit) for
# simplicity: a firefighter moving with purpose toward a known rescue
# target is not assumed to move at their unencumbered patrol speed
# either.
RESCUE_SPEED_MULTIPLIER = 0.70


def _apply_speed_multiplier(profile, multiplier: float) -> None:

    # Same idempotent-via-a-once-captured-base-speed technique
    # behavior_library.assistance_strategies._apply_speed_multiplier()
    # uses, restated here rather than imported -- see this module's own
    # docstring on why civilian and firefighter mechanisms are kept
    # fully independent.

    base_speed = profile.traits.get(TRAIT_BASE_WALKING_SPEED)

    if base_speed is None:
        base_speed = profile.walking_speed
        profile.traits[TRAIT_BASE_WALKING_SPEED] = base_speed

    if base_speed is not None:
        profile.walking_speed = base_speed * multiplier


# =====================================================


class FirefighterDecisionStrategy(DecisionStrategy):

    # A firefighter with a rescue assignment (BehaviorProfile.traits
    # carries rescue_target_occupant_id/rescue_target_zone_id, set at
    # registration time by behaviour_profile_resolver.
    # firefighter_registrar from ScenarioFirefighter's own Scenario-
    # authored fields -- see that dataclass's own docstring for why
    # this must be decided upstream of simulation, not reactively)
    # intends ActionType.FIREFIGHTER_RESCUE and reduces their own
    # walking_speed for the whole rescue trip. One with no assignment
    # intends ActionType.FIREFIGHTER_SEARCH at their own unmodified
    # (already brisk, geared) pace.

    def decide(self, context) -> ActionIntent:

        target_occupant_id = context.profile.traits.get(TRAIT_RESCUE_TARGET_OCCUPANT_ID)
        target_zone_id = context.profile.traits.get(TRAIT_RESCUE_TARGET_ZONE_ID)

        if target_zone_id is None:

            return ActionIntent(
                occupant_id=context.profile.occupant_id,
                action_type=ActionType.FIREFIGHTER_SEARCH,
                requires_movement=True,
            )

        _apply_speed_multiplier(context.profile, RESCUE_SPEED_MULTIPLIER)

        return ActionIntent(
            occupant_id=context.profile.occupant_id,
            action_type=ActionType.FIREFIGHTER_RESCUE,
            requires_movement=True,
            metadata={
                "rescue_target_occupant_id": target_occupant_id,
                "rescue_target_zone_id": target_zone_id,
            },
        )


# =====================================================


class FirefighterRouteChoiceStrategy(RouteChoiceStrategy):

    def __init__(self, fallback=None):

        self.fallback = fallback or ShortestRouteChoiceStrategy()

    # =====================================================

    def choose(self, context) -> RouteChoice:

        target_zone_id = context.profile.traits.get(TRAIT_RESCUE_TARGET_ZONE_ID)

        if target_zone_id is None:
            return self.fallback.choose(context)

        route = _combined_rescue_route(context.engine, context.start_id, target_zone_id)

        if route is None or route.goal is None:
            # The target zone is unreachable, or has no route onward to
            # any exit -- cannot honestly construct a rescue route;
            # search instead rather than fabricating one.
            return self.fallback.choose(context)

        return RouteChoice(goal_id=route.goal.id, route=route)


# =====================================================


def _combined_rescue_route(engine, start_id, via_zone_id):

    # Two legs, concatenated into one Route: start -> the rescue
    # target's zone, then that zone -> the nearest exit. Both legs are
    # ordinary PathfindingEngine.dijkstra()/nearest_exit() calls --
    # this function performs no graph search of its own, only
    # stitching. When start_id == via_zone_id (the firefighter's own
    # entry point already is the target's zone), leg_to_target is the
    # already-established trivial (zero-edge) Route shape every other
    # "already at goal" case in this codebase produces, and
    # leg_to_target.nodes[1:] below correctly contributes nothing extra.

    leg_to_target = engine.dijkstra(start_id, via_zone_id)

    if leg_to_target is None:
        return None

    leg_to_exit = engine.nearest_exit(via_zone_id)

    if leg_to_exit is None:
        return None

    total_distance = (
        leg_to_target.total_distance + leg_to_exit.total_distance
        if leg_to_target.total_distance is not None and leg_to_exit.total_distance is not None
        else None
    )

    return Route(
        nodes=leg_to_target.nodes + leg_to_exit.nodes[1:],
        edges=leg_to_target.edges + leg_to_exit.edges,
        total_cost=leg_to_target.total_cost + leg_to_exit.total_cost,
        total_distance=total_distance,
    )
