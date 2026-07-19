from simulator.decision import ActionType

from behavior.intent import ActionIntent, AlwaysEvacuateDecisionStrategy, DecisionStrategy
from behavior.route_choice import RouteChoice, RouteChoiceStrategy, ShortestRouteChoiceStrategy


# =====================================================
# Human Population & Assisted Evacuation Modeling Phase 3 -- Assistance
# Behaviors. A civilian-only module (kept separate from firefighter
# behavior -- see behavior_library.firefighter_strategies -- per that
# phase's own "keep firefighter logic separate from civilian behavior
# logic" rule; nothing here is imported by, or imports, that module).
#
# The assistance vocabulary is a plain, documented string constant set
# -- the same "opaque identifier string, interpreted only where it
# needs to be" convention behaviour_profile_id itself already
# establishes (scenario/occupant.py) -- rather than an enum, since it
# is carried on ScenarioOccupant.assistance_type (a plain-string field,
# for the same serialization-simplicity reasons behaviour_profile_id
# is a plain string there too).
# =====================================================

ASSISTANCE_HELP = "HELP"
ASSISTANCE_ESCORT = "ESCORT"
ASSISTANCE_PUSH_WHEELCHAIR = "PUSH_WHEELCHAIR"
ASSISTANCE_DRAG = "DRAG"
ASSISTANCE_CARRY = "CARRY"

ASSISTANCE_TYPES = (
    ASSISTANCE_HELP, ASSISTANCE_ESCORT, ASSISTANCE_PUSH_WHEELCHAIR, ASSISTANCE_DRAG, ASSISTANCE_CARRY,
)

# Applied to the HELPER's own walking_speed while assisting -- a
# documented, illustrative simplification (same honesty convention
# DEFAULT_PROFILE_REGISTRY's own walking speeds/compliance levels
# already carry), not a validated mobility-impact model. Escorting/
# pushing/dragging/carrying another person is modeled as pure cost to
# the helper's own pace; the assisted occupant then moves at that same
# already-reduced pace (see AssistanceAwareRouteChoiceStrategy below),
# so neither speed nor the resulting congestion/evacuation-time impact
# needs a second, separate model.
_SPEED_MULTIPLIER_BY_ASSISTANCE_TYPE = {
    ASSISTANCE_HELP: 0.85,
    ASSISTANCE_ESCORT: 0.80,
    ASSISTANCE_PUSH_WHEELCHAIR: 0.75,
    ASSISTANCE_DRAG: 0.50,
    ASSISTANCE_CARRY: 0.55,
}

_ACTION_TYPE_BY_ASSISTANCE_TYPE = {
    ASSISTANCE_HELP: ActionType.HELP,
    ASSISTANCE_ESCORT: ActionType.ESCORT,
    ASSISTANCE_PUSH_WHEELCHAIR: ActionType.PUSH_WHEELCHAIR,
    ASSISTANCE_DRAG: ActionType.DRAG,
    ASSISTANCE_CARRY: ActionType.CARRY,
}

# The role of the two traits keys every assistance-aware strategy below
# reads -- populated onto BehaviorProfile.traits at registration time
# by behaviour_profile_resolver.registrar (translated from
# ScenarioOccupant.assisting_occupant_id/assistance_type), never
# invented here. A profile with neither key set is simply not part of
# any assistance pairing this run -- every strategy below degrades to
# its `fallback` in that case, with zero effect on an ordinary
# occupant.
TRAIT_ASSISTANCE_ROLE = "assistance_role"
TRAIT_ASSISTANCE_TARGET_ID = "assistance_target_id"
TRAIT_ASSISTANCE_TYPE = "assistance_type"
TRAIT_BASE_WALKING_SPEED = "base_walking_speed"

ROLE_HELPER = "HELPER"
ROLE_ASSISTED = "ASSISTED"


def _apply_speed_multiplier(profile, multiplier: float) -> None:

    # Idempotent regardless of how many times decide()/choose() runs
    # for the same profile (e.g. simulation_interactive's replanning
    # calls the same DecisionStrategy/RouteChoiceStrategy repeatedly
    # against the same BehaviorProfile instance) -- always multiplies
    # from a once-captured base_walking_speed, never from whatever
    # walking_speed already happens to be, so repeated calls never
    # compound the reduction.

    base_speed = profile.traits.get(TRAIT_BASE_WALKING_SPEED)

    if base_speed is None:
        base_speed = profile.walking_speed
        profile.traits[TRAIT_BASE_WALKING_SPEED] = base_speed

    if base_speed is not None:
        profile.walking_speed = base_speed * multiplier


# =====================================================


class AssistanceAwareDecisionStrategy(DecisionStrategy):

    # Wraps any existing DecisionStrategy: an occupant with no
    # assistance role assigned this run delegates entirely to
    # `fallback`, with zero behavioral difference from not wrapping it
    # at all. An occupant assigned the HELPER role reduces their own
    # walking_speed by the assistance type's documented multiplier
    # (see _SPEED_MULTIPLIER_BY_ASSISTANCE_TYPE) and returns a
    # movement intent labeled with the matching ActionType; an occupant
    # assigned the ASSISTED role returns ActionType.ASSISTED --
    # AssistanceAwareRouteChoiceStrategy below is what actually couples
    # their route/speed to their helper's.

    def __init__(self, fallback=None):

        self.fallback = fallback or AlwaysEvacuateDecisionStrategy()

    # =====================================================

    def decide(self, context) -> ActionIntent:

        role = context.profile.traits.get(TRAIT_ASSISTANCE_ROLE)

        if role == ROLE_HELPER:
            return self._decide_helper(context)

        if role == ROLE_ASSISTED:
            return self._decide_assisted(context)

        return self.fallback.decide(context)

    # =====================================================

    def _decide_helper(self, context) -> ActionIntent:

        assistance_type = context.profile.traits.get(TRAIT_ASSISTANCE_TYPE, ASSISTANCE_HELP)
        target_id = context.profile.traits.get(TRAIT_ASSISTANCE_TARGET_ID)

        multiplier = _SPEED_MULTIPLIER_BY_ASSISTANCE_TYPE.get(assistance_type, 1.0)
        _apply_speed_multiplier(context.profile, multiplier)

        action_type = _ACTION_TYPE_BY_ASSISTANCE_TYPE.get(assistance_type, ActionType.HELP)

        return ActionIntent(
            occupant_id=context.profile.occupant_id,
            action_type=action_type,
            requires_movement=True,
            metadata={"assistance_type": assistance_type, "assistance_target_id": target_id},
        )

    # =====================================================

    def _decide_assisted(self, context) -> ActionIntent:

        helper_id = context.profile.traits.get(TRAIT_ASSISTANCE_TARGET_ID)

        return ActionIntent(
            occupant_id=context.profile.occupant_id,
            action_type=ActionType.ASSISTED,
            requires_movement=True,
            metadata={"helped_by_occupant_id": helper_id},
        )


# =====================================================


class AssistanceAwareRouteChoiceStrategy(RouteChoiceStrategy):

    # Companion to AssistanceAwareDecisionStrategy -- an occupant with
    # no assistance role, or in the HELPER role, delegates entirely to
    # `fallback` (their own ordinary shortest route). An occupant in
    # the ASSISTED role copies their helper's already-resolved route
    # wholesale (same RouteChoice(goal_id=..., route=...) pattern
    # FollowLeaderRouteChoiceStrategy already establishes) and aligns
    # their own walking_speed to the helper's already-reduced one, so
    # both arrive together, at the same simulated time -- the concrete
    # mechanism that gives "assistance affects movement speed,
    # congestion, and evacuation time" (Phase 3's own requirement) a
    # real effect on MultiAgentSimulation's output, not just a label.
    #
    # Requires the helper to already be registered (and resolved to a
    # movement decision) before the assisted occupant is -- same
    # ordering dependency FollowLeaderRouteChoiceStrategy already
    # documents; behaviour_profile_resolver.registrar registers every
    # HELPER before the specific occupant they assist for exactly this
    # reason. Falls back gracefully (their own independent route, at
    # their own unreduced speed) rather than raising if that ordering
    # is not honored, or if the helper could not be resolved at all
    # (e.g. the helper themselves turned out unreachable).

    def __init__(self, fallback=None):

        self.fallback = fallback or ShortestRouteChoiceStrategy()

    # =====================================================

    def choose(self, context) -> RouteChoice:

        role = context.profile.traits.get(TRAIT_ASSISTANCE_ROLE)

        if role != ROLE_ASSISTED:
            return self.fallback.choose(context)

        helper_id = context.profile.traits.get(TRAIT_ASSISTANCE_TARGET_ID)

        if helper_id is None:
            return self.fallback.choose(context)

        helper_decision = context.decisions_so_far.get(helper_id)

        if helper_decision is None or helper_decision.goal_id is None:
            return self.fallback.choose(context)

        # Recomputed from the assisted occupant's *own* start_id to the
        # helper's already-resolved goal, rather than reusing
        # helper_decision.route verbatim -- correct for both the common
        # case (a civilian assistance pairing that starts in the same
        # zone: dijkstra(same start, same goal) trivially reproduces
        # the identical route the helper got, since PathfindingEngine's
        # search is a deterministic, pure function of (start, goal,
        # graph, cost model)) and the case where it does not (Phase 4's
        # firefighter rescue: the helper's own route runs from the
        # firefighter's entry point, through the rescued occupant's
        # zone, onward to the exit -- the rescued occupant must only
        # ever be assigned the *remaining* leg from their own actual
        # position, never a route that implies they started somewhere
        # they never were). A shortest sub-path of an already-shortest
        # path is itself shortest, so this is not an approximation of
        # "riding along with the helper" -- it is the exact tail of
        # their route, recovered by asking the same question
        # (shortest path to this goal) from this occupant's own
        # position instead of slicing/re-costing the helper's Route
        # object by hand.
        route = context.engine.dijkstra(context.start_id, helper_decision.goal_id)

        if route is None:
            return self.fallback.choose(context)

        if helper_decision.walking_speed is not None:
            context.profile.walking_speed = helper_decision.walking_speed

        return RouteChoice(goal_id=helper_decision.goal_id, route=route)
