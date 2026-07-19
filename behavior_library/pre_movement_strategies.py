import math
import random

from behavior.pre_movement import NoPreMovementDelay, PreMovementDelayStrategy


class ProbabilisticPreMovementDelay(PreMovementDelayStrategy):

    # A lognormal pre-movement delay -- lognormal is the shape commonly
    # used in evacuation literature for human response/pre-movement
    # time, but this is a simple, two-parameter draw, not a validated
    # RSET pre-movement-time model -- same documented-simplification
    # honesty as every other Default*/Strategy in this codebase.
    #
    # median_delay (seconds) is the distribution's median -- the most
    # directly interpretable parameter for a scenario author ("half of
    # occupants start moving within this many seconds"). spread is the
    # underlying normal distribution's sigma, controlling how variable
    # individual response times are; larger spread means a longer,
    # heavier tail of very late movers. rng is constructor-injected
    # for the same reason as ComplianceDecisionStrategy's.

    def __init__(self, median_delay, spread=0.5, rng=None):

        if median_delay <= 0:
            raise ValueError(
                f"ProbabilisticPreMovementDelay.median_delay must be > 0, "
                f"got {median_delay!r}."
            )

        self.median_delay = median_delay
        self.spread = spread
        self.rng = rng or random.Random()

    # =====================================================

    def delay(self, context) -> float:

        # Prefers context.rng when supplied -- see
        # ComplianceDecisionStrategy's identical comment
        # (docs/architecture/reproducibility_review.md §7.2).
        rng = context.rng if context.rng is not None else self.rng

        mu = math.log(self.median_delay)

        return rng.lognormvariate(mu, self.spread)


class HesitationPreMovementDelayStrategy(PreMovementDelayStrategy):

    # Models decision hesitation as the time spent weighing several
    # comparably-attractive evacuation routes -- not literally "at a
    # blocked exit": PathfindingEngine bakes traversability into its
    # own Dijkstra relaxation (never into a CostModel), so there is no
    # honest way for a strategy to ask "would my preferred exit have
    # been reachable if nothing were blocked" without bypassing that
    # check. Genuine route ambiguity is the honestly-observable signal
    # instead -- reuses PathfindingEngine.alternative_paths() exactly
    # as FamiliarityBasedRouteChoiceStrategy/StaticHerdingRouteChoiceStrategy
    # already do, entirely independent of whichever RouteChoiceStrategy
    # actually runs alongside this one (both simply query context.engine
    # again, which is a pure, cheap, stateless function of current
    # graph state -- see PathfindingEngine's own docstring).
    #
    # tie_margin defines "comparably attractive": a candidate counts as
    # tied with the cheapest one when its own total_cost is within
    # tie_margin (a fraction, e.g. 0.15 = within 15%) of it. Delay grows
    # linearly with how many tied options exist beyond the first, capped
    # at max_hesitation so an occupant facing many similar choices still
    # hesitates a bounded amount. Entirely deterministic -- no `random`
    # anywhere, satisfying "no additional randomness" exactly; the same
    # DecisionContext always produces the same delay.

    def __init__(self, max_alternatives=5, tie_margin=0.15,
                 hesitation_per_tied_option=3.0, max_hesitation=20.0, fallback=None):

        self.max_alternatives = max_alternatives
        self.tie_margin = tie_margin
        self.hesitation_per_tied_option = hesitation_per_tied_option
        self.max_hesitation = max_hesitation
        self.fallback = fallback or NoPreMovementDelay()

    # =====================================================

    def delay(self, context) -> float:

        baseline_delay = self.fallback.delay(context)

        baseline = context.engine.nearest_exit(context.start_id)

        if baseline is None:
            return baseline_delay

        candidates = context.engine.alternative_paths(
            context.start_id, baseline.goal.id, k=self.max_alternatives,
        )

        if len(candidates) <= 1:
            return baseline_delay

        # alternative_paths() already returns candidates in
        # non-decreasing total_cost order -- candidates[0] is the
        # cheapest.
        cheapest_cost = candidates[0].total_cost

        if cheapest_cost <= 0:
            tied_count = sum(1 for route in candidates if route.total_cost <= 0)
        else:
            threshold = cheapest_cost * (1.0 + self.tie_margin)
            tied_count = sum(1 for route in candidates if route.total_cost <= threshold)

        extra_options = max(0, tied_count - 1)
        hesitation = min(extra_options * self.hesitation_per_tied_option, self.max_hesitation)

        return baseline_delay + hesitation


class FollowLeaderPreMovementDelayStrategy(PreMovementDelayStrategy):

    # Companion to FollowLeaderRouteChoiceStrategy (behavior_library/
    # route_choice_strategies.py) -- a follower shouldn't just copy
    # their leader's route, they should actually depart relative to
    # when the leader does, so a group observably moves together
    # instead of merely converging on the same exit independently.
    # Same "leader_occupant_id" trait convention, same ordering
    # dependency (the leader must already be registered and resolved
    # to a BehaviorDecision before the follower is), same graceful
    # fallback-rather-than-raise shape as that strategy.

    def __init__(self, follow_gap=1.0, fallback=None):

        self.follow_gap = follow_gap
        self.fallback = fallback or NoPreMovementDelay()

    # =====================================================

    def delay(self, context) -> float:

        leader_id = context.profile.traits.get("leader_occupant_id")

        if leader_id is None:
            return self.fallback.delay(context)

        leader_decision = context.decisions_so_far.get(leader_id)

        if leader_decision is None or leader_decision.depart_time is None:
            return self.fallback.delay(context)

        return max(0.0, leader_decision.depart_time) + self.follow_gap
