import random

from collections import Counter

from behavior.route_choice import RouteChoice, RouteChoiceStrategy, ShortestRouteChoiceStrategy


class FamiliarityBasedRouteChoiceStrategy(RouteChoiceStrategy):

    # Familiarity-based exit selection -- among the occupant's own
    # top `max_alternatives` distinct routes to Outside (Pathfinding
    # Engine's alternative_paths(), unmodified), picks whichever one
    # the occupant is most familiar with on average, using
    # BehaviorProfile.familiarity (already an existing field, keyed by
    # Node.id exactly as its own docstring specifies).
    #
    # Deliberately bounded to `max_alternatives` candidates rather than
    # every possible route -- an honest, documented simplification
    # (same "not exhaustive, just a reasonable default" spirit as
    # DefaultCapacityModel/DefaultCongestionModel), not a claim that
    # occupants somehow evaluate every conceivable path. When no
    # familiarity data applies to any candidate, every score is 0.0 and
    # the first (cheapest -- i.e. shortest) candidate wins ties, so
    # this degrades to ShortestRouteChoiceStrategy's own result rather
    # than needing a separate no-data special case.

    def __init__(self, max_alternatives=3, fallback=None):

        self.max_alternatives = max_alternatives
        self.fallback = fallback or ShortestRouteChoiceStrategy()

    # =====================================================

    def choose(self, context) -> RouteChoice:

        baseline = context.engine.nearest_exit(context.start_id)

        if baseline is None:
            return self.fallback.choose(context)

        candidates = context.engine.alternative_paths(
            context.start_id, baseline.goal.id, k=self.max_alternatives,
        )

        if not candidates:
            return self.fallback.choose(context)

        best = max(candidates, key=lambda route: self._familiarity_score(context, route))

        return RouteChoice(goal_id=best.goal.id, route=best)

    # =====================================================

    def _familiarity_score(self, context, route):

        if not route.node_ids:
            return 0.0

        familiarity = context.profile.familiarity

        return sum(
            familiarity.get(node_id, 0.0) for node_id in route.node_ids
        ) / len(route.node_ids)


class StaticHerdingRouteChoiceStrategy(RouteChoiceStrategy):

    # Static herding: routes toward whichever Exit the majority of
    # already-resolved occupants in *this registration session*
    # (context.decisions_so_far) actually used -- "static" because it
    # only ever reads that one fixed snapshot, never iterates using a
    # prior MultiAgentSimulationResult (context.prior_result is DecisionContext's
    # own documented extension point for that kind of adaptive,
    # multi-pass herding; a future "Dynamic Herding" strategy would
    # read that field instead of, or in addition to, this one).
    #
    # Tallied by exit *edge* id, not goal_id -- V1's Navigation Graph
    # shares a single "Outside" node (see navigation/node.py), so every
    # movement decision's goal_id is trivially identical regardless of
    # which Exit was actually used; Route.edges[-1].id is the specific
    # Exit, exactly as PathfindingEngine.nearest_exit()'s own docstring
    # already relies on.
    #
    # Only searches the occupant's own top `max_alternatives` routes
    # for the herd's chosen exit -- if it isn't among them, falls back
    # rather than performing an unbounded search.

    def __init__(self, follow_probability=1.0, max_alternatives=5, rng=None, fallback=None):

        self.follow_probability = follow_probability
        self.max_alternatives = max_alternatives
        self.rng = rng or random.Random()
        self.fallback = fallback or ShortestRouteChoiceStrategy()

    # =====================================================

    def choose(self, context) -> RouteChoice:

        exit_edge_counts = Counter(
            decision.route.edges[-1].id
            for decision in context.decisions_so_far.values()
            if decision.route is not None and decision.route.edges
        )

        if not exit_edge_counts:
            return self.fallback.choose(context)

        if self.rng.random() > self.follow_probability:
            return self.fallback.choose(context)

        majority_exit_edge_id, _ = exit_edge_counts.most_common(1)[0]

        baseline = context.engine.nearest_exit(context.start_id)

        if baseline is None:
            return self.fallback.choose(context)

        candidates = context.engine.alternative_paths(
            context.start_id, baseline.goal.id, k=self.max_alternatives,
        )

        for candidate in candidates:

            if candidate.edges and candidate.edges[-1].id == majority_exit_edge_id:
                return RouteChoice(goal_id=candidate.goal.id, route=candidate)

        return self.fallback.choose(context)


class FollowLeaderRouteChoiceStrategy(RouteChoiceStrategy):

    # Leader/Follower behavior -- generalizes the pattern already
    # proven in test_behavior_layer.py's CopyLeaderRouteChoice into a
    # reusable, defensive library strategy. The follower's own
    # BehaviorProfile.traits carries "leader_occupant_id" (same
    # traits-as-extension-point convention as BasicHelpingDecisionStrategy
    # above) rather than this strategy needing a group registry object
    # of its own -- one instance is reusable across every follower in
    # every group. Requires the leader to already be registered (and
    # resolved to a movement decision) before the follower is
    # registered, same ordering dependency the existing leader/follower
    # test already relies on; falls back gracefully rather than
    # raising if that ordering isn't honored.

    def __init__(self, fallback=None):

        self.fallback = fallback or ShortestRouteChoiceStrategy()

    # =====================================================

    def choose(self, context) -> RouteChoice:

        leader_id = context.profile.traits.get("leader_occupant_id")

        if leader_id is None:
            return self.fallback.choose(context)

        leader_decision = context.decisions_so_far.get(leader_id)

        if leader_decision is None or leader_decision.goal_id is None:
            return self.fallback.choose(context)

        return RouteChoice(goal_id=leader_decision.goal_id, route=leader_decision.route)


class HelpTargetRouteChoiceStrategy(RouteChoiceStrategy):

    # Companion to BasicHelpingDecisionStrategy -- once the Decision
    # Stage has produced a HELP intent, this picks *where* to go:
    # toward the person needing help's own registered location
    # (context.decisions_so_far[target_id].start_id), not toward
    # Outside. Reaching the target and then evacuating together is
    # future work ("basic" scope stops at reaching them); this only
    # ever produces a single-stage route to the target's start_id.
    # Same ordering dependency as FollowLeaderRouteChoiceStrategy: the
    # person needing help must already be registered.

    def __init__(self, fallback=None):

        self.fallback = fallback or ShortestRouteChoiceStrategy()

    # =====================================================

    def choose(self, context) -> RouteChoice:

        target_id = context.profile.traits.get("helping_occupant_id")

        if target_id is None:
            return self.fallback.choose(context)

        target_decision = context.decisions_so_far.get(target_id)

        if target_decision is None:
            return self.fallback.choose(context)

        route = context.engine.dijkstra(context.start_id, target_decision.start_id)

        if route is None:
            return self.fallback.choose(context)

        return RouteChoice(goal_id=target_decision.start_id, route=route)
