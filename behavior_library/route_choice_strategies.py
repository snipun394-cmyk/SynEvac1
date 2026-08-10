import random

from collections import Counter

from behavior.route_choice import RouteChoice, RouteChoiceStrategy, ShortestRouteChoiceStrategy

from behavior_library import kinateder_warren_2021_herding_evidence


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

        # Prefers context.rng when supplied -- see
        # ComplianceDecisionStrategy's identical comment
        # (docs/architecture/reproducibility_review.md §7.2).
        rng = context.rng if context.rng is not None else self.rng

        if rng.random() > self.follow_probability:
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


class EmpiricalProportionHerdingRouteChoiceStrategy(RouteChoiceStrategy):

    # Empirically Parameterized Proportion-Conditioned Herding --
    # Kinateder & Warren (2021), see behavior_library.
    # kinateder_warren_2021_herding_evidence's own module docstring for
    # the full citation and the published coefficients this class
    # consumes. StaticHerdingRouteChoiceStrategy (above) is completely
    # untouched by this class -- it is reused, unmodified, as the
    # actual roll-and-route-selection mechanism (see choose() below),
    # never duplicated.
    #
    # SEMANTIC LIMITATION, disclosed here rather than hidden: this
    # class's own "observed crowd proportion"/"observed crowd size"
    # (derived from context.decisions_so_far, exactly like
    # StaticHerdingRouteChoiceStrategy's own exit_edge_counts) are a
    # STRUCTURAL/FUNCTIONAL analogue of Kinateder & Warren's own
    # simultaneously-visible virtual crowd, not a perceptual
    # equivalent -- decisions_so_far has no spatial/visibility gating
    # and grows with registration order, not with what an occupant
    # could actually see at their own moment of choice. This
    # implementation is therefore EMPIRICALLY PARAMETERIZED, not
    # PERCEPTUALLY VALIDATED -- see the accompanying architecture
    # document for the full disclosure.
    #
    # `legacy_follow_probability` is the fallback used whenever the
    # observed (crowd_size, majority_proportion) pair falls outside
    # the evidence's own supported domain (see kinateder_warren_2021_
    # herding_evidence.follow_probability_for()'s own docstring for the
    # exact policy) -- in that case this class behaves EXACTLY like a
    # plain StaticHerdingRouteChoiceStrategy(follow_probability=
    # legacy_follow_probability, ...) would, since that is literally
    # what it delegates to.

    def __init__(self, legacy_follow_probability=1.0, max_alternatives=5, rng=None, fallback=None):

        self.legacy_follow_probability = legacy_follow_probability
        self.max_alternatives = max_alternatives
        self.rng = rng or random.Random()
        self.fallback = fallback or ShortestRouteChoiceStrategy()

    # =====================================================

    def choose(self, context) -> RouteChoice:

        # Steps 1-4: identical construction to StaticHerdingRouteChoice
        # Strategy's own exit_edge_counts (necessarily recomputed here,
        # not shared -- this class needs the raw counts to derive
        # crowd_size/majority_proportion BEFORE it can even decide
        # which follow_probability to hand the delegate below; this is
        # the one, cheap, pure-computation duplication accepted by the
        # approved design -- the actual roll-and-route-selection logic,
        # the much larger and more consequential method body, is never
        # duplicated).
        exit_edge_counts = Counter(
            decision.route.edges[-1].id
            for decision in context.decisions_so_far.values()
            if decision.route is not None and decision.route.edges
        )

        if not exit_edge_counts:
            return self.fallback.choose(context)

        total_observed_decisions = sum(exit_edge_counts.values())
        majority_count = exit_edge_counts.most_common(1)[0][1]
        majority_proportion = majority_count / total_observed_decisions

        # Step 5-6: empirical domain check + fixed-effects probability,
        # or None (out of domain) -> legacy constant fallback.
        follow_probability = kinateder_warren_2021_herding_evidence.follow_probability_for(
            crowd_size=total_observed_decisions, majority_proportion=majority_proportion,
        )

        if follow_probability is None:
            follow_probability = self.legacy_follow_probability

        # Steps 7-9: exactly one Bernoulli decision, majority-exit
        # selection, and fallback -- all owned, unmodified, by
        # StaticHerdingRouteChoiceStrategy itself. No new randomness,
        # no new route-selection logic, no congestion/hazard/distance/
        # familiarity weighting is introduced anywhere in this class.
        delegate = StaticHerdingRouteChoiceStrategy(
            follow_probability=follow_probability,
            max_alternatives=self.max_alternatives,
            rng=self.rng,
            fallback=self.fallback,
        )

        return delegate.choose(context)


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


class LoadBalancingRouteChoiceStrategy(RouteChoiceStrategy):

    # Anti-herding: distributes occupants across the top `max_alternatives`
    # candidate exits (PathfindingEngine.alternative_paths(), unmodified)
    # instead of everyone taking the single shortest one. Reuses the
    # exact tallying pattern StaticHerdingRouteChoiceStrategy already
    # established (Counter over context.decisions_so_far, keyed by the
    # actual Exit edge id, not goal_id -- V1's shared "Outside" node
    # makes goal_id useless for this) -- this strategy just steers away
    # from the majority instead of toward it.
    #
    # Each candidate's load is normalized by its exit's own modeled
    # capacity (Edge.capacity -- Exit models one, Door/Stair don't) so
    # a crowded low-capacity exit is recognized as more loaded than an
    # equally-crowded high-capacity one, not just compared by headcount.
    # An exit with no modeled capacity falls back to an undivided load
    # count of 1, the same "None means not applicable, not zero"
    # reading Edge.capacity's own docstring already uses. Ties (e.g. no
    # occupant has been assigned to any candidate yet, and every
    # candidate's exit has equal capacity) are broken toward the
    # cheaper route, so this degrades to ShortestRouteChoiceStrategy's
    # own choice whenever load truly can't distinguish the candidates --
    # exactly the first occupant registered against a building whose
    # exits are otherwise identical.
    #
    # No route.route ever changes after this call -- like every other
    # RouteChoiceStrategy, this runs once, at registration/decision
    # time, never mid-simulation (MultiAgentSimulation's own "no
    # dynamic rerouting of an occupant already in flight" invariant is
    # untouched).

    def __init__(self, max_alternatives=5, fallback=None):

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

        exit_edge_counts = Counter(
            decision.route.edges[-1].id
            for decision in context.decisions_so_far.values()
            if decision.route is not None and decision.route.edges
        )

        best = min(
            candidates,
            key=lambda route: (self._load_score(route, exit_edge_counts), route.total_cost),
        )

        return RouteChoice(goal_id=best.goal.id, route=best)

    # =====================================================

    def _load_score(self, route, exit_edge_counts) -> float:

        if not route.edges:
            return 0.0

        exit_edge = route.edges[-1]
        assigned = exit_edge_counts.get(exit_edge.id, 0)

        effective_capacity = exit_edge.capacity if exit_edge.capacity else 1

        return (assigned + 1) / effective_capacity


class HazardAwareRouteChoiceStrategy(RouteChoiceStrategy):

    # Prefers a safer route over a shorter one -- but only once hazard
    # information actually exists. context.hazard_snapshot is
    # DecisionContext's own documented, optional "Dynamic Hazard
    # Layer's one point of contact with Human Behavior" field (None by
    # default everywhere today); when it's None this strategy delegates
    # to `fallback` (ShortestRouteChoiceStrategy by default) with zero
    # difference in outcome from not having this strategy at all --
    # every existing caller/test that never supplies a hazard_snapshot
    # is completely unaffected.
    #
    # When hazard data is present, scores each of the top
    # `max_alternatives` candidate routes (alternative_paths(),
    # unmodified) by total_cost plus a weighted sum of
    # HazardSnapshot.node_state(node_id).hazard_score across every node
    # the route passes through -- reusing only the already-existing
    # HazardSnapshot reader, never inventing a second hazard scale.
    # Candidates arrive already sorted by ascending total_cost (Yen's
    # algorithm's own guarantee), so a tie in combined score still
    # resolves toward the cheaper/shorter one.

    def __init__(self, max_alternatives=3, hazard_score_weight=50.0, fallback=None):

        self.max_alternatives = max_alternatives
        self.hazard_score_weight = hazard_score_weight
        self.fallback = fallback or ShortestRouteChoiceStrategy()

    # =====================================================

    def choose(self, context) -> RouteChoice:

        if context.hazard_snapshot is None:
            return self.fallback.choose(context)

        baseline = context.engine.nearest_exit(context.start_id)

        if baseline is None:
            return self.fallback.choose(context)

        candidates = context.engine.alternative_paths(
            context.start_id, baseline.goal.id, k=self.max_alternatives,
        )

        if not candidates:
            return self.fallback.choose(context)

        best = min(candidates, key=lambda route: self._combined_score(context, route))

        return RouteChoice(goal_id=best.goal.id, route=best)

    # =====================================================

    def _combined_score(self, context, route) -> float:

        hazard_total = sum(
            context.hazard_snapshot.node_state(node_id).hazard_score
            for node_id in route.node_ids
        )

        return route.total_cost + self.hazard_score_weight * hazard_total
