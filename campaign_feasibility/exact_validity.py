from dataclasses import dataclass, field
from itertools import product
from typing import Dict, FrozenSet, List, Optional, Tuple

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node
from navigation.reachability import bfs_reachable

from scenario_definition.distributions import FixedValue, WeightedOptions

from campaign_feasibility.analysis import (
    _resolve_fire_eligible_zones,
    _resolve_fire_sampling_population,
)


# Scenario Campaign Feasibility Preflight -- Phase 2A/2B/2B.1: Exact
# Candidate Validity Probability Analysis.
# docs/architecture/scenario_campaign_feasibility_phase2_investigation.txt
# docs/architecture/scenario_campaign_acceptance_and_uniqueness_investigation.txt
# docs/architecture/scenario_campaign_feasibility_phase2a_implementation_report.txt
# docs/architecture/scenario_campaign_feasibility_phase2b_implementation_report.txt
# docs/architecture/scenario_campaign_feasibility_min_open_exits_occupancy_implementation_report.txt
#
# Phase 2B.1 (this extension) closes Phase 2B's own disclosed, narrow
# gap: min_open_exits' NAVIGATION-based reachable-egress check now
# accounts for every SAFE zone's occupancy uncertainty exactly (not
# only GUARANTEED-occupied zones) -- see `_combo_occupancy_factor()`
# and the accompanying implementation report.
#
# Computes P(a freshly generated candidate passes the real Scenario
# Validator) EXACTLY, for the finite/discrete engineering-state + fire-
# origin + OCCUPANCY dimension (Phase 2B), whenever the relevant,
# genuinely-uncertain engineering-state space is small enough to
# enumerate. Read-only over Building/ScenarioDefinition; never mutates
# either; never touches scenario_generator or scenario_validator's own
# code; reuses navigation.reachability.bfs_reachable() verbatim (the
# same primitive navigation_validation.py itself uses) rather than a
# second, independently-written reachability algorithm.
#
# Reuses campaign_feasibility.analysis's own fire-eligibility/sampling-
# population resolution (pure, RNG-free restatements of scenario_
# generator.generator.py::_generate_fire()'s own eligibility rule,
# already proven correct by Phase 1's own tests) rather than
# re-deriving them a third time.
#
# Door/exit/stair distributions are re-interpreted here (not reused
# from campaign_feasibility.analysis's own _bound_maps()) because Phase
# 2A needs the EXACT per-object traversable probability, not merely the
# optimistic/pessimistic BOUNDS Phase 1 needs -- the underlying
# traversable-name-sets and Building-default fallback rules are kept
# byte-identical to analysis.py's own (same constants, same fallback
# logic) specifically to avoid semantic drift between the two.
#
# Phase 2B (occupancy uncertainty): a zone's occupant COUNT (and hence
# whether it is occupied at all this attempt) is itself random whenever
# its occupancy_distribution is not a guaranteed-positive FixedValue.
# navigation_validation.py's own `_occupied_zone_ids()` reads only
# `occupant.zone_id` -- never `occupant.position` -- so, combined with
# `Zone.contains()` matching `_generate_occupant_placements()`'s own
# uniform-sampling bounds exactly (re-verified this task), the ONLY
# occupancy fact that ever affects NAVIGATION validity is "does zone Z
# end up with >=1 occupant," never occupant count, identity, or
# continuous position -- so this module never enumerates or
# discretizes anything about occupant count or position; it only needs
# each zone's exact P(zero occupants), computed analytically from the
# Definition's own distribution (see `_p_zero_occupancy()`).

DOOR_TRAVERSABLE_NAMES = frozenset({"OPEN", "CLOSED"})
STAIR_TRAVERSABLE_NAME = "AVAILABLE"

# Exact-enumeration tractability limit -- see the accompanying
# implementation report for the full justification. Bounds the total
# number of (fire-branch x engineering-state-combination) pairs this
# module will evaluate; each pair costs a small, bounded number of BFS
# calls over a graph Phase 1 already builds once per campaign, so this
# keeps worst-case exact-enumeration work in the same low-thousands-of-
# BFS-calls order of magnitude Phase 1's own fire cut-vertex analysis
# already performs routinely. A named, adjustable module constant, not
# a magic number inlined at the call site. Occupancy uncertainty
# (Phase 2B) does NOT multiply into this limit at all -- see the
# accompanying implementation report's proof that occupancy validity,
# conditional on one fixed engineering state, reduces to a closed-form
# product over per-zone P(zero) values, never a Cartesian product over
# occupant-count vectors.
DEFAULT_MAX_ENUMERATED_STATES = 4096

# Phase 2B.1 -- min_open_exits x uncertain-safe-zone-occupancy
# interaction: a SEPARATE tractability limit for the exit-coverage
# subset DP this extension adds (`_exit_coverage_probability_at_least()`),
# bounding 2^(number of open Exit edges in the Building) -- a
# combo-invariant, purely structural upper bound on that DP's own
# "which exits are covered" state space, checked once per
# `compute_exact_candidate_validity()` call, before any enumeration
# begins (never approximated or partially computed). In practice, real
# state counts stay far below this ceiling: zones sharing an identical
# reachable-exit contribution are merged into one DP group first (see
# the accompanying implementation report), so the DP's actual state
# count depends only on the number of *distinct* exit subsets that
# occur among relevant zones, not on how many zones or exits exist.
# This limit exists purely as a hard safety ceiling for the
# pathological case of many, many DISTINCT exit combinations -- kept
# entirely separate from `DEFAULT_MAX_ENUMERATED_STATES` (which bounds
# the door/exit/stair engineering-state enumeration, an unrelated
# dimension) so that neither limit can silently absorb the other's
# budget.
MAX_EXIT_COVERAGE_STATES = 4096


@dataclass(frozen=True)
class CandidateValidityResult:

    # The Phase 2A/2B result model -- deliberately a NEW, separate type
    # from campaign_feasibility.model's ZoneFeasibilityResult (which
    # answers a per-zone reachability-bound question); this answers a
    # single, campaign-wide "P(one freshly generated candidate is
    # accepted)" question, jointly, across every occupancy-relevant
    # zone at once. Carries no severity field of its own (ERROR/
    # WARNING/OK) -- Phase 1's existing severity mechanism is not
    # extended or overloaded by this phase; wiring a result into the
    # existing preflight UI/severity policy is explicitly deferred to a
    # later phase.

    # True exactly when EVERY relevant random dimension this candidate's
    # validity could depend on was included in the calculation --
    # fire origin, door/exit/stair state, min_open_exits (including, as
    # of Phase 2B.1, its interaction with every SAFE zone's occupancy
    # uncertainty -- not only guaranteed-occupied zones), AND every
    # zone's occupancy uncertainty -- with no unresolved distribution
    # anywhere, no engineering-state enumeration skipped for being too
    # large, and no exit-coverage state space skipped for being too
    # large (`state_space_too_large`, Phase 2B.1's own separate,
    # honestly-labeled bound). Before Phase 2B.1, this field could be
    # `True` even when a SAFE uncertain zone's own occupancy could, in
    # principle, have changed the min_open_exits outcome -- a disclosed
    # (in prose, not in this field) gap, now closed: `exact=True` is now
    # genuinely exact for every dimension this module analyzes.
    exact: bool

    # True whenever the ANALYZED dimensions were computed exactly --
    # even if some OTHER, unresolved randomness (a zone whose
    # occupancy distribution could not be resolved to an exact P(zero),
    # or an oversized engineering-state space) was intentionally
    # excluded from scope. Kept as the same field name and meaning
    # Phase 2A established (the task's own suggested terminology),
    # narrowed in practice by Phase 2B: far fewer occupancy
    # distributions now qualify as "unresolved" than before, since
    # FixedValue, WeightedOptions, and (new in Phase 2B) both discrete
    # and continuous UniformRange occupancy distributions are all now
    # resolvable to an exact P(zero).
    exact_for_analyzed_dimensions: bool

    p_valid: Optional[float] = None
    p_invalid: Optional[float] = None

    # Counts of enumerated ENGINEERING-STATE combinations (fire-branch x
    # door/exit/stair combination) -- unchanged in meaning from Phase
    # 2A. As of Phase 2B, a single engineering-state combination's own
    # contribution to p_valid/p_invalid can be FRACTIONAL (neither
    # fully valid nor fully invalid) whenever occupancy uncertainty is
    # present -- such a combination is counted in
    # `total_states_considered` but in NEITHER `valid_states` nor
    # `invalid_states` (which only count combinations that are, given
    # the analyzed occupancy distributions, fully deterministic one way
    # or the other). `valid_states + invalid_states <= total_states_
    # considered` therefore holds in general; equality holds exactly
    # when no analyzed zone has genuine occupancy uncertainty (e.g.
    # every analyzed zone is guaranteed-occupied, Phase 2A's own
    # regime).
    total_states_considered: int = 0
    valid_states: int = 0
    invalid_states: int = 0

    pruned: bool = False
    state_space_too_large: bool = False

    analyzed_zone_ids: FrozenSet[str] = field(default_factory=frozenset)
    unresolved_occupancy_zone_ids: FrozenSet[str] = field(default_factory=frozenset)

    warnings: Tuple[str, ...] = ()


# =====================================================
# Occupancy -- Phase 2B: exact P(zero occupants) per zone, computed
# analytically from the Definition's own occupancy_distribution, using
# the SAME rounding rule scenario_generator.generator.py::
# _generate_occupant_placements() uses (`count = max(0, int(round(
# raw_count)))`) -- re-verified against the real Generator source this
# task, not assumed.
# =====================================================


def _p_zero_occupancy(distribution) -> Optional[float]:

    try:

        if isinstance(distribution, FixedValue):
            return 1.0 if round(distribution.value) <= 0 else 0.0

        if isinstance(distribution, WeightedOptions):

            weights = distribution.weights
            total_weight = sum(weights.values())

            if total_weight <= 0:
                # No value can ever legitimately be drawn -- sample()
                # itself would be in undefined territory; this module
                # never guesses, but this degenerate case cannot arise
                # from a Definition that has already passed the
                # existing, separate structural pre-flight checks
                # (WeightedOptions with non-positive total weight is
                # already flagged there). Treated as "cannot resolve"
                # rather than fabricating an answer.
                return None

            zero_weight = sum(
                weight for value, weight in weights.items() if round(value) <= 0
            )

            return zero_weight / total_weight

        low = getattr(distribution, "low", None)
        high = getattr(distribution, "high", None)
        discrete = getattr(distribution, "discrete", None)

        if low is None or high is None or high < low:
            return None

        if discrete:

            low_i, high_i = int(low), int(high)
            total_count = high_i - low_i + 1

            if total_count <= 0:
                return None

            # rng.randint(low, high) samples uniformly over the
            # integers [low, high] inclusive -- the count of those
            # that resolve to zero (i.e. <= 0, per the Generator's own
            # `max(0, ...)` floor) is a closed-form range-intersection
            # count, not a loop.
            zero_count = max(0, min(high_i, 0) - low_i + 1)

            return zero_count / total_count

        # Continuous: raw_count ~ Uniform(low, high); round(x) <= 0 for
        # x < 0.5 (Python's round-half-to-even puts the single
        # boundary point at a measure-zero location that does not
        # affect a continuous distribution's probability mass) --
        # P(zero) is exactly the length of the sub-interval of
        # [low, high] lying below 0.5, divided by the interval's total
        # length. This is an EXACT calculation for a continuous
        # uniform distribution (an interval-length ratio), not an
        # approximation.
        if high == low:
            return 1.0 if round(low) <= 0 else 0.0

        upper_bound = min(high, 0.5)
        zero_length = max(0.0, upper_bound - low)

        return zero_length / (high - low)

    except (TypeError, ValueError):

        return None


def _occupancy_probabilities(definition, graph) -> Tuple[Dict[str, float], FrozenSet[str]]:

    # Returns (resolved, unresolved):
    #   resolved: zone_id -> exact P(zero occupants), for every zone
    #     that (a) exists on the Building, (b) has a stated occupancy_
    #     distribution, and (c) CAN possibly receive occupants
    #     (P(zero) < 1.0) -- a zone that can NEVER be occupied
    #     (P(zero) == 1.0 exactly) is dropped entirely, since it
    #     contributes a fixed, trivial factor of 1 to every combination
    #     regardless (exactly like Phase 1's own `_can_be_occupied()`
    #     filter, restated here for the same reason).
    #   unresolved: zone ids whose occupancy_distribution could not be
    #     resolved to an exact P(zero) at all -- never silently assumed
    #     either way.

    resolved: Dict[str, float] = {}
    unresolved = set()

    for zone_id, distribution in definition.occupant.occupancy_distribution.items():

        if graph.find_node(zone_id) is None:
            # References an id that doesn't exist on this Building --
            # a Definition-authoring problem already caught elsewhere
            # (Phase 1's own precedent); not this analysis's concern.
            continue

        p_zero = _p_zero_occupancy(distribution)

        if p_zero is None:
            unresolved.add(zone_id)
            continue

        if p_zero >= 1.0:
            continue

        resolved[zone_id] = p_zero

    return resolved, frozenset(unresolved)


# =====================================================
# Exact per-object traversable probability (not merely optimistic/
# pessimistic bounds) -- Door/Exit/Stair only "matter," for both
# reachability and min_open_exits purposes, through their TRAVERSABLE/
# not-traversable outcome (never through which specific named state --
# OPEN vs CLOSED are both traversable for a Door, and neither
# min_open_exits nor navigation_validation.py's own checks ever
# distinguish them) -- so every object's distribution is collapsed to
# exactly two outcomes here, each with its own exact probability. This
# is what keeps the enumerated state space 2^k (k = uncertain relevant
# object count), not 3^k.
# =====================================================


def _p_traversable_from_distribution(distribution, is_traversable_value) -> Optional[float]:

    if isinstance(distribution, FixedValue):
        return 1.0 if is_traversable_value(distribution.value) else 0.0

    if isinstance(distribution, WeightedOptions):

        weights = distribution.weights
        total_weight = sum(weights.values())

        if total_weight <= 0:
            return 0.0

        traversable_weight = sum(
            weight for value, weight in weights.items() if is_traversable_value(value)
        )

        return traversable_weight / total_weight

    # UniformRange (or an unknown kind) is not used for door/exit/stair
    # state in practice (confirmed in the Phase 2 investigation) -- no
    # exact probability can be derived from a continuous range's
    # bounds alone without fabricating one; signalling "unknown"
    # (None) rather than guessing keeps this module honest.
    return None


def _p_traversable_door(distribution, door_reference) -> Optional[float]:

    if distribution is None:
        return 0.0 if bool(getattr(door_reference, "locked", False)) else 1.0

    return _p_traversable_from_distribution(distribution, lambda value: value in DOOR_TRAVERSABLE_NAMES)


def _p_traversable_exit(distribution, exit_reference) -> Optional[float]:

    if distribution is None:
        return 0.0 if bool(getattr(exit_reference, "is_blocked", False)) else 1.0

    return _p_traversable_from_distribution(distribution, lambda value: bool(value) is True)


def _p_traversable_stair(distribution) -> Optional[float]:

    if distribution is None:
        return 1.0

    return _p_traversable_from_distribution(distribution, lambda value: value == STAIR_TRAVERSABLE_NAME)


def _p_traversable_maps(graph, definition) -> Tuple[Dict[str, Optional[float]], List[str]]:

    door_distributions = definition.engineering.door_state_distribution
    exit_distributions = definition.engineering.exit_state_distribution
    stair_distributions = definition.engineering.stair_state_distribution

    p_map: Dict[str, Optional[float]] = {}
    unresolvable_edge_ids: List[str] = []

    for edge in graph.edges:

        if edge.edge_type == Edge.DOOR:
            p = _p_traversable_door(door_distributions.get(edge.id), edge.reference)
        elif edge.edge_type == Edge.EXIT:
            p = _p_traversable_exit(exit_distributions.get(edge.id), edge.reference)
        elif edge.edge_type == Edge.STAIR:
            p = _p_traversable_stair(stair_distributions.get(edge.id))
        else:
            continue

        p_map[edge.id] = p

        if p is None:
            unresolvable_edge_ids.append(edge.id)

    return p_map, unresolvable_edge_ids


def _make_predicate_from_states(state_map: Dict[str, bool]):

    def is_traversable(edge):

        if edge.edge_type in (Edge.DOOR, Edge.EXIT, Edge.STAIR):
            return state_map.get(edge.id, True)

        return True

    return is_traversable


def _optimistic_predicate_from_p_map(p_map: Dict[str, Optional[float]]):

    def is_traversable(edge):

        if edge.edge_type in (Edge.DOOR, Edge.EXIT, Edge.STAIR):

            p = p_map.get(edge.id)

            if p is None:
                # Unresolvable distribution -- optimistically assume
                # traversable is possible, consistent with how an
                # unresolvable/uncertain object is treated everywhere
                # else in this module (never silently assumed
                # impassable).
                return True

            return p > 0.0

        return True

    return is_traversable


def _topologically_relevant_edge_ids(graph, zone_ids: FrozenSet[str]) -> FrozenSet[str]:

    # A necessary-condition prune: an edge that is not even in the same
    # connected component as any zone in `zone_ids`, UNDER THE MOST
    # PERMISSIVE POSSIBLE ASSUMPTION (every edge treated as
    # traversable), can never be on any path between that zone and
    # Outside under ANY actual state -- excluding it from the
    # enumeration is therefore always mathematically sound, regardless
    # of how the real traversability distributions resolve. This is a
    # weaker (and much cheaper to compute) prune than a tight minimal-
    # cut relevant-edge-set, but it is exactly sound and reuses the
    # same bfs_reachable() primitive, with an always-True predicate,
    # rather than a second graph algorithm.

    if not zone_ids:
        return frozenset()

    always_traversable = lambda edge: True  # noqa: E731

    reachable_from_outside = bfs_reachable(graph, [Node.OUTSIDE_NODE_ID], always_traversable)

    if not (zone_ids & reachable_from_outside):
        return frozenset()

    relevant_edge_ids = set()

    for node_id in reachable_from_outside:

        node = graph.find_node(node_id)

        if node is None:
            continue

        for _neighbor, edge in graph.find_neighbors(node):
            relevant_edge_ids.add(edge.id)

    return frozenset(relevant_edge_ids)


# =====================================================
# min_open_exits -- structural (unconditional) vs navigation
# (reachability-aware, only active when occupied_zone_ids is
# non-empty, mirroring navigation_validation.py's own early return
# exactly) reproduced here.
#
# Phase 2B.1 (this extension): navigation_validation.py's own check (4)
# (INSUFFICIENT_REACHABLE_EGRESS) counts distinct OPEN exits reachable
# from *any actually-occupied zone* (`_occupied_zone_ids(candidate)`,
# real candidate.occupants -- re-verified against the live validator
# source this task) -- not merely from the GUARANTEED-occupied zones,
# which was Phase 2B's own disclosed, narrow simplification. A SAFE
# uncertain zone (one that does not, by itself, invalidate the
# candidate if occupied) can therefore genuinely change whether the
# egress threshold is met, depending on whether it happens to be
# occupied this attempt. `_zone_reachable_open_exit_sets()` computes,
# per zone, exactly which open exits that zone can reach indoors --
# the same per-exit BFS check (4) itself performs, just also recorded
# per zone instead of collapsed into one boolean -- so this is a
# strict refinement of the existing exact computation, not a
# reinterpretation of it. See `_combo_occupancy_factor()` below for how
# this is combined with the unsafe-zone-emptiness requirement, and the
# accompanying implementation report for the full derivation.
# =====================================================


def _zone_reachable_open_exit_sets(
    graph, state_map: Dict[str, bool], zone_ids: FrozenSet[str],
) -> Dict[str, FrozenSet[str]]:

    # For each zone in `zone_ids`: the set of open Exit ids reachable
    # from that zone under an "indoor" predicate (exits excluded from
    # the traversal itself, matching navigation_validation.py's own
    # `include_exits=False` pattern for check (4)). Costs one
    # `bfs_reachable()` call per open exit (identical cost to the old,
    # single-count `_reachable_open_exit_count()` this replaces) --
    # never one call per zone.

    if not zone_ids:
        return {}

    def indoor_traversable(edge):

        if edge.edge_type == Edge.EXIT:
            return False

        if edge.edge_type in (Edge.DOOR, Edge.STAIR):
            return state_map.get(edge.id, True)

        return True

    per_zone = {zone_id: set() for zone_id in zone_ids}

    for edge in graph.edges:

        if edge.edge_type != Edge.EXIT:
            continue

        if not state_map.get(edge.id, True):
            continue

        reachable_from_exit = bfs_reachable(graph, [edge.from_node], indoor_traversable)

        for zone_id in (zone_ids & reachable_from_exit):
            per_zone[zone_id].add(edge.id)

    return {zone_id: frozenset(exits) for zone_id, exits in per_zone.items()}


def _exit_coverage_probability_at_least(
    groups: List[Tuple[FrozenSet[str], float]], need: int,
) -> float:

    # Exact P(|union of activated groups' contribution sets| >= need),
    # where each group activates independently with its own
    # probability (zones sharing an identical reachable-exit
    # contribution are pre-merged into one group by the caller -- see
    # `_combo_occupancy_factor()` -- so this DP's state count depends
    # only on the number of *distinct* contribution sets and the size
    # of their union, never on the number of underlying zones). Never
    # enumerates occupancy patterns over individual zones/occupants --
    # only over the small "which exits are covered so far" subset
    # state, each transition merging in ONE group's contribution via a
    # frozenset union, exactly the subset-DP the accompanying
    # implementation report derives.

    distribution: Dict[FrozenSet[str], float] = {frozenset(): 1.0}

    for contribution, p_activated in groups:

        next_distribution: Dict[FrozenSet[str], float] = {}

        for state, probability in distribution.items():

            next_distribution[state] = next_distribution.get(state, 0.0) + probability * (1.0 - p_activated)

            covered = state | contribution
            next_distribution[covered] = next_distribution.get(covered, 0.0) + probability * p_activated

        distribution = next_distribution

    return sum(probability for state, probability in distribution.items() if len(state) >= need)


def _combo_occupancy_factor(
    graph, state_map: Dict[str, bool], min_open_exits: int,
    occupancy_p_zero: Dict[str, float],
    unsafe_zone_ids: List[str], safe_zone_ids: List[str],
) -> float:

    # The exact P(this combination's occupancy-dependent validation
    # passes | this engineering-state combination), accounting for
    # BOTH requirements simultaneously: every UNSAFE zone gets zero
    # occupants, AND the (occupancy-dependent) min_open_exits condition
    # holds. See the accompanying implementation report for the full
    # derivation; summarized here:
    #
    # p_U = P(every unsafe zone is empty) -- an unsafe zone must never
    #   be allowed to "help" satisfy min_open_exits (Phase 2B.1's own
    #   explicit requirement): its only valid contribution is zero
    #   occupants, so it never contributes to any reachable-exit set
    #   below.
    #
    # Among the SAFE zones: `base_reachable` is the set of open exits
    # already guaranteed reachable from the SAFE zones that are
    # GUARANTEED occupied (P(zero) == 0.0) -- these are occupied with
    # certainty, so their contribution is unconditional. If
    # `base_reachable` alone already meets `min_open_exits`, no
    # uncertain zone's occupancy can possibly matter to this
    # condition -- the Phase 2B closed-form shortcut (no per-exit DP
    # needed at all) is preserved exactly for this common case.
    #
    # Otherwise, the remaining SAFE zones with genuine uncertainty
    # (0 < P(zero) < 1) each contribute their OWN reachable-exit set
    # minus what `base_reachable` already covers -- zones with an
    # empty residual contribution are dropped (they can never change
    # the outcome, Phase 2B.1's "redundant uncertain zones" case) and
    # zones with an IDENTICAL residual contribution are merged into
    # one group (Phase 2B.1's "grouping" requirement) before the exact
    # union-coverage DP (`_exit_coverage_probability_at_least()`) runs
    # -- never a Cartesian product over individual zones' occupancy.
    #
    # The one remaining sub-case: nobody among the SAFE zones ends up
    # occupied at all (probability `p_all_safe_empty`, which is
    # provably 0.0 the instant any SAFE zone is guaranteed-occupied,
    # since a guaranteed zone's own P(zero) is 0.0 and it is one of the
    # factors in that product -- so this term only ever contributes
    # when nobody could possibly be forced present). In that sub-case,
    # NAVIGATION never runs at all (matching `validate_navigation()`'s
    # own early return) and only the unconditional STRUCTURAL
    # `min_open_exits` count rule remains active.

    p_unsafe_all_empty = 1.0
    for zone_id in unsafe_zone_ids:
        p_unsafe_all_empty *= occupancy_p_zero[zone_id]

    if min_open_exits <= 0:
        return p_unsafe_all_empty

    zone_reachable = _zone_reachable_open_exit_sets(graph, state_map, frozenset(safe_zone_ids))

    base_reachable: FrozenSet[str] = frozenset()
    for zone_id in safe_zone_ids:
        if occupancy_p_zero[zone_id] == 0.0:
            base_reachable = base_reachable | zone_reachable.get(zone_id, frozenset())

    need = min_open_exits - len(base_reachable)

    if need <= 0:
        return p_unsafe_all_empty

    contribution_groups: Dict[FrozenSet[str], float] = {}

    for zone_id in safe_zone_ids:

        p_zero = occupancy_p_zero[zone_id]

        if p_zero <= 0.0 or p_zero >= 1.0:
            # Guaranteed-occupied (already folded into base_reachable
            # above) or (structurally impossible here, since
            # `_occupancy_probabilities()` already drops P(zero)>=1.0
            # zones entirely) -- neither has any remaining uncertainty
            # to contribute to this DP.
            continue

        residual = zone_reachable.get(zone_id, frozenset()) - base_reachable

        if not residual:
            # Redundant: this zone's reachable exits are already fully
            # covered by `base_reachable` (or it reaches no open exit
            # at all) -- it can never change whether the threshold is
            # met, so it is dropped before the DP ever sees it,
            # exactly as required.
            continue

        p_all_empty_for_residual = contribution_groups.get(residual, 1.0)
        contribution_groups[residual] = p_all_empty_for_residual * p_zero

    groups = [
        (residual, 1.0 - p_all_empty)
        for residual, p_all_empty in contribution_groups.items()
    ]

    coverage_probability = _exit_coverage_probability_at_least(groups, need)

    p_all_safe_empty = 1.0
    for zone_id in safe_zone_ids:
        p_all_safe_empty *= occupancy_p_zero[zone_id]

    structural_ok = _total_open_exit_count(graph, state_map) >= min_open_exits

    return p_unsafe_all_empty * (coverage_probability + (p_all_safe_empty if structural_ok else 0.0))


def _total_open_exit_count(graph, state_map: Dict[str, bool]) -> int:

    return sum(
        1 for edge in graph.edges
        if edge.edge_type == Edge.EXIT and state_map.get(edge.id, True)
    )


# =====================================================
# The main entry point.
# =====================================================


def compute_exact_candidate_validity(
    building, definition, max_states: int = DEFAULT_MAX_ENUMERATED_STATES,
) -> CandidateValidityResult:

    if building is None:

        return CandidateValidityResult(
            exact=False, exact_for_analyzed_dimensions=False,
            warnings=("No Building was supplied -- nothing to analyze.",),
        )

    graph = NavigationGraphGenerator().build(building)

    occupancy_p_zero, unresolved_occupancy_zone_ids = _occupancy_probabilities(definition, graph)
    guaranteed_zones = frozenset(zone_id for zone_id, p in occupancy_p_zero.items() if p == 0.0)

    min_open_exits = definition.engineering.min_open_exits

    p_map, unresolvable_edge_ids = _p_traversable_maps(graph, definition)

    warnings = []

    if unresolvable_edge_ids:
        warnings.append(
            f"{len(unresolvable_edge_ids)} door/exit/stair distribution(s) could not be "
            f"resolved to an exact traversable probability (not a FixedValue or "
            f"WeightedOptions) -- treated optimistically for pruning, excluded from the "
            f"exact enumeration."
        )

    if unresolved_occupancy_zone_ids:
        warnings.append(
            f"{len(unresolved_occupancy_zone_ids)} zone(s) have an occupancy distribution "
            f"that could not be resolved to an exact P(zero occupants) -- excluded from "
            f"this exact analysis, see unresolved_occupancy_zone_ids."
        )

    if not occupancy_p_zero:

        return _compute_structural_only(
            graph, definition, p_map, min_open_exits, unresolved_occupancy_zone_ids,
            max_states, warnings,
        )

    return _compute_with_occupancy(
        graph, definition, occupancy_p_zero, guaranteed_zones, unresolved_occupancy_zone_ids,
        p_map, min_open_exits, max_states, warnings,
    )


def _compute_structural_only(
    graph, definition, p_map, min_open_exits, unresolved_occupancy_zone_ids, max_states, warnings,
) -> CandidateValidityResult:

    # No zone can ever be occupied at all -- navigation_validation.py's
    # own `if not occupied_zone_ids: return report` means NAVIGATION
    # contributes nothing to validity in this case (re-verified against
    # the real code this task). The ONLY min_open_exits-relevant rule
    # still active is the unconditional STRUCTURAL count check
    # (engineering_validation.py's own MIN_OPEN_EXITS_UNSATISFIED,
    # which has no such guard).

    if min_open_exits <= 0:

        return CandidateValidityResult(
            exact=not unresolved_occupancy_zone_ids, exact_for_analyzed_dimensions=True,
            p_valid=1.0, p_invalid=0.0,
            total_states_considered=1, valid_states=1, invalid_states=0,
            analyzed_zone_ids=frozenset(),
            unresolved_occupancy_zone_ids=unresolved_occupancy_zone_ids,
            warnings=tuple(warnings) + (
                "No occupant can ever be present, and min_open_exits<=0 -- no rule this "
                "phase analyzes can reject a candidate.",
            ),
        )

    exit_edges = [edge for edge in graph.edges if edge.edge_type == Edge.EXIT]

    uncertain_exits = [
        edge for edge in exit_edges
        if p_map.get(edge.id) is not None and 0.0 < p_map[edge.id] < 1.0
    ]

    baseline_open_count = sum(
        1 for edge in exit_edges
        if p_map.get(edge.id) is not None and p_map[edge.id] >= 1.0
    )

    total_states = 2 ** len(uncertain_exits)

    if total_states > max_states:

        return CandidateValidityResult(
            exact=False, exact_for_analyzed_dimensions=False,
            total_states_considered=0, state_space_too_large=True,
            unresolved_occupancy_zone_ids=unresolved_occupancy_zone_ids,
            warnings=tuple(warnings) + (
                f"Structural min_open_exits enumeration over {len(uncertain_exits)} "
                f"uncertain exit(s) ({total_states} combinations) exceeds the exact-"
                f"analysis limit ({max_states}).",
            ),
        )

    valid_weight = 0.0
    invalid_weight = 0.0
    valid_states = 0
    invalid_states = 0

    for combo in product([False, True], repeat=len(uncertain_exits)) if uncertain_exits else [()]:

        combo_probability = 1.0
        open_count = baseline_open_count

        for edge, is_open in zip(uncertain_exits, combo):

            p = p_map[edge.id]
            combo_probability *= p if is_open else (1.0 - p)

            if is_open:
                open_count += 1

        if open_count >= min_open_exits:
            valid_weight += combo_probability
            valid_states += 1
        else:
            invalid_weight += combo_probability
            invalid_states += 1

    return CandidateValidityResult(
        exact=not unresolved_occupancy_zone_ids, exact_for_analyzed_dimensions=True,
        p_valid=valid_weight, p_invalid=invalid_weight,
        total_states_considered=valid_states + invalid_states,
        valid_states=valid_states, invalid_states=invalid_states,
        unresolved_occupancy_zone_ids=unresolved_occupancy_zone_ids,
        warnings=tuple(warnings),
    )


def _compute_with_occupancy(
    graph, definition, occupancy_p_zero, guaranteed_zones, unresolved_occupancy_zone_ids,
    p_map, min_open_exits, max_states, warnings,
) -> CandidateValidityResult:

    all_occupancy_zone_ids = frozenset(occupancy_p_zero.keys())

    # Phase 2B.1 state-space protection: the exit-coverage DP
    # (`_exit_coverage_probability_at_least()`) this extension adds is
    # bounded, per combination, by 2^(number of open Exit edges
    # actually relevant to that combination) -- always <= 2^(total
    # Exit edges in the Building), a purely structural, combo-invariant
    # upper bound. Checked ONCE here, before any engineering-state
    # enumeration begins, exactly like `DEFAULT_MAX_ENUMERATED_STATES`
    # itself is checked before its own enumeration -- an oversized
    # request is declined honestly, never partially computed or
    # approximated. Only relevant at all when `min_open_exits > 0` AND
    # at least one zone has genuine occupancy uncertainty (0 < P(zero)
    # < 1) -- otherwise the DP is never reached (see
    # `_combo_occupancy_factor()`'s own closed-form shortcuts), so the
    # check is skipped entirely in that case, preserving every existing
    # fast path unchanged.
    if min_open_exits > 0 and any(0.0 < p < 1.0 for p in occupancy_p_zero.values()):

        total_exit_edge_count = sum(1 for edge in graph.edges if edge.edge_type == Edge.EXIT)

        if 2 ** total_exit_edge_count > MAX_EXIT_COVERAGE_STATES:

            return CandidateValidityResult(
                exact=False, exact_for_analyzed_dimensions=False,
                total_states_considered=0, state_space_too_large=True,
                analyzed_zone_ids=all_occupancy_zone_ids,
                unresolved_occupancy_zone_ids=unresolved_occupancy_zone_ids,
                warnings=tuple(warnings) + (
                    f"The min_open_exits x uncertain-occupancy exit-coverage analysis over "
                    f"{total_exit_edge_count} Exit(s) (up to {2 ** total_exit_edge_count} "
                    f"reachable-exit-coverage states) exceeds the exact-analysis limit "
                    f"({MAX_EXIT_COVERAGE_STATES}).",
                ),
            )

    optimistic_predicate = _optimistic_predicate_from_p_map(p_map)

    optimistic_reachable = bfs_reachable(graph, [Node.OUTSIDE_NODE_ID], optimistic_predicate)

    if not guaranteed_zones <= optimistic_reachable:

        # Proven, exact zero-feasibility -- no combination of sampled
        # states can ever connect every guaranteed-occupied zone to
        # Outside. No enumeration is needed; this matches Phase 1's
        # own Case 1 finding for these zones. Restricted to
        # GUARANTEED zones deliberately (Phase 2A's own precedent,
        # unchanged): a zone that is merely POSSIBLY occupied might
        # simply not be occupied this attempt, so its unreachability
        # alone cannot PROVE zero feasibility the way a guaranteed
        # zone's can.
        return CandidateValidityResult(
            exact=not unresolved_occupancy_zone_ids, exact_for_analyzed_dimensions=True,
            p_valid=0.0, p_invalid=1.0,
            total_states_considered=1, valid_states=0, invalid_states=1,
            analyzed_zone_ids=all_occupancy_zone_ids,
            unresolved_occupancy_zone_ids=unresolved_occupancy_zone_ids,
            warnings=tuple(warnings) + (
                "At least one guaranteed-occupied zone is unreachable even under the "
                "most favorable door/exit/stair states -- proven zero feasibility.",
            ),
        )

    eligible_fire_zones = _resolve_fire_eligible_zones(definition.fire, graph)
    fire_population = _resolve_fire_sampling_population(definition.fire, eligible_fire_zones)

    lethal_fire_zones = set()
    safe_fire_zones = set()

    for fire_zone_id in fire_population:

        remaining = guaranteed_zones - {fire_zone_id}

        if not remaining:
            # Every guaranteed-occupied zone IS this fire zone -- the
            # fire-exclusion rule has nothing left to check (mirrors
            # navigation_validation.py's own `occupied_zone_ids -
            # {ignition_zone_id}` scoping exactly). Fire pruning
            # remains scoped to GUARANTEED zones only, per the Phase 2B
            # task's own explicit instruction to keep this a sound,
            # unchanged pruning criterion -- see the implementation
            # report for why this stays exact.
            safe_fire_zones.add(fire_zone_id)
            continue

        reachable_excluding = bfs_reachable(
            graph, [Node.OUTSIDE_NODE_ID], optimistic_predicate,
            excluded_node_ids={fire_zone_id},
        )

        if remaining <= reachable_excluding:
            safe_fire_zones.add(fire_zone_id)
        else:
            lethal_fire_zones.add(fire_zone_id)

    total_population_weight = sum(fire_population.values())
    lethal_weight = 0.0

    if total_population_weight <= 0:
        # No fire zone to consider at all -- treat as a single branch
        # with no exclusion applied (fire poses no analyzable risk).
        fire_branches = [(None, 1.0)]
    else:

        lethal_weight = sum(fire_population[f] for f in lethal_fire_zones)
        p_lethal_fire = lethal_weight / total_population_weight

        if p_lethal_fire >= 1.0:

            return CandidateValidityResult(
                exact=not unresolved_occupancy_zone_ids, exact_for_analyzed_dimensions=True,
                p_valid=0.0, p_invalid=1.0,
                total_states_considered=1, valid_states=0, invalid_states=1,
                pruned=True,
                analyzed_zone_ids=all_occupancy_zone_ids,
                unresolved_occupancy_zone_ids=unresolved_occupancy_zone_ids,
                warnings=tuple(warnings) + (
                    "Every fire-eligible ignition zone disconnects at least one "
                    "guaranteed-occupied zone -- proven zero feasibility from the fire "
                    "dimension alone.",
                ),
            )

        fire_branches = [
            (fire_zone_id, fire_population[fire_zone_id] / total_population_weight)
            for fire_zone_id in safe_fire_zones
        ]

    # Relevant, genuinely uncertain door/exit/stair objects -- the
    # topological prune now spans every occupancy-relevant zone (not
    # only guaranteed ones), since an uncertain zone's own reachability
    # under a given combination is now examined too (Phase 2B).
    relevant_edge_ids = _topologically_relevant_edge_ids(graph, all_occupancy_zone_ids)

    uncertain_objects = [
        edge for edge in graph.edges
        if edge.id in relevant_edge_ids
        and p_map.get(edge.id) is not None
        and 0.0 < p_map[edge.id] < 1.0
    ]

    combinations_per_branch = 2 ** len(uncertain_objects)
    total_states = len(fire_branches) * combinations_per_branch

    pruned = bool(lethal_fire_zones)

    if total_states > max_states:

        lethal_note = (
            f" ({len(lethal_fire_zones)} fire-eligible zone(s) already proven LETHAL "
            f"and excluded from this count.)" if lethal_fire_zones else ""
        )

        return CandidateValidityResult(
            exact=False, exact_for_analyzed_dimensions=False,
            total_states_considered=0, state_space_too_large=True,
            pruned=pruned,
            analyzed_zone_ids=all_occupancy_zone_ids,
            unresolved_occupancy_zone_ids=unresolved_occupancy_zone_ids,
            warnings=tuple(warnings) + (
                f"Exact enumeration over {len(fire_branches)} safe fire-origin "
                f"branch(es) x {combinations_per_branch} engineering-state "
                f"combination(s) ({total_states} total) exceeds the exact-analysis "
                f"limit ({max_states}).{lethal_note}",
            ),
        )

    valid_weight = 0.0
    invalid_weight = 0.0
    valid_states = 0
    invalid_states = 0

    default_states = {edge.id: (p_map[edge.id] >= 1.0) for edge in graph.edges if p_map.get(edge.id) is not None}

    _EPSILON = 1e-12

    for fire_zone_id, branch_weight in fire_branches:

        combos = product([False, True], repeat=len(uncertain_objects)) if uncertain_objects else [()]

        for combo in combos:

            state_map = dict(default_states)
            combo_probability = 1.0

            for edge, is_traversable in zip(uncertain_objects, combo):

                p = p_map[edge.id]
                state_map[edge.id] = is_traversable
                combo_probability *= p if is_traversable else (1.0 - p)

            predicate = _make_predicate_from_states(state_map)

            reachable_no_exclusion = bfs_reachable(graph, [Node.OUTSIDE_NODE_ID], predicate)

            reachable_excluding_fire = None

            if fire_zone_id is not None:
                reachable_excluding_fire = bfs_reachable(
                    graph, [Node.OUTSIDE_NODE_ID], predicate,
                    excluded_node_ids={fire_zone_id},
                )

            # Phase 2B -- classify EVERY occupancy-relevant zone as
            # SAFE or UNSAFE under this specific (fire, engineering-
            # state) combination, mirroring the exact same two-check
            # logic navigation_validation.py itself applies to a real
            # candidate's real occupied_zone_ids (checks 1 and 2), just
            # applied here to every zone that COULD be occupied, not
            # only the ones that happen to be in one sampled candidate.
            unsafe_zone_ids = []
            safe_zone_ids = []

            for zone_id in occupancy_p_zero:

                if zone_id not in reachable_no_exclusion:
                    unsafe_zone_ids.append(zone_id)
                    continue

                if (
                    fire_zone_id is not None and zone_id != fire_zone_id
                    and zone_id not in reachable_excluding_fire
                ):
                    unsafe_zone_ids.append(zone_id)
                    continue

                safe_zone_ids.append(zone_id)

            # Phase 2B.1 -- the exact, occupancy-pattern-aware
            # combination of "every unsafe zone empty" with the real
            # (occupied-zone-dependent) min_open_exits condition, see
            # `_combo_occupancy_factor()`'s own docstring for the full
            # derivation. This replaces Phase 2B's own guaranteed-
            # zones-only egress approximation with an exact treatment
            # of every SAFE zone's occupancy uncertainty.
            combo_occupancy_factor = _combo_occupancy_factor(
                graph, state_map, min_open_exits, occupancy_p_zero,
                unsafe_zone_ids, safe_zone_ids,
            )

            weight = combo_probability * branch_weight

            valid_weight += weight * combo_occupancy_factor
            invalid_weight += weight * (1.0 - combo_occupancy_factor)

            if combo_occupancy_factor >= 1.0 - _EPSILON:
                valid_states += 1
            elif combo_occupancy_factor <= _EPSILON:
                invalid_states += 1
            # else: a combination whose occupancy contribution is
            # genuinely fractional -- still counted in
            # total_states_considered below, but not attributed to
            # either bucket (see CandidateValidityResult's own
            # docstring for why valid_states + invalid_states can be
            # < total_states_considered as of Phase 2B).

    if total_population_weight > 0:
        invalid_weight += lethal_weight / total_population_weight

    total_states_considered = len(fire_branches) * combinations_per_branch

    return CandidateValidityResult(
        exact=not unresolved_occupancy_zone_ids, exact_for_analyzed_dimensions=True,
        p_valid=valid_weight, p_invalid=invalid_weight,
        total_states_considered=total_states_considered,
        valid_states=valid_states, invalid_states=invalid_states,
        pruned=pruned,
        analyzed_zone_ids=all_occupancy_zone_ids,
        unresolved_occupancy_zone_ids=unresolved_occupancy_zone_ids,
        warnings=tuple(warnings),
    )
