from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node
from navigation.reachability import bfs_reachable

from scenario_definition.distributions import FixedValue, WeightedOptions

from campaign_feasibility.model import (
    CampaignFeasibilityReport,
    ZoneFeasibilityResult,
    ZoneFeasibilityStatus,
)


# Scenario Campaign Feasibility Preflight -- Phase 1.
# docs/architecture/scenario_campaign_feasibility_preflight_investigation.txt
# Section 3.2/3.3 (Section 3.4's "uncertain middle" enumeration/sampling
# is explicitly Phase 2, not implemented here).
#
# This module is read-only over Building/ScenarioDefinition, builds the
# NavigationGraph exactly once via the unmodified NavigationGraphGenerator,
# and reuses navigation.reachability.bfs_reachable() (itself extracted
# unchanged from scenario_validator.navigation_validation, see that
# module's own docstring) rather than a second, independently written
# reachability algorithm. It never imports scenario_generator or
# scenario_validator, is never imported by either, and never mutates the
# Building or the ScenarioDefinition it is given (Part C: "do not invent
# connectivity... do not modify Building state").

_DOOR_TRAVERSABLE_NAMES = frozenset({"OPEN", "CLOSED"})
_STAIR_TRAVERSABLE_NAME = "AVAILABLE"


def analyze_campaign_feasibility(building, definition) -> CampaignFeasibilityReport:

    # The single entry point (Part A/H): call this once, before the
    # per-scenario generation loop begins, with exactly the in-memory
    # Building/ScenarioDefinition CampaignController.build_config()
    # already assembles. Nothing here samples anything, constructs a
    # Scenario, or calls scenario_validator.

    if building is None:
        return CampaignFeasibilityReport()

    graph = NavigationGraphGenerator().build(building)

    occupied_zone_ids = _potentially_occupied_zone_ids(definition, graph)

    if not occupied_zone_ids:
        return CampaignFeasibilityReport()

    optimistic_map, pessimistic_map = _bound_maps(graph, definition)
    optimistic_predicate = _make_predicate(optimistic_map)
    pessimistic_predicate = _make_predicate(pessimistic_map)

    # Part C/D -- each bound is computed exactly once for the whole
    # graph (not once per occupied zone); every zone's own
    # optimistic/pessimistic verdict is then a plain set-membership
    # check against these two shared results.
    optimistic_reachable_set = bfs_reachable(graph, [Node.OUTSIDE_NODE_ID], optimistic_predicate)
    pessimistic_reachable_set = bfs_reachable(graph, [Node.OUTSIDE_NODE_ID], pessimistic_predicate)

    eligible_zone_ids = _resolve_fire_eligible_zones(definition.fire, graph)
    sampling_population = _resolve_fire_sampling_population(definition.fire, eligible_zone_ids)

    # Part E -- one BFS per distinct fire-eligible zone in the actual
    # sampling population, not one per (occupied zone, fire zone) pair:
    # which nodes stay reachable when a given fire zone F is excluded
    # does not depend on which occupied zone is being asked about, so
    # this is computed once per F and reused for every zone below.
    reachable_excluding = {
        fire_zone_id: bfs_reachable(
            graph, [Node.OUTSIDE_NODE_ID], optimistic_predicate,
            excluded_node_ids={fire_zone_id},
        )
        for fire_zone_id in sampling_population
    }

    zone_results = tuple(
        _evaluate_zone(
            zone_id, optimistic_reachable_set, pessimistic_reachable_set,
            sampling_population, reachable_excluding,
        )
        for zone_id in sorted(occupied_zone_ids)
    )

    return CampaignFeasibilityReport(zone_results=zone_results)


# =====================================================
# Part B -- potentially occupied zones
# =====================================================


def _potentially_occupied_zone_ids(definition, graph):

    zone_ids = set()

    for zone_id, distribution in definition.occupant.occupancy_distribution.items():

        if graph.find_node(zone_id) is None:
            # References an id that doesn't exist on this Building -- a
            # Definition-authoring problem already caught by the
            # existing pre-flight's own _check_ids_exist_on_building();
            # not this analysis's concern (there is no real node to
            # evaluate reachability against).
            continue

        if _can_be_occupied(distribution):
            zone_ids.add(zone_id)

    return frozenset(zone_ids)


def _can_be_occupied(distribution) -> bool:

    # Mirrors scenario_generator.generator.py::_generate_occupant_placements()'s
    # own rounding rule exactly (`count = max(0, int(round(raw_count)))`)
    # -- a zone is "potentially occupied" only if its own declared
    # distribution support can ever round to a positive integer count.
    # A structural read of the Distribution's own declared value set,
    # never a sampled draw (Part B: "be precise here").

    try:

        if isinstance(distribution, FixedValue):
            return round(distribution.value) >= 1

        if isinstance(distribution, WeightedOptions):

            if not distribution.weights:
                return False

            return any(
                weight > 0 and round(value) >= 1
                for value, weight in distribution.weights.items()
            )

        # UniformRange (or any other kind): use its own declared upper
        # bound -- the most positive value its support could ever
        # produce.
        high = getattr(distribution, "high", None)

        if high is None:
            # Unknown distribution shape -- conservatively treat as
            # potentially occupied rather than silently excluding a
            # zone this analysis cannot prove is unoccupiable.
            return True

        return round(high) >= 1

    except (TypeError, ValueError):

        # A non-numeric value where a count was expected -- not this
        # analysis's problem to diagnose (existing structural
        # validation already covers malformed distributions);
        # conservatively treat as potentially occupied.
        return True


# =====================================================
# Part C/D -- optimistic/pessimistic traversability bounds
# =====================================================


def _bound_maps(graph, definition):

    door_distributions = definition.engineering.door_state_distribution
    exit_distributions = definition.engineering.exit_state_distribution
    stair_distributions = definition.engineering.stair_state_distribution

    optimistic = {}
    pessimistic = {}

    for edge in graph.edges:

        if edge.edge_type == Edge.DOOR:
            opt, pess = _door_bounds(door_distributions.get(edge.id), edge.reference)
        elif edge.edge_type == Edge.EXIT:
            opt, pess = _exit_bounds(exit_distributions.get(edge.id), edge.reference)
        elif edge.edge_type == Edge.STAIR:
            opt, pess = _stair_bounds(stair_distributions.get(edge.id))
        else:
            continue

        optimistic[edge.id] = opt
        pessimistic[edge.id] = pess

    return optimistic, pessimistic


def _door_bounds(distribution, door_reference):

    if distribution is None:
        # Mirrors scenario_generator.generator.py::_generate_door_states()'s
        # own default_for() exactly: LOCKED if door.locked, else OPEN if
        # door.normally_open, else CLOSED -- traversable unless locked.
        traversable = not bool(getattr(door_reference, "locked", False))
        return traversable, traversable

    return _bounds_from_distribution(distribution, lambda value: value in _DOOR_TRAVERSABLE_NAMES)


def _exit_bounds(distribution, exit_reference):

    if distribution is None:
        traversable = not bool(getattr(exit_reference, "is_blocked", False))
        return traversable, traversable

    return _bounds_from_distribution(distribution, lambda value: bool(value) is True)


def _stair_bounds(distribution):

    if distribution is None:
        # Stairs have no Building-authored blocking flag in V1 --
        # scenario_generator's own default_for() is always AVAILABLE.
        return True, True

    return _bounds_from_distribution(distribution, lambda value: value == _STAIR_TRAVERSABLE_NAME)


def _bounds_from_distribution(distribution, is_traversable_value):

    if isinstance(distribution, FixedValue):

        traversable = is_traversable_value(distribution.value)
        return traversable, traversable

    if isinstance(distribution, WeightedOptions):

        weights = distribution.weights
        total_weight = sum(weights.values())

        if total_weight <= 0:
            return False, False

        traversable_weight = sum(
            weight for value, weight in weights.items() if is_traversable_value(value)
        )

        optimistic = traversable_weight > 0
        pessimistic = traversable_weight >= total_weight

        return optimistic, pessimistic

    # UniformRange (or any other kind) is not used for door/exit/stair
    # state in practice (their schema is enum-name/bool-valued, per the
    # investigation's own Section 1.2 finding) -- if ever present,
    # degrade safely to the same "uncertain middle" classification a
    # genuinely uncertain WeightedOptions would receive, rather than
    # guessing.
    return True, False


def _make_predicate(state_map):

    def is_traversable(edge):

        if edge.edge_type in (Edge.DOOR, Edge.EXIT, Edge.STAIR):
            return state_map.get(edge.id, True)

        return True

    return is_traversable


# =====================================================
# Part E/F -- fire-origin eligibility, cut analysis, and probability
# =====================================================


def _resolve_fire_eligible_zones(fire_def, graph):

    # Mirrors scenario_generator.generator.py::_generate_fire()'s own
    # eligibility resolution exactly (allowed, minus forbidden,
    # intersected with allowed-floor if stated) -- a pure,
    # RNG-independent restatement of the same Definition-interpretation
    # logic already present there and in scenario_definition.validation's
    # own (misleadingly named) _check_allowed_floor_has_a_reachable_zone.

    zone_ids = frozenset(
        node_id for node_id, node in graph.nodes.items() if node.node_type == Node.ZONE
    )

    if fire_def.allowed_ignition_zone_ids:
        base = frozenset(fire_def.allowed_ignition_zone_ids)
    else:
        base = zone_ids

    eligible = base - frozenset(fire_def.forbidden_ignition_zone_ids)

    if fire_def.allowed_ignition_floor_ids:

        eligible = frozenset(
            zone_id for zone_id in eligible
            if graph.nodes.get(zone_id) is not None
            and graph.nodes[zone_id].floor_id in fire_def.allowed_ignition_floor_ids
        )

    return eligible


def _resolve_fire_sampling_population(fire_def, eligible_zone_ids):

    # Mirrors generator.py's own _generate_fire() sampling rule exactly:
    # a stated ignition_zone_preference is sampled from DIRECTLY (its
    # own keys/weights), *not* intersected with the eligible set --
    # generator.py's own comment names this an explicitly undecided
    # architectural question ("§13's open question... this module
    # trusts the preference distribution as authored"). This analysis
    # is faithful to that ACTUAL runtime behavior, not to what an
    # idealized architecture might otherwise resolve.

    preference = fire_def.ignition_zone_preference

    if preference is not None:

        if isinstance(preference, WeightedOptions):
            return dict(preference.weights)

        if isinstance(preference, FixedValue):
            return {preference.value: 1.0}

        # UniformRange (or an unknown kind) over zone ids is not a
        # shape this Definition schema uses for ignition_zone_preference
        # in practice (zone ids are categorical, not numeric) -- fall
        # through to the eligible-set default rather than fabricate a
        # population.

    # No stated preference -- sample_uniform_choice()'s own default
    # policy (§4.7): uniform over the eligible set.
    return {zone_id: 1.0 for zone_id in eligible_zone_ids}


def _evaluate_zone(
    zone_id, optimistic_reachable_set, pessimistic_reachable_set,
    sampling_population, reachable_excluding,
):

    optimistic_reachable = zone_id in optimistic_reachable_set

    if not optimistic_reachable:

        # Part C/D, Case 1 -- proven zero feasibility, independent of
        # fire and independent of asset-state sampling entirely.
        return ZoneFeasibilityResult(
            occupied_zone_id=zone_id,
            optimistic_reachable=False,
            pessimistic_reachable=False,
            status=ZoneFeasibilityStatus.ERROR,
            explanation=(
                f"Occupied zone {zone_id!r} cannot reach Outside even under the most "
                f"favorable door/exit/stair states, with no fire exclusion applied. "
                f"This is a proven zero-feasibility condition -- no random generation "
                f"attempt can repair it, because no combination of sampled asset "
                f"states can ever connect this zone to an open Exit."
            ),
        )

    pessimistic_reachable = zone_id in pessimistic_reachable_set

    safe_ids = set()
    lethal_ids = set()

    for fire_zone_id, reachable_set in reachable_excluding.items():

        if fire_zone_id == zone_id:
            # Fire igniting in the occupant's own zone is not evaluated
            # by this dimension -- FIRE_ORIGIN_BLOCKS_EVACUATION itself
            # excludes the ignition zone from its own occupied-zone set
            # (`occupied_zone_ids - {ignition_zone_id}`); this analysis
            # reuses the identical exclusion rather than inventing a
            # new interpretation of what fire blocking means.
            continue

        if zone_id in reachable_set:
            safe_ids.add(fire_zone_id)
        else:
            lethal_ids.add(fire_zone_id)

    lethal_probability = _lethal_probability(sampling_population, zone_id, lethal_ids)

    status, explanation = _classify(zone_id, pessimistic_reachable, lethal_ids, lethal_probability)

    return ZoneFeasibilityResult(
        occupied_zone_id=zone_id,
        optimistic_reachable=True,
        pessimistic_reachable=pessimistic_reachable,
        safe_fire_zone_ids=frozenset(safe_ids),
        lethal_fire_zone_ids=frozenset(lethal_ids),
        lethal_fire_probability=lethal_probability,
        status=status,
        explanation=explanation,
    )


def _lethal_probability(sampling_population, zone_id, lethal_ids):

    # Part F -- analytical, not sampled: the exact probability mass the
    # Definition's own actual fire-zone sampling population assigns to
    # zones classified LETHAL for this occupied zone. `zone_id` itself
    # is excluded from both numerator and denominator (see
    # _evaluate_zone's own comment on why).

    relevant = {
        fire_zone_id: weight
        for fire_zone_id, weight in sampling_population.items()
        if fire_zone_id != zone_id
    }

    total_weight = sum(relevant.values())

    if total_weight <= 0:
        # No fire-eligible zone could ever be evaluated against this
        # occupied zone (e.g. the eligible set is empty, or its only
        # member is this zone itself) -- not determinable, not
        # fabricated as 0% or 100%.
        return None

    lethal_weight = sum(
        weight for fire_zone_id, weight in relevant.items() if fire_zone_id in lethal_ids
    )

    return lethal_weight / total_weight


def _classify(zone_id, pessimistic_reachable, lethal_ids, lethal_probability):

    if lethal_ids and lethal_probability is not None and lethal_probability >= 1.0:

        # Part D/H, Case 2 -- the fire-origin distribution guarantees a
        # LETHAL draw for this zone; zero feasible space from the fire
        # dimension alone, regardless of asset-state sampling.
        return (
            ZoneFeasibilityStatus.ERROR,
            (
                f"Occupied zone {zone_id!r} can reach Outside under favorable "
                f"door/exit/stair states, but every evaluated fire-eligible ignition "
                f"zone ({sorted(lethal_ids)!r}) disconnects it "
                f"(P(fire disconnects)={lethal_probability:.2f}). No sampled fire "
                f"placement can produce an evacuable scenario for this zone."
            ),
        )

    if (not pessimistic_reachable) or lethal_ids:

        # Part D/H, Case 3 -- provably non-zero feasible space, but not
        # structurally robust either.
        parts = []

        if not pessimistic_reachable:
            parts.append(
                "reachability depends on which door/exit/stair states are sampled "
                "(unreachable under the worst-case allowed asset state)"
            )

        if lethal_ids:

            probability_text = (
                f"{lethal_probability:.2f}" if lethal_probability is not None else "unknown"
            )
            parts.append(
                f"{len(lethal_ids)} of the evaluated fire-eligible ignition zone(s) "
                f"({sorted(lethal_ids)!r}) disconnect this zone if selected "
                f"(estimated P(fire disconnects)={probability_text})"
            )

        return (
            ZoneFeasibilityStatus.WARNING,
            (
                f"Occupied zone {zone_id!r} is partially feasible: " + "; and ".join(parts) +
                ". Exact asset-state combination analysis (the 'uncertain middle') is "
                "not yet performed -- that is Phase 2. Monitor this campaign's rejection "
                "rate."
            ),
        )

    # Part D/H, Case 4 -- structurally robust.
    return (
        ZoneFeasibilityStatus.OK,
        (
            f"Occupied zone {zone_id!r} is structurally robust: reachable under the "
            f"worst-case allowed door/exit/stair state, and no evaluated fire-eligible "
            f"ignition zone can disconnect it."
        ),
    )
