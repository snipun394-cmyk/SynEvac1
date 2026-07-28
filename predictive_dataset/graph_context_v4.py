from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx

from navigation.node import Node
from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates


# =====================================================
# Predictive Dataset V4 milestone, Phase 1 -- the PROMOTED, canonical
# graph-context computation. This is the single shared implementation
# both `predictive_dataset/simulation_extractor_v4.py` (dataset
# extraction) and `predictive_dataset/live_extractor_v4.py` (live
# extraction, never wired into LiveRuntime this milestone) call --
# "use shared computation wherever practical so simulation/live
# semantics cannot silently diverge" (Phase 1's own explicit
# instruction). `predictive_dataset/experimental_features_v4.py` (the
# Cross-Topology Generalization Investigation's own module,
# commit e8d728a) is refactored to DELEGATE to this module rather than
# duplicating the algorithm -- its own public API and test expectations
# are unchanged, only its internals now reuse this canonical version.
#
# WHY THIS IS SAFE TO PROMOTE (proven, not assumed, this milestone):
# `live_runtime/factory.py:220` builds LiveRuntime's own shared
# `navigation_graph` via `NavigationGraphGenerator().build(building)` --
# the EXACT SAME class `predictive_dataset/candidate.py` already uses
# for simulation extraction (`enumerate_candidates`/`edges_by_
# candidate_id`, both called below). LiveRuntime's `building` is,
# per `designer/windows/main_window.py:242-245` ->
# `designer/live_runtime_controller.py` -> `live_runtime_launcher/
# session.py`, literally `self.canvas.scene_obj.project.building` --
# the SAME live, in-memory `models.building.Building` the Designer
# canvas is editing (this repo's own "Building Designer IS the Digital
# Twin" rule, by construction, not by convention). There is therefore
# NO separate live graph representation to keep in sync -- calling this
# module's one function against a simulation-sourced Building or a
# live-sourced Building runs the IDENTICAL code path and produces
# IDENTICAL output for an identical graph shape. See
# `tests/test_predictive_dataset_graph_context_v4.py`'s own
# `SimLiveEquivalenceTests` for the mechanical proof.
# =====================================================


@dataclass(frozen=True)
class CandidateGraphContext:
    """One candidate's static structural graph-context descriptors.
    ALL THREE fields are non-nullable BY CONSTRUCTION: every candidate
    enumerate_candidates() returns has a resolvable graph edge (it came
    from the SAME NavigationGraphGenerator), so betweenness/is_bridge/
    catchment are always real, computed values -- never None, unlike
    e.g. candidate_capacity/candidate_walking_distance, which CAN be
    None when the underlying Door/Exit/Stair object lacks geometry.
    A structurally isolated edge that lies on zero shortest paths is
    still a genuine, honest 0.0/0 -- not missing data."""

    candidate_id: str
    betweenness_centrality: float
    is_bridge: bool
    upstream_catchment_count: int


def _build_undirected_graph(building) -> Tuple[nx.Graph, Dict[Tuple[str, str], list]]:
    """One undirected nx.Graph node per zone + the single shared OUTSIDE
    node, one edge per DISTINCT (zone, zone) pair. When two or more
    candidates connect the exact same pair of nodes (parallel routes,
    e.g. two doors between the same two zones), they collapse onto one
    graph edge and share that edge's centrality/bridge/catchment value
    -- a documented simplification: they are topologically redundant
    alternatives to each other, not distinguishable by pure graph shape."""

    candidates = enumerate_candidates(building)
    edges = edges_by_candidate_id(building)

    graph = nx.Graph()
    edge_to_candidate_ids: Dict[Tuple[str, str], list] = {}

    for candidate in candidates:

        edge_obj = edges[candidate.candidate_id]
        key = tuple(sorted((edge_obj.from_node, edge_obj.to_node)))
        distance = edge_obj.walking_distance if edge_obj.walking_distance else 1.0

        if graph.has_edge(*key):
            edge_to_candidate_ids[key].append(candidate.candidate_id)
        else:
            graph.add_edge(*key, distance=distance)
            edge_to_candidate_ids[key] = [candidate.candidate_id]

    return graph, edge_to_candidate_ids


def compute_graph_context_for_building(building) -> Dict[str, CandidateGraphContext]:
    """The one canonical entry point. Physical/mathematical meaning of
    each descriptor (Phase 1's required documentation):

    betweenness_centrality -- standard graph-theoretic EDGE betweenness
        centrality (networkx.edge_betweenness_centrality, Brandes'
        algorithm, weighted by walking_distance, normalized to [0, 1]
        by node-pair count): the fraction of all-pairs shortest paths in
        the building that pass through this candidate's edge. Physical
        meaning: how structurally CENTRAL this candidate is to the whole
        evacuation graph, not just its own local queue. Applies to every
        candidate type (Door/Exit/Stair) identically -- no type-specific
        caveat. Complexity: O(V*E) once per building (Brandes'
        algorithm), never per row/tick.

    is_bridge -- standard graph-theoretic CUT-EDGE test
        (networkx.bridges): True iff removing this edge increases the
        graph's connected-component count (equivalently: the edge lies
        on no cycle). Physical meaning: whether this candidate is a
        SINGLE POINT OF FAILURE for some zone's evacuation, structurally,
        by design -- not a momentary blockage (that remains the job of
        candidate_traversable). Applies to every candidate type
        identically. Complexity: O(V+E) once per building.

    upstream_catchment_count -- count of zones z (z != OUTSIDE) such
        that this candidate's edge lies on the walking-distance-weighted
        shortest path from z to OUTSIDE (networkx.shortest_path,
        Dijkstra; ties broken by networkx's own deterministic
        implementation -- a zone with multiple equally-short paths is
        only counted once, along ONE of them, a disclosed
        simplification). Physical meaning: how much of the building's
        total occupant-carrying demand structurally DEPENDS on this
        candidate under normal (shortest-path) routing -- a STATIC
        structural proxy for "how many people could ever need this",
        never how many currently do (that remains
        candidate_queue_length/candidate_approaching_count's job).
        Applies to every candidate type identically. Complexity:
        O(V * (E + V log V)) once per building (one shortest-path call
        per zone) -- cheap at this codebase's building scale (single
        digits to low tens of candidates).

    LEAKAGE ANALYSIS (all three): computed ENTIRELY from the Building's
    static NavigationGraph shape (zones, doors, exits, stairs, walking
    distances) -- zero occupancy, fire, time, or scenario-outcome
    dependence anywhere in this function. Cannot leak future information
    by construction, the same class of guarantee
    candidate_alternative_route_count already established
    (simulation_extractor_v2_1.py's own leakage note). Computed ONCE per
    Building (per structural variant in simulation, per Designer edit in
    live operation) and reused across every row/tick that Building
    produces -- never recomputed per candidate-time observation.
    """

    graph, edge_to_candidate_ids = _build_undirected_graph(building)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return {}

    # networkx.edge_betweenness_centrality returns keys in ITS OWN
    # internal (u, v) order, which is not guaranteed to match
    # tuple(sorted(...)) -- every other lookup in this function
    # (bridge_keys, catchment_count) normalizes to a sorted key, so
    # betweenness must too, or every below-lookup silently falls back to
    # the 0.0 default for any edge networkx happened to store in the
    # opposite order. Confirmed as a REAL bug (not hypothetical) during
    # this milestone's schema-promotion audit: 30 of 92 candidate edges
    # across all 16 Dataset V3 structural variants were silently reading
    # 0.0 before this fix.
    raw_betweenness = nx.edge_betweenness_centrality(graph, weight="distance")
    betweenness = {tuple(sorted(edge)): value for edge, value in raw_betweenness.items()}
    bridge_keys = {tuple(sorted(edge)) for edge in nx.bridges(graph)}

    catchment_count: Dict[Tuple[str, str], int] = {key: 0 for key in edge_to_candidate_ids}
    outside = Node.OUTSIDE_NODE_ID

    for zone in graph.nodes():

        if zone == outside or not nx.has_path(graph, zone, outside):
            continue

        path = nx.shortest_path(graph, zone, outside, weight="distance")
        for a, b in zip(path[:-1], path[1:]):
            catchment_count[tuple(sorted((a, b)))] += 1

    result: Dict[str, CandidateGraphContext] = {}

    for key, candidate_ids in edge_to_candidate_ids.items():

        bw = float(betweenness.get(key, 0.0))
        is_bridge = key in bridge_keys
        catch = int(catchment_count.get(key, 0))

        for candidate_id in candidate_ids:
            result[candidate_id] = CandidateGraphContext(
                candidate_id=candidate_id,
                betweenness_centrality=bw,
                is_bridge=is_bridge,
                upstream_catchment_count=catch,
            )

    return result


GRAPH_CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    "candidate_betweenness_centrality",
    "candidate_is_bridge",
    "candidate_upstream_catchment_count",
)
