from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx
import pandas as pd

from navigation.node import Node
from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.topologies_v3 import StructuralVariant, all_structural_variants_v3
from predictive_dataset.topology_signature import compute_all_signatures


# =====================================================
# Cross-Topology Generalization Investigation, Phases 6 & 8 -- an
# EXPERIMENTAL feature set, deliberately kept separate from
# predictive_dataset/schema.py's frozen CANONICAL_FEATURE_SCHEMA (per
# the milestone charter: "Do not modify the canonical production
# schema yet"). Two independent families of features live here:
#
# (1) NORMALIZED features (Phase 6) -- ratios of already-canonical,
#     already-live-parity-audited raw columns against a small set of
#     STATIC per-structural-variant constants (mean walking distance,
#     candidate count). Every constant is itself honestly available
#     live: it is pure Building/NavigationGraph geometry (Zone/Door/
#     Exit/Staircase counts and distances), never simulator-only truth,
#     computed the SAME way predictive_dataset.topology_signature
#     already computes it for Dataset V3's own diversity verification.
#
# (2) GRAPH-CONTEXT features (Phase 8) -- structural descriptors of a
#     candidate's position in the evacuation graph (edge betweenness
#     centrality, cut-edge/bridge status, upstream demand catchment),
#     computed ONCE per structural variant from the Building's
#     NavigationGraph via networkx (already a project dependency,
#     newly used here). These are STATIC structural facts about the
#     graph shape -- identical for every scenario/tick that shares a
#     structural_variant_id -- never a function of occupant state,
#     fire state, or time, and therefore carry zero future-outcome
#     information (Phase 7's own leakage-audit requirement).
#
# Both are joined onto the canonical per-row feature matrix by
# (structural_variant_id, candidate_id) -- never by scenario_id or
# observation_time. NEITHER family touches predictive_dataset/schema.py,
# predictive_dataset/simulation_extractor*.py, or predictive_dataset/
# live_extractor*.py -- this module is read-only with respect to the
# canonical pipeline, an ADDITIVE experimental extractor only.
# =====================================================


NORMALIZED_FEATURE_NAMES: Tuple[str, ...] = (
    "rel_queue_to_capacity",
    "rel_flow_to_capacity",
    "rel_walking_distance_to_building_mean",
    "rel_alt_route_share",
    "rel_adjacent_occupancy_to_building_occupancy",
)

GRAPH_CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    "graph_edge_betweenness_centrality",
    "graph_is_bridge",
    "graph_upstream_catchment_count",
)


# -----------------------------------------------------
# Phase 6 -- normalized/ratio features.
# -----------------------------------------------------


def variant_structural_constants() -> pd.DataFrame:
    """One row per structural_variant_id: mean_candidate_walking_distance
    and candidate_count, reused verbatim from topology_signature.py
    (Phase 3/4's own StructuralTopologySignature) rather than
    recomputing a second, differently-defined version. Pure static
    Building geometry -- the SAME value regardless of scenario/tick,
    and computable live from the Navigation Graph exactly as it is
    here from the simulator's Building object (same call, same
    function, no simulation-only input)."""

    signatures = compute_all_signatures(all_structural_variants_v3())

    return pd.DataFrame([
        {
            "structural_variant_id": sig.variant_id,
            "variant_mean_walking_distance": sig.mean_candidate_walking_distance,
            "variant_candidate_count": sig.candidate_count,
        }
        for sig in signatures
    ])


def add_normalized_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Returns a COPY of `frame` with NORMALIZED_FEATURE_NAMES columns
    added. `frame` must already carry structural_variant_id and the
    canonical raw columns these ratios are built from. Every ratio uses
    max(denominator, small floor) to avoid divide-by-zero -- never
    silently drops or fabricates a row."""

    frame = frame.merge(variant_structural_constants(), on="structural_variant_id", how="left")

    capacity = frame["candidate_capacity"].astype(float)
    safe_capacity = capacity.where(capacity > 0, 1.0)

    frame["rel_queue_to_capacity"] = frame["candidate_queue_length"].astype(float) / safe_capacity
    frame["rel_flow_to_capacity"] = frame["candidate_recent_flow_rate"].astype(float) / safe_capacity

    safe_mean_distance = frame["variant_mean_walking_distance"].where(frame["variant_mean_walking_distance"] > 0, 1.0)
    frame["rel_walking_distance_to_building_mean"] = frame["candidate_walking_distance"].astype(float) / safe_mean_distance

    safe_route_denominator = (frame["variant_candidate_count"].astype(float) - 1.0).clip(lower=1.0)
    frame["rel_alt_route_share"] = frame["candidate_alternative_route_count"].astype(float) / safe_route_denominator

    safe_occupancy = frame["total_active_occupant_count"].astype(float).where(frame["total_active_occupant_count"] > 0, 1.0)
    frame["rel_adjacent_occupancy_to_building_occupancy"] = frame["candidate_adjacent_zone_occupancy"].astype(float) / safe_occupancy

    return frame.drop(columns=["variant_mean_walking_distance", "variant_candidate_count"])


# -----------------------------------------------------
# Phase 8 -- graph-context structural descriptors.
# -----------------------------------------------------


@dataclass(frozen=True)
class GraphContextRow:

    structural_variant_id: str
    candidate_id: str
    graph_edge_betweenness_centrality: float
    graph_is_bridge: bool
    graph_upstream_catchment_count: int


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


def compute_graph_context_for_variant(variant: StructuralVariant) -> Tuple[GraphContextRow, ...]:
    """Physical meaning of each descriptor (Phase 7 audit answers inline):

    graph_edge_betweenness_centrality -- fraction of all-pairs shortest
        (by walking distance) paths in the building that pass through
        this candidate's edge. Meaning: how structurally CENTRAL this
        candidate is to the whole evacuation graph, not just its own
        local queue. Computable identically in simulation and live (pure
        NavigationGraph geometry). Static structural context, not
        dynamic intelligence. Cannot leak future occupant/fire state --
        it never reads any of it.

    graph_is_bridge -- True if removing this candidate's edge would
        disconnect part of the building from the rest (a cut-edge in
        graph-theory terms). Meaning: whether this candidate is a SINGLE
        POINT OF FAILURE for some zone's evacuation, structurally, by
        design -- not a momentary blockage. Same live/sim parity and
        non-leakage properties as above.

    graph_upstream_catchment_count -- number of zones whose shortest
        (by walking distance) path to OUTSIDE passes through this
        candidate's edge. Meaning: how much of the building's total
        occupant-carrying demand structurally DEPENDS on this candidate
        under normal (shortest-path) routing -- a static structural
        proxy for "how many people could ever need this", not how many
        currently do (that remains the job of the existing dynamic
        candidate_queue_length/candidate_approaching_count features).
        Same live/sim parity and non-leakage properties as above.
    """

    graph, edge_to_candidate_ids = _build_undirected_graph(variant.topology.building)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return ()

    betweenness = nx.edge_betweenness_centrality(graph, weight="distance")
    bridge_keys = {tuple(sorted(edge)) for edge in nx.bridges(graph)}

    catchment_count: Dict[Tuple[str, str], int] = {key: 0 for key in edge_to_candidate_ids}
    outside = Node.OUTSIDE_NODE_ID

    for zone in graph.nodes():

        if zone == outside or not nx.has_path(graph, zone, outside):
            continue

        path = nx.shortest_path(graph, zone, outside, weight="distance")
        for a, b in zip(path[:-1], path[1:]):
            catchment_count[tuple(sorted((a, b)))] += 1

    rows = []
    for key, candidate_ids in edge_to_candidate_ids.items():

        bw = float(betweenness.get(key, 0.0))
        is_bridge = key in bridge_keys
        catch = int(catchment_count.get(key, 0))

        for candidate_id in candidate_ids:
            rows.append(GraphContextRow(
                structural_variant_id=variant.variant_id,
                candidate_id=candidate_id,
                graph_edge_betweenness_centrality=bw,
                graph_is_bridge=is_bridge,
                graph_upstream_catchment_count=catch,
            ))

    return tuple(rows)


def build_graph_context_table() -> pd.DataFrame:
    """One row per (structural_variant_id, candidate_id) across all 16
    Dataset V3 structural variants -- static, computed once, never per
    scenario or per tick."""

    rows = []
    for variant in all_structural_variants_v3():
        rows.extend(compute_graph_context_for_variant(variant))

    return pd.DataFrame([r.__dict__ for r in rows])


def add_graph_context_features(frame: pd.DataFrame, graph_context_table: pd.DataFrame = None) -> pd.DataFrame:
    """Returns a COPY of `frame` with GRAPH_CONTEXT_FEATURE_NAMES columns
    added, left-joined by (structural_variant_id, candidate_id)."""

    if graph_context_table is None:
        graph_context_table = build_graph_context_table()

    merged = frame.merge(graph_context_table, on=["structural_variant_id", "candidate_id"], how="left")

    merged["graph_is_bridge"] = merged["graph_is_bridge"].fillna(False).astype(bool)
    merged["graph_edge_betweenness_centrality"] = merged["graph_edge_betweenness_centrality"].fillna(0.0)
    merged["graph_upstream_catchment_count"] = merged["graph_upstream_catchment_count"].fillna(0).astype(int)

    return merged
