from dataclasses import dataclass
from typing import Any, Dict, Tuple

import networkx as nx

from navigation.node import Node
from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.graph_context_v4 import _build_undirected_graph, compute_graph_context_for_building
from predictive_dataset.topology_signature import StructuralTopologySignature, compute_structural_signature


# =====================================================
# Predictive Dataset V4 milestone, Phase 5/9 -- structural-signature
# EXTENSION capturing the graph-theoretic dimensions the Cross-Topology
# Generalization Investigation's own graph-context work made newly
# relevant (bridge/cut-edge prevalence, betweenness distribution,
# upstream-catchment distribution, branching factor, cycle/ring
# presence, corridor depth, exit asymmetry) -- on top of, never
# replacing, topology_signature.py's V3 fields (floor/zone/door/exit/
# stair/candidate counts, walking-distance/alternative-route stats).
#
# Pure DATASET/CAMPAIGN METADATA, exactly like its V3 predecessor --
# NOT added to predictive_dataset/schema_v4.py's trainable
# CANDIDATE_FEATURE_SCHEMA_V4 (Phase 9's own explicit "do NOT
# automatically expose all of them as ML features" instruction). Used
# ONLY for Phase 5's coverage-gap analysis and Phase 4/16's structural-
# diversity/coverage-map verification.
# =====================================================


@dataclass(frozen=True)
class StructuralTopologySignatureV4:

    v3_signature: StructuralTopologySignature

    bridge_edge_count: int
    bridge_edge_fraction: float

    mean_betweenness: float
    max_betweenness: float
    betweenness_stdev: float

    mean_upstream_catchment: float
    max_upstream_catchment: int

    mean_zone_degree: float
    max_zone_degree: int

    cyclomatic_number: int  # edges - nodes + connected_components; >0 means at least one independent cycle/ring
    has_cycle: bool

    corridor_depth_hops: int  # longest UNWEIGHTED shortest path (in edge count) from any zone to OUTSIDE

    exit_count: int
    exit_catchment_asymmetry: float  # coefficient of variation of upstream_catchment_count across Exit candidates only (0.0 if <2 exits)

    def to_dict(self) -> Dict[str, Any]:
        d = self.v3_signature.to_dict()
        d.update({
            "bridge_edge_count": self.bridge_edge_count,
            "bridge_edge_fraction": self.bridge_edge_fraction,
            "mean_betweenness": self.mean_betweenness,
            "max_betweenness": self.max_betweenness,
            "betweenness_stdev": self.betweenness_stdev,
            "mean_upstream_catchment": self.mean_upstream_catchment,
            "max_upstream_catchment": self.max_upstream_catchment,
            "mean_zone_degree": self.mean_zone_degree,
            "max_zone_degree": self.max_zone_degree,
            "cyclomatic_number": self.cyclomatic_number,
            "has_cycle": self.has_cycle,
            "corridor_depth_hops": self.corridor_depth_hops,
            "exit_count": self.exit_count,
            "exit_catchment_asymmetry": self.exit_catchment_asymmetry,
        })
        return d


def compute_structural_signature_v4(family: str, variant_id: str, building) -> StructuralTopologySignatureV4:

    v3_sig = compute_structural_signature(family, variant_id, building)

    graph, _ = _build_undirected_graph(building)

    if graph.number_of_edges() == 0:
        return StructuralTopologySignatureV4(
            v3_signature=v3_sig, bridge_edge_count=0, bridge_edge_fraction=0.0,
            mean_betweenness=0.0, max_betweenness=0.0, betweenness_stdev=0.0,
            mean_upstream_catchment=0.0, max_upstream_catchment=0,
            mean_zone_degree=0.0, max_zone_degree=0,
            cyclomatic_number=0, has_cycle=False, corridor_depth_hops=0,
            exit_count=0, exit_catchment_asymmetry=0.0,
        )

    per_candidate = compute_graph_context_for_building(building)
    bw_values = [c.betweenness_centrality for c in per_candidate.values()]
    catchment_values = [c.upstream_catchment_count for c in per_candidate.values()]
    bridge_count = sum(1 for c in per_candidate.values() if c.is_bridge)
    n_candidates = len(per_candidate)

    mean_bw = sum(bw_values) / len(bw_values) if bw_values else 0.0
    variance_bw = sum((x - mean_bw) ** 2 for x in bw_values) / len(bw_values) if bw_values else 0.0

    outside = Node.OUTSIDE_NODE_ID
    zone_nodes = [n for n in graph.nodes() if n != outside]
    degrees = dict(graph.degree(zone_nodes))

    n_components = nx.number_connected_components(graph)
    cyclomatic = graph.number_of_edges() - graph.number_of_nodes() + n_components

    depths = []
    for zone in zone_nodes:
        if outside in graph and nx.has_path(graph, zone, outside):
            depths.append(nx.shortest_path_length(graph, zone, outside))
    corridor_depth = max(depths) if depths else 0

    exit_candidate_ids = {
        candidate.candidate_id for candidate in enumerate_candidates(building)
        if candidate.candidate_type == "Exit"
    }
    exit_catchments = [
        per_candidate[cid].upstream_catchment_count
        for cid in exit_candidate_ids if cid in per_candidate
    ]
    exit_count = len(exit_candidate_ids)

    if exit_count >= 2 and exit_catchments:
        mean_exit_catch = sum(exit_catchments) / len(exit_catchments)
        if mean_exit_catch > 0:
            var_exit = sum((x - mean_exit_catch) ** 2 for x in exit_catchments) / len(exit_catchments)
            exit_asymmetry = (var_exit ** 0.5) / mean_exit_catch
        else:
            exit_asymmetry = 0.0
    else:
        exit_asymmetry = 0.0

    return StructuralTopologySignatureV4(
        v3_signature=v3_sig,
        bridge_edge_count=bridge_count,
        bridge_edge_fraction=(bridge_count / n_candidates) if n_candidates else 0.0,
        mean_betweenness=mean_bw,
        max_betweenness=max(bw_values) if bw_values else 0.0,
        betweenness_stdev=variance_bw ** 0.5,
        mean_upstream_catchment=(sum(catchment_values) / len(catchment_values)) if catchment_values else 0.0,
        max_upstream_catchment=max(catchment_values) if catchment_values else 0,
        mean_zone_degree=(sum(degrees.values()) / len(degrees)) if degrees else 0.0,
        max_zone_degree=max(degrees.values()) if degrees else 0,
        cyclomatic_number=cyclomatic,
        has_cycle=cyclomatic > 0,
        corridor_depth_hops=corridor_depth,
        exit_count=exit_count,
        exit_catchment_asymmetry=exit_asymmetry,
    )


def compute_all_signatures_v4(variants) -> Tuple[StructuralTopologySignatureV4, ...]:
    """`variants` is any iterable of predictive_dataset.topologies_v3/v4.
    StructuralVariant (duck-typed on .family/.variant_id/.topology.building)."""

    return tuple(
        compute_structural_signature_v4(v.family, v.variant_id, v.topology.building)
        for v in variants
    )
