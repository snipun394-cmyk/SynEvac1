from dataclasses import dataclass
from typing import Any, Dict, Tuple

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.simulation_extractor_v2_1 import build_alternative_route_counts


# =====================================================
# Predictive Dataset V3 milestone, Phase 3 -- StructuralTopologySignature.
# Compact, deterministic DATASET METADATA describing a Building's graph
# shape independently of any scenario state (no fire/occupant/door-state
# distribution involved -- purely floors/zones/doors/exits/stairs/
# candidates/walking-distance/alternative-route structure). Deliberately
# NOT added to predictive_dataset.schema.CANDIDATE_FEATURE_SCHEMA (Phase
# 6's feature-schema freeze) -- this is campaign/variant bookkeeping for
# Phase 4's diversity verification and Phase 14's V2-vs-V3 comparison,
# never a trainable model input.
#
# "route redundancy"/"alternative route count" reuses predictive_dataset.
# simulation_extractor_v2_1.build_alternative_route_counts VERBATIM (the
# one existing implementation in this codebase, already used throughout
# the v2.1+/v3/v3.1 feature-extraction pipeline) rather than inventing a
# second, differently-defined metric -- it counts, per candidate, how
# many OTHER candidates share at least one zone with it.
# =====================================================


@dataclass(frozen=True)
class StructuralTopologySignature:

    family: str
    variant_id: str

    floor_count: int
    zone_count: int
    door_count: int
    exit_count: int
    stair_count: int
    candidate_count: int

    # Graph-level counts: nodes = zones + 1 shared OUTSIDE node (every
    # Exit's to_node); edges = candidates (Door/Exit/Stair, the only
    # edge types enumerate_candidates() returns).
    graph_node_count: int
    graph_edge_count: int

    mean_candidate_walking_distance: float
    max_candidate_walking_distance: float

    mean_alternative_route_count: float
    max_alternative_route_count: float

    def to_dict(self) -> Dict[str, Any]:

        return {
            "family": self.family,
            "variant_id": self.variant_id,
            "floor_count": self.floor_count,
            "zone_count": self.zone_count,
            "door_count": self.door_count,
            "exit_count": self.exit_count,
            "stair_count": self.stair_count,
            "candidate_count": self.candidate_count,
            "graph_node_count": self.graph_node_count,
            "graph_edge_count": self.graph_edge_count,
            "mean_candidate_walking_distance": self.mean_candidate_walking_distance,
            "max_candidate_walking_distance": self.max_candidate_walking_distance,
            "mean_alternative_route_count": self.mean_alternative_route_count,
            "max_alternative_route_count": self.max_alternative_route_count,
        }

    def structural_key(self) -> Tuple[Any, ...]:
        """A rounded, hashable tuple of the graph-shape fields only
        (excludes family/variant_id identity) -- used by Phase 4's
        duplicate-signature detection. Walking-distance/alternative-route
        means are rounded to 1 decimal so two variants that differ only
        by floating-point noise (not real structure) still collapse to
        the SAME key, while genuinely different graphs do not."""

        return (
            self.floor_count, self.zone_count, self.door_count, self.exit_count, self.stair_count,
            self.candidate_count, self.graph_node_count, self.graph_edge_count,
            round(self.mean_candidate_walking_distance, 1), round(self.max_candidate_walking_distance, 1),
            round(self.mean_alternative_route_count, 1), self.max_alternative_route_count,
        )


def compute_structural_signature(family: str, variant_id: str, building) -> StructuralTopologySignature:

    candidates = enumerate_candidates(building)
    edges = edges_by_candidate_id(building)
    alt_route_counts = build_alternative_route_counts(candidates)

    floor_count = len(building.floors)
    zone_count = sum(len(floor.zones) for floor in building.floors)
    door_count = sum(len(floor.doors) for floor in building.floors)
    exit_count = sum(len(floor.exits) for floor in building.floors)
    stair_count = sum(len(floor.stairs) for floor in building.floors)

    distances = [
        edges[c.candidate_id].walking_distance for c in candidates
        if edges[c.candidate_id].walking_distance is not None
    ]
    alt_counts = [alt_route_counts.get(c.candidate_id, 0) for c in candidates]

    return StructuralTopologySignature(
        family=family,
        variant_id=variant_id,
        floor_count=floor_count,
        zone_count=zone_count,
        door_count=door_count,
        exit_count=exit_count,
        stair_count=stair_count,
        candidate_count=len(candidates),
        graph_node_count=zone_count + 1,  # +1 for the shared OUTSIDE node
        graph_edge_count=len(candidates),
        mean_candidate_walking_distance=(sum(distances) / len(distances)) if distances else 0.0,
        max_candidate_walking_distance=max(distances) if distances else 0.0,
        mean_alternative_route_count=(sum(alt_counts) / len(alt_counts)) if alt_counts else 0.0,
        max_alternative_route_count=max(alt_counts) if alt_counts else 0,
    )


def compute_all_signatures(variants) -> Tuple[StructuralTopologySignature, ...]:
    """`variants` is any iterable of predictive_dataset.topologies_v3.
    StructuralVariant (duck-typed on .family/.variant_id/.topology.building
    to avoid a circular import with topologies_v3.py)."""

    return tuple(
        compute_structural_signature(v.family, v.variant_id, v.topology.building)
        for v in variants
    )
