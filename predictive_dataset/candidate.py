from dataclasses import dataclass
from typing import Dict, Tuple

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node


# =====================================================
# Per-Candidate Predictive AI Data Foundation milestone, Phase 2 --
# candidate identity. A "candidate" is one Door, Exit, or Staircase a
# future predictive model would score. Identity is always the SAME
# stable id the rest of this codebase already uses for that object
# (models.base_object.BaseObject.id, a uuid4 string reused verbatim as
# navigation.edge.Edge.id -- see navigation/edge.py's own "Edge.id is
# always that object's own id" docstring) -- NEVER row order, NEVER a
# freshly generated id. This is what lets a candidate be matched
# across simulation extraction, dataset serialization, training, and
# live inference (Phase 2's own explicit requirement).
#
# enumerate_candidates() reuses navigation.graph_builder.
# NavigationGraphGenerator directly rather than re-deriving Door/Exit/
# Stair connectivity from Building.ordered_floors() a second way (as
# dataset_builder.schema.ordered_doors/exits/stairs() does for its own,
# unrelated purpose of column-name enumeration) -- the Navigation Graph
# is the one place "which zone(s) does this candidate touch" is already
# correctly resolved (endpoint validation, obstacle wiring, floor
# crossing for Stair), and this module needs exactly that, not just an
# object list.
# =====================================================


@dataclass(frozen=True)
class CandidateIdentity:

    candidate_id: str
    candidate_type: str  # Edge.DOOR | Edge.EXIT | Edge.STAIR

    floor_id: str

    # The zone(s) this candidate is adjacent to -- (from_node, to_node)
    # with the single shared "outside" node filtered out (an Exit's
    # to_node is always Node.OUTSIDE_NODE_ID, which is not a zone any
    # occupant can be "in"). A Door/Stair between two real zones keeps
    # both; an Exit keeps only its one zone.
    zone_ids: Tuple[str, ...]


def enumerate_candidates(building) -> Tuple[CandidateIdentity, ...]:

    graph = NavigationGraphGenerator().build(building)

    candidates = []

    for edge in graph.edges:

        if edge.edge_type not in Edge.EDGE_TYPES:
            continue

        from_node = graph.find_node(edge.from_node)

        zone_ids = tuple(
            node_id for node_id in (edge.from_node, edge.to_node)
            if node_id != Node.OUTSIDE_NODE_ID
        )

        candidates.append(
            CandidateIdentity(
                candidate_id=edge.id,
                candidate_type=edge.edge_type,
                floor_id=from_node.floor_id if from_node is not None else "",
                zone_ids=zone_ids,
            )
        )

    return tuple(candidates)


def edges_by_candidate_id(building) -> Dict[str, Edge]:

    # The matching Edge object for every candidate enumerate_candidates()
    # returns, keyed by the same CandidateIdentity.candidate_id -- built
    # once per Building and threaded into the feature extractors below
    # rather than re-running NavigationGraphGenerator per candidate per
    # tick.

    graph = NavigationGraphGenerator().build(building)

    return {edge.id: edge for edge in graph.edges if edge.edge_type in Edge.EDGE_TYPES}
