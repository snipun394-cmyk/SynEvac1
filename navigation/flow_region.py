from dataclasses import dataclass, field
from typing import Optional, Tuple

from navigation.edge import Edge


@dataclass(frozen=True)
class FlowRegionMember:

    # Flow Region Capacity Formula V2 -- one member edge's own live
    # Edge reference (never a copy, same convention Edge.reference
    # itself uses), plus which of its two ends is upstream (farther
    # from Outside) vs downstream (closer). This is exactly the
    # orientation FlowRegionInferencer already derives internally to
    # decide chain/merge membership (see flow_region_inference.py) --
    # retained here, instead of being discarded after inference, so a
    # region-level capacity formula can reconstruct the region's own
    # internal flow network (for a min-cut/max-flow computation)
    # without needing the whole NavigationGraph or Building passed to
    # it. Anticipated explicitly by the Flow Region Capacity Formula V2
    # design investigation: "no new topology analysis is needed beyond
    # what Milestone 1 already derives and currently discards after
    # inference."

    edge: Edge
    upstream_node_id: str
    downstream_node_id: str


@dataclass(frozen=True)
class FlowRegion:

    # Hybrid Flow Regions (Option D), Milestone 1 -- a FlowRegion groups
    # one or more Edges (Door/Exit/Stair, possibly mixed types) that
    # represent one continuous, shared physical crowding phenomenon: a
    # multi-flight stairwell, or a stairwell converging through a door
    # into one shared exit. Computed fresh by FlowRegionInferencer every
    # time the Navigation Graph is rebuilt (see
    # navigation/flow_region_inference.py and
    # NavigationGraphGenerator.build()) -- never hand-authored, never
    # persisted independently of the graph that produced it, exactly
    # like Edge/Node themselves.
    #
    # This is a pure data structure. Nothing here reads or writes
    # simulator/capacity/congestion state, and nothing in Pathfinding or
    # navigation/cost.py reads this at all -- routing continues to cost
    # individual Edges exactly as before. See
    # docs/architecture/... Hybrid Flow Regions (Option D) design
    # document, Milestones 2+, for how this object is actually consumed
    # by admission control.

    id: str
    edge_ids: Tuple[str, ...]

    # Diagnostic only, never behavior-affecting in Milestone 1 (nothing
    # yet reads flow_kind to change simulator behavior): SINGLE is a
    # trivial, one-edge region -- the overwhelming majority of edges,
    # identical in effect to today's per-edge behavior. CHAIN is two or
    # more edges linked end-to-end with no real merge point (e.g. every
    # landing of one straight stairwell). MERGE is two or more edges
    # converging into a shared choke point (e.g. several stair flights
    # feeding one lobby door).
    region_kind: str

    # Aggregate physical parameters for the region as a whole, for a
    # future region-level capacity formula to consume (Milestone 2) --
    # computed once here, at inference time, not recomputed by every
    # future caller. None means "not derivable", the same convention
    # Edge.walking_distance/Edge.width already use, since a region's own
    # members may themselves have undeterminable geometry.
    total_length: Optional[float]
    representative_width: Optional[float]

    # Flow Region Capacity Formula V2 -- each member edge's own live
    # Edge reference plus its orientation within this region (see
    # FlowRegionMember above). Defaults to an empty tuple so every
    # existing caller/test that constructs a FlowRegion with only the
    # five original fields keeps working unchanged; populated by
    # FlowRegionInferencer for every region it produces except a
    # SINGLE-kind region built from an edge whose own orientation was
    # undeterminable (non-traversable or an equal-distance tie -- see
    # FlowRegionInferencer's own docstring), which was never going to
    # reach a region-level capacity formula anyway (MultiAgentSimulation
    # resolves a SINGLE-kind region straight through to its one edge,
    # see simulator/coordinator.py::_resolve_admission()).
    member_edges: Tuple[FlowRegionMember, ...] = field(default_factory=tuple)

    SINGLE = "single"
    CHAIN = "chain"
    MERGE = "merge"

    REGION_KINDS = (
        SINGLE,
        CHAIN,
        MERGE,
    )
