from dataclasses import dataclass
from typing import Optional, Tuple


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

    SINGLE = "single"
    CHAIN = "chain"
    MERGE = "merge"

    REGION_KINDS = (
        SINGLE,
        CHAIN,
        MERGE,
    )
