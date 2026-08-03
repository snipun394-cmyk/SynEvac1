from collections import defaultdict, deque

from navigation.flow_region import FlowRegion
from simulator.capacity import CapacityModel, DefaultCapacityModel, StairCapacityModel


class FlowRegionCapacityModel(CapacityModel):

    # Hybrid Flow Regions (Option D), Milestone 2 -- a region-aware
    # CapacityModel, built entirely alongside today's per-edge models
    # without touching them. NOT yet used by MultiAgentSimulation or
    # scenario_runner (see Milestone 3+): this class exists so it can
    # be unit-tested and integrated later without any further design
    # work, exactly as the Option D design document's Milestone 2
    # requires.
    #
    # capacity() accepts EITHER a plain Edge (today's usual argument,
    # delegated straight through to `base_model` -- StairCapacityModel
    # by default, the same model scenario_runner already wires up
    # today, so nothing changes for any existing caller that never
    # passes a FlowRegion) OR a FlowRegion (the new case: one
    # aggregate capacity for the whole region). This dual acceptance
    # is what makes the model a genuine drop-in replacement candidate
    # for Milestone 3+ -- a future coordinator can hand this model
    # either an Edge or a FlowRegion and get the right answer either
    # way, without needing to know which one it has.
    #
    # The region formula reuses Option E's own area-based reasoning
    # (capacity ~ footprint area x an assumed jam density) applied to
    # the region's own total footprint instead of one edge's --
    # `total_length` and `representative_width` are exactly the two
    # aggregate fields FlowRegionInferencer already computes at
    # inference time for this purpose (see navigation/flow_region.py),
    # so no per-member-edge lookup is needed here at all.

    JAM_DENSITY_PEOPLE_PER_SQUARE_METER = 2.0

    MINIMUM_CAPACITY = DefaultCapacityModel.MINIMUM_CAPACITY

    def __init__(self, base_model: CapacityModel = None):

        self.base_model = base_model or StairCapacityModel()

    def capacity(self, edge_or_region):

        if isinstance(edge_or_region, FlowRegion):
            return self._region_capacity(edge_or_region)

        return self.base_model.capacity(edge_or_region)

    def _region_capacity(self, region: FlowRegion):

        # Same "None means not derivable, fall back to the minimum"
        # convention Edge.width/Edge.walking_distance and
        # DefaultCapacityModel already use -- a region whose members'
        # own geometry couldn't be determined gets the same one-
        # person-at-a-time floor as an edge in the same situation,
        # never a fabricated number.
        if (
            region.total_length is None
            or region.representative_width is None
        ):
            return self.MINIMUM_CAPACITY

        area = region.total_length * region.representative_width

        derived_capacity = int(
            area * self.JAM_DENSITY_PEOPLE_PER_SQUARE_METER
        )

        return max(derived_capacity, self.MINIMUM_CAPACITY)


# =====================================================
# Flow Region Capacity Formula V2 -- root-caused and designed in the
# Flow Region Capacity Formula V2 Design Investigation, prompted by
# Milestone 5's own finding: FlowRegionCapacityModel's area-summing
# formula (above) gave a 9-edge stair chain a capacity of 121 against a
# true per-flight bottleneck capacity of 1, because it computes "how
# many people could occupy this whole stretch at once" and then uses
# that as a throughput admission gate -- physically the wrong quantity.
#
# Experimental only. FlowRegionCapacityModel (V1) above is completely
# untouched, so every existing test/candidate that names it keeps
# working unchanged; V2 is a new, separate, opt-in class for the
# Calibration Benchmark path only.
# =====================================================


def _min_cut_capacity(member_edges, edge_capacities):

    # Computes the min-cut (== max-flow, by the max-flow/min-cut
    # theorem) of a FlowRegion's own internal flow network: each member
    # edge is a directed arc from its own upstream_node_id to its own
    # downstream_node_id, with capacity edge_capacities[edge.id] (the
    # SAME per-edge capacity StairCapacityModel/DefaultCapacityModel
    # would already give that edge in isolation -- nothing about
    # per-edge capacity computation changes). This single computation
    # is provably correct for both region kinds without any hand-
    # written chain/merge branching:
    #
    # - A pure CHAIN is one straight serial path from a single source
    #   to a single sink -- max-flow along a simple path always equals
    #   its narrowest single edge, exactly the tandem-queueing-theory
    #   result ("a series system's throughput is bounded by its
    #   slowest stage") the design investigation identified as the
    #   correct answer for chains.
    # - A MERGE is several branches converging on one shared discharge
    #   edge -- max-flow into a single sink node is exactly
    #   min(discharge edge's own capacity, sum of what the converging
    #   branches can each deliver), matching the merge-bottleneck
    #   literature the same investigation cited.
    # - A MIXED region (a chain feeding a merge, or vice versa) is
    #   handled automatically, with no special-casing, because it is
    #   the identical underlying graph problem.
    #
    # Region sizes seen in any real building are tiny (at most a
    # handful of converging branches per merge point) -- this is not a
    # performance concern, exactly as the design investigation noted.

    if not member_edges:
        return 0

    if len(member_edges) == 1:
        return edge_capacities[member_edges[0].edge.id]

    downstream_nodes = {member.downstream_node_id for member in member_edges}
    upstream_nodes = {member.upstream_node_id for member in member_edges}

    # Every region FlowRegionInferencer actually produces has exactly
    # one true sink (its shared discharge node) by construction --
    # grouping only ever stops at a fork or at Outside, never leaves
    # two unconnected discharge points inside one region. If that ever
    # doesn't hold (a future inferencer change, or a hand-built region
    # in a test), fall back to plain minimum-bottleneck across every
    # member -- still a safe, conservative answer, never an
    # overestimate.
    sinks = downstream_nodes - upstream_nodes
    sources = upstream_nodes - downstream_nodes

    if len(sinks) != 1 or not sources:
        return min(edge_capacities[member.edge.id] for member in member_edges)

    sink = next(iter(sinks))

    super_source = object()  # unique sentinel, can never collide with a real node id string

    adjacency = defaultdict(list)
    residual = defaultdict(int)

    def _connect(from_node, to_node, capacity):

        if to_node not in adjacency[from_node]:
            adjacency[from_node].append(to_node)

        if from_node not in adjacency[to_node]:
            adjacency[to_node].append(from_node)

        residual[(from_node, to_node)] += capacity

    for member in member_edges:
        _connect(member.upstream_node_id, member.downstream_node_id, edge_capacities[member.edge.id])

    for source in sources:

        source_capacity = sum(
            edge_capacities[member.edge.id]
            for member in member_edges
            if member.upstream_node_id == source
        )
        _connect(super_source, source, source_capacity)

    max_flow = 0

    while True:

        # Edmonds-Karp: BFS for the shortest (fewest-edge) augmenting
        # path in the residual graph, standard and sufficient at these
        # region sizes.
        parent = {super_source: None}
        queue = deque([super_source])

        while queue and sink not in parent:

            current = queue.popleft()

            for neighbor in adjacency[current]:

                if neighbor not in parent and residual[(current, neighbor)] > 0:

                    parent[neighbor] = current
                    queue.append(neighbor)

        if sink not in parent:
            break

        path_flow = float("inf")
        node = sink

        while node != super_source:

            previous = parent[node]
            path_flow = min(path_flow, residual[(previous, node)])
            node = previous

        node = sink

        while node != super_source:

            previous = parent[node]
            residual[(previous, node)] -= path_flow
            residual[(node, previous)] += path_flow
            node = previous

        max_flow += path_flow

    return int(max_flow)


class FlowRegionCapacityModelV2(CapacityModel):

    # Same dual-acceptance shape as FlowRegionCapacityModel (V1): an
    # Edge is delegated straight through to base_model unchanged; only
    # a FlowRegion triggers the region-level (here, min-cut) formula.

    MINIMUM_CAPACITY = DefaultCapacityModel.MINIMUM_CAPACITY

    def __init__(self, base_model: CapacityModel = None):

        self.base_model = base_model or StairCapacityModel()

    def capacity(self, edge_or_region):

        if isinstance(edge_or_region, FlowRegion):
            return self._region_capacity(edge_or_region)

        return self.base_model.capacity(edge_or_region)

    def _region_capacity(self, region: FlowRegion):

        if not region.member_edges:
            # No orientation data was derivable for every member (see
            # FlowRegion.member_edges' own comment) -- fall back to the
            # same safe floor DefaultCapacityModel/StairCapacityModel
            # already use rather than fabricate a number.
            return self.MINIMUM_CAPACITY

        edge_capacities = {
            member.edge.id: max(self.base_model.capacity(member.edge), 0)
            for member in region.member_edges
        }

        min_cut = _min_cut_capacity(region.member_edges, edge_capacities)

        return max(min_cut, self.MINIMUM_CAPACITY)
