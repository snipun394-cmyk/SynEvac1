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


def _collapse_sequential_ties(candidate_ids, member_edges):

    # Admission Control V10 -- Storage-Throughput Separation. Both min-
    # cut identification paths below (the clean residual-graph cut, and
    # the tie-aware minimum-capacity fallback) can legitimately return
    # MULTIPLE tied edges. Two structurally different reasons produce a
    # tie, and only one of them should survive as an active throughput-
    # gating point:
    #
    # - SEQUENTIAL tie: a uniform-width chain (e.g. every flight of one
    #   straight stairwell built to the same code-minimum width) --
    #   every member has the identical capacity purely because ONE
    #   occupant crosses ALL of them, one after another, at the same
    #   narrow width throughout. Gating every one of them would re-admit-
    #   check (and re-touch the region's shared discharge clock) at
    #   every single internal flight boundary -- exactly the self-
    #   throttling bug this whole redesign exists to eliminate, just
    #   reintroduced via a tie instead of via "any first entry". Only
    #   the LAST tied edge in the sequence (the one closest to the
    #   region's own sink) needs to remain a bottleneck -- gating it
    #   once enforces the identical rate as gating all nine would, since
    #   they are already identical in capacity.
    # - PARALLEL tie: a merge's independently narrow converging branches
    #   (e.g. three equally-narrow stairs feeding one wider shared
    #   door) -- these are NOT the same occupant crossing multiple
    #   points in sequence; they are multiple, simultaneously-active,
    #   physically independent streams that happen to tie in capacity.
    #   All of them must remain individually gated, or two of the three
    #   streams would flow completely ungated past the region's own
    #   throughput constraint.
    #
    # Distinguished structurally, not by region_kind (diagnostic-only,
    # see FlowRegion's own docstring): an edge is sequential-redundant
    # if its own downstream_node_id equals some OTHER candidate edge's
    # upstream_node_id -- i.e. it feeds directly into another candidate,
    # meaning the same occupant necessarily crosses both, one right
    # after the other. Iteratively removing such edges leaves exactly
    # the "final" edge of every sequential run, while never touching
    # edges that only share a downstream node (true parallel siblings,
    # never satisfying "my downstream feeds an eligible upstream").

    if len(candidate_ids) <= 1:
        return frozenset(candidate_ids)

    by_id = {member.edge.id: member for member in member_edges}
    candidates = set(candidate_ids)

    # Fixed, never-shrinking index: for each node, every candidate edge
    # whose OWN upstream is that node. Deliberately built once, up
    # front, against the full original candidate set -- checking each
    # candidate's redundancy against a set that shrinks as edges are
    # removed is order-dependent and wrong (removing a middle edge of a
    # run first would spare its own upstream neighbor, which no longer
    # appears to "feed into" anything once its true successor is
    # already gone). Checking against the fixed original set instead
    # means every candidate's redundancy is decided once, independent
    # of iteration order.
    upstream_index = {}

    for candidate_id in candidates:
        upstream_index.setdefault(by_id[candidate_id].upstream_node_id, set()).add(candidate_id)

    redundant = set()

    for candidate_id in candidates:

        downstream = by_id[candidate_id].downstream_node_id
        successors = upstream_index.get(downstream, set()) - {candidate_id}

        if successors:
            # Some OTHER candidate edge starts exactly where this one
            # ends -- the same occupant necessarily crosses both, in
            # sequence, so gating this earlier one is redundant with
            # gating whichever candidate comes after it.
            redundant.add(candidate_id)

    survivors = candidates - redundant

    return frozenset(survivors) if survivors else frozenset(candidate_ids)


def _min_cut_capacity_and_bottleneck_edges(member_edges, edge_capacities):

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
    #
    # Admission Control V10 -- Storage-Throughput Separation. In
    # addition to the min-cut VALUE (used, as before, only as a
    # descriptive/diagnostic number -- see FlowRegionCapacityModelV2.
    # capacity()'s own docstring for why this number is no longer the
    # active admission ceiling under V10), this now also returns the
    # SET of member edge ids that actually constitute the cut -- the
    # coordinator needs to know exactly WHERE the region's true
    # bottleneck is, not merely its capacity, so that the throughput
    # constraint can be applied at that specific crossing (Design
    # Review-mandated correction #2) instead of at every member edge or
    # only at the region's outer boundary. Recovered from the exact
    # same Edmonds-Karp residual graph already computed for the value
    # itself -- no second algorithm, no extra pass beyond one final
    # reachability BFS.

    if not member_edges:
        return 0, frozenset()

    if len(member_edges) == 1:
        only_id = member_edges[0].edge.id
        return edge_capacities[only_id], frozenset({only_id})

    downstream_nodes = {member.downstream_node_id for member in member_edges}
    upstream_nodes = {member.upstream_node_id for member in member_edges}

    # Every region FlowRegionInferencer actually produces has exactly
    # one true sink (its shared discharge node) by construction --
    # grouping only ever stops at a fork or at Outside, never leaves
    # two unconnected discharge points inside one region. If that ever
    # doesn't hold (a future inferencer change, or a hand-built region
    # in a test), fall back to plain minimum-bottleneck across every
    # member -- still a safe, conservative answer, never an
    # overestimate. The bottleneck set in this fallback is every member
    # edge that achieves that same minimum (ties included), since the
    # network structure needed to identify a single true cut isn't
    # available.
    sinks = downstream_nodes - upstream_nodes
    sources = upstream_nodes - downstream_nodes

    if len(sinks) != 1 or not sources:
        min_capacity = min(edge_capacities[member.edge.id] for member in member_edges)
        tied_ids = frozenset(
            member.edge.id for member in member_edges if edge_capacities[member.edge.id] == min_capacity
        )
        return min_capacity, _collapse_sequential_ties(tied_ids, member_edges)

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

    # Min-cut edge identification: standard max-flow/min-cut recovery --
    # one final BFS over the now-saturated residual graph from
    # super_source finds the set of nodes still reachable (S); the
    # min-cut is exactly the set of original member edges whose
    # upstream node is in S and downstream node is not (the arcs
    # crossing from the source side to the sink side). Synthetic
    # super_source->source arcs are never member edges and are
    # naturally excluded, since the loop below only ever inspects real
    # `member_edges`.
    reachable = {super_source}
    queue = deque([super_source])

    while queue:

        current = queue.popleft()

        for neighbor in adjacency[current]:

            if neighbor not in reachable and residual[(current, neighbor)] > 0:

                reachable.add(neighbor)
                queue.append(neighbor)

    cut_ids = frozenset(
        member.edge.id
        for member in member_edges
        if member.upstream_node_id in reachable and member.downstream_node_id not in reachable
    )

    if not cut_ids:
        # Degenerate case: the reachable-set BFS above finds nothing
        # (observed in practice for a uniformly-tied chain, where the
        # very first super_source arc already saturates at max_flow,
        # leaving no real node reachable at all) -- never silently
        # return "nowhere", fall back to the same tie-aware minimum-
        # capacity identification the earlier fallback uses.
        min_capacity = min(edge_capacities[member.edge.id] for member in member_edges)
        cut_ids = frozenset(
            member.edge.id for member in member_edges if edge_capacities[member.edge.id] == min_capacity
        )

    return int(max_flow), _collapse_sequential_ties(cut_ids, member_edges)


class FlowRegionCapacityModelV2(CapacityModel):

    # Same dual-acceptance shape as FlowRegionCapacityModel (V1): an
    # Edge is delegated straight through to base_model unchanged; only
    # a FlowRegion triggers the region-level (here, min-cut) formula.
    #
    # Admission Control V10 -- Storage-Throughput Separation. capacity()
    # itself is UNCHANGED (same public behavior, same return value, same
    # tests) -- kept exactly as validated by the FlowRegionCapacityModel
    # V2 Automatic Calibration Campaign, and still exposed for research/
    # diagnostic use. But the V10 coordinator no longer calls capacity()
    # on a FlowRegion at all for admission purposes: storage is always
    # local (per edge, see simulator/coordinator.py's own V10 comments),
    # so this scalar is no longer the active admission ceiling for a
    # merged region -- it remains available as a descriptive number
    # (informally: "how many people this whole footprint's true
    # bottleneck could sustain if pooled," the same historical
    # Milestone-5-derived quantity this class has always computed) but
    # is not consumed by MultiAgentSimulation's V10 admission path.
    # What V10 DOES consume is the new bottleneck_edges() method below,
    # which reuses the identical min-cut computation for what it is
    # actually reliable at: identifying WHERE a region's true bottleneck
    # sits, not asserting how many people may occupy the whole region.

    MINIMUM_CAPACITY = DefaultCapacityModel.MINIMUM_CAPACITY

    def __init__(self, base_model: CapacityModel = None):

        self.base_model = base_model or StairCapacityModel()

    def capacity(self, edge_or_region):

        if isinstance(edge_or_region, FlowRegion):
            capacity, _ = self._region_capacity_and_bottleneck(edge_or_region)
            return capacity

        return self.base_model.capacity(edge_or_region)

    def bottleneck_edges(self, edge_or_region):

        # Admission Control V10 -- returns the frozenset of member edge
        # ids that constitute this region's own true min-cut bottleneck
        # (possibly more than one, on a genuine tie). A plain Edge
        # (not a FlowRegion) is trivially its own sole bottleneck --
        # this keeps the method total over CapacityModel's own dual-
        # accepted argument shape, so a caller never needs to branch on
        # isinstance(...) itself before calling it.

        if isinstance(edge_or_region, FlowRegion):
            _, bottleneck_ids = self._region_capacity_and_bottleneck(edge_or_region)
            return bottleneck_ids

        return frozenset({edge_or_region.id})

    def _region_capacity_and_bottleneck(self, region: FlowRegion):

        if not region.member_edges:
            # No orientation data was derivable for every member (see
            # FlowRegion.member_edges' own comment) -- fall back to the
            # same safe floor DefaultCapacityModel/StairCapacityModel
            # already use rather than fabricate a number. No bottleneck
            # edge can be identified either, for the same reason.
            return self.MINIMUM_CAPACITY, frozenset()

        edge_capacities = {
            member.edge.id: max(self.base_model.capacity(member.edge), 0)
            for member in region.member_edges
        }

        min_cut, bottleneck_ids = _min_cut_capacity_and_bottleneck_edges(region.member_edges, edge_capacities)

        return max(min_cut, self.MINIMUM_CAPACITY), bottleneck_ids
