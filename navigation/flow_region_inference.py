from collections import deque
from typing import Dict, List, Tuple

from navigation.flow_region import FlowRegion, FlowRegionMember
from navigation.node import Node


class _DisjointSet:

    # A minimal union-find over a fixed, known-in-advance set of items
    # (edge ids) -- used to accumulate FlowRegion membership as chains
    # of "this edge and that edge share one admission-control fate" are
    # discovered node by node, without needing to revisit earlier
    # decisions when a later union links two already-grown groups
    # together (the transitive chaining a multi-flight stairwell needs).

    def __init__(self, items):

        self._parent = {item: item for item in items}

    def find(self, item):

        root = item

        while self._parent[root] != root:
            root = self._parent[root]

        # Path compression: flatten every visited node directly onto
        # the root so repeated find() calls during a long chain stay
        # cheap.
        while self._parent[item] != root:

            next_item = self._parent[item]
            self._parent[item] = root
            item = next_item

        return root

    def union(self, item_a, item_b):

        root_a = self.find(item_a)
        root_b = self.find(item_b)

        if root_a != root_b:
            self._parent[root_b] = root_a


class FlowRegionInferencer:

    # Hybrid Flow Regions (Option D), Milestone 1 -- derives FlowRegion
    # membership purely from the already-built NavigationGraph's own
    # topology, with no authoring step and no change to Building/Door/
    # Exit/Staircase. Every existing Building, including every one
    # already saved to disk, gets a real (if trivial) FlowRegion mapping
    # the moment its graph is rebuilt, with zero migration.
    #
    # Heuristic (see the Option D design document, "New Objects" -> 2.
    # FlowRegionInferencer, for the full rationale): walk the graph in
    # the direction of evacuation -- "toward Outside" -- and merge two
    # adjacent edges into the same region whenever the node between them
    # offers exactly one way to continue toward Outside. A node offering
    # more than one way to continue (a genuine fork) stops the merge in
    # both directions, since occupants beyond that point may not share
    # the same downstream path.
    #
    # Edges are inherently bidirectional (see Edge's own docstring), so
    # "toward Outside" isn't given directly -- it is derived once, per
    # graph, from each node's unweighted hop distance to the single
    # shared Outside node, considering only currently traversable edges
    # (a locked Door or a blocked Exit cannot carry any real evacuation
    # flow today, so it cannot be used to justify grouping anything).
    # An edge whose two endpoints are equally distant from Outside (a
    # ring corridor with two equally short ways back) or where either
    # endpoint cannot reach Outside at all has no well-defined direction
    # and is deliberately left ungrouped -- it remains its own trivial,
    # one-edge region rather than risk merging along a false direction.

    @staticmethod
    def infer(graph) -> Dict[str, FlowRegion]:

        if not graph.edges:
            return {}

        distances = FlowRegionInferencer._distances_from_outside(graph)
        out_edges, in_edges, orientation = FlowRegionInferencer._orient_edges(graph, distances)
        groups, merge_marked = FlowRegionInferencer._group_edges(graph, out_edges, in_edges)

        return FlowRegionInferencer._build_regions(graph, groups, merge_marked, orientation)

    # =====================================================

    @staticmethod
    def _distances_from_outside(graph) -> Dict[str, int]:

        adjacency: Dict[str, List[str]] = {}

        for edge in graph.edges:

            if not edge.traversable:
                continue

            adjacency.setdefault(edge.from_node, []).append(edge.to_node)
            adjacency.setdefault(edge.to_node, []).append(edge.from_node)

        if Node.OUTSIDE_NODE_ID not in graph.nodes:
            return {}

        distances = {Node.OUTSIDE_NODE_ID: 0}
        queue = deque([Node.OUTSIDE_NODE_ID])

        while queue:

            current = queue.popleft()

            for neighbor in adjacency.get(current, ()):

                if neighbor not in distances:

                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        return distances

    # =====================================================

    @staticmethod
    def _orient_edges(graph, distances):

        # out_edges[node] -- edges leaving `node` heading strictly
        # closer to Outside ("downstream"). in_edges[node] -- edges
        # arriving at `node` from strictly farther away ("upstream").
        # Outside itself (distance 0, the global minimum) can never
        # appear as a key in out_edges: no neighbor can be strictly
        # closer to Outside than Outside, so no edge can ever be
        # oriented as leaving it. This is what keeps every Exit in the
        # building from being wrongly merged into one region just
        # because they all happen to share the one synthetic Outside
        # node -- the merge rule below only ever fires on a real,
        # single physical node, never on the universal exterior sink.
        #
        # Flow Region Capacity Formula V2 -- `orientation` is the same
        # per-edge (upstream, downstream) pair as out_edges/in_edges
        # already encode in aggregate, just retained per edge id
        # instead of being discarded once grouping is done, so
        # _build_regions() can hand each FlowRegion enough topology to
        # reconstruct its own internal flow network later.
        out_edges: Dict[str, List[str]] = {}
        in_edges: Dict[str, List[str]] = {}
        orientation: Dict[str, Tuple[str, str]] = {}

        for edge in graph.edges:

            if not edge.traversable:
                continue

            dist_from = distances.get(edge.from_node)
            dist_to = distances.get(edge.to_node)

            if dist_from is None or dist_to is None or dist_from == dist_to:
                continue

            if dist_from > dist_to:
                upstream, downstream = edge.from_node, edge.to_node
            else:
                upstream, downstream = edge.to_node, edge.from_node

            out_edges.setdefault(upstream, []).append(edge.id)
            in_edges.setdefault(downstream, []).append(edge.id)
            orientation[edge.id] = (upstream, downstream)

        return out_edges, in_edges, orientation

    # =====================================================

    @staticmethod
    def _group_edges(graph, out_edges, in_edges):

        dsu = _DisjointSet(edge.id for edge in graph.edges)
        merge_marked = set()

        for node_id, outgoing in out_edges.items():

            if len(outgoing) != 1:
                # A genuine fork: more than one way to continue toward
                # Outside from here. No grouping crosses this node in
                # either direction.
                continue

            out_edge_id = outgoing[0]
            incoming = in_edges.get(node_id, [])

            for in_edge_id in incoming:
                dsu.union(out_edge_id, in_edge_id)

            if len(incoming) > 1:

                merge_marked.add(out_edge_id)
                merge_marked.update(incoming)

        groups: Dict[str, List[str]] = {}

        for edge in graph.edges:

            root = dsu.find(edge.id)
            groups.setdefault(root, []).append(edge.id)

        return groups, merge_marked

    # =====================================================

    @staticmethod
    def _build_regions(graph, groups, merge_marked, orientation) -> Dict[str, FlowRegion]:

        edges_by_id = {edge.id: edge for edge in graph.edges}
        mapping: Dict[str, FlowRegion] = {}

        for member_ids in groups.values():

            member_ids = tuple(sorted(member_ids))
            member_edges = [edges_by_id[edge_id] for edge_id in member_ids]

            if len(member_ids) == 1:
                region_kind = FlowRegion.SINGLE
            elif any(edge_id in merge_marked for edge_id in member_ids):
                region_kind = FlowRegion.MERGE
            else:
                region_kind = FlowRegion.CHAIN

            lengths = [
                edge.walking_distance
                for edge in member_edges
                if edge.walking_distance is not None
            ]
            total_length = sum(lengths) if lengths else None

            widths = [
                edge.width
                for edge in member_edges
                if edge.width is not None
            ]
            representative_width = min(widths) if widths else None

            # Flow Region Capacity Formula V2 -- an edge only has an
            # entry in `orientation` if it was actually oriented (see
            # _orient_edges()'s own docstring); an edge that was
            # non-traversable or part of an equal-distance tie
            # contributes nothing here, which for a SINGLE-kind region
            # simply leaves member_edges empty -- exactly right, since
            # that region never reaches a region-level capacity formula
            # in the first place (see FlowRegion.member_edges' own
            # comment).
            region_member_edges = tuple(
                FlowRegionMember(
                    edge=edges_by_id[edge_id],
                    upstream_node_id=orientation[edge_id][0],
                    downstream_node_id=orientation[edge_id][1],
                )
                for edge_id in member_ids
                if edge_id in orientation
            )

            region = FlowRegion(
                id=f"flow-region-{member_ids[0]}",
                edge_ids=member_ids,
                region_kind=region_kind,
                total_length=total_length,
                representative_width=representative_width,
                member_edges=region_member_edges,
            )

            for edge_id in member_ids:
                mapping[edge_id] = region

        return mapping
