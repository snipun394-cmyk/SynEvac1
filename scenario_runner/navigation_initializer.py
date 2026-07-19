from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from scenario import StairAvailability


# Navigation initialization -- architecture doc
# docs/architecture/scenario_runner.md §5 (stair exception) and §9
# ("reset" resolves to: always rebuild every layer fresh, never reuse
# a caller-supplied graph/engine from a prior run). Never alters
# routing algorithms (PathfindingEngine/NavigationGraphGenerator are
# used exactly as they exist, unmodified) -- this module only decides
# *which Building* the graph is built from and, for the one category
# with no Building-side field to set, *which edges* survive
# afterwards.
#
# exclude_stair_edge()/include_stair_edge() below are public --
# architecture doc docs/architecture/scenario_event_execution.md §6/§18:
# the Scenario Event Executor needs the identical stair-availability
# mechanism this module already established for initial (t=0) state,
# applied again, one stair at a time, as a stair-availability event
# fires mid-simulation. _exclude_closed_stair_edges() (below) now calls
# exclude_stair_edge() internally, so build_navigation()'s own behavior
# is completely unchanged.


def build_navigation(scenario, building_copy):

    graph = NavigationGraphGenerator().build(building_copy)

    _exclude_closed_stair_edges(graph, scenario.stair_states)

    engine = PathfindingEngine(graph)

    return graph, engine


def exclude_stair_edge(graph, stair_id: str) -> None:

    # Removes the matching Edge (edge_type == Edge.STAIR) from the
    # already-built graph directly -- NavigationGraph.find_neighbors()
    # iterates graph.edges fresh on every call, so filtering this list
    # once is sufficient for every later traversal query (including
    # PathfindingEngine's) to already respect it. A no-op if the edge
    # is already absent.

    graph.edges = [
        edge
        for edge in graph.edges
        if not (edge.edge_type == Edge.STAIR and edge.id == stair_id)
    ]


def include_stair_edge(graph, building, stair_id: str) -> None:

    # The inverse of exclude_stair_edge() -- re-adds a stair Edge a
    # prior CLOSE (either at t=0 initialization or from an earlier
    # event) removed. NavigationGraphGenerator.build() derives edges
    # purely from Door/Exit/Staircase *connectivity references*, never
    # from any state field (§2 of docs/architecture/scenario_runner.md)
    # -- rebuilding from the same Building therefore always reproduces
    # every stair's Edge, unaffected by any exclusion applied
    # afterwards. Reusing that existing, pure, already-approved builder
    # to obtain a correct replacement Edge is simpler and more honest
    # than caching removed edges speculatively or reimplementing
    # NavigationGraphGenerator's own edge-construction math
    # (walking_distance, endpoint resolution, ...) a second time here.
    # A no-op if the edge is already present.

    if any(edge.edge_type == Edge.STAIR and edge.id == stair_id for edge in graph.edges):
        return

    fresh_graph = NavigationGraphGenerator().build(building)

    replacement_edge = next(
        (
            edge
            for edge in fresh_graph.edges
            if edge.edge_type == Edge.STAIR and edge.id == stair_id
        ),
        None,
    )

    if replacement_edge is not None:
        graph.edges.append(replacement_edge)


def _exclude_closed_stair_edges(graph, stair_states):

    closed_stair_ids = {
        state.stair_id
        for state in stair_states
        if state.availability == StairAvailability.CLOSED
    }

    for stair_id in closed_stair_ids:
        exclude_stair_edge(graph, stair_id)
