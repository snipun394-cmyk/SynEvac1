import unittest

from navigation.edge import Edge
from navigation.graph import NavigationGraph
from navigation.node import Node

from hazard.snapshot import HazardSnapshot

from fire_growth.growth_curve import TSquaredFireGrowthCurve
from fire_growth.spread import FireSpreadModel

from smoke_propagation.graph_distance import shortest_graph_distances


class _OpenDoorRef:
    active = True
    locked = False
    normally_open = True
    door_type = "Standard"


class _ClosedDoorRef:
    active = True
    locked = False
    normally_open = False
    door_type = "Standard"


class _LockedDoorRef:
    active = True
    locked = True
    normally_open = False
    door_type = "Standard"


class _FireDoorRef:
    active = True
    locked = False
    normally_open = False
    door_type = "Fire Door"


def make_node(node_id, floor_id="F0", node_type=Node.ZONE):

    return Node(id=node_id, name=node_id, floor_id=floor_id, node_type=node_type)


def build_resistance_test_graph():

    # ignition --door(5, OPEN)--> near_open
    #     \----door(5, CLOSED)--> near_closed
    #     \----door(5, LOCKED)--> near_locked
    #     \----door(5, "Fire Door")--> near_firedoor
    #     \----stair(5)--> upstairs (floor F1)
    # Every edge shares the identical physical walking_distance (5.0)
    # so any difference in arrival time is purely attributable to
    # FireSpreadModel's own resistance weighting, not to geometry.

    graph = NavigationGraph()

    for node_id, floor_id in (
        ("ignition", "F0"), ("near_open", "F0"), ("near_closed", "F0"),
        ("near_locked", "F0"), ("near_firedoor", "F0"), ("upstairs", "F1"),
    ):
        graph.add_node(make_node(node_id, floor_id=floor_id))

    graph.add_edge(Edge(id="door_open", edge_type=Edge.DOOR, from_node="ignition", to_node="near_open", walking_distance=5.0, reference=_OpenDoorRef()))
    graph.add_edge(Edge(id="door_closed", edge_type=Edge.DOOR, from_node="ignition", to_node="near_closed", walking_distance=5.0, reference=_ClosedDoorRef()))
    graph.add_edge(Edge(id="door_locked", edge_type=Edge.DOOR, from_node="ignition", to_node="near_locked", walking_distance=5.0, reference=_LockedDoorRef()))
    graph.add_edge(Edge(id="door_firedoor", edge_type=Edge.DOOR, from_node="ignition", to_node="near_firedoor", walking_distance=5.0, reference=_FireDoorRef()))
    graph.add_edge(Edge(id="stair1", edge_type=Edge.STAIR, from_node="ignition", to_node="upstairs", walking_distance=5.0))

    return graph


class GraphDistanceEdgeWeightFnTests(unittest.TestCase):

    # Confirms the additive edge_weight_fn parameter is genuinely
    # additive: omitted, behavior is byte-for-byte the original
    # walking_distance-based Dijkstra SmokePropagationModel already
    # depends on.

    def test_default_behavior_is_unchanged_when_edge_weight_fn_omitted(self):

        graph = build_resistance_test_graph()

        distances = shortest_graph_distances(graph, "ignition", (Edge.DOOR, Edge.STAIR))

        self.assertEqual(distances["near_open"], 5.0)
        self.assertEqual(distances["near_closed"], 5.0)
        self.assertEqual(distances["upstairs"], 5.0)

    def test_custom_edge_weight_fn_scales_distance(self):

        graph = build_resistance_test_graph()

        distances = shortest_graph_distances(
            graph, "ignition", (Edge.DOOR, Edge.STAIR),
            edge_weight_fn=lambda edge: edge.walking_distance * 10.0,
        )

        self.assertEqual(distances["near_open"], 50.0)


class FireSpreadModelResistanceOrderingTests(unittest.TestCase):

    # The core Phase 2 realism requirement: open doors let fire spread
    # fastest, closed/locked doors slow it, a "Fire Door" resists it
    # far more, and vertical (stair) spread sits between an ordinary
    # closed door and a Fire Door -- all from identical physical
    # distance, so the ordering is attributable only to the door-type/
    # state-aware weighting.

    def setUp(self):

        self.graph = build_resistance_test_graph()
        self.curve = TSquaredFireGrowthCurve(growth_time=100.0)

    def _model(self):

        return FireSpreadModel(
            self.graph, ignition_node_id="ignition", ignition_time=0.0,
            growth_curve=self.curve, front_speed=1.0,
        )

    def _arrival_time(self, model, node_id):

        # Binary-search-free: step forward until the node first appears
        # in the model's own proposal.
        for candidate_time in range(0, 2000):
            contribution = model.propose(HazardSnapshot(), time=float(candidate_time), dt=0.0)
            if node_id in contribution.node_states:
                return candidate_time
        raise AssertionError(f"{node_id} never appeared within the search window")

    def test_open_door_spreads_faster_than_closed_door(self):

        model = self._model()

        self.assertLess(
            self._arrival_time(model, "near_open"), self._arrival_time(model, "near_closed"),
        )

    def test_locked_door_is_equivalent_to_closed_door(self):

        model = self._model()

        self.assertEqual(
            self._arrival_time(model, "near_locked"), self._arrival_time(model, "near_closed"),
        )

    def test_fire_door_resists_far_more_than_an_ordinary_closed_door(self):

        model = self._model()

        self.assertLess(
            self._arrival_time(model, "near_closed"), self._arrival_time(model, "near_firedoor"),
        )

    def test_stair_spread_sits_between_closed_door_and_fire_door(self):

        model = self._model()

        closed_arrival = self._arrival_time(model, "near_closed")
        stair_arrival = self._arrival_time(model, "upstairs")
        firedoor_arrival = self._arrival_time(model, "near_firedoor")

        self.assertLess(closed_arrival, stair_arrival)
        self.assertLess(stair_arrival, firedoor_arrival)

    def test_vertical_spread_reaches_a_node_on_a_different_floor(self):

        # Direct confirmation of "Vertical spread"/"Stair shafts" as a
        # genuine, working capability, not just a resistance-ordering
        # artifact.
        model = self._model()

        contribution = model.propose(HazardSnapshot(), time=100.0, dt=0.0)

        self.assertIn("upstairs", contribution.node_states)
        self.assertGreater(contribution.node_states["upstairs"].hazard_score, 0.0)


class FireSpreadModelBehaviorTests(unittest.TestCase):

    def setUp(self):

        self.graph = build_resistance_test_graph()
        self.curve = TSquaredFireGrowthCurve(growth_time=100.0)

    def _model(self, **overrides):

        fields = dict(
            graph=self.graph, ignition_node_id="ignition", ignition_time=0.0,
            growth_curve=self.curve, front_speed=1.0,
        )
        fields.update(overrides)

        return FireSpreadModel(**fields)

    def test_never_proposes_for_its_own_ignition_node(self):

        # FireGrowthModel already owns the ignition node's own growth --
        # FireSpreadModel must never also author it.
        model = self._model()

        contribution = model.propose(HazardSnapshot(), time=1000.0, dt=0.0)

        self.assertNotIn("ignition", contribution.node_states)

    def test_no_opinion_before_fire_reaches_a_node(self):

        model = self._model()
        # near_open is at effective distance 5.0 (OPEN_DOOR_RESISTANCE=1.0),
        # front_speed=1.0 -> arrival_time=5.0.
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=4.0)

        self.assertNotIn("near_open", contribution.node_states)

    def test_intensity_grows_via_the_growth_curve_after_arrival(self):

        model = self._model()

        contribution = model.propose(HazardSnapshot(), time=5.0, dt=50.0)  # arrival at t=5, elapsed=50

        self.assertAlmostEqual(contribution.node_states["near_open"].hazard_score, 0.25)

    def test_negative_or_zero_front_speed_is_rejected(self):

        with self.assertRaises(ValueError):
            self._model(front_speed=0.0)

    def test_edge_types_outside_door_and_stair_are_never_propagated_through(self):

        graph = build_resistance_test_graph()
        graph.add_node(make_node(Node.OUTSIDE_NODE_ID, floor_id="", node_type=Node.OUTSIDE))
        graph.add_edge(
            Edge(id="exit1", edge_type=Edge.EXIT, from_node="near_open", to_node=Node.OUTSIDE_NODE_ID, walking_distance=1.0),
        )
        model = FireSpreadModel(graph, ignition_node_id="ignition", ignition_time=0.0, growth_curve=self.curve, front_speed=1.0)

        contribution = model.propose(HazardSnapshot(), time=1000.0, dt=0.0)

        self.assertNotIn(Node.OUTSIDE_NODE_ID, contribution.node_states)


if __name__ == "__main__":
    unittest.main()
