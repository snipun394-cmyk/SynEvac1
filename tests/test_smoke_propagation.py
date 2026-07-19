import unittest

from navigation.edge import Edge
from navigation.graph import NavigationGraph
from navigation.node import Node

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.engine import HazardEvolutionEngine
from hazard_evolution.provider import EvolutionBackedHazardProvider
from hazard_evolution.source import HazardSource

from smoke_propagation.graph_distance import shortest_graph_distances
from smoke_propagation.growth_curve import SmokeGrowthCurve, TSquaredSmokeGrowthCurve
from smoke_propagation.model import SmokePropagationModel
from smoke_propagation.visibility import (
    CLEAR_VISIBILITY_M,
    MIN_VISIBILITY_M,
    visibility_from_smoke_level,
)


class _LockedDoorRef:
    active = True
    locked = True


class _BlockedExitRef:
    is_blocked = True


def make_node(node_id, floor_id="F0", node_type=Node.ZONE):

    return Node(id=node_id, name=node_id, floor_id=floor_id, node_type=node_type)


def build_multi_floor_graph():

    # ignition --door(5)--> near --door(10)--> far --door(2, locked)--> locked_room
    #     \--stair(8)--> upstairs (floor F1)
    #
    # near --exit(3)--> outside (Exit is never a propagated edge type)
    graph = NavigationGraph()

    graph.add_node(make_node("ignition"))
    graph.add_node(make_node("near"))
    graph.add_node(make_node("far"))
    graph.add_node(make_node("locked_room"))
    graph.add_node(make_node("upstairs", floor_id="F1"))
    graph.add_node(make_node(Node.OUTSIDE_NODE_ID, floor_id="", node_type=Node.OUTSIDE))

    graph.add_edge(Edge(id="door1", edge_type=Edge.DOOR, from_node="ignition", to_node="near", walking_distance=5.0))
    graph.add_edge(Edge(id="door2", edge_type=Edge.DOOR, from_node="near", to_node="far", walking_distance=10.0))
    graph.add_edge(
        Edge(
            id="locked_door", edge_type=Edge.DOOR, from_node="far", to_node="locked_room",
            walking_distance=2.0, reference=_LockedDoorRef(),
        )
    )
    graph.add_edge(Edge(id="stair1", edge_type=Edge.STAIR, from_node="ignition", to_node="upstairs", walking_distance=8.0))
    graph.add_edge(
        Edge(
            id="exit1", edge_type=Edge.EXIT, from_node="near", to_node=Node.OUTSIDE_NODE_ID,
            walking_distance=3.0,
        )
    )

    return graph


class TSquaredSmokeGrowthCurveTests(unittest.TestCase):

    def test_zero_elapsed_time_is_zero_intensity(self):

        curve = TSquaredSmokeGrowthCurve(growth_time=100.0)
        self.assertEqual(curve.intensity_at(0.0), 0.0)

    def test_intensity_grows_quadratically(self):

        curve = TSquaredSmokeGrowthCurve(growth_time=100.0)

        self.assertAlmostEqual(curve.intensity_at(50.0), 0.25)
        self.assertAlmostEqual(curve.intensity_at(100.0), 1.0)

    def test_intensity_saturates_at_one_past_growth_time(self):

        curve = TSquaredSmokeGrowthCurve(growth_time=100.0)
        self.assertEqual(curve.intensity_at(500.0), 1.0)


class VisibilityTests(unittest.TestCase):

    def test_zero_smoke_is_clear_visibility(self):
        self.assertEqual(visibility_from_smoke_level(0.0), CLEAR_VISIBILITY_M)

    def test_full_smoke_is_minimum_visibility(self):
        self.assertEqual(visibility_from_smoke_level(1.0), MIN_VISIBILITY_M)

    def test_midpoint_smoke_is_midpoint_visibility(self):
        expected = (CLEAR_VISIBILITY_M + MIN_VISIBILITY_M) / 2.0
        self.assertAlmostEqual(visibility_from_smoke_level(0.5), expected)

    def test_out_of_range_inputs_are_clamped(self):
        self.assertEqual(visibility_from_smoke_level(-1.0), CLEAR_VISIBILITY_M)
        self.assertEqual(visibility_from_smoke_level(2.0), MIN_VISIBILITY_M)


class ShortestGraphDistancesTests(unittest.TestCase):

    def setUp(self):
        self.graph = build_multi_floor_graph()

    def test_source_node_is_distance_zero(self):
        distances = shortest_graph_distances(self.graph, "ignition", (Edge.DOOR, Edge.STAIR))
        self.assertEqual(distances["ignition"], 0.0)

    def test_distances_accumulate_along_door_chain(self):
        distances = shortest_graph_distances(self.graph, "ignition", (Edge.DOOR, Edge.STAIR))
        self.assertEqual(distances["near"], 5.0)
        self.assertEqual(distances["far"], 15.0)

    def test_stair_edges_cross_floors(self):
        distances = shortest_graph_distances(self.graph, "ignition", (Edge.DOOR, Edge.STAIR))
        self.assertEqual(distances["upstairs"], 8.0)

    def test_locked_door_does_not_block_propagation(self):
        distances = shortest_graph_distances(self.graph, "ignition", (Edge.DOOR, Edge.STAIR))

        locked_edge = next(e for e in self.graph.edges if e.id == "locked_door")
        self.assertFalse(locked_edge.traversable)

        self.assertEqual(distances["locked_room"], 17.0)

    def test_edge_types_not_requested_are_excluded(self):
        distances = shortest_graph_distances(self.graph, "ignition", (Edge.DOOR, Edge.STAIR))
        self.assertNotIn(Node.OUTSIDE_NODE_ID, distances)

    def test_excluding_stair_type_leaves_upstairs_unreachable(self):
        distances = shortest_graph_distances(self.graph, "ignition", (Edge.DOOR,))
        self.assertNotIn("upstairs", distances)

    def test_unknown_source_node_returns_empty(self):
        distances = shortest_graph_distances(self.graph, "does_not_exist", (Edge.DOOR, Edge.STAIR))
        self.assertEqual(distances, {})


class SmokePropagationModelTests(unittest.TestCase):

    def setUp(self):
        self.graph = build_multi_floor_graph()
        self.curve = TSquaredSmokeGrowthCurve(growth_time=100.0)

    def _model(self, ignition_time=0.0):
        return SmokePropagationModel(
            self.graph, ignition_node_id="ignition", ignition_time=ignition_time,
            growth_curve=self.curve, front_speed=1.0,
        )

    def test_no_opinion_before_smoke_reaches_a_node(self):

        model = self._model()
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=4.0)  # step ends at t=4, "near" arrives at t=5

        self.assertIn("ignition", contribution.node_states)
        self.assertNotIn("near", contribution.node_states)
        self.assertNotIn("far", contribution.node_states)
        self.assertNotIn("upstairs", contribution.node_states)

    def test_propagates_through_stairs_across_floors(self):

        model = self._model()
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=8.0)  # step ends at t=8

        self.assertIn("upstairs", contribution.node_states)

    def test_never_propagates_through_exit_edges(self):

        model = self._model()
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=1000.0)

        self.assertNotIn(Node.OUTSIDE_NODE_ID, contribution.node_states)

    def test_propagates_through_a_locked_door(self):

        model = self._model()
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=17.0)  # locked_room arrives at t=17

        self.assertIn("locked_room", contribution.node_states)

    def test_smoke_level_matches_growth_curve_at_elapsed_time_since_arrival(self):

        model = self._model()
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=15.0)  # "near" arrived at t=5, elapsed=10

        self.assertAlmostEqual(
            contribution.node_state("near").smoke_level,
            self.curve.intensity_at(10.0),
        )

    def test_hazard_score_mirrors_smoke_level(self):

        model = self._model()
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=20.0)

        state = contribution.node_state("far")
        self.assertEqual(state.hazard_score, state.smoke_level)

    def test_visibility_is_derived_from_smoke_level(self):

        model = self._model()
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=20.0)

        state = contribution.node_state("far")
        self.assertEqual(state.visibility, visibility_from_smoke_level(state.smoke_level))

    def test_produces_no_edge_states_ever(self):

        model = self._model()
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=1000.0)

        self.assertEqual(len(contribution.edge_states), 0)

    def test_does_not_read_previous_snapshot(self):

        model = self._model()

        unrelated_snapshot = HazardSnapshot(
            node_states={"near": HazardNodeState(hazard_score=0.99)},
        )

        with_history = model.propose(unrelated_snapshot, time=0.0, dt=10.0)
        without_history = model.propose(HazardSnapshot(), time=0.0, dt=10.0)

        self.assertEqual(with_history, without_history)

    def test_default_growth_curve_and_front_speed_are_used_when_not_supplied(self):

        model = SmokePropagationModel(self.graph, ignition_node_id="ignition", ignition_time=0.0)

        self.assertIsInstance(model.growth_curve, TSquaredSmokeGrowthCurve)
        self.assertEqual(model.front_speed, SmokePropagationModel.DEFAULT_FRONT_SPEED_M_PER_S)

    def test_front_speed_must_be_positive(self):

        with self.assertRaises(ValueError):
            SmokePropagationModel(self.graph, ignition_node_id="ignition", ignition_time=0.0, front_speed=0.0)

    def test_repeated_stepping_matches_a_single_larger_step_at_the_same_absolute_time(self):

        model_a = self._model()
        model_b = self._model()

        engine_a = HazardEvolutionEngine(sources=[model_a])
        snapshot_a = HazardSnapshot(timestamp=0.0)
        for step in range(5):
            snapshot_a = engine_a.evolve(snapshot_a, time=step * 4.0, dt=4.0)

        engine_b = HazardEvolutionEngine(sources=[model_b])
        snapshot_b = engine_b.evolve(HazardSnapshot(timestamp=0.0), time=0.0, dt=20.0)

        self.assertAlmostEqual(
            snapshot_a.node_state("far").hazard_score,
            snapshot_b.node_state("far").hazard_score,
        )


class SmokeDilutionTests(unittest.TestCase):

    # Phase 3's own "smoke dilution" requirement: a node further from
    # the source must reach a LOWER peak concentration than one closer
    # in, not merely the identical peak reached later. dilution_half_
    # distance defaults to None (off) so every test above this class
    # -- and every existing caller -- is completely unaffected; these
    # tests are the one place it's exercised.

    def setUp(self):
        self.graph = build_multi_floor_graph()
        self.curve = TSquaredSmokeGrowthCurve(growth_time=1.0)  # saturates almost immediately

    def _model(self, dilution_half_distance):
        return SmokePropagationModel(
            self.graph, ignition_node_id="ignition", ignition_time=0.0,
            growth_curve=self.curve, front_speed=1.0,
            dilution_half_distance=dilution_half_distance,
        )

    def test_dilution_off_by_default(self):

        model = SmokePropagationModel(
            self.graph, ignition_node_id="ignition", ignition_time=0.0,
            growth_curve=self.curve, front_speed=1.0,
        )
        self.assertIsNone(model.dilution_half_distance)

        contribution = model.propose(HazardSnapshot(), time=1000.0, dt=0.0)

        # "near" (distance 5) and "far" (distance 15) both fully
        # saturate identically without dilution -- the pre-existing,
        # unchanged behavior.
        self.assertAlmostEqual(contribution.node_state("near").smoke_level, 1.0)
        self.assertAlmostEqual(contribution.node_state("far").smoke_level, 1.0)

    def test_farther_node_reaches_a_lower_peak_than_a_nearer_node(self):

        model = self._model(dilution_half_distance=10.0)

        contribution = model.propose(HazardSnapshot(), time=1000.0, dt=0.0)

        near_level = contribution.node_state("near").smoke_level  # distance 5
        far_level = contribution.node_state("far").smoke_level    # distance 15

        self.assertGreater(near_level, far_level)
        self.assertLess(far_level, 1.0)  # genuinely diluted, not merely delayed

    def test_ignition_node_itself_is_never_diluted(self):

        model = self._model(dilution_half_distance=10.0)

        contribution = model.propose(HazardSnapshot(), time=1000.0, dt=0.0)

        self.assertAlmostEqual(contribution.node_state("ignition").smoke_level, 1.0)

    def test_dilution_at_exactly_one_half_distance_halves_the_peak(self):

        model = self._model(dilution_half_distance=5.0)  # "near" is at distance 5

        contribution = model.propose(HazardSnapshot(), time=1000.0, dt=0.0)

        self.assertAlmostEqual(contribution.node_state("near").smoke_level, 0.5)

    def test_visibility_reflects_the_diluted_smoke_level_not_the_undiluted_one(self):

        model = self._model(dilution_half_distance=5.0)

        contribution = model.propose(HazardSnapshot(), time=1000.0, dt=0.0)
        state = contribution.node_state("near")

        self.assertEqual(state.visibility, visibility_from_smoke_level(state.smoke_level))
        self.assertGreater(state.visibility, MIN_VISIBILITY_M)

    def test_zero_or_negative_dilution_half_distance_is_rejected(self):

        with self.assertRaises(ValueError):
            self._model(dilution_half_distance=0.0)


class SmokePropagationModelNeverTouchesDownstreamLayersTests(unittest.TestCase):

    def test_module_imports_nothing_from_simulation_behavior_pathfinding_or_the_building_model(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "smoke_propagation"

        forbidden = r"^\s*(from|import)\s+(simulator|behavior|pathfinding|models|designer)\b"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{path.name} imports a downstream/engineering layer directly -- "
                f"SmokePropagationModel must only ever produce HazardContribution objects",
            )

            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+navigation\.cost\b", text, re.MULTILINE),
                f"{path.name} imports navigation.cost -- SmokePropagationModel must "
                f"only ever read static graph topology, never a CostModel",
            )


class MultipleIgnitionPointsComposeWithoutNewCodeTests(unittest.TestCase):

    def test_two_smoke_sources_evolve_independently_in_the_same_engine(self):

        graph = build_multi_floor_graph()
        curve = TSquaredSmokeGrowthCurve(growth_time=100.0)

        source_a = SmokePropagationModel(graph, "ignition", ignition_time=0.0, growth_curve=curve, front_speed=1.0)
        source_b = SmokePropagationModel(graph, "far", ignition_time=0.0, growth_curve=curve, front_speed=1.0)

        engine = HazardEvolutionEngine(sources=[source_a, source_b])
        result = engine.evolve(HazardSnapshot(timestamp=0.0), time=0.0, dt=20.0)

        # "far" is reached by source_a at distance 15 (elapsed 5) and by
        # source_b at distance 0 (elapsed 20) -- worst-case-wins merge
        # must pick source_b's higher value.
        self.assertAlmostEqual(
            result.node_state("far").hazard_score,
            curve.intensity_at(20.0),
        )


class SmokePropagationModelViaProviderIntegrationTests(unittest.TestCase):

    def test_evolution_backed_provider_surfaces_smoke_propagation_over_time(self):

        graph = build_multi_floor_graph()
        curve = TSquaredSmokeGrowthCurve(growth_time=200.0)
        model = SmokePropagationModel(graph, "ignition", ignition_time=0.0, growth_curve=curve, front_speed=1.0)

        engine = HazardEvolutionEngine(sources=[model])
        provider = EvolutionBackedHazardProvider(engine, HazardSnapshot(timestamp=0.0), dt=10.0)

        early = provider.snapshot_at(20.0).node_state("near").smoke_level
        later = provider.snapshot_at(150.0).node_state("near").smoke_level

        self.assertLess(early, later)


class CustomHazardSourceCanReplaceSmokePropagationModelTests(unittest.TestCase):

    def test_engine_accepts_any_hazard_source_in_place_of_smoke_propagation_model(self):

        class StubCFDSmokeModel(HazardSource):
            def propose(self, previous_snapshot, time, dt):
                return HazardContribution(
                    node_states={"near": HazardNodeState(smoke_level=0.5, visibility=10.0, hazard_score=0.5)},
                )

        engine = HazardEvolutionEngine(sources=[StubCFDSmokeModel()])
        result = engine.evolve(HazardSnapshot(timestamp=0.0), time=0.0, dt=1.0)

        self.assertEqual(result.node_state("near").smoke_level, 0.5)


if __name__ == "__main__":
    unittest.main()
