import unittest

from dataclasses import FrozenInstanceError

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.cost import DefaultCostModel
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.capacity import DefaultCapacityModel
from simulator.coordinator import MultiAgentSimulation

from behavior.context import DecisionContext
from behavior.profile import BehaviorProfile

from hazard.capacity_model import HazardAwareCapacityModel
from hazard.cost_model import HazardAwareCostModel
from hazard.edge_state import HazardEdgeState
from hazard.node_state import HazardNodeState
from hazard.severity import HazardSeverity
from hazard.snapshot import HazardSnapshot

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.engine import HazardEvolutionEngine
from hazard_evolution.merge_strategy import DefaultHazardMergeStrategy, HazardMergeStrategy
from hazard_evolution.provider import EvolutionBackedHazardProvider
from hazard_evolution.source import HazardSource, NullHazardSource


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def build_two_zone_building(door_width=4.0):

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    room = make_zone("Room", x=0.0, y=0.0)
    corridor = make_zone("Corridor", x=10.0, y=0.0)

    floor.add_zone(room)
    floor.add_zone(corridor)

    door = Door(
        name="D1", zone_a_id=room.id, zone_b_id=corridor.id,
        floor_id=floor.id, width=door_width,
    )
    floor.add_door(door)

    exit_obj = Exit(name="Ex", zone_id=corridor.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, floor, room, corridor, door, exit_obj, engine


class FixedContributionSource(HazardSource):

    # Test double -- always proposes the same, pre-baked contribution
    # regardless of previous_snapshot/time/dt. Used to make engine
    # behavior deterministic and inspectable.

    def __init__(self, contribution):
        self.contribution = contribution
        self.calls = []

    def propose(self, previous_snapshot, time, dt):
        self.calls.append((previous_snapshot, time, dt))
        return self.contribution


class LinearGrowthFireSource(HazardSource):

    # A stand-in for a *future* Fire Growth producer -- deliberately
    # not real fire physics, just enough temporal behavior (reads the
    # previous snapshot, proposes a value that increases with time) to
    # prove HazardEvolutionEngine can host a source that evolves state
    # over repeated steps without any engine change.

    def __init__(self, node_id, growth_per_second):
        self.node_id = node_id
        self.growth_per_second = growth_per_second

    def propose(self, previous_snapshot, time, dt):

        previous_score = previous_snapshot.node_state(self.node_id).hazard_score
        next_score = min(previous_score + self.growth_per_second * dt, 1.0)

        return HazardContribution(
            node_states={self.node_id: HazardNodeState(hazard_score=next_score)},
        )


class SentinelMergeStrategy(HazardMergeStrategy):

    # Proves the engine delegates merging *exclusively* to the injected
    # strategy -- returns an obviously-not-computed-by-the-engine
    # sentinel value regardless of input, so if this value shows up in
    # evolve()'s output, the engine must have gone through this
    # strategy rather than any hardcoded logic of its own.

    SENTINEL_SCORE = 0.4242

    def merge_node_states(self, states):
        return HazardNodeState(hazard_score=self.SENTINEL_SCORE)

    def merge_edge_states(self, states):
        return HazardEdgeState(cost_penalty=self.SENTINEL_SCORE)


class HazardContributionTests(unittest.TestCase):

    def test_default_is_empty(self):

        contribution = HazardContribution()

        self.assertIsNone(contribution.node_state("n1"))
        self.assertIsNone(contribution.edge_state("e1"))

    def test_missing_id_returns_none_not_a_clear_default(self):

        # Unlike HazardSnapshot (a total function defaulting absence to
        # "clear"), absence on a HazardContribution must mean "no
        # opinion" -- distinguishable from an explicit clear reading.
        contribution = HazardContribution(node_states={"n1": HazardNodeState(hazard_score=0.0)})

        self.assertEqual(contribution.node_state("n1"), HazardNodeState(hazard_score=0.0))
        self.assertIsNone(contribution.node_state("n2"))

    def test_is_frozen(self):

        contribution = HazardContribution()

        with self.assertRaises(FrozenInstanceError):
            contribution.node_states = {}

    def test_mappings_are_read_only(self):

        contribution = HazardContribution(node_states={"n1": HazardNodeState()})

        with self.assertRaises(TypeError):
            contribution.node_states["n2"] = HazardNodeState()


class NullHazardSourceTests(unittest.TestCase):

    def test_never_proposes_anything(self):

        source = NullHazardSource()
        contribution = source.propose(HazardSnapshot(), time=0.0, dt=1.0)

        self.assertEqual(contribution, HazardContribution())


class DefaultHazardMergeStrategyTests(unittest.TestCase):

    def setUp(self):
        self.strategy = DefaultHazardMergeStrategy()

    def test_node_merge_takes_the_worst_case_per_field(self):

        merged = self.strategy.merge_node_states([
            HazardNodeState(hazard_score=0.3, visibility=20.0, temperature=30.0, smoke_level=0.1),
            HazardNodeState(hazard_score=0.7, visibility=5.0, temperature=80.0, smoke_level=0.6),
        ])

        self.assertEqual(merged.hazard_score, 0.7)
        self.assertEqual(merged.visibility, 5.0)
        self.assertEqual(merged.temperature, 80.0)
        self.assertEqual(merged.smoke_level, 0.6)

    def test_node_merge_ignores_none_fields_rather_than_propagating_them(self):

        merged = self.strategy.merge_node_states([
            HazardNodeState(hazard_score=0.2, visibility=None),
            HazardNodeState(hazard_score=0.5, visibility=10.0),
        ])

        self.assertEqual(merged.visibility, 10.0)

    def test_edge_merge_any_blocking_source_wins(self):

        merged = self.strategy.merge_edge_states([
            HazardEdgeState(traversable=True),
            HazardEdgeState(traversable=False),
        ])

        self.assertFalse(merged.traversable)

    def test_edge_merge_no_opinion_stays_none(self):

        merged = self.strategy.merge_edge_states([
            HazardEdgeState(), HazardEdgeState(),
        ])

        self.assertIsNone(merged.traversable)

    def test_edge_merge_capacity_modifier_takes_the_most_restrictive(self):

        merged = self.strategy.merge_edge_states([
            HazardEdgeState(capacity_modifier=0.8),
            HazardEdgeState(capacity_modifier=0.3),
        ])

        self.assertEqual(merged.capacity_modifier, 0.3)

    def test_edge_merge_cost_penalties_stack(self):

        merged = self.strategy.merge_edge_states([
            HazardEdgeState(cost_penalty=2.0),
            HazardEdgeState(cost_penalty=3.0),
        ])

        self.assertEqual(merged.cost_penalty, 5.0)


class HazardEvolutionEngineTests(unittest.TestCase):

    def test_no_sources_carries_forward_unchanged_except_timestamp(self):

        initial = HazardSnapshot(
            timestamp=0.0, node_states={"n1": HazardNodeState(hazard_score=0.5)},
        )
        engine = HazardEvolutionEngine(sources=[])

        result = engine.evolve(initial, time=0.0, dt=10.0)

        self.assertEqual(result.timestamp, 10.0)
        self.assertEqual(result.node_state("n1"), HazardNodeState(hazard_score=0.5))
        self.assertNotEqual(result.snapshot_id, initial.snapshot_id)

    def test_original_snapshot_is_never_mutated(self):

        initial = HazardSnapshot(node_states={"n1": HazardNodeState(hazard_score=0.5)})
        engine = HazardEvolutionEngine(
            sources=[LinearGrowthFireSource("n1", growth_per_second=0.1)],
        )

        engine.evolve(initial, time=0.0, dt=1.0)

        self.assertEqual(initial.node_state("n1").hazard_score, 0.5)

    def test_source_receives_previous_snapshot_time_and_dt(self):

        source = FixedContributionSource(HazardContribution())
        engine = HazardEvolutionEngine(sources=[source])
        snapshot = HazardSnapshot()

        engine.evolve(snapshot, time=5.0, dt=2.0)

        self.assertEqual(source.calls, [(snapshot, 5.0, 2.0)])

    def test_untouched_ids_carry_forward_previous_values(self):

        initial = HazardSnapshot(
            node_states={
                "untouched": HazardNodeState(hazard_score=0.6),
                "touched": HazardNodeState(hazard_score=0.1),
            },
        )
        source = FixedContributionSource(
            HazardContribution(node_states={"touched": HazardNodeState(hazard_score=0.9)}),
        )
        engine = HazardEvolutionEngine(sources=[source])

        result = engine.evolve(initial, time=0.0, dt=1.0)

        self.assertEqual(result.node_state("untouched").hazard_score, 0.6)
        self.assertEqual(result.node_state("touched").hazard_score, 0.9)

    def test_two_non_conflicting_sources_each_apply_independently(self):

        source_a = FixedContributionSource(
            HazardContribution(node_states={"n1": HazardNodeState(hazard_score=0.4)}),
        )
        source_b = FixedContributionSource(
            HazardContribution(node_states={"n2": HazardNodeState(hazard_score=0.7)}),
        )
        engine = HazardEvolutionEngine(sources=[source_a, source_b])

        result = engine.evolve(HazardSnapshot(), time=0.0, dt=1.0)

        self.assertEqual(result.node_state("n1").hazard_score, 0.4)
        self.assertEqual(result.node_state("n2").hazard_score, 0.7)

    def test_engine_delegates_conflicting_ids_exclusively_to_the_merge_strategy(self):

        source_a = FixedContributionSource(
            HazardContribution(node_states={"n1": HazardNodeState(hazard_score=0.1)}),
        )
        source_b = FixedContributionSource(
            HazardContribution(node_states={"n1": HazardNodeState(hazard_score=0.9)}),
        )
        engine = HazardEvolutionEngine(
            sources=[source_a, source_b], merge_strategy=SentinelMergeStrategy(),
        )

        result = engine.evolve(HazardSnapshot(), time=0.0, dt=1.0)

        # Neither 0.1 nor 0.9 (what a hardcoded worst-case-wins engine
        # would produce) -- only the sentinel the injected strategy
        # always returns, proving the engine has no merge logic of its
        # own baked in.
        self.assertEqual(result.node_state("n1").hazard_score, SentinelMergeStrategy.SENTINEL_SCORE)

    def test_replacing_the_merge_strategy_changes_conflict_resolution_only(self):

        source_a = FixedContributionSource(
            HazardContribution(node_states={"n1": HazardNodeState(hazard_score=0.1)}),
        )
        source_b = FixedContributionSource(
            HazardContribution(node_states={"n1": HazardNodeState(hazard_score=0.9)}),
        )

        default_engine = HazardEvolutionEngine(sources=[source_a, source_b])
        sentinel_engine = HazardEvolutionEngine(
            sources=[source_a, source_b], merge_strategy=SentinelMergeStrategy(),
        )

        default_result = default_engine.evolve(HazardSnapshot(), time=0.0, dt=1.0)
        sentinel_result = sentinel_engine.evolve(HazardSnapshot(), time=0.0, dt=1.0)

        self.assertEqual(default_result.node_state("n1").hazard_score, 0.9)  # worst-case-wins
        self.assertEqual(sentinel_result.node_state("n1").hazard_score, SentinelMergeStrategy.SENTINEL_SCORE)
        self.assertNotEqual(default_result.node_state("n1"), sentinel_result.node_state("n1"))

    def test_default_merge_strategy_is_worst_case_wins_but_swappable(self):

        # Documents (via test, not comment alone) that worst-case-wins
        # is a *default*, not a requirement -- the engine's own
        # constructor accepts any HazardMergeStrategy.
        engine = HazardEvolutionEngine()
        self.assertIsInstance(engine.merge_strategy, DefaultHazardMergeStrategy)

        custom_engine = HazardEvolutionEngine(merge_strategy=SentinelMergeStrategy())
        self.assertIsInstance(custom_engine.merge_strategy, SentinelMergeStrategy)

    def test_repeated_evolution_accumulates_growth_from_a_source(self):

        # Simulates several steps of a future Fire-Growth-shaped source
        # without implementing any real physics -- proves the engine
        # supports repeated evolve() calls chaining correctly.
        source = LinearGrowthFireSource("n1", growth_per_second=0.1)
        engine = HazardEvolutionEngine(sources=[source])

        snapshot = HazardSnapshot(node_states={"n1": HazardNodeState(hazard_score=0.0)})

        for step in range(5):
            snapshot = engine.evolve(snapshot, time=step * 1.0, dt=1.0)

        self.assertAlmostEqual(snapshot.node_state("n1").hazard_score, 0.5)
        self.assertEqual(snapshot.timestamp, 5.0)


class EvolutionBackedHazardProviderTests(unittest.TestCase):

    def test_snapshot_at_initial_time_returns_the_initial_snapshot(self):

        initial = HazardSnapshot(timestamp=0.0)
        provider = EvolutionBackedHazardProvider(
            HazardEvolutionEngine(), initial, dt=1.0,
        )

        self.assertIs(provider.snapshot_at(0.0), initial)

    def test_snapshot_at_future_time_advances_through_the_engine(self):

        source = LinearGrowthFireSource("n1", growth_per_second=0.1)
        engine = HazardEvolutionEngine(sources=[source])
        initial = HazardSnapshot(timestamp=0.0, node_states={"n1": HazardNodeState(hazard_score=0.0)})

        provider = EvolutionBackedHazardProvider(engine, initial, dt=1.0)

        snapshot_at_5 = provider.snapshot_at(5.0)

        self.assertEqual(snapshot_at_5.timestamp, 5.0)
        self.assertAlmostEqual(snapshot_at_5.node_state("n1").hazard_score, 0.5)

    def test_repeated_query_for_the_same_time_returns_the_cached_instance(self):

        provider = EvolutionBackedHazardProvider(
            HazardEvolutionEngine(), HazardSnapshot(timestamp=0.0), dt=1.0,
        )

        first = provider.snapshot_at(3.0)
        second = provider.snapshot_at(3.0)

        self.assertIs(first, second)

    def test_stepping_lands_exactly_on_a_non_multiple_of_dt(self):

        provider = EvolutionBackedHazardProvider(
            HazardEvolutionEngine(), HazardSnapshot(timestamp=0.0), dt=2.0,
        )

        snapshot = provider.snapshot_at(2.5)

        self.assertEqual(snapshot.timestamp, 2.5)

    def test_requesting_an_unvisited_earlier_time_raises(self):

        provider = EvolutionBackedHazardProvider(
            HazardEvolutionEngine(), HazardSnapshot(timestamp=0.0), dt=1.0,
        )

        provider.snapshot_at(10.0)

        with self.assertRaises(ValueError):
            provider.snapshot_at(4.5)

    def test_previously_visited_earlier_time_is_still_retrievable(self):

        provider = EvolutionBackedHazardProvider(
            HazardEvolutionEngine(), HazardSnapshot(timestamp=0.0), dt=1.0,
        )

        provider.snapshot_at(3.0)
        replay = provider.snapshot_at(1.0)

        self.assertEqual(replay.timestamp, 1.0)


class EvolutionBackedProviderV1CompatibilityTests(unittest.TestCase):

    # Confirms every existing V1 hazard consumer (HazardAwareCostModel,
    # HazardAwareCapacityModel, DecisionContext) operates completely
    # unchanged when handed a snapshot sourced from an evolving
    # provider instead of ManualHazardProvider's fixed one -- none of
    # them can tell the difference, by construction (they only ever
    # see a HazardSnapshot).

    def setUp(self):

        (
            self.building, self.floor, self.room, self.corridor,
            self.door, self.exit_obj, self.engine,
        ) = build_two_zone_building(door_width=4.0)

        self.graph_edge = self.engine.graph.find_neighbors(
            self.engine.graph.find_node(self.room.id)
        )[0][1]

    def _evolving_snapshot_at(self, time):

        source = FixedContributionSource(
            HazardContribution(
                node_states={self.room.id: HazardNodeState(hazard_score=0.9)},
                edge_states={self.graph_edge.id: HazardEdgeState(cost_penalty=50.0, capacity_modifier=0.5)},
            ),
        )
        evolution_engine = HazardEvolutionEngine(sources=[source])
        provider = EvolutionBackedHazardProvider(
            evolution_engine, HazardSnapshot(timestamp=0.0), dt=1.0,
        )

        return provider.snapshot_at(time)

    def test_cost_model_consumes_an_evolved_snapshot_unchanged(self):

        snapshot = self._evolving_snapshot_at(1.0)
        hazard_cost_model = HazardAwareCostModel(DefaultCostModel(), snapshot)

        cost = hazard_cost_model.cost(self.graph_edge)

        self.assertEqual(cost, DefaultCostModel().cost(self.graph_edge) + 50.0)

    def test_capacity_model_consumes_an_evolved_snapshot_unchanged(self):

        snapshot = self._evolving_snapshot_at(1.0)
        hazard_capacity_model = HazardAwareCapacityModel(DefaultCapacityModel(), snapshot)

        base_capacity = DefaultCapacityModel().capacity(self.graph_edge)
        self.assertEqual(hazard_capacity_model.capacity(self.graph_edge), int(base_capacity * 0.5))

    def test_decision_context_consumes_an_evolved_snapshot_unchanged(self):

        snapshot = self._evolving_snapshot_at(1.0)

        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="p1"),
            start_id=self.room.id, hazard_snapshot=snapshot,
        )

        self.assertEqual(
            context.hazard_snapshot.node_state(self.room.id).severity,
            HazardSeverity.CRITICAL,
        )

    def test_full_pathfinding_and_simulation_run_against_an_evolved_snapshot(self):

        snapshot = self._evolving_snapshot_at(1.0)
        hazard_cost_model = HazardAwareCostModel(DefaultCostModel(), snapshot)
        hazard_capacity_model = HazardAwareCapacityModel(DefaultCapacityModel(), snapshot)

        hazard_engine = PathfindingEngine(self.engine.graph, cost_model=hazard_cost_model)
        sim = MultiAgentSimulation(hazard_engine, capacity_model=hazard_capacity_model)

        sim.add_occupant(self.room.id, occupant_id="p1")
        result = sim.run()

        self.assertIsNotNone(result.occupants["p1"].arrival_time)


class HazardEvolutionIndependenceTests(unittest.TestCase):

    def test_hazard_evolution_package_never_touches_reference_or_designer(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "hazard_evolution"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertNotIn(
                ".reference", text, f"{path.name} touches .reference directly"
            )
            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+(models|designer)\b", text, re.MULTILINE),
                f"{path.name} imports models/designer directly -- the Building "
                f"Model must stay untouched by Hazard Evolution",
            )

    def test_hazard_package_never_imports_hazard_evolution(self):

        import pathlib
        import re

        hazard_dir = pathlib.Path(__file__).resolve().parent.parent / "hazard"

        for path in hazard_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+hazard_evolution\b", text, re.MULTILINE),
                f"hazard/{path.name} imports hazard_evolution/ -- reverses the "
                f"dependency direction; hazard/ (V1, frozen) must stay unaware "
                f"of hazard_evolution/",
            )

    def test_frozen_subsystems_never_import_hazard_evolution(self):

        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent

        frozen_dirs = ("navigation", "pathfinding", "simulator", "behavior", "analysis")

        for dir_name in frozen_dirs:

            for path in (root / dir_name).glob("*.py"):

                text = path.read_text()

                self.assertIsNone(
                    re.search(r"^\s*(from|import)\s+hazard_evolution\b", text, re.MULTILINE),
                    f"{dir_name}/{path.name} imports hazard_evolution/ -- "
                    f"reverses the dependency direction",
                )

    def test_engine_has_no_merge_policy_of_its_own(self):

        # Structural guard against the merge policy quietly creeping
        # back into the engine as hardcoded logic -- max/min/sum/`in`
        # membership checks belong only in a HazardMergeStrategy.
        import pathlib

        text = (
            pathlib.Path(__file__).resolve().parent.parent
            / "hazard_evolution" / "engine.py"
        ).read_text()

        for forbidden in ("max(", "min(", "sum("):
            self.assertNotIn(
                forbidden, text,
                f"engine.py contains '{forbidden}' -- merge policy must live "
                f"only in HazardMergeStrategy",
            )


if __name__ == "__main__":
    unittest.main()
