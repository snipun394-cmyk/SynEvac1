import unittest

from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.obstacle import Obstacle
from models.staircase import Staircase
from models.zone import Zone

from navigation.node import Node

from pathfinding.route import Route

from scenario import Scenario, ScenarioEvent, ScenarioFire, ScenarioMetadata, ScenarioOccupant
from scenario_runner import run
from behaviour_profile_resolver import register_occupants

from simulator import MultiAgentSimulationResult, OccupantState, OccupantTimeline, OccupantTimelineStep

from ai_decision.engine import AIDecisionEngine

from perception.models.building_observation import BuildingObservation

from simulation_runtime import MovementTimelineOccupancyProvider, SimulationClock, SimulationRuntime
from simulation_runtime.clock import resolve_default_end_time


# =====================================================
# Fixtures -- mirrors tests/test_scenario_event_executor.py's own
# make_building()/make_metadata()/make_event()/make_scenario() shape, so
# a reader already familiar with that suite recognizes this one
# immediately. zone-2 -> door-1 -> zone-1 -> exit-1 gives every occupant a
# genuine, non-trivial, reachable route with real OccupantTimelineSteps.
# =====================================================


def make_building():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
        obstacles=[Obstacle(id="obs-1", active=False)],
        cameras=[Camera(id="cam-1", active=True)],
        detectors=[Detector(id="det-1", active=True)],
        stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-3", to_floor_id="floor-2")],
    )
    floor2 = Floor(name="Upper", id="floor-2", zones=[Zone(id="zone-3", name="Attic")])

    return Building(name="Test Building", id="building-1", floors=[floor1, floor2])


def make_metadata(**overrides):

    defaults = dict(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="hash-abc",
        generation_version="scenario_generator/1", seed=42, created_at="2026-07-13T00:00:00",
    )
    defaults.update(overrides)

    return ScenarioMetadata(**defaults)


def make_occupant(**overrides):

    # Staff_Default resolves to AlwaysEvacuateDecisionStrategy +
    # NoPreMovementDelay -- the one profile in the default registry with
    # zero randomness (behavior/intent.py, behavior/pre_movement.py), so
    # these tests get a reproducible depart_time/arrival_time rather than
    # being at the mercy of ComplianceDecisionStrategy's/
    # ProbabilisticPreMovementDelay's own unseeded sampling.
    defaults = dict(
        occupant_id="occ-1", zone_id="zone-2", floor_id="floor-1",
        position=(21.0, 1.0), behaviour_profile_id="Staff_Default",
    )
    defaults.update(overrides)

    return ScenarioOccupant(**defaults)


def make_event(**overrides):

    defaults = dict(
        event_id="evt-1", target_type="camera", target_id="cam-1",
        event_type="fail", time=10.0, parameters={"availability": "FAILED"},
    )
    defaults.update(overrides)

    return ScenarioEvent(**defaults)


def make_scenario(**overrides):

    defaults = dict(
        metadata=make_metadata(),
        occupants=(make_occupant(),),
        fire=ScenarioFire(
            ignition_zone_id="zone-2", ignition_floor_id="floor-1",
            fire_profile="Electrical", growth_parameters={"growth_time": 200.0},
        ),
        events=(),
    )
    defaults.update(overrides)

    return Scenario(**defaults)


def make_context(scenario=None, building=None):

    context = run(scenario or make_scenario(), building or make_building())
    register_occupants(context)

    return context


def make_decision_engine(context):

    return AIDecisionEngine(base_engine=context.engine)


class RecordingPerceptionProvider:

    # A minimal stand-in for a real PerceptionProvider (§13 -- this
    # Runtime does not build a concrete one; a caller supplies whatever
    # implementation it has). Records the Building's live camera.active
    # flag at the instant it is called, which is exactly what
    # SubsystemOrderingTests below needs to observe.

    def __init__(self, context):

        self.context = context
        self.calls = []

    def observation_at(self, time):

        camera_active = self.context.building.floors[0].cameras[0].active
        self.calls.append((time, camera_active))

        return BuildingObservation(timestamp=time)


class RecordingDatasetLogger:

    def __init__(self):

        self.calls = []

    def on_tick(self, time, fired_events, hazard_snapshot, occupancy_snapshot, decision, observation):

        self.calls.append(
            (time, fired_events, hazard_snapshot, occupancy_snapshot, decision, observation),
        )


class InitializationTests(unittest.TestCase):

    def test_constructing_the_runtime_runs_the_simulation_exactly_once(self):

        context = make_context()

        self.assertEqual(
            context.simulation._occupants["occ-1"].state, OccupantState.AT_NODE,
        )

        SimulationRuntime(context, make_decision_engine(context), dt=5.0)

        self.assertEqual(
            context.simulation._occupants["occ-1"].state, OccupantState.ARRIVED,
        )

    def test_movement_result_is_exposed_and_populated(self):

        context = make_context()
        runtime = SimulationRuntime(context, make_decision_engine(context), dt=5.0)

        self.assertIsInstance(runtime.movement_result, MultiAgentSimulationResult)
        self.assertIsNotNone(runtime.movement_result.total_evacuation_time)
        self.assertIn("occ-1", runtime.movement_result.occupants)

    def test_default_end_time_is_the_later_of_arrival_and_last_event(self):

        scenario = make_scenario(events=(make_event(time=1000.0),))
        context = make_context(scenario=scenario)
        runtime = SimulationRuntime(context, make_decision_engine(context), dt=5.0)

        self.assertEqual(runtime.end_time, 1000.0)

    def test_default_end_time_falls_back_to_arrival_when_it_is_later(self):

        scenario = make_scenario(events=(make_event(time=1.0),))
        context = make_context(scenario=scenario)
        runtime = SimulationRuntime(context, make_decision_engine(context), dt=5.0)

        self.assertEqual(runtime.end_time, runtime.movement_result.total_evacuation_time)
        self.assertGreater(runtime.end_time, 1.0)

    def test_caller_supplied_end_time_overrides_the_default(self):

        context = make_context()
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=5.0, end_time=3.0,
        )

        self.assertEqual(runtime.end_time, 3.0)

    def test_perception_provider_and_dataset_logger_default_to_none(self):

        context = make_context()
        runtime = SimulationRuntime(context, make_decision_engine(context), dt=5.0)

        self.assertIsNone(runtime.perception_provider)
        self.assertIsNone(runtime.dataset_logger)

    def test_no_occupants_and_no_events_means_runtime_starts_finished(self):

        scenario = make_scenario(occupants=(), events=())
        context = make_context(scenario=scenario)
        runtime = SimulationRuntime(context, make_decision_engine(context), dt=5.0)

        self.assertEqual(runtime.end_time, 0.0)
        self.assertTrue(runtime.is_finished)
        self.assertEqual(runtime.run(), ())


class ClockAdvancementTests(unittest.TestCase):

    def test_current_time_starts_at_zero(self):

        context = make_context()
        runtime = SimulationRuntime(context, make_decision_engine(context), dt=5.0)

        self.assertEqual(runtime.current_time, 0.0)

    def test_tick_advances_current_time_by_dt(self):

        context = make_context()
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=5.0, end_time=100.0,
        )

        runtime.tick()
        self.assertEqual(runtime.current_time, 5.0)

        runtime.tick()
        self.assertEqual(runtime.current_time, 10.0)

    def test_tick_is_capped_at_end_time(self):

        context = make_context()
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=20.0, end_time=47.0,
        )

        result = runtime.run()

        self.assertEqual([r.time for r in result], [20.0, 40.0, 47.0])
        self.assertEqual(runtime.current_time, 47.0)
        self.assertTrue(runtime.is_finished)

    def test_tick_after_finished_raises(self):

        context = make_context()
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=5.0, end_time=5.0,
        )

        runtime.tick()
        self.assertTrue(runtime.is_finished)

        with self.assertRaises(ValueError):
            runtime.tick()

    def test_current_time_unchanged_after_a_tick_that_raises(self):

        # Event execution is the first subsystem call inside tick() (§8),
        # so an unsupported target_type raised there must leave the clock
        # untouched (§15: "no partial-tick recovery").
        scenario = make_scenario(
            events=(make_event(target_type="ai_action", target_id="agent-1", parameters={}),),
        )
        context = make_context(scenario=scenario)
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=10.0, end_time=100.0,
        )

        with self.assertRaises(LookupError):
            runtime.tick()

        self.assertEqual(runtime.current_time, 0.0)
        self.assertFalse(runtime.is_finished)

    def test_simulation_clock_rejects_non_positive_dt(self):

        with self.assertRaises(ValueError):
            SimulationClock(dt=0.0, end_time=10.0)

        with self.assertRaises(ValueError):
            SimulationClock(dt=-1.0, end_time=10.0)

    def test_simulation_clock_rejects_negative_end_time(self):

        with self.assertRaises(ValueError):
            SimulationClock(dt=1.0, end_time=-1.0)

    def test_runtime_construction_rejects_non_positive_dt(self):

        context = make_context()

        with self.assertRaises(ValueError):
            SimulationRuntime(context, make_decision_engine(context), dt=0.0)


class StopConditionTests(unittest.TestCase):

    def test_run_produces_a_tick_for_every_dt_up_to_end_time(self):

        context = make_context()
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=10.0, end_time=25.0,
        )

        results = runtime.run()

        self.assertEqual([r.time for r in results], [10.0, 20.0, 25.0])

    def test_end_time_of_zero_produces_no_ticks(self):

        context = make_context()
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=5.0, end_time=0.0,
        )

        self.assertTrue(runtime.is_finished)
        self.assertEqual(runtime.run(), ())


class SubsystemOrderingTests(unittest.TestCase):

    def test_event_execution_happens_before_perception_is_queried_the_same_tick(self):

        scenario = make_scenario(events=(make_event(time=10.0),))
        context = make_context(scenario=scenario)

        perception_provider = RecordingPerceptionProvider(context)
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=10.0, end_time=10.0,
            perception_provider=perception_provider,
        )

        self.assertTrue(context.building.floors[0].cameras[0].active)

        runtime.tick()

        self.assertEqual(perception_provider.calls, [(10.0, False)])

    def test_fired_events_reflected_in_building_state_before_tick_returns(self):

        scenario = make_scenario(events=(make_event(time=10.0),))
        context = make_context(scenario=scenario)
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=10.0, end_time=10.0,
        )

        result = runtime.tick()

        self.assertEqual([e.event_id for e in result.fired_events], ["evt-1"])
        self.assertFalse(context.building.floors[0].cameras[0].active)

    def test_decision_is_computed_from_ground_truth_snapshots_not_observation(self):

        # §14 -- AIDecisionEngine.decide() is called with this tick's
        # hazard_snapshot/occupancy_snapshot directly, never with the
        # BuildingObservation a configured perception_provider produces.
        context = make_context()
        perception_provider = RecordingPerceptionProvider(context)
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=5.0, end_time=5.0,
            perception_provider=perception_provider,
        )

        result = runtime.tick()

        self.assertEqual(result.decision.timestamp, 5.0)
        self.assertIsNotNone(result.observation)
        self.assertIsNot(result.observation, result.decision)

    def test_dataset_logger_receives_every_tick_after_every_other_subsystem_call(self):

        scenario = make_scenario(events=(make_event(time=10.0),))
        context = make_context(scenario=scenario)

        logger = RecordingDatasetLogger()
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=10.0, end_time=20.0,
            dataset_logger=logger,
        )

        results = runtime.run()

        self.assertEqual(len(logger.calls), len(results))

        for call, result in zip(logger.calls, results):

            time, fired_events, hazard_snapshot, occupancy_snapshot, decision, observation = call

            self.assertEqual(time, result.time)
            self.assertEqual(fired_events, result.fired_events)
            self.assertIs(hazard_snapshot, result.hazard_snapshot)
            self.assertIs(occupancy_snapshot, result.occupancy_snapshot)
            self.assertIs(decision, result.decision)
            self.assertEqual(observation, result.observation)

    def test_observation_is_none_when_no_perception_provider_is_configured(self):

        context = make_context()
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=5.0, end_time=5.0,
        )

        result = runtime.tick()

        self.assertIsNone(result.observation)

    def test_hazard_evolves_across_ticks(self):

        context = make_context()
        runtime = SimulationRuntime(
            context, make_decision_engine(context), dt=20.0, end_time=100.0,
        )

        first = runtime.tick()
        second = runtime.tick()

        first_score = first.hazard_snapshot.node_state("zone-2").hazard_score
        second_score = second.hazard_snapshot.node_state("zone-2").hazard_score

        self.assertGreaterEqual(second_score, first_score)
        self.assertEqual(first.hazard_snapshot.timestamp, 20.0)
        self.assertEqual(second.hazard_snapshot.timestamp, 40.0)


class DeterministicReplayTests(unittest.TestCase):

    def test_two_independent_runtimes_over_equivalent_scenarios_agree(self):

        scenario = make_scenario(events=(make_event(time=10.0),))

        first_context = make_context(scenario=scenario, building=make_building())
        second_context = make_context(scenario=scenario, building=make_building())

        first_runtime = SimulationRuntime(
            first_context, make_decision_engine(first_context), dt=10.0,
        )
        second_runtime = SimulationRuntime(
            second_context, make_decision_engine(second_context), dt=10.0,
        )

        first_results = first_runtime.run()
        second_results = second_runtime.run()

        self.assertEqual(len(first_results), len(second_results))

        for first, second in zip(first_results, second_results):

            self.assertEqual(first.time, second.time)
            self.assertEqual(
                [e.event_id for e in first.fired_events], [e.event_id for e in second.fired_events],
            )
            self.assertEqual(
                first.hazard_snapshot.node_state("zone-2").hazard_score,
                second.hazard_snapshot.node_state("zone-2").hazard_score,
            )
            self.assertEqual(
                first.occupancy_snapshot.observation_at("zone-1").occupant_count,
                second.occupancy_snapshot.observation_at("zone-1").occupant_count,
            )
            self.assertEqual(first.decision.unsafe_zone_ids, second.decision.unsafe_zone_ids)

    def test_stepping_in_smaller_increments_ends_at_the_same_state_as_one_big_step(self):

        scenario = make_scenario(events=(make_event(time=10.0),))

        stepped_context = make_context(scenario=scenario, building=make_building())
        stepped_runtime = SimulationRuntime(
            stepped_context, make_decision_engine(stepped_context), dt=5.0, end_time=40.0,
        )
        stepped_runtime.run()

        one_shot_context = make_context(scenario=scenario, building=make_building())
        one_shot_runtime = SimulationRuntime(
            one_shot_context, make_decision_engine(one_shot_context), dt=40.0, end_time=40.0,
        )
        one_shot_runtime.run()

        self.assertEqual(
            stepped_context.building.floors[0].cameras[0].active,
            one_shot_context.building.floors[0].cameras[0].active,
        )
        self.assertFalse(stepped_context.building.floors[0].cameras[0].active)


class MovementTimelineOccupancyProviderTests(unittest.TestCase):

    NODE_A = Node(id="zone-a", name="A", floor_id="floor-1", node_type=Node.ZONE)
    NODE_B = Node(id="zone-b", name="B", floor_id="floor-1", node_type=Node.ZONE)
    NODE_C = Node(id="zone-c", name="C", floor_id="floor-1", node_type=Node.ZONE)

    def _result(self, *timelines):

        return MultiAgentSimulationResult(
            occupants={t.occupant_id: t for t in timelines},
            total_evacuation_time=None,
        )

    def test_stationary_occupant_with_no_route_contributes_nothing(self):

        timeline = OccupantTimeline(
            occupant_id="occ-1", route=None, steps=[],
            state=OccupantState.STATIONARY, depart_time=0.0, arrival_time=None,
        )
        provider = MovementTimelineOccupancyProvider(self._result(timeline))

        snapshot = provider.snapshot_at(50.0)

        self.assertEqual(dict(snapshot.observations), {})

    def test_trivial_route_arrived_occupant_contributes_nothing(self):

        route = Route(nodes=[self.NODE_A], edges=[], total_cost=0.0, total_distance=0.0)
        timeline = OccupantTimeline(
            occupant_id="occ-1", route=route, steps=[],
            state=OccupantState.ARRIVED, depart_time=0.0, arrival_time=0.0,
        )
        provider = MovementTimelineOccupancyProvider(self._result(timeline))

        snapshot = provider.snapshot_at(50.0)

        self.assertEqual(dict(snapshot.observations), {})

    def test_unreachable_occupant_contributes_nothing(self):

        route = Route(
            nodes=[self.NODE_A, self.NODE_B], edges=["not-empty"], total_cost=0.0, total_distance=0.0,
        )
        timeline = OccupantTimeline(
            occupant_id="occ-1", route=route, steps=[],
            state=OccupantState.UNREACHABLE, depart_time=0.0, arrival_time=None,
        )
        provider = MovementTimelineOccupancyProvider(self._result(timeline))

        snapshot = provider.snapshot_at(50.0)

        self.assertEqual(dict(snapshot.observations), {})

    def _traveling_timeline(self, occupant_id="occ-1"):

        route = Route(
            nodes=[self.NODE_A, self.NODE_B, self.NODE_C], edges=["e1", "e2"],
            total_cost=0.0, total_distance=0.0,
        )
        steps = [
            OccupantTimelineStep(
                index=0, from_node=self.NODE_A, to_node=self.NODE_B, edge=None,
                queue_wait_time=0.0, start_time=10.0, end_time=20.0,
            ),
            OccupantTimelineStep(
                index=1, from_node=self.NODE_B, to_node=self.NODE_C, edge=None,
                queue_wait_time=0.0, start_time=25.0, end_time=35.0,
            ),
        ]

        return OccupantTimeline(
            occupant_id=occupant_id, route=route, steps=steps,
            state=OccupantState.ARRIVED, depart_time=10.0, arrival_time=35.0,
        )

    def test_before_first_step_counts_toward_the_start_node(self):

        provider = MovementTimelineOccupancyProvider(self._result(self._traveling_timeline()))

        snapshot = provider.snapshot_at(5.0)

        self.assertEqual(snapshot.observation_at("zone-a").occupant_count, 1.0)
        self.assertIsNone(snapshot.observation_at("zone-b").occupant_count)

    def test_mid_traversal_counts_toward_neither_node(self):

        provider = MovementTimelineOccupancyProvider(self._result(self._traveling_timeline()))

        snapshot = provider.snapshot_at(15.0)

        self.assertEqual(dict(snapshot.observations), {})

    def test_waiting_between_steps_counts_toward_the_intermediate_node(self):

        provider = MovementTimelineOccupancyProvider(self._result(self._traveling_timeline()))

        snapshot = provider.snapshot_at(22.0)

        self.assertEqual(snapshot.observation_at("zone-b").occupant_count, 1.0)

    def test_after_arrival_contributes_nothing(self):

        provider = MovementTimelineOccupancyProvider(self._result(self._traveling_timeline()))

        at_arrival = provider.snapshot_at(35.0)
        after_arrival = provider.snapshot_at(1000.0)

        self.assertEqual(dict(at_arrival.observations), {})
        self.assertEqual(dict(after_arrival.observations), {})

    def test_multiple_occupants_aggregate_at_the_same_node(self):

        first = self._traveling_timeline(occupant_id="occ-1")
        second = self._traveling_timeline(occupant_id="occ-2")

        provider = MovementTimelineOccupancyProvider(self._result(first, second))

        snapshot = provider.snapshot_at(5.0)

        self.assertEqual(snapshot.observation_at("zone-a").occupant_count, 2.0)

    def test_snapshot_carries_the_requested_timestamp(self):

        provider = MovementTimelineOccupancyProvider(self._result(self._traveling_timeline()))

        snapshot = provider.snapshot_at(42.0)

        self.assertEqual(snapshot.timestamp, 42.0)


class ResolveDefaultEndTimeTests(unittest.TestCase):

    def test_uses_arrival_time_when_no_events(self):

        result = MultiAgentSimulationResult(occupants={}, total_evacuation_time=30.0)

        self.assertEqual(resolve_default_end_time(result, ()), 30.0)

    def test_uses_last_event_time_when_later_than_arrival(self):

        result = MultiAgentSimulationResult(occupants={}, total_evacuation_time=30.0)
        events = (
            ScenarioEvent(
                event_id="e1", target_type="door", target_id="d1",
                event_type="close", time=100.0,
            ),
        )

        self.assertEqual(resolve_default_end_time(result, events), 100.0)

    def test_defaults_to_zero_when_nothing_happens(self):

        result = MultiAgentSimulationResult(occupants={}, total_evacuation_time=None)

        self.assertEqual(resolve_default_end_time(result, ()), 0.0)


class SimulationRuntimePackageDependencyDirectionTests(unittest.TestCase):

    def test_package_never_imports_forbidden_packages(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "simulation_runtime"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(scenario_generator|scenario_validator|scenario_pipeline|scenario_storage|"
            r"behaviour_profile_resolver|behavior\b|behavior_library|sandbox|designer|"
            r"random|gymnasium|torch|tensorflow)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"simulation_runtime/{path.name} imports a package the architecture "
                f"(docs/architecture/simulation_runtime.md §19) forbids",
            )

    @staticmethod
    def _non_comment_code(text):

        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

    def test_package_never_touches_behavior_apis_directly(self):

        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "simulation_runtime"

        for path in sorted(package_dir.glob("*.py")):

            code = self._non_comment_code(path.read_text())

            self.assertNotIn("HumanBehaviorLayer", code, f"simulation_runtime/{path.name}")
            self.assertNotIn("BehaviorProfile", code, f"simulation_runtime/{path.name}")
            self.assertNotIn(".submit_decision", code, f"simulation_runtime/{path.name}")
            self.assertNotIn(".add_occupant", code, f"simulation_runtime/{path.name}")

    def test_package_calls_simulation_run_at_most_once_per_module(self):

        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "simulation_runtime"

        occurrences = 0

        for path in sorted(package_dir.glob("*.py")):

            occurrences += self._non_comment_code(path.read_text()).count(".simulation.run()")

        self.assertEqual(occurrences, 1)

    def test_package_performs_no_file_io(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "simulation_runtime"

        forbidden = r"\bopen\s*\(|\.write\s*\(|\.read\s*\(|\bjson\.(load|dump)\s*\("

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text),
                f"simulation_runtime/{path.name} appears to perform file I/O",
            )


if __name__ == "__main__":
    unittest.main()
