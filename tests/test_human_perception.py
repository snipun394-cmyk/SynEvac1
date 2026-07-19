import sys
import unittest
from dataclasses import FrozenInstanceError

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine
from pathfinding.route import Route

from simulator.coordinator import MultiAgentSimulation
from simulator.decision import ActionType, BehaviorDecision

from perception.models.building_observation import BuildingObservation
from perception.models.human_observation import (
    BehaviorEvent,
    HumanClassification,
    HumanObservation,
    HumanState,
)
from perception.human_inference import InferenceFlag, derive_inference_flags
from perception.fusion.sensor_fusion import SensorFusion
from perception.providers.human_observation_provider import HumanObservationProvider

from simulation_runtime.human_observation_bridge import GroundTruthHumanObservationProvider

from ground_truth.labels import GroundTruth

from scenario.metadata import ScenarioMetadata
from scenario.scenario import Scenario

from decision_policy.policy import DecisionInputs, DecisionPolicy, generate_policy
from decision_policy.human_priority_policy import compute_human_rescue_priorities

from live_system.state_manager import LiveBuildingSnapshot, StateManager


# =====================================================
# Shared fixtures
# =====================================================


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def build_two_zone_building():

    # zone_a --door--> zone_b --exit--> Outside, mirroring the exact
    # shape tests/test_behavior_layer.py's own build_two_zone_building()
    # already uses.

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    zone_a = make_zone("A", x=0.0, y=0.0)
    zone_b = make_zone("B", x=5.0, y=0.0)
    floor.add_zone(zone_a)
    floor.add_zone(zone_b)

    door = Door(name="D1", zone_a_id=zone_a.id, zone_b_id=zone_b.id, floor_id=floor.id, width=1.0)
    floor.add_door(door)
    exit_obj = Exit(name="Ex", zone_id=zone_b.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, floor, zone_a, zone_b, door, exit_obj, engine


def make_observation(**overrides):

    fields = dict(person_id="p1")
    fields.update(overrides)

    return HumanObservation(**fields)


# =====================================================
# Phase 1/2/3/4 -- HumanObservation model
# =====================================================


class HumanObservationModelTests(unittest.TestCase):

    def test_minimal_construction_uses_documented_defaults(self):

        observation = HumanObservation(person_id="p1")

        self.assertEqual(observation.person_id, "p1")
        self.assertIsNone(observation.zone_id)
        self.assertIsNone(observation.floor_id)
        self.assertIsNone(observation.confidence)
        self.assertEqual(observation.classification, HumanClassification.UNKNOWN)
        self.assertIsNone(observation.state)
        self.assertEqual(observation.behavior_events, frozenset())

    def test_every_field_can_be_supplied(self):

        observation = HumanObservation(
            person_id="p42",
            zone_id="zone-1",
            floor_id="floor-1",
            confidence=0.87,
            classification=HumanClassification.WHEELCHAIR_USER,
            state=HumanState.HELPING_ANOTHER_OCCUPANT,
            behavior_events=frozenset({BehaviorEvent.PUSHING_WHEELCHAIR}),
            last_observed_time=12.5,
        )

        self.assertEqual(observation.person_id, "p42")
        self.assertEqual(observation.classification, HumanClassification.WHEELCHAIR_USER)
        self.assertEqual(observation.state, HumanState.HELPING_ANOTHER_OCCUPANT)
        self.assertIn(BehaviorEvent.PUSHING_WHEELCHAIR, observation.behavior_events)

    def test_behavior_events_accepts_a_plain_set_and_stores_a_frozenset(self):

        observation = HumanObservation(
            person_id="p1", behavior_events={BehaviorEvent.QUEUEING, BehaviorEvent.GROUPED_MOVEMENT},
        )

        self.assertIsInstance(observation.behavior_events, frozenset)
        self.assertEqual(
            observation.behavior_events, {BehaviorEvent.QUEUEING, BehaviorEvent.GROUPED_MOVEMENT},
        )

    def test_a_person_can_exhibit_more_than_one_behavior_event_at_once(self):

        observation = HumanObservation(
            person_id="p1",
            behavior_events=frozenset(
                {BehaviorEvent.HELPING_ANOTHER_PERSON, BehaviorEvent.GROUPED_MOVEMENT},
            ),
        )

        self.assertEqual(len(observation.behavior_events), 2)

    def test_is_frozen(self):

        observation = HumanObservation(person_id="p1")

        with self.assertRaises(FrozenInstanceError):
            observation.zone_id = "zone-2"

    def test_confidence_out_of_range_is_rejected(self):

        with self.assertRaises(ValueError):
            HumanObservation(person_id="p1", confidence=1.5)

        with self.assertRaises(ValueError):
            HumanObservation(person_id="p1", confidence=-0.1)

    def test_confidence_boundary_values_are_accepted(self):

        HumanObservation(person_id="p1", confidence=0.0)
        HumanObservation(person_id="p1", confidence=1.0)

    def test_every_documented_classification_value_exists(self):

        for name in ("ADULT", "CHILD", "WHEELCHAIR_USER", "FIREFIGHTER", "FIRE_WARDEN", "UNKNOWN"):
            self.assertTrue(hasattr(HumanClassification, name))

    def test_every_documented_state_value_exists(self):

        for name in (
            "WALKING", "RUNNING", "STANDING", "FALLEN", "CRAWLING", "WAITING",
            "BEING_ASSISTED", "HELPING_ANOTHER_OCCUPANT", "NEVER_MOVING_YET",
        ):
            self.assertTrue(hasattr(HumanState, name))

    def test_every_documented_behavior_event_value_exists(self):

        for name in (
            "HELPING_ANOTHER_PERSON", "DRAGGING_ANOTHER_PERSON", "CARRYING_ANOTHER_PERSON",
            "PUSHING_WHEELCHAIR", "GROUPED_MOVEMENT", "QUEUEING",
        ):
            self.assertTrue(hasattr(BehaviorEvent, name))


# =====================================================
# Phase 5 -- Inference flags
# =====================================================


class DeriveInferenceFlagsTests(unittest.TestCase):

    def test_never_moving_yet_flags_possible_pre_movement_delay_only(self):

        observation = make_observation(state=HumanState.NEVER_MOVING_YET)
        flags = derive_inference_flags(observation)

        self.assertIn(InferenceFlag.POSSIBLE_PRE_MOVEMENT_DELAY, flags)
        self.assertNotIn(InferenceFlag.POSSIBLE_INJURY, flags)

    def test_fallen_flags_possible_injury_and_high_rescue_priority(self):

        observation = make_observation(state=HumanState.FALLEN)
        flags = derive_inference_flags(observation)

        self.assertIn(InferenceFlag.POSSIBLE_INJURY, flags)
        self.assertIn(InferenceFlag.HIGH_RESCUE_PRIORITY, flags)
        self.assertNotIn(InferenceFlag.POSSIBLE_PRE_MOVEMENT_DELAY, flags)

    def test_crawling_flags_possible_injury(self):

        observation = make_observation(state=HumanState.CRAWLING)
        self.assertIn(InferenceFlag.POSSIBLE_INJURY, derive_inference_flags(observation))

    def test_being_assisted_flags_possible_injury_and_high_priority(self):

        observation = make_observation(state=HumanState.BEING_ASSISTED)
        flags = derive_inference_flags(observation)

        self.assertIn(InferenceFlag.POSSIBLE_INJURY, flags)
        self.assertIn(InferenceFlag.HIGH_RESCUE_PRIORITY, flags)

    def test_walking_adult_produces_no_flags(self):

        observation = make_observation(
            state=HumanState.WALKING, classification=HumanClassification.ADULT,
        )

        self.assertEqual(derive_inference_flags(observation), frozenset())

    def test_child_classification_flags_high_rescue_priority_even_while_walking(self):

        observation = make_observation(
            state=HumanState.WALKING, classification=HumanClassification.CHILD,
        )
        flags = derive_inference_flags(observation)

        self.assertIn(InferenceFlag.HIGH_RESCUE_PRIORITY, flags)
        self.assertNotIn(InferenceFlag.POSSIBLE_INJURY, flags)

    def test_wheelchair_user_classification_flags_high_rescue_priority(self):

        observation = make_observation(
            state=HumanState.WALKING, classification=HumanClassification.WHEELCHAIR_USER,
        )
        self.assertIn(InferenceFlag.HIGH_RESCUE_PRIORITY, derive_inference_flags(observation))

    def test_dragging_another_person_flags_high_rescue_priority(self):

        observation = make_observation(
            state=HumanState.WALKING,
            behavior_events=frozenset({BehaviorEvent.DRAGGING_ANOTHER_PERSON}),
        )
        self.assertIn(InferenceFlag.HIGH_RESCUE_PRIORITY, derive_inference_flags(observation))

    def test_carrying_another_person_flags_high_rescue_priority(self):

        observation = make_observation(
            state=HumanState.WALKING,
            behavior_events=frozenset({BehaviorEvent.CARRYING_ANOTHER_PERSON}),
        )
        self.assertIn(InferenceFlag.HIGH_RESCUE_PRIORITY, derive_inference_flags(observation))

    def test_queueing_alone_does_not_flag_high_rescue_priority(self):

        observation = make_observation(
            state=HumanState.WAITING, behavior_events=frozenset({BehaviorEvent.QUEUEING}),
        )
        self.assertEqual(derive_inference_flags(observation), frozenset())

    def test_no_state_and_unknown_classification_produces_no_flags(self):

        observation = make_observation()
        self.assertEqual(derive_inference_flags(observation), frozenset())

    def test_flags_are_never_read_from_a_field_that_does_not_exist(self):

        # Structural proof of Phase 5's own rule: derive_inference_flags()
        # is a pure function of an already-built HumanObservation --
        # calling it twice on the same (immutable) observation must
        # always agree.
        observation = make_observation(state=HumanState.FALLEN)

        self.assertEqual(derive_inference_flags(observation), derive_inference_flags(observation))


# =====================================================
# Phase 6a -- Perception integration
# =====================================================


class BuildingObservationHumanIntegrationTests(unittest.TestCase):

    def test_defaults_to_no_human_observations(self):

        observation = BuildingObservation()
        self.assertEqual(dict(observation.human_observations), {})

    def test_human_observation_accessor_returns_none_when_absent(self):

        observation = BuildingObservation()
        self.assertIsNone(observation.human_observation("nobody"))

    def test_human_observation_accessor_returns_the_stored_value(self):

        person = make_observation(person_id="p1", zone_id="zone-a")
        observation = BuildingObservation(human_observations={"p1": person})

        self.assertIs(observation.human_observation("p1"), person)

    def test_human_observations_mapping_is_read_only(self):

        observation = BuildingObservation(human_observations={"p1": make_observation()})

        with self.assertRaises(TypeError):
            observation.human_observations["p2"] = make_observation(person_id="p2")

    def test_existing_building_observation_construction_is_unaffected(self):

        # No human_observations argument at all -- every pre-existing
        # caller/test must still work unchanged.
        observation = BuildingObservation(timestamp=5.0)
        self.assertEqual(observation.timestamp, 5.0)
        self.assertEqual(observation.human_observations, {})


class SensorFusionHumanIntegrationTests(unittest.TestCase):

    def test_fuse_without_human_observations_argument_still_works(self):

        sensor_fusion = SensorFusion(
            smoke_detector_zone_assignments={}, heat_detector_zone_assignments={},
            camera_zone_assignments={},
        )
        observation = sensor_fusion.fuse(timestamp=0.0, occupancy_estimates={})

        self.assertEqual(dict(observation.human_observations), {})

    def test_fuse_threads_human_observations_straight_through(self):

        sensor_fusion = SensorFusion(
            smoke_detector_zone_assignments={}, heat_detector_zone_assignments={},
            camera_zone_assignments={},
        )
        person = make_observation(person_id="p1", zone_id="zone-a")

        observation = sensor_fusion.fuse(
            timestamp=0.0, occupancy_estimates={}, human_observations={"p1": person},
        )

        self.assertIs(observation.human_observation("p1"), person)


class HumanObservationProviderInterfaceTests(unittest.TestCase):

    def test_base_interface_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            HumanObservationProvider().observations_at(0.0)


# =====================================================
# Phase 6c -- Ground Truth / Simulation Snapshot bridge
# =====================================================


class GroundTruthHumanObservationProviderTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.zone_a, self.zone_b,
            self.door, self.exit_obj, self.engine,
        ) = build_two_zone_building()

    def test_before_depart_time_is_never_moving_yet(self):

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.zone_a.id, occupant_id="p1", depart_time=100.0)
        result = sim.run()

        provider = GroundTruthHumanObservationProvider(result)
        observations = provider.observations_at(0.0)

        self.assertEqual(observations["p1"].zone_id, self.zone_a.id)
        self.assertEqual(observations["p1"].state, HumanState.NEVER_MOVING_YET)

    def test_mid_traversal_is_walking(self):

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.zone_a.id, occupant_id="p1", depart_time=0.0)
        result = sim.run()

        step = result.occupants["p1"].steps[0]
        mid_time = (step.start_time + step.end_time) / 2.0

        provider = GroundTruthHumanObservationProvider(result)
        observation = provider.observations_at(mid_time)["p1"]

        self.assertEqual(observation.state, HumanState.WALKING)
        self.assertEqual(observation.zone_id, step.from_node.id)

    def test_queued_occupant_is_waiting(self):

        # A capacity-1 door with two occupants departing at the same
        # time forces the second to queue -- same fixture shape
        # tests/test_multi_agent_simulation.py's own QueueFormationTests
        # already use.
        self.door.width = 0.4  # capacity floors to 1 (see DefaultCapacityModel)

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.zone_a.id, occupant_id="first", depart_time=0.0)
        sim.add_occupant(self.zone_a.id, occupant_id="second", depart_time=0.0)
        result = sim.run()

        second_step = result.occupants["second"].steps[0]
        self.assertGreater(second_step.queue_wait_time, 0.0)

        join_time = second_step.start_time - second_step.queue_wait_time
        query_time = (join_time + second_step.start_time) / 2.0

        provider = GroundTruthHumanObservationProvider(result)
        observation = provider.observations_at(query_time)["second"]

        self.assertEqual(observation.state, HumanState.WAITING)

    def test_arrived_occupant_is_no_longer_observed(self):

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.zone_a.id, occupant_id="p1", depart_time=0.0)
        result = sim.run()

        arrival_time = result.occupants["p1"].arrival_time
        self.assertIsNotNone(arrival_time)

        provider = GroundTruthHumanObservationProvider(result)
        observations = provider.observations_at(arrival_time)

        self.assertNotIn("p1", observations)

    def test_trivial_route_occupant_reports_their_own_goal_zone(self):

        sim = MultiAgentSimulation(self.engine)
        node = self.engine.graph.find_node(self.zone_a.id)
        trivial_route = Route(nodes=[node], edges=[], total_cost=0.0, total_distance=0.0)
        sim.add_occupant(self.zone_a.id, occupant_id="p1", route=trivial_route)
        result = sim.run()

        provider = GroundTruthHumanObservationProvider(result)
        observation = provider.observations_at(0.0)["p1"]

        self.assertEqual(observation.zone_id, self.zone_a.id)
        self.assertEqual(observation.state, HumanState.WAITING)

    def test_stationary_occupant_is_observed_with_unknown_location(self):

        sim = MultiAgentSimulation(self.engine)
        sim.submit_decision(
            BehaviorDecision(occupant_id="p1", action_type=ActionType.WAIT, start_id=self.zone_a.id),
        )
        result = sim.run()

        provider = GroundTruthHumanObservationProvider(result)
        observation = provider.observations_at(0.0)["p1"]

        self.assertIsNone(observation.zone_id)
        self.assertIsNone(observation.floor_id)
        self.assertEqual(observation.state, HumanState.WAITING)

    def test_unreachable_occupant_is_observed_with_unknown_location(self):

        sim = MultiAgentSimulation(self.engine)
        sim.submit_decision(
            BehaviorDecision(
                occupant_id="p1", action_type=ActionType.EVACUATE, start_id=self.zone_a.id,
                route_unavailable=True,
            ),
        )
        result = sim.run()

        provider = GroundTruthHumanObservationProvider(result)
        observation = provider.observations_at(0.0)["p1"]

        self.assertIsNone(observation.zone_id)
        self.assertEqual(observation.state, HumanState.WAITING)

    def test_never_fabricates_classification_or_behavior_events(self):

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.zone_a.id, occupant_id="p1", depart_time=0.0)
        result = sim.run()

        provider = GroundTruthHumanObservationProvider(result)
        observation = provider.observations_at(0.0)["p1"]

        self.assertEqual(observation.classification, HumanClassification.UNKNOWN)
        self.assertEqual(observation.behavior_events, frozenset())

    def test_confidence_is_always_reported_as_certain(self):

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.zone_a.id, occupant_id="p1", depart_time=0.0)
        result = sim.run()

        provider = GroundTruthHumanObservationProvider(result)
        observation = provider.observations_at(0.0)["p1"]

        self.assertEqual(observation.confidence, 1.0)

    def test_repeated_calls_at_the_same_time_are_deterministic(self):

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.zone_a.id, occupant_id="p1", depart_time=0.0)
        result = sim.run()

        provider = GroundTruthHumanObservationProvider(result)

        first = provider.observations_at(0.0)
        second = provider.observations_at(0.0)

        self.assertEqual(first["p1"].state, second["p1"].state)
        self.assertEqual(first["p1"].zone_id, second["p1"].zone_id)


# =====================================================
# Phase 6b -- Live System integration
# =====================================================


class LiveBuildingSnapshotHumanIntegrationTests(unittest.TestCase):

    def test_human_observation_for_returns_none_with_no_perception_yet(self):

        snapshot = LiveBuildingSnapshot()
        self.assertIsNone(snapshot.human_observation_for("p1"))
        self.assertEqual(snapshot.human_observations(), {})

    def test_human_observation_for_reads_through_building_observation(self):

        person = make_observation(person_id="p1", zone_id="zone-a")
        observation = BuildingObservation(human_observations={"p1": person})

        snapshot = StateManager().update_perception(observation, time=1.0)

        self.assertIs(snapshot.human_observation_for("p1"), person)
        self.assertEqual(dict(snapshot.human_observations()), {"p1": person})


# =====================================================
# Phase 6d -- Decision Policy integration
# =====================================================


class HumanPriorityPolicyTests(unittest.TestCase):

    def test_no_observations_produces_no_priorities(self):

        self.assertEqual(compute_human_rescue_priorities({}), ())

    def test_high_priority_person_is_included(self):

        observations = {
            "p1": make_observation(
                person_id="p1", zone_id="zone-a", floor_id="floor-1", confidence=0.9,
                classification=HumanClassification.WHEELCHAIR_USER, state=HumanState.WAITING,
            ),
        }
        rows = compute_human_rescue_priorities(observations)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["person_id"], "p1")
        self.assertEqual(rows[0]["zone_id"], "zone-a")
        self.assertIn("HIGH_RESCUE_PRIORITY", rows[0]["inference_flags"])

    def test_ordinary_adult_walking_is_excluded(self):

        observations = {
            "p1": make_observation(
                person_id="p1", classification=HumanClassification.ADULT, state=HumanState.WALKING,
            ),
        }
        self.assertEqual(compute_human_rescue_priorities(observations), ())

    def test_rows_are_sorted_by_person_id_for_determinism(self):

        observations = {
            "p2": make_observation(person_id="p2", state=HumanState.FALLEN),
            "p1": make_observation(person_id="p1", state=HumanState.FALLEN),
        }
        rows = compute_human_rescue_priorities(observations)

        self.assertEqual([row["person_id"] for row in rows], ["p1", "p2"])


class DecisionPolicyHumanIntegrationTests(unittest.TestCase):

    def _building(self):

        building, floor, zone_a, zone_b, door, exit_obj, engine = build_two_zone_building()
        return building

    def _scenario(self):

        return Scenario(
            metadata=ScenarioMetadata(
                scenario_id="scn-human-1", definition_id="def-1",
                definition_content_hash="hash", generation_version="v1", seed=1,
                created_at="2026-07-15T00:00:00",
            ),
            occupants=(),
        )

    def test_decision_inputs_defaults_to_no_human_observations(self):

        inputs = DecisionInputs(
            building=self._building(), scenario=self._scenario(),
            ground_truth=GroundTruth(scenario_id="scn-human-1", definition_id="def-1"),
        )
        self.assertEqual(dict(inputs.human_observations), {})

    def test_generate_policy_produces_no_human_rescue_priorities_when_omitted(self):

        inputs = DecisionInputs(
            building=self._building(), scenario=self._scenario(),
            ground_truth=GroundTruth(scenario_id="scn-human-1", definition_id="def-1"),
        )
        policy = generate_policy(inputs)

        self.assertEqual(policy.human_rescue_priorities, ())

    def test_generate_policy_includes_human_rescue_priorities_when_supplied(self):

        human_observations = {
            "p1": make_observation(person_id="p1", zone_id="zone-a", state=HumanState.FALLEN),
        }
        inputs = DecisionInputs(
            building=self._building(), scenario=self._scenario(),
            ground_truth=GroundTruth(scenario_id="scn-human-1", definition_id="def-1"),
            human_observations=human_observations,
        )
        policy = generate_policy(inputs)

        self.assertEqual(len(policy.human_rescue_priorities), 1)
        self.assertEqual(policy.human_rescue_priorities[0]["person_id"], "p1")

    def test_existing_decision_inputs_construction_without_human_observations_is_unaffected(self):

        # The exact construction shape every pre-existing caller/test
        # already uses -- must keep working with zero changes.
        inputs = DecisionInputs(
            building=self._building(), scenario=self._scenario(),
            ground_truth=GroundTruth(scenario_id="scn-human-1", definition_id="def-1"),
        )
        self.assertIsInstance(inputs, DecisionInputs)


class DecisionPolicySerializationTests(unittest.TestCase):

    def test_round_trip_preserves_human_rescue_priorities(self):

        policy = DecisionPolicy(
            scenario_id="scn-1",
            human_rescue_priorities=(
                {"person_id": "p1", "zone_id": "zone-a", "floor_id": None,
                 "classification": "CHILD", "state": "WALKING",
                 "inference_flags": ["HIGH_RESCUE_PRIORITY"], "confidence": 0.8},
            ),
        )
        restored = DecisionPolicy.from_dict(policy.to_dict())

        self.assertEqual(restored.human_rescue_priorities, policy.human_rescue_priorities)

    def test_from_dict_defaults_missing_human_rescue_priorities(self):

        # A payload serialized before this integration existed --
        # from_dict() must not raise, and must default to ().
        legacy_payload = {"scenario_id": "scn-1"}
        restored = DecisionPolicy.from_dict(legacy_payload)

        self.assertEqual(restored.human_rescue_priorities, ())


# =====================================================
# Phase 7 -- Command Center display
# =====================================================


class HumanPanelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):

        from command_center.human_panel import HumanPanel
        from command_center.incident_data import IncidentData, IncidentFrame

        self.HumanPanel = HumanPanel
        self.IncidentData = IncidentData
        self.IncidentFrame = IncidentFrame

        building, floor, zone_a, zone_b, door, exit_obj, engine = build_two_zone_building()
        self.building = building
        self.zone_a = zone_a

        self.scenario = Scenario(
            metadata=ScenarioMetadata(
                scenario_id="scn-panel-1", definition_id="def-1",
                definition_content_hash="hash", generation_version="v1", seed=1,
                created_at="2026-07-15T00:00:00",
            ),
            occupants=(),
        )

    def test_empty_frame_shows_no_rows_and_placeholder_detail(self):

        panel = self.HumanPanel()
        panel.show_frame(None)

        self.assertEqual(panel.people_table.rowCount(), 0)
        self.assertEqual(panel.person_id_label.text(), "Person: -")

    def test_show_frame_populates_one_row_per_person(self):

        person = make_observation(
            person_id="person-42", zone_id=self.zone_a.id,
            classification=HumanClassification.WHEELCHAIR_USER,
            state=HumanState.HELPING_ANOTHER_OCCUPANT, confidence=0.75,
        )
        frame = self.IncidentFrame(human_observations={"person-42": person})

        panel = self.HumanPanel()
        panel.set_incident(self.IncidentData(building=self.building, scenario=self.scenario))
        panel.show_frame(frame)

        self.assertEqual(panel.people_table.rowCount(), 1)
        self.assertEqual(panel.people_table.item(0, 0).text(), "person-42")
        self.assertEqual(panel.people_table.item(0, 1).text(), "WHEELCHAIR_USER")
        self.assertEqual(panel.people_table.item(0, 2).text(), "HELPING_ANOTHER_OCCUPANT")

    def test_selecting_a_person_shows_their_full_detail_including_derived_flags(self):

        person = make_observation(
            person_id="person-42", zone_id=self.zone_a.id,
            classification=HumanClassification.WHEELCHAIR_USER,
            state=HumanState.NEVER_MOVING_YET,
            behavior_events=frozenset({BehaviorEvent.QUEUEING}),
            confidence=0.75,
        )
        frame = self.IncidentFrame(human_observations={"person-42": person})

        panel = self.HumanPanel()
        panel.set_incident(self.IncidentData(building=self.building, scenario=self.scenario))
        panel.show_frame(frame)

        panel.people_table.selectRow(0)

        self.assertIn("person-42", panel.person_id_label.text())
        self.assertIn("WHEELCHAIR_USER", panel.classification_label.text())
        self.assertIn("NEVER_MOVING_YET", panel.state_label.text())
        self.assertIn("QUEUEING", panel.behavior_events_label.text())
        self.assertIn("POSSIBLE_PRE_MOVEMENT_DELAY", panel.inference_flags_label.text())
        self.assertIn("HIGH_RESCUE_PRIORITY", panel.inference_flags_label.text())
        self.assertIn("75%", panel.confidence_label.text())

    def test_incident_frame_defaults_to_no_human_observations(self):

        frame = self.IncidentFrame()
        self.assertEqual(dict(frame.human_observations), {})


if __name__ == "__main__":
    unittest.main()
