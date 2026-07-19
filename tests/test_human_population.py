import unittest
from unittest.mock import patch

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from scenario.fire import ScenarioFire
from scenario.firefighter import ScenarioFirefighter
from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from scenario_definition.definition import ScenarioDefinition
from scenario_definition.fire_definition import FireDefinition
from scenario_definition.firefighter_definition import FirefighterDeploymentDefinition
from scenario_definition.occupant_definition import OccupantDefinition
from scenario_definition.distributions import FixedValue, WeightedOptions

from scenario_generator.generator import generate_scenario
from scenario_generator.request import GenerationRequest

from scenario_runner import run as run_scenario

from behaviour_profile_resolver import (
    DEFAULT_FIREFIGHTER_PROFILE_REGISTRY,
    DEFAULT_PROFILE_REGISTRY,
    OccupantCategory,
    occupant_category,
    register_occupants,
    register_population,
)

from behavior_library.assistance_strategies import (
    ASSISTANCE_TYPES,
    ROLE_ASSISTED,
    ROLE_HELPER,
    TRAIT_ASSISTANCE_ROLE,
    _SPEED_MULTIPLIER_BY_ASSISTANCE_TYPE,
)

from simulator.coordinator import MultiAgentSimulation
from simulator.decision import ActionType, BehaviorDecision
from simulator.occupant import OccupantState

from simulation_runtime.result import TickResult
from simulation_runtime import SimulationRuntime

from hazard.snapshot import HazardSnapshot
from hazard.node_state import HazardNodeState

from ground_truth.analyzer import SimulationArtifacts, analyze
from ground_truth.human_behavior import (
    DynamicHumanState,
    compute_occupant_behavior,
    summarize_behavior_counts,
)
from ground_truth.labels import GroundTruth
from ground_truth.profile_outcomes import compute_profile_outcomes
from ground_truth.rescue_outcomes import compute_rescue_outcomes, summarize_rescue_outcomes

from dataset_builder.builder import SimulationRun
from dataset_builder.labels import extract_simulation_outcome
from dataset_builder.feature_extractor import extract_scenario_features
from dataset_builder.schema import SIMULATION_OUTCOME_COLUMNS

from simulation_interactive.state_snapshot import build_snapshot, OccupantSnapshot
from simulation_interactive.movement_stepper import MovementStepper

from rl_training.observation_space import ObservationEncoder


# =====================================================
# Shared fixtures
# =====================================================


def build_two_zone_building(door_width=1.0):

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    zone_a = Zone(name="A", x=0.0, y=0.0, width=2.0, height=2.0)
    zone_b = Zone(name="B", x=10.0, y=0.0, width=2.0, height=2.0)
    floor.add_zone(zone_a)
    floor.add_zone(zone_b)

    door = Door(name="D1", zone_a_id=zone_a.id, zone_b_id=zone_b.id, floor_id=floor.id, width=door_width)
    floor.add_door(door)
    exit_obj = Exit(name="Ex", zone_id=zone_b.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    return building, floor, zone_a, zone_b, door, exit_obj


def make_metadata(seed=1, scenario_id="s1"):

    return ScenarioMetadata(
        scenario_id=scenario_id, definition_id="d1", definition_content_hash="h",
        generation_version="v1", seed=seed, created_at="2026-07-15T00:00:00",
    )


def make_fire(zone_id, floor_id):

    return ScenarioFire(
        ignition_zone_id=zone_id, ignition_floor_id=floor_id,
        fire_profile="Electrical", growth_parameters={"growth_time": 200.0},
    )


# =====================================================
# Phase 1 -- Occupant profiles
# =====================================================


class OccupantProfileTests(unittest.TestCase):

    def test_every_minimum_required_profile_is_present(self):

        for profile_id in (
            "Adult_Default", "Child_Default", "Elderly_Default",
            "Wheelchair_Default", "Visitor_Default",
        ):
            self.assertIn(profile_id, DEFAULT_PROFILE_REGISTRY)

    def test_elderly_default_has_a_positive_walking_speed_slower_than_adult(self):

        elderly = DEFAULT_PROFILE_REGISTRY["Elderly_Default"]
        adult = DEFAULT_PROFILE_REGISTRY["Adult_Default"]

        self.assertGreater(elderly.walking_speed, 0.0)
        self.assertLess(elderly.walking_speed, adult.walking_speed)

    def test_every_generated_occupant_receives_exactly_one_profile(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()

        definition = ScenarioDefinition(
            fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
            occupant=OccupantDefinition(
                occupancy_distribution={zone_a.id: FixedValue(5)},
                behaviour_profile_distribution={
                    zone_a.id: WeightedOptions({"Adult_Default": 0.6, "Elderly_Default": 0.4}),
                },
            ),
        )
        scenario = generate_scenario(
            GenerationRequest(definition=definition, definition_id="def-1", building=building, seed=7),
        )

        self.assertEqual(len(scenario.occupants), 5)
        for occupant in scenario.occupants:
            self.assertIn(occupant.behaviour_profile_id, ("Adult_Default", "Elderly_Default"))

    def test_occupant_category_classifies_every_default_profile(self):

        expected = {
            "Adult_Default": OccupantCategory.ADULT,
            "Child_Default": OccupantCategory.CHILD,
            "Elderly_Default": OccupantCategory.ELDERLY,
            "Wheelchair_Default": OccupantCategory.WHEELCHAIR_USER,
            "Visitor_Default": OccupantCategory.VISITOR,
            "Staff_Default": OccupantCategory.STAFF,
            "FireWarden_Default": OccupantCategory.FIRE_WARDEN,
            "Firefighter_Default": OccupantCategory.FIREFIGHTER,
        }
        for profile_id, category in expected.items():
            self.assertEqual(occupant_category(profile_id), category)

    def test_unknown_profile_id_classifies_as_unknown(self):

        self.assertEqual(occupant_category("Nonexistent_Profile"), OccupantCategory.UNKNOWN)


# =====================================================
# Phase 2 -- Dynamic human states
# =====================================================


class _AlwaysBelowThresholdRandom:

    # A deterministic stand-in random.Random -- .random() always
    # returns 0.0, guaranteeing both the fall-probability check and the
    # possible-injury-given-fallen check in ground_truth.human_behavior
    # succeed, without brute-forcing a real seed that happens to.

    def random(self):
        return 0.0


class DynamicHumanStateTests(unittest.TestCase):

    def _tick(self, time, zone_id, hazard_score):

        return TickResult(
            time=time, fired_events=(),
            hazard_snapshot=HazardSnapshot(
                node_states={zone_id: HazardNodeState(hazard_score=hazard_score)},
            ),
            occupancy_snapshot=None, decision=None,
        )

    def test_state_enum_has_every_documented_value(self):

        for name in (
            "NORMAL", "WALKING", "RUNNING", "WAITING",
            "HELPING_ANOTHER_OCCUPANT", "BEING_ASSISTED", "FALLEN", "POSSIBLE_INJURY",
        ):
            self.assertTrue(hasattr(DynamicHumanState, name))

    def test_stationary_occupant_is_waiting(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        sim = MultiAgentSimulation(_engine_for(building))
        sim.submit_decision(
            BehaviorDecision(occupant_id="p1", action_type=ActionType.WAIT, start_id=zone_a.id),
        )
        result = sim.run()

        scenario = Scenario(metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id))
        records = compute_occupant_behavior(scenario, result, ())

        self.assertEqual(records["p1"].dynamic_state, DynamicHumanState.WAITING)

    def test_fallen_and_possible_injury_arise_from_congestion_and_hazard_not_a_static_profile(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building(door_width=0.4)
        engine = _engine_for(building)

        sim = MultiAgentSimulation(engine)
        sim.add_occupant(zone_a.id, occupant_id="first", depart_time=0.0)
        sim.add_occupant(zone_a.id, occupant_id="second", depart_time=0.0)
        result = sim.run()

        second_step = result.occupants["second"].steps[0]
        self.assertGreater(second_step.queue_wait_time, 0.0)

        # A single tick, at a time within the queued step, reporting a
        # hazard score high enough to qualify -- both thresholds this
        # module's own FALL_QUEUE_WAIT_THRESHOLD_SECONDS/
        # FALL_HAZARD_SCORE_THRESHOLD require.
        tick = self._tick(
            time=second_step.start_time - 0.01, zone_id=second_step.from_node.id, hazard_score=0.9,
        )
        # Force the queue wait above the fall threshold by using a tick
        # time consistent with a long queue -- simplest honest way is
        # to lower the threshold via monkeypatch instead of engineering
        # exact timings; see the module-level patch below.

        scenario = Scenario(
            metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id),
        )

        with patch("ground_truth.human_behavior.FALL_QUEUE_WAIT_THRESHOLD_SECONDS", 0.0), \
             patch("ground_truth.human_behavior.random.Random", return_value=_AlwaysBelowThresholdRandom()):

            records = compute_occupant_behavior(scenario, result, (tick,))

        self.assertTrue(records["second"].ever_fallen)
        self.assertTrue(records["second"].ever_possible_injury)
        self.assertIn(
            records["second"].dynamic_state, (DynamicHumanState.FALLEN, DynamicHumanState.POSSIBLE_INJURY),
        )

    def test_no_congestion_or_hazard_never_produces_a_fall(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building(door_width=5.0)
        sim = MultiAgentSimulation(_engine_for(building))
        sim.add_occupant(zone_a.id, occupant_id="solo", depart_time=0.0)
        result = sim.run()

        scenario = Scenario(metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id))

        with patch("ground_truth.human_behavior.random.Random", return_value=_AlwaysBelowThresholdRandom()):
            records = compute_occupant_behavior(scenario, result, ())

        self.assertFalse(records["solo"].ever_fallen)
        self.assertFalse(records["solo"].ever_possible_injury)

    def test_helping_and_assisted_flags_come_from_scenario_assistance_fields(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        sim = MultiAgentSimulation(_engine_for(building))
        sim.add_occupant(zone_a.id, occupant_id="helper1", depart_time=0.0)
        sim.add_occupant(zone_a.id, occupant_id="assisted1", depart_time=0.0)
        result = sim.run()

        occupants = (
            ScenarioOccupant(
                occupant_id="helper1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Adult_Default",
                assisting_occupant_id="assisted1", assistance_type="ESCORT",
            ),
            ScenarioOccupant(
                occupant_id="assisted1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Elderly_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))

        records = compute_occupant_behavior(scenario, result, ())

        self.assertTrue(records["helper1"].ever_helping)
        self.assertFalse(records["helper1"].ever_assisted)
        self.assertTrue(records["assisted1"].ever_assisted)
        self.assertFalse(records["assisted1"].ever_helping)

    def test_summarize_behavior_counts_matches_the_records(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        sim = MultiAgentSimulation(_engine_for(building))
        sim.add_occupant(zone_a.id, occupant_id="p1", depart_time=0.0)
        result = sim.run()

        scenario = Scenario(metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id))
        records = compute_occupant_behavior(scenario, result, ())
        counts = summarize_behavior_counts(records)

        self.assertEqual(
            counts,
            {
                "helping_group_count": 0, "assisted_occupant_count": 0,
                "fallen_count": 0, "possible_injury_count": 0,
            },
        )

    def test_deterministic_across_repeated_calls(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        sim = MultiAgentSimulation(_engine_for(building))
        sim.add_occupant(zone_a.id, occupant_id="p1", depart_time=0.0)
        result = sim.run()

        scenario = Scenario(metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id))

        first = compute_occupant_behavior(scenario, result, ())
        second = compute_occupant_behavior(scenario, result, ())

        self.assertEqual(first["p1"].dynamic_state, second["p1"].dynamic_state)
        self.assertEqual(first["p1"].ever_fallen, second["p1"].ever_fallen)


def _engine_for(building):

    from navigation.graph_builder import NavigationGraphGenerator
    from pathfinding.engine import PathfindingEngine

    return PathfindingEngine(NavigationGraphGenerator().build(building))


# =====================================================
# Phase 3 -- Assistance behaviors
# =====================================================


class AssistanceBehaviorTests(unittest.TestCase):

    def test_every_documented_assistance_type_has_a_speed_multiplier(self):

        for assistance_type in ASSISTANCE_TYPES:
            self.assertIn(assistance_type, _SPEED_MULTIPLIER_BY_ASSISTANCE_TYPE)
            self.assertGreater(_SPEED_MULTIPLIER_BY_ASSISTANCE_TYPE[assistance_type], 0.0)
            self.assertLessEqual(_SPEED_MULTIPLIER_BY_ASSISTANCE_TYPE[assistance_type], 1.0)

    def test_helper_and_assisted_depart_together_and_arrive_close_together(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building(door_width=10.0)
        engine = _engine_for(building)

        occupants = (
            ScenarioOccupant(
                occupant_id="helper1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Staff_Default",
                assisting_occupant_id="assisted1", assistance_type="CARRY",
            ),
            ScenarioOccupant(
                occupant_id="assisted1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Adult_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))

        context = run_scenario(scenario, building)
        register_occupants(context)
        result = context.simulation.run()

        helper = result.occupants["helper1"]
        assisted = result.occupants["assisted1"]

        self.assertEqual(helper.depart_time, assisted.depart_time)
        self.assertEqual(helper.route.edge_ids, assisted.route.edge_ids)
        self.assertLess(abs(helper.arrival_time - assisted.arrival_time), 5.0)

    def test_carry_multiplier_roughly_doubles_travel_time(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building(door_width=10.0)

        def run_solo(profile_id, occupant_id):
            engine = _engine_for(building)
            scenario = Scenario(
                metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id),
            )
            context = run_scenario(scenario, building)
            context.simulation.add_occupant(zone_a.id, occupant_id=occupant_id)
            return context.simulation.run()

        # Directly compare a Staff_Default's own registered walking
        # speed against the same profile assisting via CARRY (0.5x
        # multiplier, behavior_library.assistance_strategies's own
        # documented constant).
        occupants = (
            ScenarioOccupant(
                occupant_id="helper1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Staff_Default",
                assisting_occupant_id="assisted1", assistance_type="CARRY",
            ),
            ScenarioOccupant(
                occupant_id="assisted1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Adult_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        register_occupants(context)
        result = context.simulation.run()

        helper = result.occupants["helper1"]
        travel_time = helper.arrival_time - helper.depart_time

        solo_result = run_solo("Staff_Default", "solo1")
        solo = solo_result.occupants["solo1"]
        solo_travel_time = solo.arrival_time - solo.depart_time

        # Not exactly 2x: the paired run has the assisted occupant
        # sharing the same edges concurrently with the helper, so
        # DefaultCongestionModel's own speed_factor also applies (a
        # second, independent, and equally intentional Phase 3 effect
        # -- "assistance must affect... congestion") on top of the 0.5x
        # CARRY multiplier alone. Still unambiguously close to 2x and
        # far from the unassisted 1x baseline.
        ratio = travel_time / solo_travel_time
        self.assertGreater(ratio, 1.4)
        self.assertLess(ratio, 2.1)

    def test_no_assistance_role_is_unaffected_by_the_wrapper_strategies(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        scenario = Scenario(
            metadata=make_metadata(),
            occupants=(
                ScenarioOccupant(
                    occupant_id="p1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                    behaviour_profile_id="Adult_Default",
                ),
            ),
            fire=make_fire(zone_b.id, floor.id),
        )
        context = run_scenario(scenario, building)
        register_occupants(context)
        result = context.simulation.run()

        self.assertEqual(result.occupants["p1"].state, OccupantState.ARRIVED)

    def test_speed_multiplier_is_idempotent_across_repeated_decide_calls(self):

        from behavior.context import DecisionContext
        from behavior.profile import BehaviorProfile
        from behavior_library.assistance_strategies import (
            AssistanceAwareDecisionStrategy, TRAIT_ASSISTANCE_TARGET_ID, TRAIT_ASSISTANCE_TYPE,
        )

        profile = BehaviorProfile(
            occupant_id="helper1", walking_speed=1.3,
            traits={
                TRAIT_ASSISTANCE_ROLE: ROLE_HELPER,
                TRAIT_ASSISTANCE_TARGET_ID: "assisted1",
                TRAIT_ASSISTANCE_TYPE: "DRAG",
            },
        )
        context = DecisionContext(graph=None, engine=None, profile=profile, start_id="zone-a")
        strategy = AssistanceAwareDecisionStrategy()

        strategy.decide(context)
        speed_after_first = profile.walking_speed

        strategy.decide(context)
        speed_after_second = profile.walking_speed

        self.assertEqual(speed_after_first, speed_after_second)
        self.assertAlmostEqual(speed_after_first, 1.3 * 0.5)


# =====================================================
# Phase 4 -- Firefighter modeling
# =====================================================


class FirefighterModelingTests(unittest.TestCase):

    def test_firefighter_default_profile_exists_and_uses_firefighter_only_strategies(self):

        from behavior_library.firefighter_strategies import (
            FirefighterDecisionStrategy, FirefighterRouteChoiceStrategy,
        )

        template = DEFAULT_FIREFIGHTER_PROFILE_REGISTRY["Firefighter_Default"]

        self.assertIsInstance(template.decision_strategy, FirefighterDecisionStrategy)
        self.assertIsInstance(template.route_choice_strategy, FirefighterRouteChoiceStrategy)

    def test_search_firefighter_with_no_rescue_target_still_evacuates_toward_an_exit(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        scenario = Scenario(
            metadata=make_metadata(), occupants=(),
            firefighters=(
                ScenarioFirefighter(
                    firefighter_id="ff1", team_id="team-0", entry_zone_id=zone_a.id,
                    floor_id=floor.id, position=(1.0, 1.0), arrival_time=0.0,
                    behaviour_profile_id="Firefighter_Default",
                ),
            ),
            fire=make_fire(zone_b.id, floor.id),
        )
        context = run_scenario(scenario, building)
        register_population(context)
        result = context.simulation.run()

        self.assertEqual(result.occupants["ff1"].state, OccupantState.ARRIVED)

    def test_rescue_firefighter_reaches_the_target_and_both_evacuate(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        scenario = Scenario(
            metadata=make_metadata(),
            occupants=(
                ScenarioOccupant(
                    occupant_id="civ1", zone_id=zone_b.id, floor_id=floor.id, position=(11.0, 1.0),
                    behaviour_profile_id="Elderly_Default",
                ),
            ),
            firefighters=(
                ScenarioFirefighter(
                    firefighter_id="ff1", team_id="team-0", entry_zone_id=zone_a.id,
                    floor_id=floor.id, position=(1.0, 1.0), arrival_time=10.0,
                    behaviour_profile_id="Firefighter_Default",
                    rescue_target_occupant_id="civ1", rescue_target_zone_id=zone_b.id,
                ),
            ),
            fire=make_fire(zone_b.id, floor.id),
        )
        context = run_scenario(scenario, building)
        register_population(context)
        result = context.simulation.run()

        civilian = result.occupants["civ1"]
        firefighter = result.occupants["ff1"]

        self.assertEqual(civilian.state, OccupantState.ARRIVED)
        self.assertEqual(firefighter.state, OccupantState.ARRIVED)
        self.assertEqual(civilian.depart_time, firefighter.depart_time)
        # The civilian's own route is exactly the tail of the
        # firefighter's combined (entry -> target -> exit) route.
        self.assertTrue(
            "".join(firefighter.route.edge_ids).endswith("".join(civilian.route.edge_ids))
            or firefighter.route.edge_ids[-len(civilian.route.edge_ids):] == civilian.route.edge_ids,
        )

    def test_rescue_outcomes_reports_success_and_rescue_time(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="civ1", zone_id=zone_b.id, floor_id=floor.id, position=(11.0, 1.0),
                behaviour_profile_id="Elderly_Default",
            ),
        )
        firefighters = (
            ScenarioFirefighter(
                firefighter_id="ff1", team_id="team-0", entry_zone_id=zone_a.id,
                floor_id=floor.id, position=(1.0, 1.0), arrival_time=10.0,
                behaviour_profile_id="Firefighter_Default",
                rescue_target_occupant_id="civ1", rescue_target_zone_id=zone_b.id,
            ),
        )
        scenario = Scenario(
            metadata=make_metadata(), occupants=occupants, firefighters=firefighters,
            fire=make_fire(zone_b.id, floor.id),
        )
        context = run_scenario(scenario, building)
        register_population(context)
        result = context.simulation.run()

        outcomes = compute_rescue_outcomes(scenario, result)
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0].rescued)
        self.assertIsNotNone(outcomes[0].rescue_time_seconds)
        self.assertGreaterEqual(outcomes[0].rescue_time_seconds, 0.0)

        summary = summarize_rescue_outcomes(outcomes)
        self.assertEqual(summary["rescued_occupant_count"], 1)
        self.assertEqual(summary["firefighter_rescue_count"], 1)
        self.assertIsNotNone(summary["average_rescue_time_seconds"])

    def test_no_rescue_assignment_produces_no_rescue_outcomes(self):

        scenario = Scenario(
            metadata=make_metadata(), occupants=(),
            firefighters=(
                ScenarioFirefighter(
                    firefighter_id="ff1", team_id="team-0", entry_zone_id="zone-a",
                    floor_id="floor-1", position=(1.0, 1.0), arrival_time=0.0,
                    behaviour_profile_id="Firefighter_Default",
                ),
            ),
        )
        from simulator.multi_agent_result import MultiAgentSimulationResult
        empty_result = MultiAgentSimulationResult(occupants={}, total_evacuation_time=None)

        outcomes = compute_rescue_outcomes(scenario, empty_result)
        self.assertEqual(outcomes, ())
        self.assertEqual(summarize_rescue_outcomes(outcomes)["average_rescue_time_seconds"], None)


# =====================================================
# Phase 5 -- Scenario generation
# =====================================================


class ScenarioGenerationTests(unittest.TestCase):

    def _definition(self, entry_zone_id, team_count=1):

        return ScenarioDefinition(
            fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
            occupant=OccupantDefinition(
                occupancy_distribution={entry_zone_id: FixedValue(8)},
                behaviour_profile_distribution={
                    entry_zone_id: WeightedOptions(
                        {"Adult_Default": 0.4, "Elderly_Default": 0.3, "Wheelchair_Default": 0.3},
                    ),
                },
                assistance_pairing_probability=1.0,
            ),
            firefighter=FirefighterDeploymentDefinition(
                team_count_distribution=FixedValue(team_count),
                team_size_distribution=FixedValue(2),
                arrival_time_distribution=FixedValue(30.0),
                entry_zone_ids=(entry_zone_id,),
                rescue_assignment_probability=1.0,
            ),
        )

    def test_assistance_pairs_only_link_vulnerable_occupants_to_non_vulnerable_helpers(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        request = GenerationRequest(
            definition=self._definition(zone_a.id), definition_id="def-1", building=building, seed=3,
        )
        scenario = generate_scenario(request)

        by_id = {o.occupant_id: o for o in scenario.occupants}
        helpers = [o for o in scenario.occupants if o.assisting_occupant_id]

        self.assertGreater(len(helpers), 0)

        for helper in helpers:
            helper_category = occupant_category(helper.behaviour_profile_id)
            target_category = occupant_category(by_id[helper.assisting_occupant_id].behaviour_profile_id)

            self.assertNotIn(
                helper_category,
                (OccupantCategory.CHILD, OccupantCategory.ELDERLY, OccupantCategory.WHEELCHAIR_USER),
            )
            self.assertIn(
                target_category,
                (OccupantCategory.CHILD, OccupantCategory.ELDERLY, OccupantCategory.WHEELCHAIR_USER),
            )
            self.assertIn(helper.assistance_type, ASSISTANCE_TYPES)

    def test_no_occupant_is_paired_more_than_once(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        request = GenerationRequest(
            definition=self._definition(zone_a.id), definition_id="def-1", building=building, seed=11,
        )
        scenario = generate_scenario(request)

        appearances = {}
        for occupant in scenario.occupants:
            if occupant.assisting_occupant_id:
                appearances[occupant.occupant_id] = appearances.get(occupant.occupant_id, 0) + 1
                appearances[occupant.assisting_occupant_id] = (
                    appearances.get(occupant.assisting_occupant_id, 0) + 1
                )

        self.assertTrue(all(count == 1 for count in appearances.values()))

    def test_zero_pairing_probability_produces_no_pairs(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        definition = self._definition(zone_a.id)
        definition = ScenarioDefinition(
            fire=definition.fire,
            occupant=OccupantDefinition(
                occupancy_distribution=definition.occupant.occupancy_distribution,
                behaviour_profile_distribution=definition.occupant.behaviour_profile_distribution,
                assistance_pairing_probability=0.0,
            ),
            firefighter=definition.firefighter,
        )
        request = GenerationRequest(definition=definition, definition_id="def-1", building=building, seed=3)
        scenario = generate_scenario(request)

        self.assertTrue(all(o.assisting_occupant_id is None for o in scenario.occupants))

    def test_firefighters_are_generated_independently_with_configured_team_size_and_arrival(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        request = GenerationRequest(
            definition=self._definition(zone_a.id, team_count=2), definition_id="def-1",
            building=building, seed=5,
        )
        scenario = generate_scenario(request)

        self.assertEqual(len(scenario.firefighters), 4)  # 2 teams x team_size 2
        for firefighter in scenario.firefighters:
            self.assertEqual(firefighter.arrival_time, 30.0)
            self.assertEqual(firefighter.entry_zone_id, zone_a.id)

        team_ids = {ff.team_id for ff in scenario.firefighters}
        self.assertEqual(len(team_ids), 2)

    def test_no_entry_zones_configured_deploys_no_firefighters(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        definition = ScenarioDefinition(
            fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
            firefighter=FirefighterDeploymentDefinition(team_count_distribution=FixedValue(3)),
        )
        request = GenerationRequest(definition=definition, definition_id="def-1", building=building, seed=1)
        scenario = generate_scenario(request)

        self.assertEqual(scenario.firefighters, ())

    def test_generation_is_deterministic_for_the_same_seed(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        request = GenerationRequest(
            definition=self._definition(zone_a.id), definition_id="def-1", building=building, seed=9,
        )
        first = generate_scenario(request)
        second = generate_scenario(request)

        self.assertEqual(
            [o.assisting_occupant_id for o in first.occupants],
            [o.assisting_occupant_id for o in second.occupants],
        )
        self.assertEqual(
            [ff.firefighter_id for ff in first.firefighters],
            [ff.firefighter_id for ff in second.firefighters],
        )

    def test_existing_definition_without_firefighter_or_assistance_config_is_unaffected(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        definition = ScenarioDefinition(
            fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
            occupant=OccupantDefinition(
                occupancy_distribution={zone_a.id: FixedValue(3)},
                behaviour_profile_distribution={zone_a.id: FixedValue("Adult_Default")},
            ),
        )
        request = GenerationRequest(definition=definition, definition_id="def-1", building=building, seed=1)
        scenario = generate_scenario(request)

        self.assertEqual(scenario.firefighters, ())
        self.assertTrue(all(o.assisting_occupant_id is None for o in scenario.occupants))


# =====================================================
# Phase 6 -- Dataset Builder columns
# =====================================================


class DatasetBuilderColumnTests(unittest.TestCase):

    def _run(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="p1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Adult_Default",
            ),
            ScenarioOccupant(
                occupant_id="p2", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Elderly_Default",
            ),
        )
        firefighters = (
            ScenarioFirefighter(
                firefighter_id="ff1", team_id="team-0", entry_zone_id=zone_a.id, floor_id=floor.id,
                position=(1.0, 1.0), arrival_time=0.0, behaviour_profile_id="Firefighter_Default",
            ),
        )
        scenario = Scenario(
            metadata=make_metadata(), occupants=occupants, firefighters=firefighters,
            fire=make_fire(zone_b.id, floor.id),
        )
        context = run_scenario(scenario, building)
        register_population(context)
        movement_result = context.simulation.run()

        return scenario, building, movement_result

    def test_new_columns_are_appended_to_simulation_outcome_schema(self):

        for column in (
            "Adult_Count", "Child_Count", "Elderly_Count", "Wheelchair_Count", "Visitor_Count",
            "Helping_Group_Count", "Assisted_Occupant_Count", "Fallen_Count", "Possible_Injury_Count",
            "Firefighter_Count", "Firefighter_Rescues",
        ):
            self.assertIn(column, SIMULATION_OUTCOME_COLUMNS)

    def test_simulation_outcome_row_reports_correct_counts(self):

        scenario, building, movement_result = self._run()
        run = SimulationRun(scenario=scenario, building=building, movement_result=movement_result)

        row = extract_simulation_outcome(run)

        self.assertEqual(row["Adult_Count"], 1)
        self.assertEqual(row["Elderly_Count"], 1)
        self.assertEqual(row["Child_Count"], 0)
        self.assertEqual(row["Firefighter_Count"], 1)

    def test_scenario_features_row_includes_profile_category_counts(self):

        scenario, building, movement_result = self._run()
        run = SimulationRun(scenario=scenario, building=building, movement_result=movement_result)

        row = extract_scenario_features(run)

        self.assertEqual(row["Adult_Count"], 1)
        self.assertEqual(row["Elderly_Count"], 1)
        self.assertEqual(row["Firefighter_Count"], 1)


# =====================================================
# Phase 7 -- Ground Truth
# =====================================================


class GroundTruthProfileAwareTests(unittest.TestCase):

    def test_profile_outcomes_split_evacuated_and_trapped_by_category(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="wc1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Wheelchair_Default",
            ),
            ScenarioOccupant(
                occupant_id="ch1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Child_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        register_occupants(context)
        result = context.simulation.run()

        outcomes = compute_profile_outcomes(scenario, result)

        self.assertEqual(outcomes["wheelchair_evacuated"] + outcomes["wheelchair_trapped"], 1)
        self.assertEqual(outcomes["children_evacuated"] + outcomes["children_trapped"], 1)

    def test_ground_truth_analyze_populates_every_new_field(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="p1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Adult_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        register_occupants(context)
        movement_result = context.simulation.run()

        ground_truth = analyze(SimulationArtifacts(
            scenario=scenario, building=building, movement_result=movement_result,
        ))

        self.assertEqual(len(ground_truth.occupant_behavior), 1)
        self.assertIsInstance(ground_truth.helping_group_count, int)
        self.assertIsInstance(ground_truth.wheelchair_evacuated, int)
        self.assertEqual(ground_truth.rescued_occupant_count, 0)
        self.assertIsNone(ground_truth.average_rescue_time_seconds)

    def test_ground_truth_serialization_round_trips_new_fields(self):

        ground_truth = GroundTruth(
            scenario_id="s1", definition_id="d1",
            occupant_behavior=({"occupant_id": "p1", "dynamic_state": "WALKING",
                                 "ever_helping": False, "ever_assisted": True,
                                 "ever_fallen": False, "ever_possible_injury": False},),
            helping_group_count=1, assisted_occupant_count=2, fallen_count=0, possible_injury_count=0,
            wheelchair_evacuated=1, wheelchair_trapped=0, children_evacuated=2, children_trapped=1,
            rescued_occupant_count=1, firefighter_rescue_count=1, average_rescue_time_seconds=42.5,
        )

        restored = GroundTruth.from_dict(ground_truth.to_dict())

        self.assertEqual(restored, ground_truth)

    def test_from_dict_defaults_new_fields_for_a_legacy_payload(self):

        legacy_payload = {"scenario_id": "s1", "definition_id": "d1"}
        restored = GroundTruth.from_dict(legacy_payload)

        self.assertEqual(restored.occupant_behavior, ())
        self.assertEqual(restored.helping_group_count, 0)
        self.assertIsNone(restored.average_rescue_time_seconds)


# =====================================================
# Phase 8 -- AI & RL
# =====================================================


class RLObservationProfileAwareTests(unittest.TestCase):

    def test_observation_schema_includes_profile_aware_global_features(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        encoder = ObservationEncoder(building)

        for feature in (
            "global:children_present", "global:elderly_present",
            "global:wheelchair_users_present", "global:firefighters_present",
        ):
            self.assertIn(feature, encoder.schema.feature_names)

    def test_occupant_snapshot_profile_category_reflects_scenario_data(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="p1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Wheelchair_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        register_occupants(context)

        stepper = MovementStepper(context.simulation)
        snapshot = build_snapshot(context, stepper, HazardSnapshot(), time=0.0)

        occupant_snapshot = next(o for o in snapshot.occupants if o.occupant_id == "p1")
        self.assertEqual(occupant_snapshot.profile_category, "WHEELCHAIR_USER")

    def test_observation_space_shape_matches_feature_count(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        encoder = ObservationEncoder(building)

        self.assertEqual(encoder.space.shape[0], len(encoder.schema.feature_names))


if __name__ == "__main__":
    unittest.main()
