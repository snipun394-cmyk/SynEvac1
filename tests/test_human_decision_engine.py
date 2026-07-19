import unittest

from behavior.context import DecisionContext
from behavior.profile import BehaviorProfile

from behaviour_profile_resolver.category import OccupantCategory
from behaviour_profile_resolver.dynamic_registrar import register_population_dynamic
from behaviour_profile_resolver.registrar import register_occupants

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from scenario.fire import ScenarioFire
from scenario.firefighter import ScenarioFirefighter
from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from scenario_runner import run as run_scenario

from simulator.occupant import OccupantState

from command_center.incident_data import IncidentFrame

from dataset_builder.builder import SimulationRun
from dataset_builder.labels import extract_simulation_outcome
from dataset_builder.schema import SIMULATION_OUTCOME_COLUMNS

from ground_truth.analyzer import SimulationArtifacts, analyze
from ground_truth.decision_events import summarize_decision_events
from ground_truth.labels import GroundTruth

from human_decision_engine.engine import (
    ACTION_BE_ASSISTED,
    ACTION_HELP,
    ACTION_IGNORE,
    ACTION_DELAY,
    HumanDecisionEngine,
)
from human_decision_engine.events import DecisionEventLog, GROUP_FORMED, HELP_DECISION, HELP_REJECTED
from human_decision_engine.firefighter_engine import (
    CivilianRosterEntry,
    FirefighterDecisionEngine,
    TASK_ASSIST_WHEELCHAIR_USER,
    TASK_CARRY_FALLEN,
    TASK_CONTINUE_SEARCH,
    TASK_GUIDE_CIVILIANS,
    TASK_REPORT_HAZARD,
    TASK_RESCUE,
    TASK_RETURN_OUTSIDE,
    TASK_SEARCH_UNCLEARED_ZONE,
)
from human_decision_engine.groups import GroupRegistry, compute_group_dissolution
from human_decision_engine.pairing import (
    PAIRING_ASSIST,
    PAIRING_GROUP_FOLLOW,
    compute_dynamic_pairings,
    ordered_for_dynamic_pairing,
)
from human_decision_engine.priority import RescuePriorityFactors, compute_rescue_priority
from human_decision_engine.view import build_human_decisions_view


# =====================================================
# Shared fixtures -- mirrors tests.test_human_population's own
# build_two_zone_building()/make_metadata()/make_fire().
# =====================================================


def build_two_zone_building(door_width=5.0):

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


def _hazard_snapshot(zone_id, hazard_score):

    return HazardSnapshot(node_states={zone_id: HazardNodeState(hazard_score=hazard_score)})


# =====================================================
# Phase 2 -- Pairing (structural, perceivable facts)
# =====================================================


class PairingTests(unittest.TestCase):

    def _occupant(self, occupant_id, zone_id, profile_id):

        return ScenarioOccupant(
            occupant_id=occupant_id, zone_id=zone_id, floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id=profile_id,
        )

    def test_adult_paired_with_wheelchair_user_in_same_zone(self):

        occupants = (
            self._occupant("adult1", "zone-a", "Adult_Default"),
            self._occupant("wheel1", "zone-a", "Wheelchair_Default"),
        )
        pairings = compute_dynamic_pairings(occupants)

        self.assertEqual(len(pairings), 1)
        self.assertEqual(pairings[0].helper_id, "adult1")
        self.assertEqual(pairings[0].target_id, "wheel1")
        self.assertEqual(pairings[0].kind, PAIRING_ASSIST)

    def test_child_paired_with_adult_as_group_follow(self):

        occupants = (
            self._occupant("adult1", "zone-a", "Adult_Default"),
            self._occupant("child1", "zone-a", "Child_Default"),
        )
        pairings = compute_dynamic_pairings(occupants)

        self.assertEqual(len(pairings), 1)
        self.assertEqual(pairings[0].kind, PAIRING_GROUP_FOLLOW)

    def test_different_zones_never_paired(self):

        occupants = (
            self._occupant("adult1", "zone-a", "Adult_Default"),
            self._occupant("wheel1", "zone-b", "Wheelchair_Default"),
        )
        self.assertEqual(compute_dynamic_pairings(occupants), ())

    def test_one_helper_claims_at_most_one_target(self):

        occupants = (
            self._occupant("adult1", "zone-a", "Adult_Default"),
            self._occupant("wheel1", "zone-a", "Wheelchair_Default"),
            self._occupant("wheel2", "zone-a", "Wheelchair_Default"),
        )
        pairings = compute_dynamic_pairings(occupants)

        self.assertEqual(len(pairings), 1)
        claimed_targets = {p.target_id for p in pairings}
        self.assertEqual(claimed_targets, {"wheel1"})

    def test_deterministic_across_repeated_calls(self):

        occupants = (
            self._occupant("adult1", "zone-a", "Adult_Default"),
            self._occupant("wheel1", "zone-a", "Wheelchair_Default"),
        )
        self.assertEqual(compute_dynamic_pairings(occupants), compute_dynamic_pairings(occupants))

    def test_ordered_for_dynamic_pairing_places_helper_before_target(self):

        occupants = (
            self._occupant("wheel1", "zone-a", "Wheelchair_Default"),
            self._occupant("adult1", "zone-a", "Adult_Default"),
        )
        pairings = compute_dynamic_pairings(occupants)
        ordered = ordered_for_dynamic_pairing(occupants, pairings)

        ids = [o.occupant_id for o in ordered]
        self.assertLess(ids.index("adult1"), ids.index("wheel1"))


# =====================================================
# Phase 4 -- Deterministic priority scoring
# =====================================================


class PriorityTests(unittest.TestCase):

    def test_ordinary_adult_scores_zero(self):

        factors = RescuePriorityFactors(category=OccupantCategory.ADULT)
        self.assertEqual(compute_rescue_priority(factors), 0.0)

    def test_child_outranks_adult_with_identical_other_factors(self):

        child_score = compute_rescue_priority(RescuePriorityFactors(category=OccupantCategory.CHILD))
        adult_score = compute_rescue_priority(RescuePriorityFactors(category=OccupantCategory.ADULT))
        self.assertGreater(child_score, adult_score)

    def test_fallen_outweighs_possible_injury_alone(self):

        fallen = compute_rescue_priority(
            RescuePriorityFactors(category=OccupantCategory.ADULT, fallen=True),
        )
        injured = compute_rescue_priority(
            RescuePriorityFactors(category=OccupantCategory.ADULT, possible_injury=True),
        )
        self.assertGreater(fallen, injured)

    def test_smoke_exposure_and_distance_increase_score_monotonically(self):

        low = compute_rescue_priority(
            RescuePriorityFactors(category=OccupantCategory.ADULT, smoke_exposure=0.1, distance_from_hazard=0.9),
        )
        high = compute_rescue_priority(
            RescuePriorityFactors(category=OccupantCategory.ADULT, smoke_exposure=0.9, distance_from_hazard=0.1),
        )
        self.assertGreater(high, low)

    def test_deterministic_for_identical_factors(self):

        factors = RescuePriorityFactors(
            category=OccupantCategory.WHEELCHAIR_USER, is_wheelchair_user=True, isolation_score=1.0,
        )
        self.assertEqual(compute_rescue_priority(factors), compute_rescue_priority(factors))


# =====================================================
# Phase 1/2 -- Civilian decision engine
# =====================================================


class CivilianEngineTests(unittest.TestCase):

    def _context(self, occupant_id, zone_id, hazard_score=None):

        profile = BehaviorProfile(occupant_id=occupant_id, walking_speed=1.2)
        hazard_snapshot = _hazard_snapshot(zone_id, hazard_score) if hazard_score is not None else None

        return DecisionContext(
            graph=None, engine=None, profile=profile, start_id=zone_id, hazard_snapshot=hazard_snapshot,
        )

    def _pairings(self):

        from human_decision_engine.pairing import AssistancePairing

        return (AssistancePairing(helper_id="adult1", target_id="wheel1", kind=PAIRING_ASSIST),)

    def test_helper_helps_when_zone_is_clear(self):

        engine = HumanDecisionEngine(self._pairings())
        decision = engine.evaluate(self._context("adult1", "zone-a", hazard_score=0.0))

        self.assertEqual(decision.action, ACTION_HELP)
        self.assertEqual(decision.target_id, "wheel1")

    def test_helper_delays_under_high_hazard(self):

        engine = HumanDecisionEngine(self._pairings())
        decision = engine.evaluate(self._context("adult1", "zone-a", hazard_score=0.7))

        self.assertEqual(decision.action, ACTION_DELAY)

    def test_helper_ignores_under_critical_hazard(self):

        engine = HumanDecisionEngine(self._pairings())
        decision = engine.evaluate(self._context("adult1", "zone-a", hazard_score=0.9))

        self.assertEqual(decision.action, ACTION_IGNORE)

    def test_target_is_assisted_only_when_helper_actually_helped(self):

        engine = HumanDecisionEngine(self._pairings())
        engine.evaluate(self._context("adult1", "zone-a", hazard_score=0.0))
        target_decision = engine.evaluate(self._context("wheel1", "zone-a"))

        self.assertEqual(target_decision.action, ACTION_BE_ASSISTED)
        self.assertEqual(target_decision.target_id, "adult1")

    def test_target_proceeds_independently_when_helper_declined(self):

        from human_decision_engine.engine import ACTION_EVACUATE

        engine = HumanDecisionEngine(self._pairings())
        engine.evaluate(self._context("adult1", "zone-a", hazard_score=0.9))  # IGNORE
        target_decision = engine.evaluate(self._context("wheel1", "zone-a"))

        self.assertEqual(target_decision.action, ACTION_EVACUATE)

    def test_unpaired_occupant_evacuates_independently(self):

        from human_decision_engine.engine import ACTION_EVACUATE

        engine = HumanDecisionEngine(())
        decision = engine.evaluate(self._context("solo1", "zone-a"))

        self.assertEqual(decision.action, ACTION_EVACUATE)

    def test_evaluate_is_cached_and_idempotent(self):

        engine = HumanDecisionEngine(self._pairings())
        context = self._context("adult1", "zone-a", hazard_score=0.0)

        first = engine.evaluate(context)
        second = engine.evaluate(context)

        self.assertIs(first, second)

    def test_help_and_reject_events_are_logged(self):

        event_log = DecisionEventLog()
        engine = HumanDecisionEngine(self._pairings(), event_log=event_log)
        engine.evaluate(self._context("adult1", "zone-a", hazard_score=0.0))

        event_types = [event.event_type for event in event_log.events]
        self.assertIn(HELP_DECISION, event_types)
        self.assertNotIn(HELP_REJECTED, event_types)


# =====================================================
# Phase 3 -- Firefighter decision engine
# =====================================================


class FirefighterEngineTests(unittest.TestCase):

    def _context(self, occupant_id, zone_id, hazard_score=None):

        profile = BehaviorProfile(occupant_id=occupant_id, walking_speed=1.4)
        hazard_snapshot = _hazard_snapshot(zone_id, hazard_score) if hazard_score is not None else None

        return DecisionContext(
            graph=None, engine=None, profile=profile, start_id=zone_id, hazard_snapshot=hazard_snapshot,
        )

    def test_claims_the_highest_priority_candidate(self):

        roster = (
            CivilianRosterEntry("adult1", "zone-a", OccupantCategory.ADULT),
            CivilianRosterEntry("child1", "zone-a", OccupantCategory.CHILD),
        )
        engine = FirefighterDecisionEngine(roster)
        decision = engine.evaluate(self._context("ff1", "zone-a"))

        self.assertEqual(decision.target_occupant_id, "child1")

    def test_fallen_civilian_produces_carry_fallen_task(self):

        roster = (CivilianRosterEntry("adult1", "zone-a", OccupantCategory.ADULT),)
        engine = FirefighterDecisionEngine(roster, known_fallen_ids=frozenset({"adult1"}))
        decision = engine.evaluate(self._context("ff1", "zone-a"))

        self.assertEqual(decision.task, TASK_CARRY_FALLEN)

    def test_wheelchair_user_produces_assist_wheelchair_task(self):

        roster = (CivilianRosterEntry("wheel1", "zone-a", OccupantCategory.WHEELCHAIR_USER),)
        engine = FirefighterDecisionEngine(roster)
        decision = engine.evaluate(self._context("ff1", "zone-a"))

        self.assertEqual(decision.task, TASK_ASSIST_WHEELCHAIR_USER)

    def test_ordinary_child_produces_rescue_task(self):

        roster = (CivilianRosterEntry("child1", "zone-a", OccupantCategory.CHILD),)
        engine = FirefighterDecisionEngine(roster)
        decision = engine.evaluate(self._context("ff1", "zone-a"))

        self.assertEqual(decision.task, TASK_RESCUE)

    def test_two_firefighters_never_claim_the_same_civilian(self):

        roster = (CivilianRosterEntry("child1", "zone-a", OccupantCategory.CHILD),)
        engine = FirefighterDecisionEngine(roster)

        first = engine.evaluate(self._context("ff1", "zone-a"))
        second = engine.evaluate(self._context("ff2", "zone-a"))

        self.assertEqual(first.target_occupant_id, "child1")
        self.assertIsNone(second.target_occupant_id)

    def test_no_candidates_first_decision_is_search(self):

        engine = FirefighterDecisionEngine(())
        decision = engine.evaluate(self._context("ff1", "zone-a"))

        self.assertEqual(decision.task, TASK_SEARCH_UNCLEARED_ZONE)

    def test_report_hazard_when_own_zone_is_severely_hazardous(self):

        engine = FirefighterDecisionEngine(())
        decision = engine.evaluate(self._context("ff1", "zone-a", hazard_score=0.9))

        self.assertEqual(decision.task, TASK_REPORT_HAZARD)

    def test_priority_scores_covers_the_full_roster_regardless_of_claims(self):

        roster = (
            CivilianRosterEntry("adult1", "zone-a", OccupantCategory.ADULT),
            CivilianRosterEntry("child1", "zone-a", OccupantCategory.CHILD),
        )
        engine = FirefighterDecisionEngine(roster)
        engine.evaluate(self._context("ff1", "zone-a"))

        scores = engine.priority_scores(hazard_snapshot=None)
        self.assertIn("adult1", scores)
        self.assertIn("child1", scores)


# =====================================================
# Phase 5 -- Group behavior
# =====================================================


class GroupTests(unittest.TestCase):

    def test_form_and_group_for(self):

        registry = GroupRegistry()
        group_id = registry.form(("adult1", "wheel1"), "ASSIST")

        group = registry.group_for("wheel1")
        self.assertIsNotNone(group)
        self.assertEqual(group.group_id, group_id)
        self.assertEqual(group.member_ids, ("adult1", "wheel1"))

    def test_form_logs_group_formed_event(self):

        event_log = DecisionEventLog()
        registry = GroupRegistry(event_log=event_log)
        registry.form(("adult1", "wheel1"), "ASSIST")

        self.assertEqual(event_log.events[0].event_type, GROUP_FORMED)

    def test_dissolution_when_both_members_arrive(self):

        from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline
        from simulator.occupant import OccupantState

        registry = GroupRegistry()
        registry.form(("adult1", "wheel1"), "ASSIST")

        result = MultiAgentSimulationResult(
            occupants={
                "adult1": OccupantTimeline(
                    occupant_id="adult1", route=None, steps=[],
                    state=OccupantState.ARRIVED, depart_time=0.0, arrival_time=10.0,
                ),
                "wheel1": OccupantTimeline(
                    occupant_id="wheel1", route=None, steps=[],
                    state=OccupantState.ARRIVED, depart_time=0.0, arrival_time=12.0,
                ),
            },
            total_evacuation_time=12.0,
        )
        records = compute_group_dissolution(registry.groups, result)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["dissolved"])
        self.assertEqual(records[0]["dissolution_time"], 12.0)

    def test_no_dissolution_when_a_member_never_terminates(self):

        from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline
        from simulator.occupant import OccupantState

        registry = GroupRegistry()
        registry.form(("adult1", "wheel1"), "ASSIST")

        result = MultiAgentSimulationResult(
            occupants={
                "adult1": OccupantTimeline(
                    occupant_id="adult1", route=None, steps=[],
                    state=OccupantState.ARRIVED, depart_time=0.0, arrival_time=10.0,
                ),
                "wheel1": OccupantTimeline(
                    occupant_id="wheel1", route=None, steps=[],
                    state=OccupantState.TRAVERSING, depart_time=0.0, arrival_time=None,
                ),
            },
            total_evacuation_time=None,
        )
        records = compute_group_dissolution(registry.groups, result)

        self.assertFalse(records[0]["dissolved"])
        self.assertIsNone(records[0]["dissolution_time"])


# =====================================================
# End-to-end -- register_population_dynamic
# =====================================================


class DynamicRegistrarEndToEndTests(unittest.TestCase):

    def test_adult_dynamically_helps_wheelchair_user_in_same_zone(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="adult1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Adult_Default",
            ),
            ScenarioOccupant(
                occupant_id="wheel1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Wheelchair_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)

        registration_result = register_population_dynamic(context)
        result = context.simulation.run()

        adult = result.occupants["adult1"]
        wheelchair = result.occupants["wheel1"]

        self.assertEqual(adult.state, OccupantState.ARRIVED)
        self.assertEqual(wheelchair.state, OccupantState.ARRIVED)
        self.assertEqual(adult.depart_time, wheelchair.depart_time)

        event_types = {event.event_type for event in registration_result.events}
        self.assertIn(HELP_DECISION, event_types)
        self.assertIn(GROUP_FORMED, event_types)
        self.assertTrue(registration_result.groups)
        self.assertIn("adult1", registration_result.priority_scores)
        self.assertIn("wheel1", registration_result.priority_scores)

    def test_unpaired_occupant_still_evacuates(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="solo1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Adult_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)

        register_population_dynamic(context)
        result = context.simulation.run()

        self.assertEqual(result.occupants["solo1"].state, OccupantState.ARRIVED)

    def test_firefighter_dynamically_rescues_a_civilian_in_a_different_zone(self):

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
            ),
        )
        scenario = Scenario(
            metadata=make_metadata(), occupants=occupants, firefighters=firefighters,
            fire=make_fire(zone_b.id, floor.id),
        )
        context = run_scenario(scenario, building)

        registration_result = register_population_dynamic(context)
        result = context.simulation.run()

        civilian = result.occupants["civ1"]
        firefighter = result.occupants["ff1"]

        self.assertEqual(civilian.state, OccupantState.ARRIVED)
        self.assertEqual(firefighter.state, OccupantState.ARRIVED)
        self.assertEqual(civilian.depart_time, firefighter.depart_time)

        decision = registration_result.decisions_by_occupant_id["ff1"]
        self.assertEqual(decision.target_occupant_id, "civ1")

    def test_search_firefighter_with_no_candidates_still_evacuates(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        firefighters = (
            ScenarioFirefighter(
                firefighter_id="ff1", team_id="team-0", entry_zone_id=zone_a.id,
                floor_id=floor.id, position=(1.0, 1.0), arrival_time=0.0,
                behaviour_profile_id="Firefighter_Default",
            ),
        )
        scenario = Scenario(
            metadata=make_metadata(), occupants=(), firefighters=firefighters,
            fire=make_fire(zone_b.id, floor.id),
        )
        context = run_scenario(scenario, building)

        register_population_dynamic(context)
        result = context.simulation.run()

        self.assertEqual(result.occupants["ff1"].state, OccupantState.ARRIVED)

    def test_deterministic_across_repeated_registrations(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="adult1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Adult_Default",
            ),
            ScenarioOccupant(
                occupant_id="wheel1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Wheelchair_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))

        first_context = run_scenario(scenario, building)
        register_population_dynamic(first_context)
        first_result = first_context.simulation.run()

        second_context = run_scenario(scenario, building)
        register_population_dynamic(second_context)
        second_result = second_context.simulation.run()

        self.assertEqual(
            first_result.occupants["wheel1"].route.edge_ids,
            second_result.occupants["wheel1"].route.edge_ids,
        )
        self.assertEqual(
            first_result.occupants["wheel1"].depart_time,
            second_result.occupants["wheel1"].depart_time,
        )


# =====================================================
# Backward compatibility -- scenario-authored path unaffected
# =====================================================


class BackwardCompatibilityTests(unittest.TestCase):

    def test_scenario_authored_assistance_path_still_works_unmodified(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="helper1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Staff_Default",
                assisting_occupant_id="assisted1", assistance_type="ESCORT",
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

        self.assertEqual(result.occupants["helper1"].depart_time, result.occupants["assisted1"].depart_time)

    def test_register_population_dynamic_is_a_new_opt_in_function(self):

        import inspect

        from behaviour_profile_resolver.registrar import register_occupants as bound_register_occupants

        # register_occupants's own public signature is unaffected --
        # this phase adds a completely separate function rather than
        # touching it.
        self.assertEqual(
            list(inspect.signature(bound_register_occupants).parameters), ["context", "registry"],
        )


# =====================================================
# Ground Truth / Dataset Builder integration
# =====================================================


class GroundTruthDecisionEventsTests(unittest.TestCase):

    def test_summarize_decision_events_counts_each_type(self):

        events = [
            {"event_type": HELP_DECISION}, {"event_type": HELP_DECISION},
            {"event_type": HELP_REJECTED}, {"event_type": GROUP_FORMED},
            {"event_type": "Unknown_Type"},
        ]
        counts = summarize_decision_events(events)

        self.assertEqual(counts["help_decision_count"], 2)
        self.assertEqual(counts["help_rejected_count"], 1)
        self.assertEqual(counts["group_formed_count"], 1)
        self.assertEqual(counts["rescue_initiated_count"], 0)

    def test_empty_events_produce_all_zero_counts(self):

        counts = summarize_decision_events(())
        self.assertTrue(all(value == 0 for value in counts.values()))

    def test_ground_truth_round_trips_decision_event_fields(self):

        gt = GroundTruth(
            scenario_id="s1", definition_id="d1",
            help_decision_count=3, group_formed_count=1, rescue_completed_count=2,
        )
        restored = GroundTruth.from_dict(gt.to_dict())

        self.assertEqual(restored.help_decision_count, 3)
        self.assertEqual(restored.group_formed_count, 1)
        self.assertEqual(restored.rescue_completed_count, 2)

    def test_simulation_artifacts_decision_events_default_to_all_zero_ground_truth_counts(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        scenario = Scenario(metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        context.simulation.add_occupant(zone_a.id, occupant_id="p1")
        result = context.simulation.run()

        artifacts = SimulationArtifacts(scenario=scenario, building=building, movement_result=result)
        ground_truth = analyze(artifacts)

        self.assertEqual(ground_truth.help_decision_count, 0)
        self.assertEqual(ground_truth.rescue_initiated_count, 0)


class DatasetBuilderDecisionColumnsTests(unittest.TestCase):

    def test_new_columns_are_appended_to_simulation_outcome_columns(self):

        for column in (
            "Help_Decision_Count", "Help_Rejected_Count", "Firefighter_Task_Count",
            "Group_Formed_Count", "Group_Dissolved_Count",
            "Rescue_Initiated_Count", "Rescue_Completed_Count",
        ):
            self.assertIn(column, SIMULATION_OUTCOME_COLUMNS)

    def test_extract_simulation_outcome_defaults_decision_columns_to_zero(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        scenario = Scenario(metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        context.simulation.add_occupant(zone_a.id, occupant_id="p1")
        result = context.simulation.run()

        run = SimulationRun(scenario=scenario, building=building, movement_result=result)
        row = extract_simulation_outcome(run)

        self.assertEqual(row["Help_Decision_Count"], 0)
        self.assertEqual(row["Rescue_Completed_Count"], 0)

    def test_extract_simulation_outcome_reflects_supplied_decision_events(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        scenario = Scenario(metadata=make_metadata(), occupants=(), fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)
        context.simulation.add_occupant(zone_a.id, occupant_id="p1")
        result = context.simulation.run()

        events = ({"event_type": HELP_DECISION},) * 3
        run = SimulationRun(
            scenario=scenario, building=building, movement_result=result, decision_events=events,
        )
        row = extract_simulation_outcome(run)

        self.assertEqual(row["Help_Decision_Count"], 3)


# =====================================================
# Command Center integration
# =====================================================


class CommandCenterViewTests(unittest.TestCase):

    def test_incident_frame_human_decisions_defaults_to_empty(self):

        frame = IncidentFrame()
        self.assertEqual(dict(frame.human_decisions), {})

    def test_incident_frame_carries_supplied_human_decisions(self):

        frame = IncidentFrame(human_decisions={"p1": {"decision": "HELP"}})
        self.assertEqual(frame.human_decisions["p1"]["decision"], "HELP")

    def test_build_human_decisions_view_translates_civilian_and_firefighter_decisions(self):

        building, floor, zone_a, zone_b, door, exit_obj = build_two_zone_building()
        occupants = (
            ScenarioOccupant(
                occupant_id="adult1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Adult_Default",
            ),
            ScenarioOccupant(
                occupant_id="wheel1", zone_id=zone_a.id, floor_id=floor.id, position=(1.0, 1.0),
                behaviour_profile_id="Wheelchair_Default",
            ),
        )
        scenario = Scenario(metadata=make_metadata(), occupants=occupants, fire=make_fire(zone_b.id, floor.id))
        context = run_scenario(scenario, building)

        registration_result = register_population_dynamic(context)
        view = build_human_decisions_view(registration_result)

        self.assertEqual(view["adult1"]["decision"], ACTION_HELP)
        self.assertEqual(view["adult1"]["goal"], "wheel1")
        self.assertEqual(view["wheel1"]["decision"], ACTION_BE_ASSISTED)
        self.assertIn("group", view["adult1"])
        self.assertIn("rescue_priority", view["wheel1"])


if __name__ == "__main__":
    unittest.main()
