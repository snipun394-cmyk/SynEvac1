import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from navigation.edge import Edge

from scenario import DoorState, ScenarioDoorState
from scenario_definition import (
    EngineeringConstraints,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
)
from scenario_definition import DoorState as DefDoorState

from scenario_generator import GenerationRequest, generate_scenario
from scenario_validator import validate, validate_navigation
from scenario_runner import run

from behaviour_profile_resolver import register_occupants
from ai_decision.engine import AIDecisionEngine
from simulation_runtime import SimulationRuntime
from simulator import OccupantState


# Canonical DoorState semantics (single source of truth:
# scenario/engineering_state.py::DoorState.is_traversable):
#   OPEN    -- already open, traversable.
#   CLOSED  -- physically closed but unlocked, occupants may open it,
#              traversable.
#   LOCKED  -- cannot be opened, not traversable.
#
# This module proves every consumer of DoorState agrees: the unit-level
# predicate itself, the Navigation Validator, and the full
# Building -> Scenario Generation -> Validation -> Runner -> Navigation
# -> Simulation pipeline all treat a CLOSED-but-unlocked door exactly
# like an OPEN one, and only a LOCKED door as impassable.


class DoorStateIsTraversableTests(unittest.TestCase):

    def test_open_is_traversable(self):
        self.assertTrue(DoorState.OPEN.is_traversable)

    def test_closed_is_traversable(self):
        self.assertTrue(DoorState.CLOSED.is_traversable)

    def test_locked_is_not_traversable(self):
        self.assertFalse(DoorState.LOCKED.is_traversable)


def _make_building():

    # zone-1 (Lobby, has the only exit) <-- door-1 --> zone-2 (Office,
    # occupied). door-1 is the sole route out of zone-2.
    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def _make_definition(door_state_name):

    return ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(100.0, 400.0),
            # Restricted to the occupant's own zone-2 -- ignition in
            # zone-1 would swallow the building's only exit and fail
            # Navigation Validation's unrelated FIRE_ORIGIN_BLOCKS_EVACUATION
            # check, which is not what this suite is exercising.
            allowed_ignition_zone_ids={"zone-2"},
            allowed_fire_profiles={"Electrical"},
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={"door-1": FixedValue(door_state_name)},
            exit_state_distribution={"exit-1": FixedValue(True)},
            min_open_exits=1,
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-2": FixedValue(1)},
            behaviour_profile_distribution={"zone-2": FixedValue("Staff_Default")},
        ),
    )


class NavigationValidationClosedDoorRegressionTests(unittest.TestCase):

    # The bug being fixed: _door_traversable_map used to treat only
    # DoorState.OPEN as traversable, so a candidate with a CLOSED
    # (unlocked) door on the only route out was wrongly rejected with
    # NO_EVACUATION_ROUTE.

    def test_closed_unlocked_door_is_traversable_for_navigation_validation(self):

        building = _make_building()
        definition = _make_definition(DefDoorState.CLOSED.name)

        scenario = generate_scenario(
            GenerationRequest(definition=definition, definition_id="def-1", building=building, seed=1),
        )

        report = validate_navigation(scenario, definition, building)

        self.assertTrue(report.accepted)
        self.assertEqual(report.by_code("NO_EVACUATION_ROUTE"), [])

    def test_locked_door_still_fails_navigation_validation(self):

        building = _make_building()
        definition = _make_definition(DefDoorState.LOCKED.name)

        scenario = generate_scenario(
            GenerationRequest(definition=definition, definition_id="def-1", building=building, seed=1),
        )

        report = validate_navigation(scenario, definition, building)

        self.assertFalse(report.accepted)
        self.assertTrue(report.by_code("NO_EVACUATION_ROUTE"))


class EndToEndPipelineAgreementTests(unittest.TestCase):

    # Building -> Scenario Generation -> Validation -> Runner ->
    # Navigation -> Simulation, all on a building whose only interior
    # door is CLOSED (unlocked) -- every stage must agree the door is
    # traversable and the occupant evacuates through it.

    def test_only_closed_unlocked_interior_door_validates_and_simulates_successfully(self):

        building = _make_building()
        definition = _make_definition(DefDoorState.CLOSED.name)

        # -- Scenario Generation --
        scenario = generate_scenario(
            GenerationRequest(definition=definition, definition_id="def-1", building=building, seed=1),
        )

        door_state = next(s for s in scenario.door_states if s.door_id == "door-1")
        self.assertEqual(door_state.state, DoorState.CLOSED)

        # -- Validation --
        report = validate(scenario, definition, building)
        self.assertTrue(report.accepted, report.summary())

        # -- Runner --
        context = run(scenario, building)

        door = context.building.floors[0].doors[0]
        self.assertFalse(door.locked)

        # -- Navigation --
        door_edge = next(e for e in context.graph.edges if e.edge_type == Edge.DOOR)
        self.assertTrue(door_edge.traversable)

        # -- Simulation --
        register_occupants(context)
        decision_engine = AIDecisionEngine(base_engine=context.engine)
        runtime = SimulationRuntime(context, decision_engine, dt=5.0)

        occupant_id = scenario.occupants[0].occupant_id

        self.assertEqual(context.simulation._occupants[occupant_id].state, OccupantState.ARRIVED)
        self.assertIsNotNone(runtime.movement_result.total_evacuation_time)
        self.assertIn(occupant_id, runtime.movement_result.occupants)


class ScenarioDoorStateRoundTripTraversabilityTests(unittest.TestCase):

    def test_every_enum_member_reports_the_canonical_traversability(self):

        expected = {
            DoorState.OPEN: True,
            DoorState.CLOSED: True,
            DoorState.LOCKED: False,
        }

        for state, traversable in expected.items():

            door_state = ScenarioDoorState(door_id="door-1", state=state)
            self.assertEqual(door_state.state.is_traversable, traversable)


if __name__ == "__main__":
    unittest.main()
