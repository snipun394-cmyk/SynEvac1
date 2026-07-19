import unittest

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario import (
    DeviceAvailability,
    DoorState,
    PresenceState,
    Scenario,
    ScenarioCameraState,
    ScenarioDetectorState,
    ScenarioDoorState,
    ScenarioEvent,
    ScenarioExitState,
    ScenarioFire,
    ScenarioMetadata,
    ScenarioObstacleState,
    ScenarioOccupant,
    ScenarioStairState,
    StairAvailability,
)
from scenario_definition import (
    EngineeringConstraints,
    EventTemplate,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
    WeightedOptions,
)
from scenario_definition import DoorState as DefDoorState

from scenario_validator import (
    FailureCategory,
    ScenarioValidationIssue,
    ScenarioValidationReport,
    ScenarioValidator,
    compute_candidate_content_hash,
    validate,
    validate_building,
    validate_dataset,
    validate_engineering,
    validate_events,
    validate_fire,
    validate_navigation,
    validate_occupants,
)


def make_building():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[
            Exit(id="exit-1", zone_id="zone-1"),
            Exit(id="exit-2", zone_id="zone-2"),
        ],
        cameras=[Camera(id="cam-1", active=True)],
    )

    return Building(name="Test Building", id="building-1", floors=[floor1])


def make_metadata(**overrides):

    defaults = dict(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="hash-abc",
        generation_version="scenario_generator/1", seed=42, created_at="2026-07-13T00:00:00",
    )
    defaults.update(overrides)

    return ScenarioMetadata(**defaults)


def make_definition(**overrides):

    defaults = dict(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(100.0, 400.0),
            allowed_ignition_zone_ids={"zone-1", "zone-2"},
            allowed_fire_profiles={"Electrical", "Flaming"},
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={"door-1": FixedValue(DefDoorState.OPEN.name)},
            exit_state_distribution={"exit-1": FixedValue(True), "exit-2": FixedValue(True)},
            min_open_exits=1,
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-1": FixedValue(2)},
            behaviour_profile_distribution={
                "zone-1": WeightedOptions({"Adult_Default": 0.8, "Child_Default": 0.2}),
            },
        ),
        seed=1,
    )
    defaults.update(overrides)

    return ScenarioDefinition(**defaults)


def make_occupant(**overrides):

    defaults = dict(
        occupant_id="occ-1", zone_id="zone-1", floor_id="floor-1",
        position=(3.0, 3.0), behaviour_profile_id="Adult_Default",
    )
    defaults.update(overrides)

    return ScenarioOccupant(**defaults)


def make_scenario(**overrides):

    defaults = dict(
        metadata=make_metadata(),
        occupants=(make_occupant(), make_occupant(occupant_id="occ-2")),
        fire=ScenarioFire(
            ignition_zone_id="zone-2", ignition_floor_id="floor-1",
            fire_profile="Electrical", growth_parameters={"growth_time": 200.0},
        ),
        door_states=(ScenarioDoorState(door_id="door-1", state=DoorState.OPEN),),
        exit_states=(
            ScenarioExitState(exit_id="exit-1", is_open=True),
            ScenarioExitState(exit_id="exit-2", is_open=True),
        ),
        stair_states=(),
        obstacle_states=(),
        camera_states=(),
        detector_states=(),
        events=(),
    )
    defaults.update(overrides)

    return Scenario(**defaults)


class IssueAndReportTests(unittest.TestCase):

    def test_issue_construction(self):

        issue = ScenarioValidationIssue(
            category=FailureCategory.FIRE, severity=ScenarioValidationReport.ERROR,
            code="X", message="bad", object_id="zone-1",
        )
        self.assertEqual(issue.category, FailureCategory.FIRE)
        self.assertEqual(issue.object_id, "zone-1")

    def test_report_accepted_and_is_valid_are_the_same_property(self):

        report = ScenarioValidationReport()
        report.add(FailureCategory.FIRE, ScenarioValidationReport.WARNING, "W", "warn only")

        self.assertTrue(report.accepted)
        self.assertTrue(report.is_valid)

        report.add(FailureCategory.FIRE, ScenarioValidationReport.ERROR, "E", "an error")

        self.assertFalse(report.accepted)
        self.assertFalse(report.is_valid)

    def test_errors_and_warnings_split_correctly(self):

        report = ScenarioValidationReport()
        report.add(FailureCategory.DATASET, ScenarioValidationReport.ERROR, "E1", "err")
        report.add(FailureCategory.DATASET, ScenarioValidationReport.WARNING, "W1", "warn")

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(len(report.warnings), 1)

    def test_by_code_and_by_category(self):

        report = ScenarioValidationReport()
        report.add(FailureCategory.FIRE, ScenarioValidationReport.ERROR, "E1", "err")
        report.add(FailureCategory.NAVIGATION, ScenarioValidationReport.ERROR, "E2", "err2")

        self.assertEqual(len(report.by_code("E1")), 1)
        self.assertEqual(len(report.by_category(FailureCategory.NAVIGATION)), 1)

    def test_single_category_attributable(self):

        single = ScenarioValidationReport()
        single.add(FailureCategory.FIRE, ScenarioValidationReport.ERROR, "E1", "err")
        single.add(FailureCategory.FIRE, ScenarioValidationReport.ERROR, "E2", "err2")
        self.assertTrue(single.is_single_category_attributable())

        multi = ScenarioValidationReport()
        multi.add(FailureCategory.FIRE, ScenarioValidationReport.ERROR, "E1", "err")
        multi.add(FailureCategory.NAVIGATION, ScenarioValidationReport.ERROR, "E2", "err2")
        self.assertFalse(multi.is_single_category_attributable())

        accepted = ScenarioValidationReport()
        self.assertTrue(accepted.is_single_category_attributable())

    def test_summary_counts(self):

        report = ScenarioValidationReport()
        report.add(FailureCategory.FIRE, ScenarioValidationReport.ERROR, "E1", "err")
        report.add(FailureCategory.FIRE, ScenarioValidationReport.WARNING, "W1", "warn")

        summary = report.summary()
        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["warning_count"], 1)
        self.assertEqual(summary["categories"], [FailureCategory.FIRE])

    def test_len_and_iter(self):

        report = ScenarioValidationReport()
        report.add(FailureCategory.FIRE, ScenarioValidationReport.ERROR, "E1", "err")
        report.add(FailureCategory.FIRE, ScenarioValidationReport.WARNING, "W1", "warn")

        self.assertEqual(len(report), 2)
        self.assertEqual(len(list(report)), 2)


class BuildingValidationTests(unittest.TestCase):

    def test_valid_candidate_passes(self):

        report = validate_building(make_scenario(), make_building())
        self.assertTrue(report.accepted)

    def test_missing_building_is_an_error(self):

        report = validate_building(make_scenario(), None)
        self.assertFalse(report.accepted)
        self.assertTrue(report.by_code("MISSING_BUILDING"))

    def test_unknown_occupant_zone_id_is_detected(self):

        scenario = make_scenario(occupants=(make_occupant(zone_id="zone-does-not-exist"),))
        report = validate_building(scenario, make_building())

        self.assertFalse(report.accepted)
        issue = report.by_code("UNKNOWN_ID")[0]
        self.assertEqual(issue.category, FailureCategory.STRUCTURAL)

    def test_unknown_door_id_is_detected(self):

        scenario = make_scenario(
            door_states=(ScenarioDoorState(door_id="door-does-not-exist", state=DoorState.OPEN),),
        )
        report = validate_building(scenario, make_building())
        self.assertTrue(report.by_code("UNKNOWN_ID"))

    def test_unknown_event_target_id_is_detected(self):

        scenario = make_scenario(
            events=(
                ScenarioEvent(
                    event_id="evt-1", target_type="door", target_id="door-ghost",
                    event_type="close", time=10.0,
                ),
            ),
        )
        report = validate_building(scenario, make_building())
        self.assertTrue(report.by_code("UNKNOWN_ID"))


class OccupantValidationTests(unittest.TestCase):

    def test_valid_occupant_passes(self):

        report = validate_occupants(make_scenario(), make_definition(), make_building())
        self.assertTrue(report.accepted)

    def test_occupant_outside_zone_is_geometry_error(self):

        scenario = make_scenario(occupants=(make_occupant(position=(999.0, 999.0)),))
        report = validate_occupants(scenario, make_definition(), make_building())

        self.assertFalse(report.accepted)
        issue = report.by_code("OCCUPANT_OUTSIDE_ZONE")[0]
        self.assertEqual(issue.category, FailureCategory.GEOMETRY)

    def test_count_out_of_support_is_detected(self):

        definition = make_definition(
            occupant=OccupantDefinition(occupancy_distribution={"zone-1": FixedValue(5)}),
        )
        scenario = make_scenario(occupants=(make_occupant(),))  # only 1 occupant, not 5
        report = validate_occupants(scenario, definition, make_building())

        self.assertTrue(report.by_code("OCCUPANT_COUNT_OUT_OF_SUPPORT"))

    def test_missing_behaviour_profile_id_is_an_error(self):

        scenario = make_scenario(occupants=(make_occupant(behaviour_profile_id=""),))
        report = validate_occupants(scenario, make_definition(), make_building())

        self.assertTrue(report.by_code("MISSING_BEHAVIOUR_PROFILE_ID"))

    def test_behaviour_profile_not_in_declared_support_is_an_error(self):

        scenario = make_scenario(
            occupants=(make_occupant(behaviour_profile_id="FireWarden_Default"),),
        )
        report = validate_occupants(scenario, make_definition(), make_building())

        self.assertTrue(report.by_code("BEHAVIOUR_PROFILE_NOT_IN_SUPPORT"))

    def test_high_density_is_a_warning_not_an_error(self):

        # zone-1 is 10x8=80 sqm; 300 occupants far exceeds the density
        # threshold but must remain a WARNING (accepted stays True).
        occupants = tuple(
            make_occupant(occupant_id=f"occ-{i}", position=(1.0, 1.0)) for i in range(300)
        )
        definition = make_definition(
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": FixedValue(300)},
                behaviour_profile_distribution={"zone-1": FixedValue("Adult_Default")},
            ),
        )
        scenario = make_scenario(occupants=occupants)
        report = validate_occupants(scenario, definition, make_building())

        warning = report.by_code("HIGH_OCCUPANCY_DENSITY")
        self.assertTrue(warning)
        self.assertEqual(warning[0].severity, ScenarioValidationReport.WARNING)
        self.assertTrue(report.accepted)

    def test_proportion_skew_warning_never_rejects(self):

        # 10 occupants, all "Adult_Default" -- Child_Default (weight
        # 0.2 >= threshold) never drawn.
        occupants = tuple(
            make_occupant(occupant_id=f"occ-{i}", behaviour_profile_id="Adult_Default")
            for i in range(10)
        )
        definition = make_definition(
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": FixedValue(10)},
                behaviour_profile_distribution={
                    "zone-1": WeightedOptions({"Adult_Default": 0.8, "Child_Default": 0.2}),
                },
            ),
        )
        scenario = make_scenario(occupants=occupants)
        report = validate_occupants(scenario, definition, make_building())

        warning = report.by_code("BEHAVIOUR_PROFILE_PROPORTION_SKEW")
        self.assertTrue(warning)
        self.assertEqual(warning[0].severity, ScenarioValidationReport.WARNING)
        self.assertTrue(report.accepted)


class FireValidationTests(unittest.TestCase):

    def test_valid_fire_passes(self):

        report = validate_fire(make_scenario(), make_definition())
        self.assertTrue(report.accepted)

    def test_missing_fire_is_an_error(self):

        report = validate_fire(make_scenario(fire=None), make_definition())
        self.assertTrue(report.by_code("MISSING_FIRE"))

    def test_zone_not_allowed_is_detected(self):

        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(200.0),
                allowed_ignition_zone_ids={"zone-1"},
            ),
        )
        scenario = make_scenario(
            fire=ScenarioFire(
                ignition_zone_id="zone-2", ignition_floor_id="floor-1",
                fire_profile="", growth_parameters={"growth_time": 200.0},
            ),
        )
        report = validate_fire(scenario, definition)
        self.assertTrue(report.by_code("IGNITION_ZONE_NOT_ALLOWED"))

    def test_zone_forbidden_is_detected(self):

        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(200.0),
                forbidden_ignition_zone_ids={"zone-2"},
            ),
        )
        scenario = make_scenario(
            fire=ScenarioFire(
                ignition_zone_id="zone-2", ignition_floor_id="floor-1",
                fire_profile="", growth_parameters={"growth_time": 200.0},
            ),
        )
        report = validate_fire(scenario, definition)
        self.assertTrue(report.by_code("IGNITION_ZONE_FORBIDDEN"))

    def test_floor_not_allowed_is_detected(self):

        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(200.0),
                allowed_ignition_floor_ids={"floor-99"},
            ),
        )
        report = validate_fire(make_scenario(), definition)
        self.assertTrue(report.by_code("IGNITION_FLOOR_NOT_ALLOWED"))

    def test_fire_profile_not_allowed_is_detected(self):

        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(200.0),
                allowed_fire_profiles={"Electrical"},
            ),
        )
        scenario = make_scenario(
            fire=ScenarioFire(
                ignition_zone_id="zone-2", ignition_floor_id="floor-1",
                fire_profile="Flaming", growth_parameters={"growth_time": 200.0},
            ),
        )
        report = validate_fire(scenario, definition)
        self.assertTrue(report.by_code("FIRE_PROFILE_NOT_ALLOWED"))

    def test_growth_parameters_out_of_support_is_detected(self):

        definition = make_definition(
            fire=FireDefinition(growth_parameter_distribution=UniformRange(100.0, 200.0)),
        )
        scenario = make_scenario(
            fire=ScenarioFire(
                ignition_zone_id="zone-2", ignition_floor_id="floor-1",
                fire_profile="", growth_parameters={"growth_time": 999.0},
            ),
        )
        report = validate_fire(scenario, definition)
        self.assertTrue(report.by_code("GROWTH_PARAMETERS_OUT_OF_SUPPORT"))


class EngineeringValidationTests(unittest.TestCase):

    def test_valid_engineering_state_passes(self):

        report = validate_engineering(make_scenario(), make_definition())
        self.assertTrue(report.accepted)

    def test_min_open_exits_unsatisfied_is_detected(self):

        definition = make_definition(
            engineering=EngineeringConstraints(min_open_exits=5),
        )
        report = validate_engineering(make_scenario(), definition)
        self.assertTrue(report.by_code("MIN_OPEN_EXITS_UNSATISFIED"))

    def test_pinned_door_mismatch_is_detected(self):

        definition = make_definition(
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue(DefDoorState.LOCKED.name)},
                min_open_exits=1,
            ),
        )
        scenario = make_scenario(
            door_states=(ScenarioDoorState(door_id="door-1", state=DoorState.OPEN),),
        )
        report = validate_engineering(scenario, definition)
        self.assertTrue(report.by_code("PINNED_STATE_MISMATCH"))

    def test_pinned_exit_mismatch_is_detected(self):

        definition = make_definition(
            engineering=EngineeringConstraints(
                exit_state_distribution={"exit-1": FixedValue(True)}, min_open_exits=1,
            ),
        )
        scenario = make_scenario(
            exit_states=(ScenarioExitState(exit_id="exit-1", is_open=False),),
        )
        report = validate_engineering(scenario, definition)
        self.assertTrue(report.by_code("PINNED_STATE_MISMATCH"))

    def test_pinned_stair_camera_detector_obstacle_mismatches_are_detected(self):

        definition = make_definition(
            engineering=EngineeringConstraints(
                stair_state_distribution={"stair-1": FixedValue(StairAvailability.AVAILABLE.name)},
                obstacle_state_distribution={"obs-1": FixedValue(PresenceState.ACTIVE.name)},
                camera_state_distribution={"cam-1": FixedValue(DeviceAvailability.AVAILABLE.name)},
                detector_state_distribution={"det-1": FixedValue(DeviceAvailability.AVAILABLE.name)},
                min_open_exits=1,
            ),
        )
        scenario = make_scenario(
            stair_states=(ScenarioStairState(stair_id="stair-1", availability=StairAvailability.CLOSED),),
            obstacle_states=(ScenarioObstacleState(obstacle_id="obs-1", presence=PresenceState.INACTIVE),),
            camera_states=(ScenarioCameraState(camera_id="cam-1", availability=DeviceAvailability.FAILED),),
            detector_states=(ScenarioDetectorState(detector_id="det-1", availability=DeviceAvailability.FAILED),),
        )
        report = validate_engineering(scenario, definition)

        mismatches = report.by_code("PINNED_STATE_MISMATCH")
        self.assertEqual(len(mismatches), 4)


class NavigationValidationTests(unittest.TestCase):

    def _chain_building(self):

        # zone-1 (has exit-1) <-> zone-2 <-> zone-3, via door-1/door-2.
        # zone-2 is the only connector between zone-3 and any exit.
        floor = Floor(
            name="Ground", id="floor-1",
            zones=[
                Zone(id="zone-1", name="Z1", x=0.0, y=0.0, width=5.0, height=5.0),
                Zone(id="zone-2", name="Z2", x=10.0, y=0.0, width=5.0, height=5.0),
                Zone(id="zone-3", name="Z3", x=20.0, y=0.0, width=5.0, height=5.0),
            ],
            doors=[
                Door(id="door-1", zone_a_id="zone-1", zone_b_id="zone-2"),
                Door(id="door-2", zone_a_id="zone-2", zone_b_id="zone-3"),
            ],
            exits=[Exit(id="exit-1", zone_id="zone-1")],
        )
        return Building(name="Chain", id="chain-1", floors=[floor])

    def _isolated_exit_building(self):

        # zone-1 (occupied, exit-1) and an isolated zone-2 (exit-2, no
        # occupants, no door to zone-1).
        floor = Floor(
            name="Ground", id="floor-1",
            zones=[
                Zone(id="zone-1", name="Z1", x=0.0, y=0.0, width=5.0, height=5.0),
                Zone(id="zone-2", name="Z2", x=10.0, y=0.0, width=5.0, height=5.0),
            ],
            doors=[],
            exits=[Exit(id="exit-1", zone_id="zone-1"), Exit(id="exit-2", zone_id="zone-2")],
        )
        return Building(name="Isolated", id="isolated-1", floors=[floor])

    def test_reachable_occupied_zone_passes(self):

        report = validate_navigation(make_scenario(), make_definition(), make_building())
        self.assertTrue(report.accepted)

    def test_no_evacuation_route_when_only_door_locked_and_exit_closed(self):

        scenario = make_scenario(
            door_states=(ScenarioDoorState(door_id="door-1", state=DoorState.LOCKED),),
            exit_states=(
                ScenarioExitState(exit_id="exit-1", is_open=False),
                ScenarioExitState(exit_id="exit-2", is_open=True),
            ),
            occupants=(make_occupant(zone_id="zone-1", position=(3.0, 3.0)),),
        )
        report = validate_navigation(scenario, make_definition(), make_building())

        self.assertFalse(report.accepted)
        issue = report.by_code("NO_EVACUATION_ROUTE")[0]
        self.assertEqual(issue.category, FailureCategory.NAVIGATION)

    def test_unoccupied_unreachable_zone_is_not_flagged(self):

        # zone-2 has no door and no occupants in the base building --
        # only zone-1 (occupied) is checked.
        building = self._isolated_exit_building()
        scenario = make_scenario(
            door_states=(),
            exit_states=(
                ScenarioExitState(exit_id="exit-1", is_open=True),
                ScenarioExitState(exit_id="exit-2", is_open=True),
            ),
            occupants=(make_occupant(zone_id="zone-1", position=(1.0, 1.0)),),
        )
        definition = make_definition(
            engineering=EngineeringConstraints(min_open_exits=1),
        )
        report = validate_navigation(scenario, definition, building)

        self.assertTrue(report.accepted)

    def test_fire_origin_blocking_the_only_route_is_detected(self):

        building = self._chain_building()
        scenario = make_scenario(
            door_states=(
                ScenarioDoorState(door_id="door-1", state=DoorState.OPEN),
                ScenarioDoorState(door_id="door-2", state=DoorState.OPEN),
            ),
            exit_states=(ScenarioExitState(exit_id="exit-1", is_open=True),),
            occupants=(make_occupant(zone_id="zone-3", position=(21.0, 1.0)),),
            fire=ScenarioFire(
                ignition_zone_id="zone-2", ignition_floor_id="floor-1",
                fire_profile="", growth_parameters={},
            ),
        )
        definition = make_definition(engineering=EngineeringConstraints(min_open_exits=1))
        report = validate_navigation(scenario, definition, building)

        self.assertFalse(report.accepted)
        self.assertTrue(report.by_code("FIRE_ORIGIN_BLOCKS_EVACUATION"))

    def test_insufficient_reachable_egress_when_one_open_exit_is_unreachable(self):

        building = self._isolated_exit_building()
        scenario = make_scenario(
            door_states=(),
            exit_states=(
                ScenarioExitState(exit_id="exit-1", is_open=True),
                ScenarioExitState(exit_id="exit-2", is_open=True),
            ),
            occupants=(make_occupant(zone_id="zone-1", position=(1.0, 1.0)),),
        )
        definition = make_definition(engineering=EngineeringConstraints(min_open_exits=2))
        report = validate_navigation(scenario, definition, building)

        self.assertFalse(report.accepted)
        self.assertTrue(report.by_code("INSUFFICIENT_REACHABLE_EGRESS"))

    def test_no_occupants_means_no_checks_run(self):

        scenario = make_scenario(occupants=())
        report = validate_navigation(scenario, make_definition(), make_building())

        self.assertTrue(report.accepted)
        self.assertEqual(len(report), 0)


class EventValidationTests(unittest.TestCase):

    def test_no_events_passes(self):

        report = validate_events(make_scenario(), make_definition())
        self.assertTrue(report.accepted)

    def test_missing_required_field_is_detected(self):

        scenario = make_scenario(
            events=(
                ScenarioEvent(
                    event_id="evt-1", target_type="door", target_id="", event_type="close",
                    time=10.0,
                ),
            ),
        )
        report = validate_events(scenario, make_definition())
        self.assertTrue(report.by_code("EVENT_MISSING_REQUIRED_FIELD"))

    def test_negative_timestamp_is_detected(self):

        scenario = make_scenario(
            events=(
                ScenarioEvent(
                    event_id="evt-1", target_type="door", target_id="door-1",
                    event_type="close", time=-5.0,
                ),
            ),
        )
        report = validate_events(scenario, make_definition())
        self.assertTrue(report.by_code("EVENT_INVALID_TIMESTAMP"))

    def test_unordered_events_are_detected(self):

        scenario = make_scenario(
            events=(
                ScenarioEvent(
                    event_id="evt-1", target_type="door", target_id="door-1",
                    event_type="close", time=50.0,
                ),
                ScenarioEvent(
                    event_id="evt-2", target_type="door", target_id="door-1",
                    event_type="open", time=10.0,
                ),
            ),
        )
        report = validate_events(scenario, make_definition())
        self.assertTrue(report.by_code("EVENTS_NOT_ORDERED"))

    def test_conflicting_events_at_the_same_instant_are_detected(self):

        scenario = make_scenario(
            events=(
                ScenarioEvent(
                    event_id="evt-1", target_type="door", target_id="door-1",
                    event_type="close", time=10.0,
                ),
                ScenarioEvent(
                    event_id="evt-2", target_type="door", target_id="door-1",
                    event_type="open", time=10.0,
                ),
            ),
        )
        report = validate_events(scenario, make_definition())
        self.assertTrue(report.by_code("CONFLICTING_EVENTS"))

    def test_event_contradicting_pinned_door_state_is_detected(self):

        definition = make_definition(
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue(DefDoorState.OPEN.name)},
                min_open_exits=1,
            ),
        )
        scenario = make_scenario(
            events=(
                ScenarioEvent(
                    event_id="evt-1", target_type="door", target_id="door-1",
                    event_type="close", time=90.0,
                ),
            ),
        )
        report = validate_events(scenario, definition)
        self.assertTrue(report.by_code("EVENT_CONTRADICTS_PINNED_STATE"))

    def test_event_not_contradicting_an_unpinned_door_is_not_flagged(self):

        definition = make_definition(
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-1": WeightedOptions({DefDoorState.OPEN.name: 0.5, DefDoorState.CLOSED.name: 0.5}),
                },
                min_open_exits=1,
            ),
        )
        scenario = make_scenario(
            events=(
                ScenarioEvent(
                    event_id="evt-1", target_type="door", target_id="door-1",
                    event_type="close", time=90.0,
                ),
            ),
        )
        report = validate_events(scenario, definition)
        self.assertEqual(report.by_code("EVENT_CONTRADICTS_PINNED_STATE"), [])

    def test_pinned_event_missing_is_detected(self):

        definition = make_definition(
            event_templates=(
                EventTemplate(
                    target_type="door", target_id="door-1", event_type="close",
                    occurs=FixedValue(True), time=FixedValue(90.0),
                ),
            ),
        )
        report = validate_events(make_scenario(events=()), definition)
        self.assertTrue(report.by_code("PINNED_EVENT_MISSING"))

    def test_pinned_event_time_mismatch_is_detected(self):

        definition = make_definition(
            event_templates=(
                EventTemplate(
                    target_type="door", target_id="door-1", event_type="close",
                    occurs=FixedValue(True), time=FixedValue(90.0),
                ),
            ),
        )
        scenario = make_scenario(
            events=(
                ScenarioEvent(
                    event_id="evt-1", target_type="door", target_id="door-1",
                    event_type="close", time=45.0,
                ),
            ),
        )
        report = validate_events(scenario, definition)
        self.assertTrue(report.by_code("PINNED_EVENT_TIME_MISMATCH"))

    def test_pinned_event_present_with_matching_time_passes(self):

        definition = make_definition(
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-1": WeightedOptions({DefDoorState.OPEN.name: 0.5, DefDoorState.CLOSED.name: 0.5}),
                },
                exit_state_distribution={"exit-1": FixedValue(True), "exit-2": FixedValue(True)},
                min_open_exits=1,
            ),
            event_templates=(
                EventTemplate(
                    target_type="door", target_id="door-1", event_type="close",
                    occurs=FixedValue(True), time=FixedValue(90.0),
                ),
            ),
        )
        scenario = make_scenario(
            events=(
                ScenarioEvent(
                    event_id="evt-1", target_type="door", target_id="door-1",
                    event_type="close", time=90.0,
                ),
            ),
        )
        report = validate_events(scenario, definition)
        self.assertTrue(report.accepted)


class DatasetValidationTests(unittest.TestCase):

    def test_valid_candidate_passes(self):

        report = validate_dataset(make_scenario())
        self.assertTrue(report.accepted)

    def test_missing_metadata_field_is_detected(self):

        scenario = make_scenario(metadata=make_metadata(definition_id=""))
        report = validate_dataset(scenario)
        self.assertTrue(report.by_code("MISSING_METADATA_FIELD"))

    def test_unsupported_generation_version_is_detected(self):

        scenario = make_scenario(metadata=make_metadata(generation_version="totally-unknown/99"))
        report = validate_dataset(scenario)
        self.assertTrue(report.by_code("UNSUPPORTED_GENERATION_VERSION"))

    def test_duplicate_hash_is_detected(self):

        scenario = make_scenario()
        content_hash = compute_candidate_content_hash(scenario)

        report = validate_dataset(scenario, accepted_hashes=frozenset({content_hash}))
        self.assertTrue(report.by_code("DUPLICATE"))

    def test_content_hash_is_stable(self):

        scenario = make_scenario()
        self.assertEqual(
            compute_candidate_content_hash(scenario), compute_candidate_content_hash(scenario),
        )

    def test_content_hash_differs_for_different_candidates(self):

        first = make_scenario()
        second = make_scenario(
            occupants=(make_occupant(), make_occupant(occupant_id="occ-3")),
        )

        self.assertNotEqual(
            compute_candidate_content_hash(first), compute_candidate_content_hash(second),
        )

    def test_content_hash_ignores_metadata_so_identical_content_from_different_runs_still_collides(self):

        # The actual bug this pass fixed: two candidates with
        # byte-identical *sampled* content but different metadata
        # (a different scenario_id/seed/created_at, e.g. from two
        # separate generation runs or two different seeds that happen
        # to land on the same content, §4.9's "finite legal-and-unique
        # space" case) must still hash equal -- metadata is provenance,
        # never part of what "duplicate" means here.
        first = make_scenario()
        second = make_scenario(
            metadata=make_metadata(
                scenario_id="scn-2", seed=999, created_at="2099-01-01T00:00:00",
            ),
        )

        self.assertEqual(
            compute_candidate_content_hash(first), compute_candidate_content_hash(second),
        )

    def test_validate_dataset_duplicate_detection_agrees_with_compute_candidate_content_hash(self):

        # Platform-refinement performance fix (docs/validation/
        # technical_report.md §7.1): validate_dataset() now reuses one
        # candidate.to_dict() call for both the serialization
        # round-trip check and the content hash, instead of calling
        # to_dict() a second time inside compute_candidate_content_hash().
        # The hash validate_dataset() actually checks duplicates against
        # must still be byte-identical to calling
        # compute_candidate_content_hash() directly.

        scenario = make_scenario()

        report = validate_dataset(
            scenario, accepted_hashes=frozenset({compute_candidate_content_hash(scenario)}),
        )
        self.assertTrue(report.by_code("DUPLICATE"))

    def test_candidate_to_dict_is_called_once_per_validate_dataset_call(self):

        scenario = make_scenario()

        call_count = {"n": 0}
        original_to_dict = type(scenario).to_dict

        def counting_to_dict(self_scenario):
            call_count["n"] += 1
            return original_to_dict(self_scenario)

        type(scenario).to_dict = counting_to_dict
        try:
            validate_dataset(scenario)
        finally:
            type(scenario).to_dict = original_to_dict

        self.assertEqual(call_count["n"], 1)


class FullValidatorTests(unittest.TestCase):

    def test_accepted_scenario_end_to_end(self):

        report = validate(make_scenario(), make_definition(), make_building())
        self.assertTrue(report.accepted)

    def test_building_validation_failure_short_circuits_other_modules(self):

        scenario = make_scenario(occupants=(make_occupant(zone_id="zone-ghost"),))
        report = validate(scenario, make_definition(), make_building())

        self.assertFalse(report.accepted)
        # Only the STRUCTURAL/UNKNOWN_ID issue from Building Validation
        # should be present -- every other module never ran.
        self.assertEqual(report.error_categories(), {FailureCategory.STRUCTURAL})
        self.assertTrue(report.by_code("UNKNOWN_ID"))

    def test_report_metadata_contains_scenario_id(self):

        report = validate(make_scenario(), make_definition(), make_building())
        self.assertEqual(report.metadata["scenario_id"], "scn-1")

    def test_multi_category_rejection_is_not_single_category_attributable(self):

        scenario = make_scenario(
            fire=ScenarioFire(
                ignition_zone_id="zone-1", ignition_floor_id="floor-1",
                fire_profile="Unlisted", growth_parameters={"growth_time": 9999.0},
            ),
            door_states=(ScenarioDoorState(door_id="door-1", state=DoorState.LOCKED),),
        )
        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=UniformRange(100.0, 400.0),
                allowed_ignition_zone_ids={"zone-1", "zone-2"},
                allowed_fire_profiles={"Electrical", "Flaming"},
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue(DefDoorState.OPEN.name)},
                exit_state_distribution={"exit-1": FixedValue(True), "exit-2": FixedValue(True)},
                min_open_exits=1,
            ),
        )
        report = validate(scenario, definition, make_building())

        self.assertFalse(report.accepted)
        self.assertIn(FailureCategory.FIRE, report.error_categories())
        self.assertIn(FailureCategory.STRUCTURAL, report.error_categories())
        self.assertGreater(len(report.error_categories()), 1)
        self.assertFalse(report.is_single_category_attributable())

    def test_single_category_rejection_is_single_category_attributable(self):

        # A pinned camera mismatch affects only Engineering Validation
        # (STRUCTURAL) -- it has no bearing on Navigation, Fire, or any
        # other category.
        definition = make_definition(
            engineering=EngineeringConstraints(
                camera_state_distribution={"cam-1": FixedValue(DeviceAvailability.AVAILABLE.name)},
                exit_state_distribution={"exit-1": FixedValue(True), "exit-2": FixedValue(True)},
                min_open_exits=1,
            ),
        )
        scenario = make_scenario(
            camera_states=(ScenarioCameraState(camera_id="cam-1", availability=DeviceAvailability.FAILED),),
        )
        report = validate(scenario, definition, make_building())

        self.assertFalse(report.accepted)
        self.assertEqual(report.error_categories(), {FailureCategory.STRUCTURAL})
        self.assertTrue(report.is_single_category_attributable())

    def test_scenario_validator_class_wraps_the_same_function(self):

        report = ScenarioValidator().validate(make_scenario(), make_definition(), make_building())
        self.assertTrue(report.accepted)


class SerializationCompatibilityTests(unittest.TestCase):

    def test_validation_is_identical_before_and_after_round_tripping_the_candidate(self):

        scenario = make_scenario()
        restored = Scenario.from_dict(scenario.to_dict())

        original_report = validate(scenario, make_definition(), make_building())
        restored_report = validate(restored, make_definition(), make_building())

        self.assertEqual(original_report.accepted, restored_report.accepted)
        self.assertEqual(
            [(i.category, i.code) for i in original_report.issues],
            [(i.category, i.code) for i in restored_report.issues],
        )

    def test_validation_is_identical_before_and_after_round_tripping_the_definition(self):

        definition = make_definition()
        restored = ScenarioDefinition.from_dict(definition.to_dict())

        original_report = validate(make_scenario(), definition, make_building())
        restored_report = validate(make_scenario(), restored, make_building())

        self.assertEqual(original_report.accepted, restored_report.accepted)


class ScenarioValidatorPackageDependencyDirectionTests(unittest.TestCase):

    # scenario_validator/ may import scenario_definition, scenario,
    # models, navigation; must not import scenario_generator, sandbox,
    # designer, simulator, behavior, behavior_library, ai_decision, or
    # random (architecture doc §5/§5.9/§12, unchanged).

    def test_package_never_imports_forbidden_packages(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_validator"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(scenario_generator|sandbox|designer|simulator|behavior|behavior_library|"
            r"ai_decision|perception|rl|random|fire_growth)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario_validator/{path.name} imports a package this Validator "
                f"must never depend on (§5.9/§12)",
            )

    def test_package_performs_no_file_io(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_validator"

        forbidden = r"\bopen\s*\(|\.write\s*\(|\.read\s*\(|\bjson\.(load|dump)\s*\("

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text),
                f"scenario_validator/{path.name} appears to perform file I/O",
            )

    def test_package_never_modifies_the_candidate_it_receives(self):

        # Not a mutator (§5.1) -- exercised structurally: validating the
        # same candidate twice must leave it unchanged and produce an
        # identical report both times.
        scenario = make_scenario()
        before = scenario.to_dict()

        validate(scenario, make_definition(), make_building())

        self.assertEqual(scenario.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
