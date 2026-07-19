import unittest

from dataclasses import FrozenInstanceError

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario_definition import (
    DeviceAvailability,
    DoorState,
    EngineeringConstraints,
    EventTemplate,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    PresenceState,
    ScenarioDefinition,
    StairAvailability,
    UniformRange,
    WeightedOptions,
    validate_definition,
)
from scenario_definition.distributions import distribution_from_dict


def make_building():

    floor = Floor(
        name="Ground",
        id="floor-1",
        zones=[Zone(id="zone-1", name="Lobby"), Zone(id="zone-2", name="Office")],
        doors=[Door(id="door-1")],
        exits=[Exit(id="exit-1"), Exit(id="exit-2")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_fire_definition(**overrides):

    defaults = dict(
        growth_parameter_distribution=UniformRange(100.0, 400.0),
        allowed_ignition_zone_ids={"zone-1"},
        forbidden_ignition_zone_ids=set(),
        allowed_ignition_floor_ids={"floor-1"},
        allowed_fire_profiles={"Electrical", "Flaming"},
    )
    defaults.update(overrides)

    return FireDefinition(**defaults)


def make_definition(**overrides):

    defaults = dict(
        fire=make_fire_definition(),
        engineering=EngineeringConstraints(
            door_state_distribution={"door-1": FixedValue(DoorState.OPEN.name)},
            exit_state_distribution={
                "exit-1": FixedValue(True),
                "exit-2": WeightedOptions({True: 0.6, False: 0.4}),
            },
            min_open_exits=1,
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-1": UniformRange(3, 8, discrete=True)},
            behaviour_profile_distribution={"zone-1": FixedValue("Adult_Default")},
        ),
        event_templates=(
            EventTemplate(
                target_type="door",
                target_id="door-1",
                event_type="close",
                occurs=FixedValue(True),
                time=UniformRange(60, 120),
            ),
        ),
        seed=42,
    )
    defaults.update(overrides)

    return ScenarioDefinition(**defaults)


class DistributionsTests(unittest.TestCase):

    def test_fixed_value_round_trip(self):

        value = FixedValue(value=7)
        restored = distribution_from_dict(value.to_dict())

        self.assertEqual(value, restored)

    def test_uniform_range_round_trip(self):

        value = UniformRange(low=1.0, high=5.0, discrete=True)
        restored = distribution_from_dict(value.to_dict())

        self.assertEqual(value, restored)

    def test_weighted_options_round_trip(self):

        value = WeightedOptions({"OPEN": 0.7, "CLOSED": 0.3})
        restored = distribution_from_dict(value.to_dict())

        self.assertEqual(value, restored)

    def test_weighted_options_weights_are_read_only(self):

        value = WeightedOptions({"OPEN": 0.7, "CLOSED": 0.3})

        with self.assertRaises(TypeError):
            value.weights["OPEN"] = 1.0

    def test_all_three_kinds_are_frozen(self):

        with self.assertRaises(FrozenInstanceError):
            FixedValue(value=1).value = 2

        with self.assertRaises(FrozenInstanceError):
            UniformRange(0, 1).low = 5

        with self.assertRaises(FrozenInstanceError):
            WeightedOptions({"a": 1.0}).weights = {}

    def test_equality(self):

        self.assertEqual(FixedValue(value=3), FixedValue(value=3))
        self.assertNotEqual(FixedValue(value=3), FixedValue(value=4))
        self.assertEqual(UniformRange(1, 2), UniformRange(1, 2))
        self.assertEqual(WeightedOptions({"a": 1.0}), WeightedOptions({"a": 1.0}))

    def test_distribution_kinds_have_no_sample_method(self):

        # Data only -- never interpreted here (this implementation
        # phase's own instruction).
        self.assertFalse(hasattr(FixedValue(value=1), "sample"))
        self.assertFalse(hasattr(UniformRange(0, 1), "sample"))
        self.assertFalse(hasattr(WeightedOptions({"a": 1.0}), "sample"))

    def test_from_dict_rejects_unknown_kind(self):

        with self.assertRaises(ValueError):
            distribution_from_dict({"kind": "NotARealDistribution"})


class FireDefinitionTests(unittest.TestCase):

    def test_construction(self):

        fire = make_fire_definition()

        self.assertEqual(fire.allowed_ignition_zone_ids, {"zone-1"})
        self.assertEqual(fire.allowed_fire_profiles, {"Electrical", "Flaming"})
        self.assertIsNone(fire.ignition_zone_preference)

    def test_sets_are_coerced_to_frozenset(self):

        fire = make_fire_definition(allowed_ignition_zone_ids=["zone-1", "zone-2"])

        self.assertIsInstance(fire.allowed_ignition_zone_ids, frozenset)

    def test_is_frozen(self):

        fire = make_fire_definition()

        with self.assertRaises(FrozenInstanceError):
            fire.allowed_ignition_zone_ids = frozenset()

    def test_equality(self):

        self.assertEqual(make_fire_definition(), make_fire_definition())
        self.assertNotEqual(
            make_fire_definition(), make_fire_definition(allowed_fire_profiles={"Flaming"}),
        )

    def test_round_trip(self):

        fire = make_fire_definition(
            ignition_zone_preference=WeightedOptions({"zone-1": 0.9, "zone-2": 0.1}),
        )
        restored = FireDefinition.from_dict(fire.to_dict())

        self.assertEqual(fire, restored)

    def test_round_trip_with_no_preference(self):

        fire = make_fire_definition()
        restored = FireDefinition.from_dict(fire.to_dict())

        self.assertEqual(fire, restored)
        self.assertIsNone(restored.ignition_zone_preference)


class OccupantDefinitionTests(unittest.TestCase):

    def test_construction(self):

        occupant = OccupantDefinition(
            occupancy_distribution={"zone-1": FixedValue(0)},
            behaviour_profile_distribution={"zone-1": FixedValue("Adult_Default")},
        )

        self.assertEqual(occupant.occupancy_distribution["zone-1"], FixedValue(0))

    def test_forbidden_zone_is_expressed_as_fixed_value_zero(self):

        # No separate "forbidden occupancy zones" field exists --
        # §3.1's compliance review already settled this collapse.
        occupant = OccupantDefinition(occupancy_distribution={"zone-3": FixedValue(0)})

        self.assertIsInstance(occupant.occupancy_distribution["zone-3"], FixedValue)
        self.assertEqual(occupant.occupancy_distribution["zone-3"].value, 0)

    def test_is_frozen(self):

        occupant = OccupantDefinition()

        with self.assertRaises(FrozenInstanceError):
            occupant.occupancy_distribution = {}

    def test_mapping_is_read_only(self):

        occupant = OccupantDefinition(occupancy_distribution={"zone-1": FixedValue(5)})

        with self.assertRaises(TypeError):
            occupant.occupancy_distribution["zone-1"] = FixedValue(6)

    def test_equality(self):

        self.assertEqual(
            OccupantDefinition(occupancy_distribution={"zone-1": FixedValue(5)}),
            OccupantDefinition(occupancy_distribution={"zone-1": FixedValue(5)}),
        )

    def test_round_trip(self):

        occupant = OccupantDefinition(
            occupancy_distribution={"zone-1": UniformRange(3, 8, discrete=True)},
            behaviour_profile_distribution={
                "zone-1": WeightedOptions({"Adult_Default": 0.8, "Child_Default": 0.2}),
            },
        )
        restored = OccupantDefinition.from_dict(occupant.to_dict())

        self.assertEqual(occupant, restored)


class EngineeringConstraintsTests(unittest.TestCase):

    def test_construction(self):

        engineering = EngineeringConstraints(
            door_state_distribution={"door-1": FixedValue(DoorState.LOCKED.name)},
            min_open_exits=2,
        )

        self.assertEqual(engineering.min_open_exits, 2)

    def test_always_open_and_always_closed_are_fixed_value_not_separate_fields(self):

        engineering = EngineeringConstraints(
            exit_state_distribution={
                "exit-1": FixedValue(True),  # "Always Open"
                "exit-2": FixedValue(False),  # "Always Closed"
                "exit-3": WeightedOptions({True: 0.5, False: 0.5}),  # "Random"
            },
        )

        self.assertIsInstance(engineering.exit_state_distribution["exit-1"], FixedValue)
        self.assertIsInstance(engineering.exit_state_distribution["exit-3"], WeightedOptions)

    def test_is_frozen(self):

        engineering = EngineeringConstraints()

        with self.assertRaises(FrozenInstanceError):
            engineering.min_open_exits = 5

    def test_every_category_mapping_is_read_only(self):

        engineering = EngineeringConstraints(
            door_state_distribution={"door-1": FixedValue(DoorState.OPEN.name)},
        )

        with self.assertRaises(TypeError):
            engineering.door_state_distribution["door-1"] = FixedValue(DoorState.CLOSED.name)

    def test_round_trip(self):

        engineering = EngineeringConstraints(
            door_state_distribution={
                "door-1": WeightedOptions(
                    {DoorState.OPEN.name: 0.5, DoorState.CLOSED.name: 0.3,
                     DoorState.LOCKED.name: 0.2},
                ),
            },
            exit_state_distribution={"exit-1": FixedValue(True)},
            min_open_exits=1,
            stair_state_distribution={"stair-1": FixedValue(StairAvailability.AVAILABLE.name)},
            obstacle_state_distribution={"obs-1": FixedValue(PresenceState.ACTIVE.name)},
            camera_state_distribution={"cam-1": FixedValue(DeviceAvailability.AVAILABLE.name)},
            detector_state_distribution={"det-1": FixedValue(DeviceAvailability.FAILED.name)},
        )
        restored = EngineeringConstraints.from_dict(engineering.to_dict())

        self.assertEqual(engineering, restored)

    def test_round_trip_of_defaults(self):

        restored = EngineeringConstraints.from_dict(EngineeringConstraints().to_dict())

        self.assertEqual(restored, EngineeringConstraints())


class EventTemplateTests(unittest.TestCase):

    def test_construction(self):

        template = EventTemplate(
            target_type="camera",
            target_id="cam-1",
            event_type="fail",
            occurs=WeightedOptions({True: 0.3, False: 0.7}),
            time=UniformRange(90, 600),
        )

        self.assertEqual(template.target_type, "camera")
        self.assertIsInstance(template.occurs, WeightedOptions)

    def test_is_not_a_resolved_scenario_event(self):

        # This module must never import scenario.ScenarioEvent --
        # verified structurally: EventTemplate has occurs/time
        # Distributions, not a single resolved time: float.
        template = EventTemplate(
            target_type="door", target_id="door-1", event_type="close",
            occurs=FixedValue(True), time=FixedValue(90.0),
        )

        self.assertIsInstance(template.time, FixedValue)
        self.assertNotIsInstance(template.time, float)

    def test_is_frozen(self):

        template = EventTemplate(
            target_type="door", target_id="door-1", event_type="close",
            occurs=FixedValue(True), time=FixedValue(90.0),
        )

        with self.assertRaises(FrozenInstanceError):
            template.event_type = "open"

    def test_parameters_mapping_is_read_only(self):

        template = EventTemplate(
            target_type="door", target_id="door-1", event_type="close",
            occurs=FixedValue(True), time=FixedValue(90.0), parameters={"reason": "drill"},
        )

        with self.assertRaises(TypeError):
            template.parameters["reason"] = "changed"

    def test_round_trip(self):

        template = EventTemplate(
            target_type="camera",
            target_id="cam-1",
            event_type="fail",
            occurs=WeightedOptions({True: 0.3, False: 0.7}),
            time=UniformRange(90, 600),
            parameters={"severity": "partial"},
        )
        restored = EventTemplate.from_dict(template.to_dict())

        self.assertEqual(template, restored)


class ScenarioDefinitionTests(unittest.TestCase):

    def test_construction_aggregates_every_category(self):

        definition = make_definition()

        self.assertIsInstance(definition.fire, FireDefinition)
        self.assertIsInstance(definition.engineering, EngineeringConstraints)
        self.assertIsInstance(definition.occupant, OccupantDefinition)
        self.assertEqual(len(definition.event_templates), 1)
        self.assertEqual(definition.seed, 42)

    def test_defaults(self):

        definition = ScenarioDefinition(fire=make_fire_definition())

        self.assertEqual(definition.engineering, EngineeringConstraints())
        self.assertEqual(definition.occupant, OccupantDefinition())
        self.assertEqual(definition.event_templates, ())
        self.assertIsNone(definition.seed)

    def test_event_templates_coerced_to_tuple(self):

        definition = make_definition()

        self.assertIsInstance(definition.event_templates, tuple)

    def test_is_frozen(self):

        definition = make_definition()

        with self.assertRaises(FrozenInstanceError):
            definition.seed = 99

    def test_equality(self):

        self.assertEqual(make_definition(), make_definition())
        self.assertNotEqual(make_definition(), make_definition(seed=7))

    def test_round_trip(self):

        definition = make_definition()
        restored = ScenarioDefinition.from_dict(definition.to_dict())

        self.assertEqual(definition, restored)

    def test_nested_serialization_shape(self):

        data = make_definition().to_dict()

        self.assertIsInstance(data["fire"], dict)
        self.assertIsInstance(data["engineering"], dict)
        self.assertIsInstance(data["occupant"], dict)
        self.assertIsInstance(data["event_templates"], list)
        self.assertIsInstance(data["event_templates"][0], dict)
        self.assertEqual(data["event_templates"][0]["occurs"]["kind"], "FixedValue")

    def test_to_dict_contains_only_plain_python_types(self):

        def assert_plain(value):

            if isinstance(value, dict):
                for v in value.values():
                    assert_plain(v)
            elif isinstance(value, list):
                for v in value:
                    assert_plain(v)
            else:
                self.assertIsInstance(value, (str, int, float, bool, type(None)))

        assert_plain(make_definition().to_dict())


class SelfValidationTests(unittest.TestCase):

    def test_valid_definition_reports_no_errors(self):

        report = validate_definition(make_definition(), building=make_building())

        self.assertTrue(report.is_valid)
        self.assertEqual(report.errors, [])

    def test_invalid_range_is_detected(self):

        fire = make_fire_definition(growth_parameter_distribution=UniformRange(400.0, 100.0))
        report = validate_definition(make_definition(fire=fire))

        self.assertFalse(report.is_valid)
        self.assertTrue(report.by_code("invalid_range"))

    def test_negative_weight_is_detected(self):

        engineering = EngineeringConstraints(
            exit_state_distribution={"exit-1": WeightedOptions({True: -0.5, False: 1.5})},
        )
        report = validate_definition(make_definition(engineering=engineering))

        self.assertTrue(report.by_code("negative_weight"))

    def test_non_positive_weight_total_is_detected(self):

        engineering = EngineeringConstraints(
            exit_state_distribution={"exit-1": WeightedOptions({True: 0.0, False: 0.0})},
        )
        report = validate_definition(make_definition(engineering=engineering))

        self.assertTrue(report.by_code("non_positive_weight_total"))

    def test_empty_weighted_options_is_detected(self):

        occupant = OccupantDefinition(occupancy_distribution={"zone-1": WeightedOptions({})})
        report = validate_definition(make_definition(occupant=occupant))

        self.assertTrue(report.by_code("empty_weighted_options"))

    def test_negative_fixed_occupant_count_is_detected(self):

        occupant = OccupantDefinition(occupancy_distribution={"zone-1": FixedValue(-3)})
        report = validate_definition(make_definition(occupant=occupant))

        self.assertTrue(report.by_code("negative_occupant_count"))

    def test_negative_occupant_range_is_detected(self):

        occupant = OccupantDefinition(occupancy_distribution={"zone-1": UniformRange(-2, 5)})
        report = validate_definition(make_definition(occupant=occupant))

        self.assertTrue(report.by_code("negative_occupant_count"))

    def test_missing_event_template_field_is_detected(self):

        template = EventTemplate(
            target_type="door", target_id="", event_type="close",
            occurs=FixedValue(True), time=FixedValue(60.0),
        )
        report = validate_definition(make_definition(event_templates=(template,)))

        self.assertTrue(report.by_code("missing_required_field"))

    def test_ignition_allow_forbid_overlap_is_detected(self):

        fire = make_fire_definition(
            allowed_ignition_zone_ids={"zone-1"}, forbidden_ignition_zone_ids={"zone-1"},
        )
        report = validate_definition(make_definition(fire=fire))

        self.assertTrue(report.by_code("ignition_allow_forbid_overlap"))

    def test_unknown_id_is_detected_when_building_supplied(self):

        fire = make_fire_definition(allowed_ignition_zone_ids={"zone-does-not-exist"})
        report = validate_definition(make_definition(fire=fire), building=make_building())

        self.assertTrue(report.by_code("unknown_id"))

    def test_building_dependent_checks_are_skipped_without_a_building(self):

        fire = make_fire_definition(allowed_ignition_zone_ids={"zone-does-not-exist"})
        report = validate_definition(make_definition(fire=fire))

        self.assertEqual(report.by_code("unknown_id"), [])

    def test_min_open_exits_infeasible_is_detected(self):

        engineering = EngineeringConstraints(
            exit_state_distribution={
                "exit-1": FixedValue(False),
                "exit-2": FixedValue(False),
            },
            min_open_exits=1,
        )
        report = validate_definition(
            make_definition(engineering=engineering), building=make_building(),
        )

        self.assertTrue(report.by_code("min_open_exits_infeasible"))

    def test_min_open_exits_feasible_with_a_weighted_open_chance_is_not_flagged(self):

        engineering = EngineeringConstraints(
            exit_state_distribution={
                "exit-1": WeightedOptions({True: 0.1, False: 0.9}),
                "exit-2": FixedValue(False),
            },
            min_open_exits=1,
        )
        report = validate_definition(
            make_definition(engineering=engineering), building=make_building(),
        )

        self.assertEqual(report.by_code("min_open_exits_infeasible"), [])

    def test_allowed_floor_with_no_eligible_zone_is_detected(self):

        fire = make_fire_definition(
            allowed_ignition_floor_ids={"floor-1"},
            forbidden_ignition_zone_ids={"zone-1", "zone-2"},
            allowed_ignition_zone_ids=set(),
        )
        report = validate_definition(make_definition(fire=fire), building=make_building())

        self.assertTrue(report.by_code("allowed_floor_has_no_eligible_zone"))

    def test_deliberately_does_not_check_behaviour_profile_registry_membership(self):

        # §3.4/§8: an unrecognized profile id is caught at simulation
        # time by the Behaviour Layer, never here.
        occupant = OccupantDefinition(
            behaviour_profile_distribution={"zone-1": FixedValue("Totally_Made_Up_Profile")},
        )
        report = validate_definition(make_definition(occupant=occupant), building=make_building())

        self.assertTrue(report.is_valid)


class ScenarioDefinitionPackageDependencyDirectionTests(unittest.TestCase):

    # scenario_definition/ must stay a rulebook only -- architecture
    # doc §3/§12: must not import scenario_generator, scenario_validator,
    # scenario (the resolved-output model package), simulator, sandbox,
    # behavior, designer, navigation, or random/numpy. May import
    # models. Enforced the same regex-scan way tests/test_sensors.py
    # already enforces its own package's dependency direction.

    def test_package_never_imports_generation_validation_simulation_or_randomness(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_definition"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(scenario_generator|scenario_validator|scenario\b|simulator|sandbox|"
            r"behavior|behavior_library|designer|navigation|perception|ai_decision|"
            r"rl|random|numpy)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario_definition/{path.name} imports a package this rulebook "
                f"must never depend on -- it declares what may be sampled and samples "
                f"nothing itself",
            )

    def test_package_may_import_models_but_nothing_deeper(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_definition"

        forbidden = r"^\s*(from|import)\s+(hazard|hazard_evolution|occupancy|sensors|fire_growth|pathfinding)\b"

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario_definition/{path.name} imports simulation/runtime machinery "
                f"-- self-validation only needs models for id existence checks",
            )

    def test_distributions_module_never_imports_random(self):

        import pathlib

        text = (
            pathlib.Path(__file__).resolve().parent.parent
            / "scenario_definition" / "distributions.py"
        ).read_text()

        self.assertNotIn("import random", text)

    def test_package_performs_no_file_io(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_definition"

        forbidden = r"\bopen\s*\(|\.write\s*\(|\.read\s*\(|\bjson\.(load|dump)\s*\("

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text),
                f"scenario_definition/{path.name} appears to perform file I/O",
            )


if __name__ == "__main__":
    unittest.main()
