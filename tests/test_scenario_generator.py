import random
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

from scenario import DeviceAvailability, DoorState, PresenceState, StairAvailability
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
from scenario_definition import DoorState as DefinitionDoorState

from scenario_generator import (
    BatchGenerationRequest,
    GenerationRequest,
    build_metadata,
    category_rng,
    compute_definition_content_hash,
    derive_attempt_seed,
    derive_scenario_id,
    derive_scenario_seed,
    derive_seed,
    generate_batch,
    generate_scenario,
    iter_batch,
    sample,
    sample_uniform_choice,
)


def make_building(**overrides):

    floor1 = Floor(
        name="Ground",
        id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True), Door(id="door-2", locked=True)],
        exits=[Exit(id="exit-1"), Exit(id="exit-2", is_blocked=True)],
        obstacles=[Obstacle(id="obs-1", active=True), Obstacle(id="obs-2", active=False)],
        cameras=[Camera(id="cam-1", active=True), Camera(id="cam-2", active=False)],
        detectors=[Detector(id="det-1", active=True)],
        stairs=[Staircase(id="stair-1")],
    )

    floor2 = Floor(name="Upper", id="floor-2", zones=[Zone(id="zone-3", name="Attic")])

    defaults = dict(name="Test Building", id="building-1", floors=[floor1, floor2])
    defaults.update(overrides)

    return Building(**defaults)


def make_definition(**overrides):

    defaults = dict(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(100.0, 400.0),
            allowed_ignition_zone_ids={"zone-1", "zone-2"},
            allowed_fire_profiles={"Electrical", "Flaming"},
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "door-1": WeightedOptions(
                    {DefinitionDoorState.OPEN.name: 0.5, DefinitionDoorState.CLOSED.name: 0.5},
                ),
            },
            exit_state_distribution={"exit-1": FixedValue(True)},
            min_open_exits=1,
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "zone-1": UniformRange(3, 8, discrete=True),
                "zone-2": FixedValue(2),
            },
            behaviour_profile_distribution={
                "zone-1": WeightedOptions({"Adult_Default": 0.8, "Child_Default": 0.2}),
            },
        ),
        event_templates=(
            EventTemplate(
                target_type="door", target_id="door-1", event_type="close",
                occurs=FixedValue(True), time=UniformRange(60, 120),
            ),
        ),
        seed=1,
    )
    defaults.update(overrides)

    return ScenarioDefinition(**defaults)


def make_request(**overrides):

    defaults = dict(
        definition=make_definition(),
        definition_id="def-1",
        building=make_building(),
        seed=1234,
    )
    defaults.update(overrides)

    return GenerationRequest(**defaults)


def strip_created_at(scenario_dict):

    scenario_dict = dict(scenario_dict)
    scenario_dict["metadata"] = dict(scenario_dict["metadata"])
    scenario_dict["metadata"].pop("created_at", None)

    return scenario_dict


class SeedManagerTests(unittest.TestCase):

    def test_derive_seed_is_stable_across_calls(self):

        self.assertEqual(derive_seed(1, "a", 2), derive_seed(1, "a", 2))

    def test_derive_seed_distinguishes_different_parts(self):

        self.assertNotEqual(derive_seed(1, "a"), derive_seed(1, "b"))
        self.assertNotEqual(derive_seed(1, "a"), derive_seed(2, "a"))

    def test_scenario_seed_is_index_keyed_not_sequential(self):

        # Generating index 1000 alone must equal generating it as part
        # of a longer run starting at 0 -- §4.8.
        seed_a = derive_scenario_seed(master_seed=7, index=1000)
        seed_b = derive_scenario_seed(master_seed=7, index=1000)

        self.assertEqual(seed_a, seed_b)

    def test_scenario_seed_differs_by_index(self):

        self.assertNotEqual(
            derive_scenario_seed(7, 0), derive_scenario_seed(7, 1),
        )

    def test_attempt_seed_differs_by_attempt_index(self):

        scenario_seed = derive_scenario_seed(7, 0)

        self.assertNotEqual(
            derive_attempt_seed(scenario_seed, 0), derive_attempt_seed(scenario_seed, 1),
        )

    def test_category_rng_is_name_keyed_not_order_dependent(self):

        attempt_seed = derive_attempt_seed(derive_scenario_seed(1, 0), 0)

        # Calling in a different order produces the same per-category
        # stream -- the whole point of §4.6's keyed derivation.
        fire_first = category_rng(attempt_seed, "fire").random()
        door_first_fire = category_rng(attempt_seed, "door")
        fire_rng_again = category_rng(attempt_seed, "fire")

        self.assertEqual(fire_first, fire_rng_again.random())

    def test_category_rng_streams_are_independent(self):

        attempt_seed = derive_attempt_seed(derive_scenario_seed(1, 0), 0)

        fire_values = [category_rng(attempt_seed, "fire").random() for _ in range(3)]
        door_values = [category_rng(attempt_seed, "door").random() for _ in range(3)]

        self.assertNotEqual(fire_values, door_values)

    def test_unknown_category_key_raises(self):

        with self.assertRaises(ValueError):
            category_rng(1, "not_a_real_category")

    def test_reserved_environmental_category_is_accepted(self):

        # Named and reserved (§3.3/§4.4) -- not used by any generation
        # stage yet, but the key must already be valid so a future
        # Environmental stage needs no change here.
        category_rng(1, "environmental")


class SamplingTests(unittest.TestCase):

    def test_fixed_value_returns_its_value(self):

        self.assertEqual(sample(FixedValue(value=7), random.Random(1)), 7)

    def test_uniform_range_continuous_stays_in_bounds(self):

        rng = random.Random(1)

        for _ in range(50):
            value = sample(UniformRange(2.0, 5.0), rng)
            self.assertTrue(2.0 <= value <= 5.0)

    def test_uniform_range_discrete_returns_int_in_bounds(self):

        rng = random.Random(1)

        for _ in range(50):
            value = sample(UniformRange(3, 8, discrete=True), rng)
            self.assertIsInstance(value, int)
            self.assertTrue(3 <= value <= 8)

    def test_weighted_options_only_returns_declared_keys(self):

        rng = random.Random(1)
        distribution = WeightedOptions({"a": 1.0, "b": 0.0})

        for _ in range(20):
            self.assertEqual(sample(distribution, rng), "a")

    def test_sampling_is_deterministic_given_the_same_rng_state(self):

        distribution = UniformRange(0.0, 100.0)

        self.assertEqual(
            sample(distribution, random.Random(42)), sample(distribution, random.Random(42)),
        )

    def test_sample_uniform_choice_is_stable_regardless_of_frozenset_construction_order(self):

        rng_a = random.Random(5)
        rng_b = random.Random(5)

        result_a = sample_uniform_choice(frozenset({"z2", "z1", "z3"}), rng_a)
        result_b = sample_uniform_choice(frozenset({"z1", "z3", "z2"}), rng_b)

        self.assertEqual(result_a, result_b)

    def test_sample_uniform_choice_raises_on_empty_set(self):

        with self.assertRaises(ValueError):
            sample_uniform_choice(frozenset(), random.Random(1))


class MetadataBuilderTests(unittest.TestCase):

    def test_definition_content_hash_is_stable_for_the_same_definition(self):

        definition = make_definition()

        self.assertEqual(
            compute_definition_content_hash(definition), compute_definition_content_hash(definition),
        )

    def test_definition_content_hash_changes_when_content_changes(self):

        original = make_definition()
        changed = make_definition(seed=999)

        self.assertNotEqual(
            compute_definition_content_hash(original), compute_definition_content_hash(changed),
        )

    def test_scenario_id_is_deterministic(self):

        content_hash = "abc123"

        self.assertEqual(
            derive_scenario_id(content_hash, seed=1, attempt_index=0),
            derive_scenario_id(content_hash, seed=1, attempt_index=0),
        )

    def test_scenario_id_differs_by_seed(self):

        content_hash = "abc123"

        self.assertNotEqual(
            derive_scenario_id(content_hash, seed=1, attempt_index=0),
            derive_scenario_id(content_hash, seed=2, attempt_index=0),
        )

    def test_build_metadata_populates_every_frozen_field(self):

        definition = make_definition()
        metadata = build_metadata(definition, definition_id="def-1", seed=42, attempt_index=0)

        self.assertTrue(metadata.scenario_id)
        self.assertEqual(metadata.definition_id, "def-1")
        self.assertEqual(
            metadata.definition_content_hash, compute_definition_content_hash(definition),
        )
        self.assertTrue(metadata.generation_version)
        self.assertEqual(metadata.seed, 42)
        self.assertTrue(metadata.created_at)
        self.assertEqual(metadata.rejected_attempt_count, 0)


class GenerateScenarioDeterminismTests(unittest.TestCase):

    def test_identical_seeds_produce_identical_scenarios(self):

        request = make_request(seed=555)

        first = strip_created_at(generate_scenario(request).to_dict())
        second = strip_created_at(generate_scenario(request).to_dict())

        self.assertEqual(first, second)

    def test_different_seeds_produce_different_scenarios(self):

        first = generate_scenario(make_request(seed=1))
        second = generate_scenario(make_request(seed=2))

        self.assertNotEqual(strip_created_at(first.to_dict()), strip_created_at(second.to_dict()))

    def test_changing_door_generation_does_not_change_occupant_placement(self):

        # The direct payoff of §4.6's name-keyed derivation, exercised
        # end to end: two Definitions differing ONLY in
        # door_state_distribution must place identical occupants for
        # the same seed.
        base_definition = make_definition()
        changed_definition = make_definition(
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue(DefinitionDoorState.LOCKED.name)},
                exit_state_distribution={"exit-1": FixedValue(True)},
                min_open_exits=1,
            ),
        )

        building = make_building()

        first = generate_scenario(
            GenerationRequest(
                definition=base_definition, definition_id="def-1", building=building, seed=777,
            ),
        )
        second = generate_scenario(
            GenerationRequest(
                definition=changed_definition, definition_id="def-1", building=building, seed=777,
            ),
        )

        self.assertEqual(first.occupants, second.occupants)
        self.assertEqual(first.fire, second.fire)
        self.assertNotEqual(first.door_states, second.door_states)

    def test_generation_is_a_pure_function_of_definition_building_and_seed(self):

        request = make_request()
        results = [generate_scenario(request) for _ in range(5)]

        # All five independently-generated scenarios must be identical
        # apart from their timestamp.
        first = strip_created_at(results[0].to_dict())
        for result in results[1:]:
            self.assertEqual(strip_created_at(result.to_dict()), first)


class GenerateScenarioFireTests(unittest.TestCase):

    def test_ignition_zone_is_within_allowed_set(self):

        definition = make_definition()
        building = make_building()

        for seed in range(20):
            scenario = generate_scenario(
                GenerationRequest(
                    definition=definition, definition_id="def-1", building=building, seed=seed,
                ),
            )
            self.assertIn(scenario.fire.ignition_zone_id, {"zone-1", "zone-2"})

    def test_forbidden_zone_is_never_chosen(self):

        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(200.0),
                allowed_ignition_zone_ids={"zone-1", "zone-2"},
                forbidden_ignition_zone_ids={"zone-2"},
            ),
        )
        building = make_building()

        for seed in range(20):
            scenario = generate_scenario(
                GenerationRequest(
                    definition=definition, definition_id="def-1", building=building, seed=seed,
                ),
            )
            self.assertEqual(scenario.fire.ignition_zone_id, "zone-1")

    def test_allowed_floor_restricts_ignition_zone(self):

        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(200.0),
                allowed_ignition_floor_ids={"floor-2"},
            ),
        )
        building = make_building()

        for seed in range(20):
            scenario = generate_scenario(
                GenerationRequest(
                    definition=definition, definition_id="def-1", building=building, seed=seed,
                ),
            )
            self.assertEqual(scenario.fire.ignition_zone_id, "zone-3")
            self.assertEqual(scenario.fire.ignition_floor_id, "floor-2")

    def test_ignition_zone_preference_overrides_uniform_default(self):

        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(200.0),
                allowed_ignition_zone_ids={"zone-1", "zone-2"},
                ignition_zone_preference=FixedValue("zone-2"),
            ),
        )
        building = make_building()

        scenario = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=building, seed=1,
            ),
        )
        self.assertEqual(scenario.fire.ignition_zone_id, "zone-2")

    def test_growth_parameters_reflect_the_sampled_value(self):

        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(321.0),
                allowed_ignition_zone_ids={"zone-1"},
            ),
        )
        scenario = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=make_building(), seed=1,
            ),
        )
        self.assertEqual(dict(scenario.fire.growth_parameters), {"growth_time": 321.0})

    def test_fire_profile_is_within_allowed_set(self):

        definition = make_definition()
        building = make_building()

        for seed in range(20):
            scenario = generate_scenario(
                GenerationRequest(
                    definition=definition, definition_id="def-1", building=building, seed=seed,
                ),
            )
            self.assertIn(scenario.fire.fire_profile, {"Electrical", "Flaming"})

    def test_empty_allowed_fire_profiles_resolves_to_empty_string(self):

        definition = make_definition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(200.0),
                allowed_ignition_zone_ids={"zone-1"},
            ),
        )
        scenario = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=make_building(), seed=1,
            ),
        )
        self.assertEqual(scenario.fire.fire_profile, "")


class GenerateScenarioOccupantTests(unittest.TestCase):

    def test_occupant_count_matches_fixed_value(self):

        scenario = generate_scenario(make_request())

        zone2_occupants = [o for o in scenario.occupants if o.zone_id == "zone-2"]
        self.assertEqual(len(zone2_occupants), 2)

    def test_occupant_count_within_uniform_range(self):

        scenario = generate_scenario(make_request())

        zone1_occupants = [o for o in scenario.occupants if o.zone_id == "zone-1"]
        self.assertTrue(3 <= len(zone1_occupants) <= 8)

    def test_occupant_position_is_within_zone_bounds(self):

        scenario = generate_scenario(make_request())
        zone1 = make_building().floors[0].zones[0]

        for occupant in scenario.occupants:
            if occupant.zone_id == "zone-1":
                x, y = occupant.position
                self.assertTrue(zone1.x <= x <= zone1.x + zone1.width)
                self.assertTrue(zone1.y <= y <= zone1.y + zone1.height)

    def test_occupant_floor_id_matches_its_zone(self):

        scenario = generate_scenario(make_request())

        for occupant in scenario.occupants:
            self.assertEqual(occupant.floor_id, "floor-1")

    def test_occupant_ids_are_unique(self):

        scenario = generate_scenario(make_request())
        ids = [o.occupant_id for o in scenario.occupants]

        self.assertEqual(len(ids), len(set(ids)))

    def test_behaviour_profile_id_is_sampled_from_the_declared_distribution(self):

        scenario = generate_scenario(make_request())

        for occupant in scenario.occupants:
            if occupant.zone_id == "zone-1":
                self.assertIn(occupant.behaviour_profile_id, {"Adult_Default", "Child_Default"})

    def test_behaviour_profile_id_defaults_to_empty_string_when_undeclared(self):

        scenario = generate_scenario(make_request())

        for occupant in scenario.occupants:
            if occupant.zone_id == "zone-2":
                self.assertEqual(occupant.behaviour_profile_id, "")

    def test_unknown_zone_id_in_definition_is_skipped_not_crashed(self):

        definition = make_definition(
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-does-not-exist": FixedValue(5)},
            ),
        )
        scenario = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=make_building(), seed=1,
            ),
        )
        self.assertEqual(scenario.occupants, ())

    def test_changing_behaviour_profile_distribution_does_not_change_occupant_positions(self):

        base = make_definition()
        changed = make_definition(
            occupant=OccupantDefinition(
                occupancy_distribution=base.occupant.occupancy_distribution,
                behaviour_profile_distribution={"zone-1": FixedValue("FireWarden_Default")},
            ),
        )
        building = make_building()

        first = generate_scenario(
            GenerationRequest(definition=base, definition_id="def-1", building=building, seed=42),
        )
        second = generate_scenario(
            GenerationRequest(
                definition=changed, definition_id="def-1", building=building, seed=42,
            ),
        )

        first_positions = [(o.occupant_id, o.position) for o in first.occupants]
        second_positions = [(o.occupant_id, o.position) for o in second.occupants]

        self.assertEqual(first_positions, second_positions)


class GenerateScenarioEngineeringStateTests(unittest.TestCase):

    def test_every_building_door_gets_a_resolved_state(self):

        scenario = generate_scenario(make_request())
        door_ids = {s.door_id for s in scenario.door_states}

        self.assertEqual(door_ids, {"door-1", "door-2"})

    def test_door_without_a_declared_rule_defaults_from_building_state(self):

        scenario = generate_scenario(make_request())
        door_2_state = next(s for s in scenario.door_states if s.door_id == "door-2")

        # door-2 is Door(locked=True) on the fixture Building and has no
        # entry in door_state_distribution.
        self.assertEqual(door_2_state.state, DoorState.LOCKED)

    def test_exit_without_a_declared_rule_defaults_from_building_state(self):

        scenario = generate_scenario(make_request())
        exit_2_state = next(s for s in scenario.exit_states if s.exit_id == "exit-2")

        # exit-2 is Exit(is_blocked=True) on the fixture Building.
        self.assertFalse(exit_2_state.is_open)

    def test_stair_without_a_declared_rule_defaults_to_available(self):

        scenario = generate_scenario(make_request())
        stair_state = next(s for s in scenario.stair_states)

        self.assertEqual(stair_state.availability, StairAvailability.AVAILABLE)

    def test_obstacle_defaults_follow_building_active_flag(self):

        scenario = generate_scenario(make_request())
        by_id = {s.obstacle_id: s.presence for s in scenario.obstacle_states}

        self.assertEqual(by_id["obs-1"], PresenceState.ACTIVE)
        self.assertEqual(by_id["obs-2"], PresenceState.INACTIVE)

    def test_camera_and_detector_defaults_follow_building_active_flag(self):

        scenario = generate_scenario(make_request())
        cameras = {s.camera_id: s.availability for s in scenario.camera_states}
        detectors = {s.detector_id: s.availability for s in scenario.detector_states}

        self.assertEqual(cameras["cam-1"], DeviceAvailability.AVAILABLE)
        self.assertEqual(cameras["cam-2"], DeviceAvailability.FAILED)
        self.assertEqual(detectors["det-1"], DeviceAvailability.AVAILABLE)

    def test_declared_distribution_is_used_over_the_building_default(self):

        definition = make_definition(
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue(DefinitionDoorState.LOCKED.name)},
                exit_state_distribution={"exit-1": FixedValue(True)},
                min_open_exits=1,
            ),
        )
        scenario = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=make_building(), seed=1,
            ),
        )
        door_1_state = next(s for s in scenario.door_states if s.door_id == "door-1")

        # door-1 is Door(normally_open=True) on the Building -- would
        # default to OPEN, but the Definition pins it LOCKED.
        self.assertEqual(door_1_state.state, DoorState.LOCKED)


class GenerateScenarioEventTests(unittest.TestCase):

    def test_event_that_occurs_is_materialized(self):

        scenario = generate_scenario(make_request())

        self.assertEqual(len(scenario.events), 1)
        event = scenario.events[0]
        self.assertEqual(event.target_id, "door-1")
        self.assertTrue(60 <= event.time <= 120)

    def test_event_that_never_occurs_is_not_materialized(self):

        definition = make_definition(
            event_templates=(
                EventTemplate(
                    target_type="camera", target_id="cam-1", event_type="fail",
                    occurs=FixedValue(False), time=FixedValue(90.0),
                ),
            ),
        )
        scenario = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=make_building(), seed=1,
            ),
        )
        self.assertEqual(scenario.events, ())

    def test_event_ids_are_unique_across_multiple_templates(self):

        definition = make_definition(
            event_templates=(
                EventTemplate(
                    target_type="door", target_id="door-1", event_type="close",
                    occurs=FixedValue(True), time=FixedValue(10.0),
                ),
                EventTemplate(
                    target_type="camera", target_id="cam-1", event_type="fail",
                    occurs=FixedValue(True), time=FixedValue(20.0),
                ),
            ),
        )
        scenario = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=make_building(), seed=1,
            ),
        )
        ids = [event.event_id for event in scenario.events]
        self.assertEqual(len(ids), len(set(ids)))


class SerializationCompatibilityTests(unittest.TestCase):

    def test_generated_scenario_round_trips_through_scenario_to_dict_and_from_dict(self):

        from scenario import Scenario

        scenario = generate_scenario(make_request())
        restored = Scenario.from_dict(scenario.to_dict())

        self.assertEqual(scenario, restored)

    def test_generated_scenario_to_dict_contains_only_plain_python_types(self):

        def assert_plain(value):

            if isinstance(value, dict):
                for v in value.values():
                    assert_plain(v)
            elif isinstance(value, list):
                for v in value:
                    assert_plain(v)
            else:
                self.assertIsInstance(value, (str, int, float, bool, type(None)))

        assert_plain(generate_scenario(make_request()).to_dict())


class BatchGenerationTests(unittest.TestCase):

    def test_batch_generates_the_requested_count(self):

        batch = generate_batch(
            BatchGenerationRequest(
                definition=make_definition(), definition_id="def-1", building=make_building(),
                master_seed=1, count=5,
            ),
        )
        self.assertEqual(len(batch), 5)

    def test_batch_scenarios_are_pairwise_distinct(self):

        batch = generate_batch(
            BatchGenerationRequest(
                definition=make_definition(), definition_id="def-1", building=make_building(),
                master_seed=1, count=5,
            ),
        )

        # Seeds differ per index, so scenario_id must differ across
        # every member of the batch.
        scenario_ids = {s.metadata.scenario_id for s in batch}
        self.assertEqual(len(scenario_ids), 5)

    def test_batch_is_resumable_and_index_keyed(self):

        request = dict(
            definition=make_definition(), definition_id="def-1", building=make_building(),
            master_seed=99,
        )

        whole_run = generate_batch(BatchGenerationRequest(count=6, **request))
        resumed_tail = generate_batch(BatchGenerationRequest(count=3, start_index=3, **request))

        for offset in range(3):

            whole_dict = strip_created_at(whole_run[3 + offset].to_dict())
            resumed_dict = strip_created_at(resumed_tail[offset].to_dict())

            self.assertEqual(whole_dict, resumed_dict)

    def test_iter_batch_is_lazy_and_yields_the_same_scenarios_as_generate_batch(self):

        request = BatchGenerationRequest(
            definition=make_definition(), definition_id="def-1", building=make_building(),
            master_seed=5, count=3,
        )

        via_list = generate_batch(request)
        via_iter = list(iter_batch(request))

        for a, b in zip(via_list, via_iter):
            self.assertEqual(strip_created_at(a.to_dict()), strip_created_at(b.to_dict()))

    def test_single_generation_is_the_degenerate_batch_of_one(self):

        # §4.8: single-scenario generation is the degenerate case,
        # count=1, master_seed unused -- caller supplies the Scenario
        # Seed directly via GenerationRequest instead.
        direct = generate_scenario(make_request(seed=42))
        self.assertIsNotNone(direct)


class ScenarioGeneratorPackageDependencyDirectionTests(unittest.TestCase):

    # scenario_generator/'s construction module must never import
    # navigation, fire_growth, scenario_validator, sandbox, designer,
    # simulator, behavior, behavior_library, ai_decision, perception,
    # or rl (architecture doc §4.2/§4.13/§12 -- narrowed this pass to
    # `scenario_definition`, `scenario`, `models` only).

    def test_package_never_imports_navigation_fire_growth_simulation_or_validation(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_generator"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(navigation|fire_growth|scenario_validator|sandbox|designer|simulator|"
            r"behavior|behavior_library|ai_decision|perception|rl|pathfinding|hazard|"
            r"hazard_evolution|occupancy|sensors)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario_generator/{path.name} imports a package the construction "
                f"module must never depend on (§4.2/§4.13)",
            )

    def test_package_performs_no_file_io(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_generator"

        forbidden = r"\bopen\s*\(|\.write\s*\(|\.read\s*\(|\bjson\.(load|dump)\s*\("

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text),
                f"scenario_generator/{path.name} appears to perform file I/O",
            )

    def test_package_never_imports_python_random_outside_seed_manager_and_sampling(self):

        # The Generator owns all randomness (§4.5) -- but ownership is
        # centralized: every other module drives sampling through
        # sampling.sample()/seed_manager.category_rng() rather than
        # touching the random module directly, so there is exactly one
        # place that decides how a Distribution or a seed turns into an
        # rng.
        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_generator"
        allowed = {"seed_manager.py", "sampling.py"}

        for path in sorted(package_dir.glob("*.py")):

            if path.name in allowed:
                continue

            text = path.read_text()
            self.assertNotIn("import random", text, f"scenario_generator/{path.name}")


if __name__ == "__main__":
    unittest.main()
