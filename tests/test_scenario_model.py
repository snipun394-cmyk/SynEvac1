import unittest

from dataclasses import FrozenInstanceError

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


def make_metadata(**overrides):

    defaults = dict(
        scenario_id="scn-1",
        definition_id="def-1",
        definition_content_hash="abc123",
        generation_version="v1",
        seed=42,
        created_at="2026-07-13T00:00:00",
        rejected_attempt_count=3,
        extra={"note": "worked example"},
    )
    defaults.update(overrides)

    return ScenarioMetadata(**defaults)


def make_occupant(**overrides):

    defaults = dict(
        occupant_id="occ-1",
        zone_id="zone-1",
        floor_id="floor-1",
        position=(1.5, 2.5),
        behaviour_profile_id="Adult_Default",
    )
    defaults.update(overrides)

    return ScenarioOccupant(**defaults)


def make_fire(**overrides):

    defaults = dict(
        ignition_zone_id="zone-1",
        ignition_floor_id="floor-1",
        fire_profile="Electrical",
        growth_parameters={"growth_time": 300.0},
    )
    defaults.update(overrides)

    return ScenarioFire(**defaults)


def make_event(**overrides):

    defaults = dict(
        event_id="evt-1",
        target_type="door",
        target_id="door-1",
        event_type="close",
        time=90.0,
        parameters={"reason": "drill"},
    )
    defaults.update(overrides)

    return ScenarioEvent(**defaults)


class ScenarioMetadataTests(unittest.TestCase):

    def test_construction(self):

        metadata = make_metadata()

        self.assertEqual(metadata.scenario_id, "scn-1")
        self.assertEqual(metadata.definition_content_hash, "abc123")
        self.assertEqual(metadata.seed, 42)
        self.assertEqual(metadata.rejected_attempt_count, 3)

    def test_default_rejected_attempt_count_and_extra(self):

        metadata = ScenarioMetadata(
            scenario_id="scn-1",
            definition_id="def-1",
            definition_content_hash="abc123",
            generation_version="v1",
            seed=42,
            created_at="2026-07-13T00:00:00",
        )

        self.assertEqual(metadata.rejected_attempt_count, 0)
        self.assertEqual(dict(metadata.extra), {})

    def test_equality(self):

        self.assertEqual(make_metadata(), make_metadata())
        self.assertNotEqual(make_metadata(), make_metadata(seed=43))

    def test_is_frozen(self):

        metadata = make_metadata()

        with self.assertRaises(FrozenInstanceError):
            metadata.seed = 99

    def test_extra_mapping_is_read_only(self):

        metadata = make_metadata()

        with self.assertRaises(TypeError):
            metadata.extra["note"] = "changed"

    def test_round_trip(self):

        metadata = make_metadata()
        restored = ScenarioMetadata.from_dict(metadata.to_dict())

        self.assertEqual(metadata, restored)

    def test_to_dict_is_plain_python_types(self):

        data = make_metadata().to_dict()

        self.assertIsInstance(data, dict)
        self.assertIsInstance(data["extra"], dict)
        self.assertNotIsInstance(data["extra"], type(make_metadata().extra))

    def test_from_dict_defaults_missing_optional_fields(self):

        data = make_metadata().to_dict()
        del data["rejected_attempt_count"]
        del data["extra"]

        restored = ScenarioMetadata.from_dict(data)

        self.assertEqual(restored.rejected_attempt_count, 0)
        self.assertEqual(dict(restored.extra), {})


class ScenarioOccupantTests(unittest.TestCase):

    def test_construction(self):

        occupant = make_occupant()

        self.assertEqual(occupant.occupant_id, "occ-1")
        self.assertEqual(occupant.position, (1.5, 2.5))
        self.assertEqual(occupant.behaviour_profile_id, "Adult_Default")

    def test_equality(self):

        self.assertEqual(make_occupant(), make_occupant())
        self.assertNotEqual(make_occupant(), make_occupant(zone_id="zone-2"))

    def test_is_frozen(self):

        occupant = make_occupant()

        with self.assertRaises(FrozenInstanceError):
            occupant.position = (0.0, 0.0)

    def test_round_trip(self):

        occupant = make_occupant()
        restored = ScenarioOccupant.from_dict(occupant.to_dict())

        self.assertEqual(occupant, restored)

    def test_behaviour_profile_id_is_carried_opaquely(self):

        # This package must never interpret the id -- round-tripping an
        # arbitrary, unregistered-looking string must work exactly like
        # any other string (§8: opaque, never resolved).
        occupant = make_occupant(behaviour_profile_id="Some_Unregistered_Profile")
        restored = ScenarioOccupant.from_dict(occupant.to_dict())

        self.assertEqual(restored.behaviour_profile_id, occupant.behaviour_profile_id)


class ScenarioFireTests(unittest.TestCase):

    def test_construction(self):

        fire = make_fire()

        self.assertEqual(fire.ignition_zone_id, "zone-1")
        self.assertEqual(fire.fire_profile, "Electrical")
        self.assertEqual(dict(fire.growth_parameters), {"growth_time": 300.0})

    def test_default_growth_parameters_is_empty(self):

        fire = ScenarioFire(
            ignition_zone_id="zone-1", ignition_floor_id="floor-1", fire_profile="Flaming",
        )

        self.assertEqual(dict(fire.growth_parameters), {})

    def test_equality(self):

        self.assertEqual(make_fire(), make_fire())
        self.assertNotEqual(make_fire(), make_fire(fire_profile="Flaming"))

    def test_is_frozen(self):

        fire = make_fire()

        with self.assertRaises(FrozenInstanceError):
            fire.fire_profile = "Flaming"

    def test_growth_parameters_mapping_is_read_only(self):

        fire = make_fire()

        with self.assertRaises(TypeError):
            fire.growth_parameters["growth_time"] = 999.0

    def test_round_trip(self):

        fire = make_fire()
        restored = ScenarioFire.from_dict(fire.to_dict())

        self.assertEqual(fire, restored)


class ScenarioEventTests(unittest.TestCase):

    def test_construction(self):

        event = make_event()

        self.assertEqual(event.target_type, "door")
        self.assertEqual(event.time, 90.0)
        self.assertEqual(dict(event.parameters), {"reason": "drill"})

    def test_equality(self):

        self.assertEqual(make_event(), make_event())
        self.assertNotEqual(make_event(), make_event(time=91.0))

    def test_is_frozen(self):

        event = make_event()

        with self.assertRaises(FrozenInstanceError):
            event.time = 100.0

    def test_parameters_mapping_is_read_only(self):

        event = make_event()

        with self.assertRaises(TypeError):
            event.parameters["reason"] = "changed"

    def test_round_trip(self):

        event = make_event()
        restored = ScenarioEvent.from_dict(event.to_dict())

        self.assertEqual(event, restored)


class EngineeringObjectStateTests(unittest.TestCase):

    def test_door_state_round_trip_for_every_enum_member(self):

        for state in DoorState:

            door_state = ScenarioDoorState(door_id="door-1", state=state)
            restored = ScenarioDoorState.from_dict(door_state.to_dict())

            self.assertEqual(door_state, restored)
            self.assertEqual(door_state.to_dict()["state"], state.name)

    def test_exit_state_is_plain_bool_shaped(self):

        exit_state = ScenarioExitState(exit_id="exit-1", is_open=True)
        restored = ScenarioExitState.from_dict(exit_state.to_dict())

        self.assertEqual(exit_state, restored)
        self.assertIs(restored.is_open, True)

    def test_stair_state_round_trip_for_every_enum_member(self):

        for availability in StairAvailability:

            stair_state = ScenarioStairState(stair_id="stair-1", availability=availability)
            restored = ScenarioStairState.from_dict(stair_state.to_dict())

            self.assertEqual(stair_state, restored)

    def test_obstacle_state_round_trip_for_every_enum_member(self):

        for presence in PresenceState:

            obstacle_state = ScenarioObstacleState(obstacle_id="obs-1", presence=presence)
            restored = ScenarioObstacleState.from_dict(obstacle_state.to_dict())

            self.assertEqual(obstacle_state, restored)

    def test_camera_and_detector_states_share_device_availability(self):

        for availability in DeviceAvailability:

            camera_state = ScenarioCameraState(camera_id="cam-1", availability=availability)
            detector_state = ScenarioDetectorState(
                detector_id="det-1", availability=availability,
            )

            self.assertEqual(
                ScenarioCameraState.from_dict(camera_state.to_dict()), camera_state,
            )
            self.assertEqual(
                ScenarioDetectorState.from_dict(detector_state.to_dict()), detector_state,
            )

    def test_all_six_state_records_are_frozen(self):

        door_state = ScenarioDoorState(door_id="door-1", state=DoorState.OPEN)
        exit_state = ScenarioExitState(exit_id="exit-1", is_open=True)
        stair_state = ScenarioStairState(
            stair_id="stair-1", availability=StairAvailability.AVAILABLE,
        )
        obstacle_state = ScenarioObstacleState(obstacle_id="obs-1", presence=PresenceState.ACTIVE)
        camera_state = ScenarioCameraState(
            camera_id="cam-1", availability=DeviceAvailability.AVAILABLE,
        )
        detector_state = ScenarioDetectorState(
            detector_id="det-1", availability=DeviceAvailability.AVAILABLE,
        )

        for record, field_name, value in (
            (door_state, "door_id", "changed"),
            (exit_state, "exit_id", "changed"),
            (stair_state, "stair_id", "changed"),
            (obstacle_state, "obstacle_id", "changed"),
            (camera_state, "camera_id", "changed"),
            (detector_state, "detector_id", "changed"),
        ):
            with self.assertRaises(FrozenInstanceError):
                setattr(record, field_name, value)


class ScenarioAggregateTests(unittest.TestCase):

    def _build_scenario(self):

        return Scenario(
            metadata=make_metadata(),
            occupants=[make_occupant(), make_occupant(occupant_id="occ-2")],
            fire=make_fire(),
            door_states=[ScenarioDoorState(door_id="door-1", state=DoorState.LOCKED)],
            exit_states=[ScenarioExitState(exit_id="exit-1", is_open=True)],
            stair_states=[
                ScenarioStairState(stair_id="stair-1", availability=StairAvailability.AVAILABLE),
            ],
            obstacle_states=[
                ScenarioObstacleState(obstacle_id="obs-1", presence=PresenceState.ACTIVE),
            ],
            camera_states=[
                ScenarioCameraState(camera_id="cam-1", availability=DeviceAvailability.FAILED),
            ],
            detector_states=[
                ScenarioDetectorState(
                    detector_id="det-1", availability=DeviceAvailability.AVAILABLE,
                ),
            ],
            events=[make_event()],
            difficulty=0.42,
        )

    # =====================================================

    def test_construction_aggregates_every_category(self):

        scenario = self._build_scenario()

        self.assertEqual(scenario.metadata.scenario_id, "scn-1")
        self.assertEqual(len(scenario.occupants), 2)
        self.assertEqual(scenario.fire.fire_profile, "Electrical")
        self.assertEqual(len(scenario.door_states), 1)
        self.assertEqual(len(scenario.exit_states), 1)
        self.assertEqual(len(scenario.stair_states), 1)
        self.assertEqual(len(scenario.obstacle_states), 1)
        self.assertEqual(len(scenario.camera_states), 1)
        self.assertEqual(len(scenario.detector_states), 1)
        self.assertEqual(len(scenario.events), 1)
        self.assertEqual(scenario.difficulty, 0.42)

    def test_defaults_produce_an_empty_but_valid_scenario(self):

        scenario = Scenario(metadata=make_metadata())

        self.assertEqual(scenario.occupants, ())
        self.assertIsNone(scenario.fire)
        self.assertEqual(scenario.events, ())
        self.assertIsNone(scenario.difficulty)

    def test_sequence_fields_are_coerced_to_tuples(self):

        scenario = self._build_scenario()

        self.assertIsInstance(scenario.occupants, tuple)
        self.assertIsInstance(scenario.door_states, tuple)
        self.assertIsInstance(scenario.events, tuple)

    def test_equality(self):

        self.assertEqual(self._build_scenario(), self._build_scenario())

        other = self._build_scenario()
        self.assertNotEqual(
            self._build_scenario(),
            Scenario(
                metadata=make_metadata(scenario_id="scn-2"),
                occupants=other.occupants,
                fire=other.fire,
            ),
        )

    def test_is_frozen(self):

        scenario = self._build_scenario()

        with self.assertRaises(FrozenInstanceError):
            scenario.difficulty = 0.9

    def test_occupants_tuple_cannot_be_mutated_in_place(self):

        scenario = self._build_scenario()

        with self.assertRaises(AttributeError):
            scenario.occupants.append(make_occupant(occupant_id="occ-3"))

    def test_nested_serialization_shape(self):

        data = self._build_scenario().to_dict()

        self.assertIsInstance(data["metadata"], dict)
        self.assertIsInstance(data["occupants"], list)
        self.assertIsInstance(data["occupants"][0], dict)
        self.assertIsInstance(data["fire"], dict)
        self.assertIsInstance(data["door_states"][0], dict)
        self.assertEqual(data["door_states"][0]["state"], "LOCKED")
        self.assertIsInstance(data["events"][0], dict)

    def test_round_trip_correctness(self):

        scenario = self._build_scenario()
        restored = Scenario.from_dict(scenario.to_dict())

        self.assertEqual(scenario, restored)

    def test_round_trip_with_no_fire_and_no_events(self):

        scenario = Scenario(metadata=make_metadata())
        restored = Scenario.from_dict(scenario.to_dict())

        self.assertEqual(scenario, restored)
        self.assertIsNone(restored.fire)
        self.assertEqual(restored.events, ())

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

        assert_plain(self._build_scenario().to_dict())


class ScenarioPackageDependencyDirectionTests(unittest.TestCase):

    # scenario/ must be plain, standalone immutable data models -- no
    # generation, validation, simulation, perception, navigation, RL,
    # behaviour logic, fire physics, randomness, or file I/O
    # (architecture doc §7; this implementation phase's own brief).
    # Enforced the same way tests/test_sensors.py already enforces its
    # own package's dependency direction: a regex scan over the
    # package's own source, not a runtime import-graph check.

    def test_package_imports_nothing_from_the_rest_of_the_repository(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(models|navigation|pathfinding|fire_growth|hazard|hazard_evolution|"
            r"occupancy|sensors|perception|behavior|behavior_library|simulator|"
            r"ai_decision|designer|sandbox|rl|scenario_generator|scenario_validator|"
            r"scenario_definition|random|numpy)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario/{path.name} imports a package outside plain immutable "
                f"data models -- this package must stay a standalone leaf with no "
                f"generation, validation, simulation, or randomness of its own",
            )

    def test_package_performs_no_file_io(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario"

        forbidden = r"\bopen\s*\(|\.write\s*\(|\.read\s*\(|\bjson\.(load|dump)\s*\("

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text),
                f"scenario/{path.name} appears to perform file I/O -- this package "
                f"must only reshape already-in-memory values",
            )


if __name__ == "__main__":
    unittest.main()
