import tempfile
import unittest
from pathlib import Path

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario import Scenario
from scenario_definition import (
    EngineeringConstraints,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
)
from scenario_pipeline import run_batch_pipeline, run_pipeline
from scenario_validator import compute_candidate_content_hash

from scenario_storage import (
    CATALOG_COLUMNS,
    SHARD_KEY_LENGTH,
    append_catalog_row,
    catalog_path,
    catalog_row_for,
    load_accepted_hashes,
    load_scenario_by_filename,
    load_scenario_by_id,
    read_catalog_rows,
    save_scenario,
    scenario_json_path,
    shard_key,
)


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0)],
        doors=[],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_definition():

    return ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(100.0, 400.0),
            allowed_ignition_zone_ids={"zone-1"},
            allowed_fire_profiles={"Electrical"},
        ),
        engineering=EngineeringConstraints(
            exit_state_distribution={"exit-1": FixedValue(True)}, min_open_exits=1,
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-1": FixedValue(2)},
            behaviour_profile_distribution={"zone-1": FixedValue("Adult_Default")},
        ),
    )


def make_accepted_scenario(seed=42):

    result = run_pipeline(make_definition(), "def-1", make_building(), seed=seed)
    assert result.accepted
    return result.scenario


class ShardKeyTests(unittest.TestCase):

    def test_shard_key_strips_the_scn_label_prefix(self):

        self.assertEqual(shard_key("scn-abcdef1234567890"), "ab")

    def test_shard_key_length_matches_configured_constant(self):

        self.assertEqual(len(shard_key("scn-abcdef1234567890")), SHARD_KEY_LENGTH)

    def test_shard_key_falls_back_to_raw_id_without_the_label_prefix(self):

        self.assertEqual(shard_key("abcdef"), "ab")

    def test_scenario_ids_do_not_all_collide_into_one_shard(self):

        # The bug this module's docstring warns against: sharding on
        # the literal string's own first two characters would put
        # every "scn-..." id in the same "sc" bucket. Real scenario_ids
        # are sha256-derived (high entropy throughout, including the
        # leading digits) -- hashlib here stands in for that, rather
        # than zero-padded sequential integers whose *leading* hex
        # digits would themselves all collide at "00" regardless of
        # this module's own logic.
        import hashlib

        def fake_scenario_id(index):
            digest = hashlib.sha256(str(index).encode()).hexdigest()[:16]
            return f"scn-{digest}"

        keys = {shard_key(fake_scenario_id(index)) for index in range(50)}
        self.assertGreater(len(keys), 1)

    def test_scenario_json_path_is_under_the_shard_directory(self):

        path = scenario_json_path("/storage", "scn-abcdef1234567890")
        self.assertEqual(Path(path).parent.name, "ab")
        self.assertEqual(Path(path).name, "scn-abcdef1234567890.json")


class SaveAndLoadTests(unittest.TestCase):

    def test_save_writes_a_json_file_that_exists(self):

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            path = save_scenario(scenario, storage_root)

            self.assertTrue(Path(path).exists())

    def test_save_reuses_scenario_to_dict_shape(self):

        import json

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            path = save_scenario(scenario, storage_root)

            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            self.assertEqual(data, scenario.to_dict())

    def test_load_by_id_round_trips_correctly(self):

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            save_scenario(scenario, storage_root)

            loaded = load_scenario_by_id(scenario.metadata.scenario_id, storage_root)

            self.assertEqual(loaded, scenario)
            self.assertIsInstance(loaded, Scenario)

    def test_load_by_filename_round_trips_correctly(self):

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            path = save_scenario(scenario, storage_root)

            loaded = load_scenario_by_filename(Path(path).name, storage_root)

            self.assertEqual(loaded, scenario)

    def test_load_by_id_and_load_by_filename_agree(self):

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            save_scenario(scenario, storage_root)

            by_id = load_scenario_by_id(scenario.metadata.scenario_id, storage_root)
            by_filename = load_scenario_by_filename(
                f"{scenario.metadata.scenario_id}.json", storage_root,
            )

            self.assertEqual(by_id, by_filename)

    def test_load_missing_scenario_raises(self):

        with tempfile.TemporaryDirectory() as storage_root:

            with self.assertRaises(FileNotFoundError):
                load_scenario_by_id("scn-doesnotexist0000", storage_root)

    def test_saving_the_same_scenario_twice_raises(self):

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            save_scenario(scenario, storage_root)

            with self.assertRaises(FileExistsError):
                save_scenario(scenario, storage_root)

    def test_two_different_scenarios_land_in_potentially_different_shards_but_both_load(self):

        with tempfile.TemporaryDirectory() as storage_root:

            first = make_accepted_scenario(seed=1)
            second = make_accepted_scenario(seed=2)

            save_scenario(first, storage_root)
            save_scenario(second, storage_root)

            self.assertEqual(load_scenario_by_id(first.metadata.scenario_id, storage_root), first)
            self.assertEqual(load_scenario_by_id(second.metadata.scenario_id, storage_root), second)


class ShardingDirectoryStructureTests(unittest.TestCase):

    def test_saved_file_lives_under_scenarios_shard_subdirectory(self):

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            path = Path(save_scenario(scenario, storage_root))

            expected_shard = shard_key(scenario.metadata.scenario_id)

            self.assertEqual(path.parent.name, expected_shard)
            self.assertEqual(path.parent.parent.name, "scenarios")

    def test_many_scenarios_are_distributed_across_more_than_one_shard_directory(self):

        with tempfile.TemporaryDirectory() as storage_root:

            definition = make_definition()
            building = make_building()

            batch = run_batch_pipeline(definition, "def-1", building, master_seed=7, count=40)

            for scenario in batch.scenarios:
                save_scenario(scenario, storage_root)

            shard_dirs = {p.name for p in (Path(storage_root) / "scenarios").iterdir() if p.is_dir()}

            self.assertGreater(len(shard_dirs), 1)


class CatalogTests(unittest.TestCase):

    def test_first_save_writes_a_header_row(self):

        with tempfile.TemporaryDirectory() as storage_root:

            save_scenario(make_accepted_scenario(), storage_root)

            with open(catalog_path(storage_root), "r", encoding="utf-8") as handle:
                first_line = handle.readline().strip()

            self.assertEqual(first_line, ",".join(CATALOG_COLUMNS))

    def test_catalog_row_contains_every_required_field(self):

        scenario = make_accepted_scenario()
        row = catalog_row_for(scenario, "scn-example.json")

        self.assertEqual(row["scenario_id"], scenario.metadata.scenario_id)
        self.assertEqual(row["json_filename"], "scn-example.json")
        self.assertEqual(row["definition_id"], scenario.metadata.definition_id)
        self.assertEqual(row["definition_content_hash"], scenario.metadata.definition_content_hash)
        self.assertEqual(row["generation_version"], scenario.metadata.generation_version)
        self.assertEqual(row["seed"], scenario.metadata.seed)
        self.assertEqual(row["fire_profile"], scenario.fire.fire_profile)
        self.assertEqual(row["fire_origin"], scenario.fire.ignition_zone_id)
        self.assertEqual(row["occupant_count"], len(scenario.occupants))
        self.assertEqual(row["difficulty"], "")
        self.assertEqual(row["created_at"], scenario.metadata.created_at)

    def test_append_is_additive_not_a_rewrite(self):

        with tempfile.TemporaryDirectory() as storage_root:

            definition = make_definition()
            building = make_building()

            batch = run_batch_pipeline(definition, "def-1", building, master_seed=3, count=3)

            for scenario in batch.scenarios:
                save_scenario(scenario, storage_root)

            rows = read_catalog_rows(storage_root)

            self.assertEqual(len(rows), 3)
            self.assertEqual(
                {row["scenario_id"] for row in rows},
                {s.metadata.scenario_id for s in batch.scenarios},
            )

    def test_reading_a_nonexistent_catalog_returns_an_empty_list(self):

        with tempfile.TemporaryDirectory() as storage_root:
            self.assertEqual(read_catalog_rows(storage_root), [])

    def test_append_catalog_row_directly_without_saving_json_still_reads_back(self):

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            append_catalog_row(storage_root, scenario, "scn-manual.json")

            rows = read_catalog_rows(storage_root)
            self.assertEqual(rows[0]["json_filename"], "scn-manual.json")


class DeduplicationTests(unittest.TestCase):

    def test_load_accepted_hashes_is_empty_for_a_fresh_storage_root(self):

        with tempfile.TemporaryDirectory() as storage_root:
            self.assertEqual(load_accepted_hashes(storage_root), frozenset())

    def test_load_accepted_hashes_matches_compute_candidate_content_hash(self):

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            save_scenario(scenario, storage_root)

            hashes = load_accepted_hashes(storage_root)

            self.assertEqual(hashes, frozenset({compute_candidate_content_hash(scenario)}))

    def test_load_accepted_hashes_covers_every_saved_scenario(self):

        with tempfile.TemporaryDirectory() as storage_root:

            definition = make_definition()
            building = make_building()

            batch = run_batch_pipeline(definition, "def-1", building, master_seed=11, count=5)

            for scenario in batch.scenarios:
                save_scenario(scenario, storage_root)

            hashes = load_accepted_hashes(storage_root)

            self.assertEqual(len(hashes), len(batch.scenarios))

    def test_seeded_accepted_hashes_can_be_fed_back_into_the_pipeline(self):

        # This is the actual integration this mechanism exists for
        # (§4.9) -- exercised without redesigning scenario_pipeline at
        # all: load_accepted_hashes()'s return value is passed straight
        # into run_pipeline()'s existing accepted_hashes parameter.
        with tempfile.TemporaryDirectory() as storage_root:

            definition = make_definition()
            building = make_building()

            first = run_pipeline(definition, "def-1", building, seed=555)
            save_scenario(first.scenario, storage_root)

            seeded_hashes = load_accepted_hashes(storage_root)

            retried = run_pipeline(
                definition, "def-1", building, seed=555, accepted_hashes=seeded_hashes,
            )

            # Same seed, but the first attempt is now a known duplicate
            # -- the pipeline must retry past it.
            self.assertTrue(retried.accepted)
            self.assertGreaterEqual(retried.statistics.attempts, 2)
            self.assertNotEqual(
                compute_candidate_content_hash(retried.scenario),
                compute_candidate_content_hash(first.scenario),
            )


class ReplayIntegrityTests(unittest.TestCase):

    def test_loaded_scenario_validates_identically_to_the_original(self):

        from scenario_validator import validate

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            save_scenario(scenario, storage_root)

            loaded = load_scenario_by_id(scenario.metadata.scenario_id, storage_root)

            definition = make_definition()
            building = make_building()

            original_report = validate(scenario, definition, building)
            loaded_report = validate(loaded, definition, building)

            self.assertEqual(original_report.accepted, loaded_report.accepted)
            self.assertEqual(
                [(i.category, i.code) for i in original_report.issues],
                [(i.category, i.code) for i in loaded_report.issues],
            )

    def test_round_trip_through_disk_preserves_every_field(self):

        with tempfile.TemporaryDirectory() as storage_root:

            scenario = make_accepted_scenario()
            save_scenario(scenario, storage_root)

            loaded = load_scenario_by_id(scenario.metadata.scenario_id, storage_root)

            self.assertEqual(loaded.to_dict(), scenario.to_dict())
            self.assertEqual(loaded.metadata, scenario.metadata)
            self.assertEqual(loaded.occupants, scenario.occupants)
            self.assertEqual(loaded.fire, scenario.fire)
            self.assertEqual(loaded.events, scenario.events)


class ScenarioStoragePackageDependencyDirectionTests(unittest.TestCase):

    # scenario_storage/ performs only persistence -- it must never
    # import scenario_generator (generation), navigation/fire_growth
    # (simulation-adjacent), sandbox, designer, simulator, behavior,
    # behavior_library, ai_decision, perception, rl, or random.
    # scenario_validator is imported deliberately, for exactly one pure
    # function (compute_candidate_content_hash) -- reused, not
    # reimplemented a third time (see storage.py's own docstring).

    def test_package_never_imports_forbidden_packages(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_storage"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(scenario_generator|scenario_pipeline|navigation|fire_growth|sandbox|"
            r"designer|simulator|behavior|behavior_library|ai_decision|perception|"
            r"rl|random)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario_storage/{path.name} imports a package this persistence-"
                f"only package must never depend on",
            )

    def test_package_never_performs_generation_or_validation_logic_itself(self):

        # Structural proxy for "no generation/validation logic of its
        # own": the only scenario_validator symbol this package may
        # touch is the content-hash function.
        import pathlib

        text = (
            pathlib.Path(__file__).resolve().parent.parent
            / "scenario_storage" / "storage.py"
        ).read_text()

        self.assertIn("compute_candidate_content_hash", text)
        self.assertNotIn("validate(", text)
        self.assertNotIn("validate_", text)


if __name__ == "__main__":
    unittest.main()
