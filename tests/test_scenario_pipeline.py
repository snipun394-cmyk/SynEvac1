import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario_definition import (
    EngineeringConstraints,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
)
from scenario_definition import DoorState as DefDoorState

from scenario_pipeline import (
    BatchPipelineResult,
    GenerationStatistics,
    PipelineResult,
    aggregate_statistics,
    build_statistics,
    run_batch_pipeline,
    run_pipeline,
)


def make_building():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0)],
        doors=[],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor1])


def make_feasible_definition(**overrides):

    defaults = dict(
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
    defaults.update(overrides)

    return ScenarioDefinition(**defaults)


def make_infeasible_definition():

    # min_open_exits=5 against a building with exactly one exit --
    # every candidate is rejected by Navigation Validation
    # (INSUFFICIENT_REACHABLE_EGRESS) and Engineering Validation
    # (MIN_OPEN_EXITS_UNSATISFIED), deterministically, forever.
    return make_feasible_definition(
        engineering=EngineeringConstraints(
            exit_state_distribution={"exit-1": FixedValue(True)}, min_open_exits=5,
        ),
    )


class RunPipelineSuccessTests(unittest.TestCase):

    def test_accepted_scenario_is_returned(self):

        result = run_pipeline(make_feasible_definition(), "def-1", make_building(), seed=42)

        self.assertTrue(result.accepted)
        self.assertIsNotNone(result.scenario)
        self.assertIsNone(result.failure_reason)

    def test_first_attempt_success_uses_exactly_one_attempt(self):

        result = run_pipeline(make_feasible_definition(), "def-1", make_building(), seed=42)

        self.assertEqual(result.statistics.attempts, 1)
        self.assertEqual(result.statistics.accepted_count, 1)
        self.assertEqual(result.statistics.rejection_count, 0)
        self.assertEqual(result.statistics.acceptance_rate, 1.0)

    def test_last_report_is_the_accepting_report(self):

        result = run_pipeline(make_feasible_definition(), "def-1", make_building(), seed=42)

        self.assertTrue(result.last_report.accepted)

    def test_identical_seed_produces_identical_scenario_content(self):

        definition = make_feasible_definition()
        building = make_building()

        first = run_pipeline(definition, "def-1", building, seed=777)
        second = run_pipeline(definition, "def-1", building, seed=777)

        first_data = first.scenario.to_dict()
        second_data = second.scenario.to_dict()
        first_data["metadata"].pop("created_at")
        second_data["metadata"].pop("created_at")

        self.assertEqual(first_data, second_data)

    def test_never_mutates_a_returned_scenario_across_calls(self):

        definition = make_feasible_definition()
        building = make_building()

        first = run_pipeline(definition, "def-1", building, seed=1)
        snapshot = first.scenario.to_dict()

        run_pipeline(definition, "def-1", building, seed=2)

        self.assertEqual(first.scenario.to_dict(), snapshot)


class RunPipelineTerminationTests(unittest.TestCase):

    def test_max_attempts_exceeded_returns_a_structured_failure(self):

        result = run_pipeline(
            make_infeasible_definition(), "def-1", make_building(), seed=1, max_attempts=3,
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.scenario)
        self.assertEqual(result.failure_reason, "MAX_ATTEMPTS_EXCEEDED")
        self.assertEqual(result.statistics.attempts, 3)
        self.assertEqual(result.statistics.accepted_count, 0)
        self.assertEqual(result.statistics.rejection_count, 3)
        self.assertEqual(result.statistics.acceptance_rate, 0.0)

    def test_last_report_on_failure_is_the_final_rejection(self):

        result = run_pipeline(
            make_infeasible_definition(), "def-1", make_building(), seed=1, max_attempts=3,
        )
        self.assertFalse(result.last_report.accepted)

    def test_rejection_categories_are_populated_on_failure(self):

        result = run_pipeline(
            make_infeasible_definition(), "def-1", make_building(), seed=1, max_attempts=3,
        )
        self.assertTrue(dict(result.statistics.rejection_categories))

    def test_never_loops_forever_even_with_max_attempts_one(self):

        result = run_pipeline(
            make_infeasible_definition(), "def-1", make_building(), seed=1, max_attempts=1,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.statistics.attempts, 1)

    def test_max_attempts_below_one_is_rejected(self):

        with self.assertRaises(ValueError):
            run_pipeline(make_feasible_definition(), "def-1", make_building(), seed=1, max_attempts=0)

    def test_repeated_failed_runs_are_deterministic(self):

        definition = make_infeasible_definition()
        building = make_building()

        first = run_pipeline(definition, "def-1", building, seed=9, max_attempts=3)
        second = run_pipeline(definition, "def-1", building, seed=9, max_attempts=3)

        self.assertEqual(first.accepted, second.accepted)
        self.assertEqual(first.statistics.attempts, second.statistics.attempts)
        self.assertEqual(
            dict(first.statistics.rejection_categories), dict(second.statistics.rejection_categories),
        )


class RunPipelineRetryTests(unittest.TestCase):

    def test_retry_uses_a_fresh_attempt_index_and_can_eventually_succeed(self):

        # A duplicate-hash rejection forces at least one retry: seed the
        # accepted_hashes set with the content hash the *first* attempt
        # (attempt_index=0) would produce, forcing that attempt to be
        # rejected as DUPLICATE and a second, different attempt to run.
        from scenario_generator import GenerationRequest, generate_scenario
        from scenario_validator import compute_candidate_content_hash

        definition = make_feasible_definition()
        building = make_building()

        first_attempt_candidate = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=building,
                seed=555, attempt_index=0,
            ),
        )
        forced_duplicate_hash = compute_candidate_content_hash(first_attempt_candidate)

        result = run_pipeline(
            definition, "def-1", building, seed=555,
            accepted_hashes=frozenset({forced_duplicate_hash}),
        )

        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.statistics.attempts, 2)
        self.assertGreaterEqual(result.statistics.rejection_count, 1)
        self.assertIn("DATASET", dict(result.statistics.rejection_categories))

    def test_retried_candidate_differs_from_the_rejected_one(self):

        from scenario_generator import GenerationRequest, generate_scenario
        from scenario_validator import compute_candidate_content_hash

        definition = make_feasible_definition()
        building = make_building()

        first_attempt_candidate = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=building,
                seed=555, attempt_index=0,
            ),
        )
        forced_duplicate_hash = compute_candidate_content_hash(first_attempt_candidate)

        result = run_pipeline(
            definition, "def-1", building, seed=555,
            accepted_hashes=frozenset({forced_duplicate_hash}),
        )

        self.assertNotEqual(
            compute_candidate_content_hash(result.scenario), forced_duplicate_hash,
        )


class BatchPipelineTests(unittest.TestCase):

    def test_generates_the_requested_count_when_all_succeed(self):

        result = run_batch_pipeline(
            make_feasible_definition(), "def-1", make_building(), master_seed=1, count=5,
        )
        self.assertEqual(len(result.scenarios), 5)
        self.assertEqual(len(result.results), 5)

    def test_rejected_candidates_never_appear_in_scenarios(self):

        result = run_batch_pipeline(
            make_infeasible_definition(), "def-1", make_building(), master_seed=1, count=3,
            max_attempts_per_scenario=2,
        )
        self.assertEqual(result.scenarios, ())
        self.assertEqual(len(result.results), 3)
        self.assertTrue(all(not r.accepted for r in result.results))

    def test_accepted_scenarios_are_deterministic_across_two_runs(self):

        definition = make_feasible_definition()
        building = make_building()

        first = run_batch_pipeline(definition, "def-1", building, master_seed=42, count=4)
        second = run_batch_pipeline(definition, "def-1", building, master_seed=42, count=4)

        for a, b in zip(first.scenarios, second.scenarios):

            a_data = a.to_dict()
            b_data = b.to_dict()
            a_data["metadata"].pop("created_at")
            b_data["metadata"].pop("created_at")

            self.assertEqual(a_data, b_data)

    def test_batch_is_index_keyed_and_resumable(self):

        definition = make_feasible_definition()
        building = make_building()

        whole_run = run_batch_pipeline(definition, "def-1", building, master_seed=99, count=6)
        resumed_tail = run_batch_pipeline(
            definition, "def-1", building, master_seed=99, count=3, start_index=3,
        )

        for offset in range(3):

            whole_data = whole_run.scenarios[3 + offset].to_dict()
            resumed_data = resumed_tail.scenarios[offset].to_dict()
            whole_data["metadata"].pop("created_at")
            resumed_data["metadata"].pop("created_at")

            self.assertEqual(whole_data, resumed_data)

    def test_duplicate_scenarios_within_a_batch_are_rejected_and_retried(self):

        # With a very small, fully-discrete Definition (few possible
        # unique scenarios), successive batch members are likely to
        # collide and must be retried rather than silently duplicated.
        # Exercised indirectly: accepted_hashes grows across the batch,
        # so every accepted scenario's content hash must be unique.
        from scenario_validator import compute_candidate_content_hash

        definition = make_feasible_definition()
        building = make_building()

        result = run_batch_pipeline(
            definition, "def-1", building, master_seed=1, count=5, max_attempts_per_scenario=20,
        )

        hashes = [compute_candidate_content_hash(s) for s in result.scenarios]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_zero_count_returns_an_empty_batch(self):

        result = run_batch_pipeline(
            make_feasible_definition(), "def-1", make_building(), master_seed=1, count=0,
        )
        self.assertEqual(result.scenarios, ())
        self.assertEqual(result.results, ())

    def test_negative_count_is_rejected(self):

        with self.assertRaises(ValueError):
            run_batch_pipeline(
                make_feasible_definition(), "def-1", make_building(), master_seed=1, count=-1,
            )

    def test_batch_statistics_aggregate_across_every_scenario(self):

        result = run_batch_pipeline(
            make_feasible_definition(), "def-1", make_building(), master_seed=1, count=4,
        )
        self.assertEqual(result.statistics.attempts, sum(r.statistics.attempts for r in result.results))
        self.assertEqual(result.statistics.accepted_count, 4)

    def test_externally_seeded_accepted_hashes_prevent_the_first_duplicate(self):

        from scenario_generator import GenerationRequest, generate_scenario
        from scenario_generator.seed_manager import derive_scenario_seed
        from scenario_validator import compute_candidate_content_hash

        definition = make_feasible_definition()
        building = make_building()

        scenario_seed = derive_scenario_seed(master_seed=1, index=0)
        first_candidate = generate_scenario(
            GenerationRequest(
                definition=definition, definition_id="def-1", building=building,
                seed=scenario_seed, attempt_index=0,
            ),
        )
        pre_seeded_hash = compute_candidate_content_hash(first_candidate)

        result = run_batch_pipeline(
            definition, "def-1", building, master_seed=1, count=1,
            accepted_hashes=frozenset({pre_seeded_hash}),
            max_attempts_per_scenario=5,
        )

        self.assertEqual(len(result.scenarios), 1)
        self.assertNotEqual(
            compute_candidate_content_hash(result.scenarios[0]), pre_seeded_hash,
        )


class StatisticsTests(unittest.TestCase):

    def test_build_statistics_derives_rejection_count_and_acceptance_rate(self):

        stats = build_statistics(
            attempts=4, accepted_count=1, rejection_categories={"FIRE": 3},
            generation_time_seconds=0.1, validation_time_seconds=0.2, total_time_seconds=0.3,
        )
        self.assertEqual(stats.rejection_count, 3)
        self.assertEqual(stats.acceptance_rate, 0.25)

    def test_build_statistics_handles_zero_attempts(self):

        stats = build_statistics(
            attempts=0, accepted_count=0, rejection_categories={},
            generation_time_seconds=0.0, validation_time_seconds=0.0, total_time_seconds=0.0,
        )
        self.assertEqual(stats.acceptance_rate, 0.0)

    def test_rejection_categories_mapping_is_read_only(self):

        stats = build_statistics(
            attempts=1, accepted_count=1, rejection_categories={"FIRE": 1},
            generation_time_seconds=0.0, validation_time_seconds=0.0, total_time_seconds=0.0,
        )
        with self.assertRaises(TypeError):
            stats.rejection_categories["FIRE"] = 99

    def test_statistics_is_frozen(self):

        from dataclasses import FrozenInstanceError

        stats = build_statistics(
            attempts=1, accepted_count=1, rejection_categories={},
            generation_time_seconds=0.0, validation_time_seconds=0.0, total_time_seconds=0.0,
        )
        with self.assertRaises(FrozenInstanceError):
            stats.attempts = 99

    def test_aggregate_statistics_sums_across_scenarios(self):

        first = build_statistics(
            attempts=2, accepted_count=1, rejection_categories={"FIRE": 1},
            generation_time_seconds=0.1, validation_time_seconds=0.1, total_time_seconds=0.2,
        )
        second = build_statistics(
            attempts=3, accepted_count=1, rejection_categories={"FIRE": 1, "NAVIGATION": 1},
            generation_time_seconds=0.2, validation_time_seconds=0.2, total_time_seconds=0.4,
        )

        aggregated = aggregate_statistics([first, second])

        self.assertEqual(aggregated.attempts, 5)
        self.assertEqual(aggregated.accepted_count, 2)
        self.assertEqual(dict(aggregated.rejection_categories), {"FIRE": 2, "NAVIGATION": 1})
        self.assertAlmostEqual(aggregated.generation_time_seconds, 0.3)


class ResultObjectTests(unittest.TestCase):

    def test_pipeline_result_rejects_accepted_true_with_no_scenario(self):

        stats = build_statistics(
            attempts=1, accepted_count=1, rejection_categories={},
            generation_time_seconds=0.0, validation_time_seconds=0.0, total_time_seconds=0.0,
        )
        with self.assertRaises(ValueError):
            PipelineResult(accepted=True, scenario=None, statistics=stats)

    def test_pipeline_result_rejects_accepted_false_with_a_scenario(self):

        result = run_pipeline(make_feasible_definition(), "def-1", make_building(), seed=1)
        stats = result.statistics

        with self.assertRaises(ValueError):
            PipelineResult(accepted=False, scenario=result.scenario, statistics=stats)

    def test_pipeline_result_is_frozen(self):

        from dataclasses import FrozenInstanceError

        result = run_pipeline(make_feasible_definition(), "def-1", make_building(), seed=1)

        with self.assertRaises(FrozenInstanceError):
            result.accepted = False

    def test_batch_pipeline_result_sequences_are_tuples(self):

        result = run_batch_pipeline(
            make_feasible_definition(), "def-1", make_building(), master_seed=1, count=2,
        )
        self.assertIsInstance(result.scenarios, tuple)
        self.assertIsInstance(result.results, tuple)

    def test_batch_pipeline_result_is_frozen(self):

        from dataclasses import FrozenInstanceError

        result = run_batch_pipeline(
            make_feasible_definition(), "def-1", make_building(), master_seed=1, count=1,
        )
        with self.assertRaises(FrozenInstanceError):
            result.scenarios = ()


class ScenarioPipelinePackageDependencyDirectionTests(unittest.TestCase):

    # scenario_pipeline/ coordinates scenario_generator and
    # scenario_validator only -- it must never import navigation,
    # fire_growth, sandbox, designer, simulator, behavior,
    # behavior_library, ai_decision, perception, rl, or random
    # directly; anything from those packages it needs must come
    # through scenario_generator/scenario_validator's own public API.

    def test_package_never_imports_forbidden_packages_directly(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_pipeline"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(navigation|fire_growth|sandbox|designer|simulator|behavior|"
            r"behavior_library|ai_decision|perception|rl|random)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario_pipeline/{path.name} imports a package it must reach only "
                f"through scenario_generator/scenario_validator, if at all",
            )

    def test_package_performs_no_file_io(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_pipeline"

        forbidden = r"\bopen\s*\(|\.write\s*\(|\.read\s*\(|\bjson\.(load|dump)\s*\("

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text),
                f"scenario_pipeline/{path.name} appears to perform file I/O -- JSON "
                f"persistence belongs to a later phase",
            )


if __name__ == "__main__":
    unittest.main()
