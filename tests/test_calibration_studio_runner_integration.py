import unittest

from calibration_benchmark import ParameterCandidate, WalkingSpeedCandidate

from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, PublishedValue
from calibration_studio.benchmark_library import BenchmarkNotFoundError
from calibration_studio.session import SessionStatus
from calibration_studio.studio import CalibrationStudio

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


# =====================================================
# Calibration Studio Phase 4 -- Calibration Runner integration tests.
# Reuses tests/calibration_benchmark_fixtures.py in full (same
# building, same definition, same DEFINITION_ID/MASTER_SEED every
# existing calibration_benchmark test already uses) -- no new
# benchmark dataset is created anywhere in this file. n_scenarios=2-3,
# dt=1.0 matches the established fast-test convention (tests/
# test_calibration_benchmark_harness.py's own real-run tests).
# =====================================================


class _BrokenCandidate(ParameterCandidate):

    # Deliberately raises mid-run -- a controlled, deterministic way to
    # exercise the FAILED path without depending on some fragile,
    # environment-dependent real failure condition.

    def __init__(self):

        super().__init__(
            name="Broken.candidate", subsystem="Test", calibration_tier="Tier 2",
            dataset_source="test", current_value=1.0, candidate_value=2.0, unit="x", rationale="test",
        )

    def candidate_capacity_model(self):

        raise RuntimeError("deliberately broken for testing")


def _make_building_benchmark(**overrides):

    defaults = dict(
        title="Calibration Benchmark Test Fixture",
        source_citation="internal test fixture",
        dataset="synthetic",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref="tests.calibration_benchmark_fixtures.make_building"),
        published_values={"evacuation_time_s": PublishedValue(value=700.0, unit="s")},
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


def _make_dataset_benchmark(**overrides):

    defaults = dict(
        title="Synthetic Walking Speed Reference",
        source_citation="internal test fixture",
        dataset="synthetic",
        benchmark_type=BenchmarkType.DATASET_VALIDATION,
        published_values={"walking_speed_ms": PublishedValue(value=0.65, unit="m/s")},
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


class RunPublishedBenchmarkTests(unittest.TestCase):

    def setUp(self):

        self.studio = CalibrationStudio()
        self.project = self.studio.create_project(name="Phase 4 Integration")
        self.definition = make_definition()

    def _candidate(self):

        return WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")

    def test_run_completes_and_attaches_a_real_calibration_benchmark_result(self):

        benchmark = _make_building_benchmark()
        self.studio.benchmarks.register(benchmark)

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark.benchmark_id, candidate=self._candidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertIsNotNone(session.result)
        self.assertEqual(session.result.n_scenarios_requested, 2)
        self.assertIn("evacuation_time", session.result.comparisons)

    def test_progress_reaches_full_completion(self):

        benchmark = _make_building_benchmark()
        self.studio.benchmarks.register(benchmark)

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark.benchmark_id, candidate=self._candidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        self.assertEqual(session.n_scenarios_completed, 2)
        self.assertEqual(session.progress, 1.0)

    def test_reproducible_is_true_for_a_real_completed_run(self):

        benchmark = _make_building_benchmark()
        self.studio.benchmarks.register(benchmark)

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark.benchmark_id, candidate=self._candidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        self.assertIs(session.reproducible, True)

    def test_benchmark_calibration_history_is_updated(self):

        benchmark = _make_building_benchmark()
        self.studio.benchmarks.register(benchmark)

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark.benchmark_id, candidate=self._candidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        self.assertEqual(benchmark.calibration_history, (session.session_id,))

    def test_session_is_attached_to_the_project(self):

        benchmark = _make_building_benchmark()
        self.studio.benchmarks.register(benchmark)

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark.benchmark_id, candidate=self._candidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        self.assertIn(session, self.project.sessions)

    def test_unknown_benchmark_id_raises(self):

        with self.assertRaises(BenchmarkNotFoundError):
            self.studio.run_published_benchmark(
                project=self.project, benchmark_id="does-not-exist", candidate=self._candidate(),
                definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                n_scenarios=2, dt=1.0,
            )

    def test_dataset_validation_benchmark_without_explicit_building_raises_value_error(self):

        benchmark = _make_dataset_benchmark()
        self.studio.benchmarks.register(benchmark)

        with self.assertRaises(ValueError):
            self.studio.run_published_benchmark(
                project=self.project, benchmark_id=benchmark.benchmark_id, candidate=self._candidate(),
                definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                n_scenarios=2, dt=1.0,
            )

    def test_dataset_validation_benchmark_with_explicit_building_runs(self):

        benchmark = _make_dataset_benchmark()
        self.studio.benchmarks.register(benchmark)

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark.benchmark_id, candidate=self._candidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0, building=make_building(),
        )

        self.assertEqual(session.status, SessionStatus.COMPLETED)

    def test_explicit_building_overrides_geometry_reference(self):

        # Even a BUILDING_RECREATION benchmark's own geometry can be
        # overridden by an explicitly-supplied building -- the caller
        # always wins.
        benchmark = _make_building_benchmark()
        self.studio.benchmarks.register(benchmark)

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark.benchmark_id, candidate=self._candidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=1, dt=1.0, building=make_building(),
        )

        self.assertEqual(session.status, SessionStatus.COMPLETED)


class RunParameterSweepTests(unittest.TestCase):

    def setUp(self):

        self.studio = CalibrationStudio()
        self.project = self.studio.create_project(name="Sweep")
        self.building = make_building()
        self.definition = make_definition()

    def test_sweep_runs_one_session_per_candidate(self):

        candidates = [
            WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test"),
            WalkingSpeedCandidate("Adult_Default", 0.9, "test", "test"),
        ]

        sessions = self.studio.run_parameter_sweep(
            project=self.project, candidates=candidates, building=self.building,
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        self.assertEqual(len(sessions), 2)
        self.assertTrue(all(s.status == SessionStatus.COMPLETED for s in sessions))
        self.assertEqual(len({s.session_id for s in sessions}), 2)

    def test_sweep_without_benchmark_id_works_and_leaves_no_benchmark_reference(self):

        candidates = [WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")]

        sessions = self.studio.run_parameter_sweep(
            project=self.project, candidates=candidates, building=self.building,
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=1, dt=1.0,
        )

        self.assertIsNone(sessions[0].benchmark_id)

    def test_sweep_with_benchmark_id_updates_history_for_every_session(self):

        benchmark = _make_building_benchmark()
        self.studio.benchmarks.register(benchmark)

        candidates = [
            WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test"),
            WalkingSpeedCandidate("Adult_Default", 0.9, "test", "test"),
        ]

        sessions = self.studio.run_parameter_sweep(
            project=self.project, candidates=candidates, building=self.building,
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=1, dt=1.0, benchmark_id=benchmark.benchmark_id,
        )

        self.assertEqual(set(benchmark.calibration_history), {s.session_id for s in sessions})

    def test_sweep_with_unknown_benchmark_id_raises(self):

        candidates = [WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")]

        with self.assertRaises(BenchmarkNotFoundError):
            self.studio.run_parameter_sweep(
                project=self.project, candidates=candidates, building=self.building,
                definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                n_scenarios=1, dt=1.0, benchmark_id="does-not-exist",
            )


class FailureHandlingTests(unittest.TestCase):

    def setUp(self):

        self.studio = CalibrationStudio()
        self.project = self.studio.create_project(name="Failure handling")
        self.building = make_building()
        self.definition = make_definition()

    def test_a_failing_candidate_marks_the_session_failed_not_completed(self):

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=self._register_benchmark(), candidate=_BrokenCandidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=1, dt=1.0,
        )

        self.assertEqual(session.status, SessionStatus.FAILED)
        self.assertIn("deliberately broken", session.failure_reason)

    def test_a_failing_candidate_does_not_raise_out_of_run_published_benchmark(self):

        # No exception should propagate -- the method call itself
        # completes normally, returning a FAILED session.
        try:
            session = self.studio.run_published_benchmark(
                project=self.project, benchmark_id=self._register_benchmark(), candidate=_BrokenCandidate(),
                definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                n_scenarios=1, dt=1.0,
            )
        except Exception as exc:  # noqa: BLE001 -- explicitly asserting this must NOT happen
            self.fail(f"run_published_benchmark() raised {exc!r} instead of returning a FAILED session")

        self.assertEqual(session.status, SessionStatus.FAILED)

    def test_a_failing_candidate_still_updates_benchmark_calibration_history(self):

        benchmark_id = self._register_benchmark()

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark_id, candidate=_BrokenCandidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=1, dt=1.0,
        )

        benchmark = self.studio.benchmarks.get(benchmark_id)
        self.assertEqual(benchmark.calibration_history, (session.session_id,))

    def test_a_failed_session_has_no_result(self):

        session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=self._register_benchmark(), candidate=_BrokenCandidate(),
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=1, dt=1.0,
        )

        self.assertIsNone(session.result)
        self.assertIsNone(session.reproducible)

    def _register_benchmark(self) -> str:

        benchmark = _make_building_benchmark(title="Failure test benchmark")
        self.studio.benchmarks.register(benchmark)
        return benchmark.benchmark_id


if __name__ == "__main__":
    unittest.main()
