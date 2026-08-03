import tempfile
import unittest
from pathlib import Path

from calibration_benchmark import (
    ComplianceLevelCandidate,
    HerdingFollowProbabilityCandidate,
    StairCounterflowPenaltyCandidate,
    run_calibration_benchmark,
)

from scenario_pipeline import run_batch_pipeline

from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, PublishedValue
from calibration_studio.replay_integration import record_session_replay
from calibration_studio.session import CalibrationSession, SessionStatus
from calibration_studio.studio import CalibrationStudio

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


# =====================================================
# Calibration Studio Phase 8 -- Complete ParameterCandidate Coverage.
# Proves the three new candidates (ComplianceLevelCandidate,
# HerdingFollowProbabilityCandidate, StairCounterflowPenaltyCandidate)
# are full, first-class citizens of Calibration Studio -- runnable,
# reportable, replayable, persistable, and dashboard-visible -- exactly
# like every one of the six pre-existing candidates, with zero special
# casing anywhere in Calibration Studio needed to support them. Reuses
# tests/calibration_benchmark_fixtures.py throughout, same as every
# other Calibration Studio test file.
# =====================================================


def _new_candidates():

    return (
        ComplianceLevelCandidate("Adult_Default", 0.5, "test", "test"),
        HerdingFollowProbabilityCandidate("Adult_Default", 0.8, "test", "test"),
        StairCounterflowPenaltyCandidate(0.4, "test", "test"),
    )


def _make_benchmark():

    return PublishedBenchmark(
        title="Phase 8 Test Fixture",
        source_citation="internal test fixture",
        dataset="synthetic",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref="tests.calibration_benchmark_fixtures.make_building"),
        published_values={"evacuation_time_s": PublishedValue(value=700.0, unit="s")},
    )


class NewCandidatesRunThroughCalibrationStudioTests(unittest.TestCase):

    def setUp(self):

        self.studio = CalibrationStudio()
        self.project = self.studio.create_project(name="Phase 8 Coverage")
        self.definition = make_definition()

    def test_every_new_candidate_completes_a_real_run(self):

        for candidate in _new_candidates():
            with self.subTest(candidate=candidate.name):

                benchmark = _make_benchmark()
                self.studio.benchmarks.register(benchmark)

                session = self.studio.run_published_benchmark(
                    project=self.project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
                    definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                    n_scenarios=2, dt=1.0,
                )

                self.assertEqual(session.status, SessionStatus.COMPLETED)
                self.assertIsNotNone(session.result)
                self.assertIn("evacuation_time", session.result.comparisons)

    def test_every_new_candidate_is_reproducible(self):

        for candidate in _new_candidates():
            with self.subTest(candidate=candidate.name):

                benchmark = _make_benchmark()
                self.studio.benchmarks.register(benchmark)

                session = self.studio.run_published_benchmark(
                    project=self.project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
                    definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                    n_scenarios=2, dt=1.0,
                )

                self.assertIs(session.reproducible, True)

    def test_every_new_candidate_runs_via_parameter_sweep_too(self):

        building = make_building()

        sessions = self.studio.run_parameter_sweep(
            project=self.project, candidates=list(_new_candidates()), building=building,
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        self.assertEqual(len(sessions), 3)
        self.assertTrue(all(s.status == SessionStatus.COMPLETED for s in sessions))


class NewCandidatesReportingTests(unittest.TestCase):

    def setUp(self):

        self.studio = CalibrationStudio()
        self.project = self.studio.create_project(name="Phase 8 Reporting")
        self.definition = make_definition()

    def test_session_report_generates_and_names_the_new_candidate(self):

        for candidate in _new_candidates():
            with self.subTest(candidate=candidate.name):

                benchmark = _make_benchmark()
                self.studio.benchmarks.register(benchmark)

                session = self.studio.run_published_benchmark(
                    project=self.project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
                    definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                    n_scenarios=2, dt=1.0,
                )

                report = self.studio.generate_session_report(session.session_id)

                self.assertIn(candidate.name, report)

    def test_validation_dashboard_generates_with_a_new_candidates_session_present(self):

        candidate = _new_candidates()[0]
        benchmark = _make_benchmark()
        self.studio.benchmarks.register(benchmark)

        self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        dashboard = self.studio.generate_validation_dashboard()

        self.assertEqual(dashboard.total_benchmarks, 1)


class NewCandidatesPersistenceTests(unittest.TestCase):

    def test_session_survives_a_save_load_roundtrip(self):

        for candidate in _new_candidates():
            with self.subTest(candidate=candidate.name):

                with tempfile.TemporaryDirectory() as tmp:

                    studio = CalibrationStudio(storage_root=tmp)
                    project = studio.create_project(name="Phase 8 Persistence")
                    definition = make_definition()

                    benchmark = _make_benchmark()
                    studio.benchmarks.register(benchmark)

                    session = studio.run_published_benchmark(
                        project=project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
                        definition=definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                        n_scenarios=2, dt=1.0,
                    )

                    studio.save_session(session)
                    reloaded = studio.load_session(session.session_id)

                    self.assertEqual(reloaded.candidate_snapshot["name"], candidate.name)
                    self.assertEqual(reloaded.status, SessionStatus.COMPLETED)
                    self.assertIs(reloaded.candidate, None)  # live object never survives reload, by design
                    self.assertIsNotNone(reloaded.result_snapshot)


class NewCandidatesReplayTests(unittest.TestCase):

    def _completed_session_and_scenario(self, candidate, n_scenarios=2):

        building = make_building()
        definition = make_definition()

        result = run_calibration_benchmark(
            candidate, building, definition, DEFINITION_ID, MASTER_SEED, n_scenarios=n_scenarios, dt=1.0,
        )

        session = CalibrationSession(candidate=candidate, master_seed=MASTER_SEED)
        session.mark_running(n_scenarios_total=n_scenarios)
        session.mark_completed(result)

        batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, n_scenarios)

        return session, batch.scenarios[0], building

    def test_replay_artifacts_are_recorded_for_every_new_candidate(self):

        for candidate in _new_candidates():
            with self.subTest(candidate=candidate.name):

                session, scenario, building = self._completed_session_and_scenario(candidate)

                with tempfile.TemporaryDirectory() as tmp:

                    record_session_replay(session, scenario, building, Path(tmp), arm="candidate", dt=1.0)

                    self.assertIsNotNone(session.replay_output_dir)
                    self.assertTrue((Path(tmp) / "building.syn").exists())


if __name__ == "__main__":
    unittest.main()
