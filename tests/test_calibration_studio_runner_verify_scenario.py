import tempfile
import unittest
from pathlib import Path

from calibration_benchmark import WalkingSpeedCandidate

from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, PublishedValue
from calibration_studio.session import SessionStatus
from calibration_studio.studio import CalibrationStudio

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_definition


class FullVerifyScenarioTests(unittest.TestCase):

    # This milestone's own VERIFY section, as one end-to-end test:
    # running a benchmark from Calibration Studio, progress updates,
    # session completion, result storage, and reopening the completed
    # session -- through the real CalibrationStudio facade, the real
    # PublishedBenchmarkLibrary, and the real calibration_benchmark
    # execution path, reusing the existing test fixture building/
    # definition throughout.

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.storage_root = Path(self._tmpdir.name)

    def tearDown(self):

        self._tmpdir.cleanup()

    def test_run_progress_completion_storage_reopen(self):

        studio = CalibrationStudio(storage_root=self.storage_root)

        benchmark = PublishedBenchmark(
            title="Calibration Benchmark Test Fixture",
            source_citation="internal test fixture",
            dataset="synthetic",
            benchmark_type=BenchmarkType.BUILDING_RECREATION,
            geometry_reference=GeometryVersion(
                version="v1", ref="tests.calibration_benchmark_fixtures.make_building",
            ),
            published_values={"evacuation_time_s": PublishedValue(value=700.0, unit="s")},
        )
        studio.benchmarks.register(benchmark)

        project = studio.create_project(name="Phase 4 VERIFY")
        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test rationale", "test")
        definition = make_definition()

        # --- Running a benchmark from Calibration Studio ---
        session = studio.run_published_benchmark(
            project=project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
            definition=definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        # --- Progress updates ---
        self.assertEqual(session.n_scenarios_completed, 2)
        self.assertEqual(session.progress, 1.0)

        # --- Session completion ---
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertIsNotNone(session.result)
        original_reproducible = session.reproducible
        self.assertIs(original_reproducible, True)

        # --- Result storage ---
        studio.save_project(project)
        studio.save_session(session)
        studio.benchmarks.save_benchmark(benchmark)

        # --- Reopening the completed session ---
        reopened_studio = CalibrationStudio(storage_root=self.storage_root)
        reopened_project = reopened_studio.load_project(project.project_id)

        self.assertEqual(len(reopened_project.sessions), 1)
        reopened_session = reopened_project.sessions[0]

        self.assertEqual(reopened_session.session_id, session.session_id)
        self.assertEqual(reopened_session.status, SessionStatus.COMPLETED)
        self.assertEqual(reopened_session.progress, 1.0)
        self.assertEqual(reopened_session.reproducible, original_reproducible)
        self.assertIsNotNone(reopened_session.result_snapshot)
        self.assertEqual(
            reopened_session.result_snapshot["comparisons"]["evacuation_time"]["n_pairs"], 2,
        )

        reopened_benchmark = reopened_studio.benchmarks.load_benchmark(benchmark.benchmark_id)
        self.assertEqual(reopened_benchmark.calibration_history, (session.session_id,))


if __name__ == "__main__":
    unittest.main()
