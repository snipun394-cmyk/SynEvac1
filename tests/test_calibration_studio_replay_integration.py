import ast
import tempfile
import unittest
from pathlib import Path

from calibration_benchmark import WalkingSpeedCandidate, run_calibration_benchmark

from command_center.incident_data import IncidentData

from scenario_pipeline import run_batch_pipeline

import calibration_studio
from calibration_studio.replay_integration import ReplayArtifactsUnavailableError, open_in_replay_studio, record_session_replay
from calibration_studio.session import CalibrationSession
from calibration_studio.studio import CalibrationStudio

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


# =====================================================
# Calibration Studio Phase 5 -- Replay Studio Integration tests.
# Reuses tests/calibration_benchmark_fixtures.py (same building/
# definition/DEFINITION_ID/MASTER_SEED every existing calibration_
# benchmark test already uses) and Replay Studio's own real, unmodified
# production functions (replay_studio.session.resolve_scenario_artifacts,
# command_center.incident_data.load_incident) to produce and read real
# replay data -- no new replay dataset format, no mocked artifacts.
# =====================================================


def _completed_session_and_scenario(n_scenarios=2):

    building = make_building()
    definition = make_definition()
    candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")

    result = run_calibration_benchmark(
        candidate, building, definition, DEFINITION_ID, MASTER_SEED, n_scenarios=n_scenarios, dt=1.0,
    )

    session = CalibrationSession(candidate=candidate, master_seed=MASTER_SEED)
    session.mark_running(n_scenarios_total=n_scenarios)
    session.mark_completed(result)

    # Deterministically regenerates the identical batch (Phase 0's own
    # verified guarantee) to recover a real Scenario object -- the same
    # scenario calibration_benchmark's own harness itself ran, not a
    # newly/independently sampled one.
    batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, n_scenarios)

    return session, batch.scenarios[0], building


class NoVisualizationCodeArchitectureGuardTests(unittest.TestCase):

    # "Calibration Studio must never render simulations itself... No
    # visualization code should be introduced" -- a real, checkable
    # guard, not just a comment.

    def test_replay_integration_module_does_not_import_pyqt(self):

        path = Path(calibration_studio.__file__).parent / "replay_integration.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        offending = []

        for node in ast.walk(tree):

            module_names = []

            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names = [node.module]

            for name in module_names:
                if "PyQt" in name or "PySide" in name:
                    offending.append(name)

        self.assertEqual(offending, [])

    def test_studio_module_does_not_import_pyqt_either(self):

        path = Path(calibration_studio.__file__).parent / "studio.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):

            module_names = []

            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names = [node.module]

            for name in module_names:
                self.assertFalse("PyQt" in name or "PySide" in name, f"studio.py imports {name!r}")

    def test_open_in_replay_studio_delegates_to_replay_studio_session_resolver(self):

        # Confirms the real delegation point exists (AST-based, not a
        # naming convention this test just trusts).
        path = Path(calibration_studio.__file__).parent / "replay_integration.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        imported_from_replay_studio = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "replay_studio.session":
                imported_from_replay_studio.update(alias.name for alias in node.names)

        self.assertIn("resolve_scenario_artifacts", imported_from_replay_studio)


class RecordSessionReplayTests(unittest.TestCase):

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmpdir.name)

    def tearDown(self):

        self._tmpdir.cleanup()

    def test_records_a_real_loadable_incident(self):

        session, scenario, building = _completed_session_and_scenario()

        record_session_replay(session, scenario, building, self.output_dir, arm="candidate", dt=1.0)

        self.assertEqual(session.replay_output_dir, str(self.output_dir))
        self.assertEqual(session.replay_scenario_id, scenario.metadata.scenario_id)

    def test_recorded_incident_has_occupant_routes(self):

        session, scenario, building = _completed_session_and_scenario()

        record_session_replay(session, scenario, building, self.output_dir, arm="candidate", dt=1.0)
        incident = open_in_replay_studio(session)

        self.assertGreater(len(incident.occupant_routes), 0)

    def test_recording_a_scenario_the_session_never_ran_is_rejected(self):

        session, _scenario, building = _completed_session_and_scenario()

        definition = make_definition()
        other_batch = run_batch_pipeline(definition, "a-different-definition-id", building, MASTER_SEED + 999, 1)
        unrelated_scenario = other_batch.scenarios[0]

        with self.assertRaises(ValueError):
            record_session_replay(session, unrelated_scenario, building, self.output_dir, arm="candidate", dt=1.0)

    def test_recording_without_a_live_candidate_raises(self):

        # A reloaded session's candidate is None (calibration_benchmark
        # has no from_dict() -- see CalibrationSession's own docstring)
        # -- recording replay for it needs the live candidate to select
        # baseline_*()/candidate_*() overrides, so this must fail
        # honestly rather than silently falling back to something else.
        session, scenario, building = _completed_session_and_scenario()
        reloaded = CalibrationSession.from_dict(session.to_dict())

        with self.assertRaises(ValueError):
            record_session_replay(reloaded, scenario, building, self.output_dir, arm="candidate", dt=1.0)

    def test_invalid_arm_raises(self):

        session, scenario, building = _completed_session_and_scenario()

        with self.assertRaises(ValueError):
            record_session_replay(session, scenario, building, self.output_dir, arm="not-a-real-arm", dt=1.0)

    def test_baseline_arm_also_records(self):

        session, scenario, building = _completed_session_and_scenario()

        record_session_replay(session, scenario, building, self.output_dir, arm="baseline", dt=1.0)

        incident = open_in_replay_studio(session)
        self.assertIsInstance(incident, IncidentData)


class OpenInReplayStudioValidationTests(unittest.TestCase):

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmpdir.name)

    def tearDown(self):

        self._tmpdir.cleanup()

    def test_no_replay_reference_recorded_raises_a_meaningful_error(self):

        session, _scenario, _building = _completed_session_and_scenario()

        with self.assertRaises(ReplayArtifactsUnavailableError) as ctx:
            open_in_replay_studio(session)

        self.assertIn(session.session_id, str(ctx.exception))

    def test_replay_reference_pointing_at_an_empty_directory_raises(self):

        session, scenario, _building = _completed_session_and_scenario()
        # A real scenario_id this session's own result actually
        # produced, but with no artifacts ever recorded for it --
        # exercising the "reference set, directory empty" case
        # specifically, distinct from set_replay_reference()'s own
        # integrity check (already covered by
        # CalibrationSessionReplayReferenceIntegrityTests).
        session.set_replay_reference(str(self.output_dir), scenario.metadata.scenario_id)

        with self.assertRaises(ReplayArtifactsUnavailableError) as ctx:
            open_in_replay_studio(session)

        self.assertIn("building.syn", str(ctx.exception))

    def test_missing_scenario_file_raises_even_when_building_syn_exists(self):

        session, scenario, building = _completed_session_and_scenario()

        # Record a real replay (writes building.syn + the scenario +
        # occupant_routes.json for THIS scenario_id), then point the
        # session at a DIFFERENT scenario_id that was never saved --
        # exercising the "building.syn exists, but this scenario
        # doesn't" branch specifically, without a wrong-session's-own-
        # scenario integrity check getting in the way (set directly on
        # the session's private state is not needed -- a second,
        # unsaved scenario_id from the same result is enough).
        record_session_replay(session, scenario, building, self.output_dir, arm="candidate", dt=1.0)

        result = session.result
        other_scenario_id = next(
            s.scenario_id for s in result.candidate_samples if s.scenario_id != scenario.metadata.scenario_id
        )
        session.set_replay_reference(str(self.output_dir), other_scenario_id)

        with self.assertRaises(ReplayArtifactsUnavailableError) as ctx:
            open_in_replay_studio(session)

        self.assertIn(other_scenario_id, str(ctx.exception))


class CalibrationSessionReplayReferenceIntegrityTests(unittest.TestCase):

    def test_set_replay_reference_accepts_a_scenario_id_from_the_result(self):

        session, scenario, _building = _completed_session_and_scenario()

        session.set_replay_reference("/some/dir", scenario.metadata.scenario_id)

        self.assertEqual(session.replay_scenario_id, scenario.metadata.scenario_id)

    def test_set_replay_reference_rejects_an_unrelated_scenario_id(self):

        session, _scenario, _building = _completed_session_and_scenario()

        with self.assertRaises(ValueError):
            session.set_replay_reference("/some/dir", "totally-unrelated-scenario-id")

    def test_set_replay_reference_is_unconstrained_before_any_result_exists(self):

        # No result yet -- nothing to validate against, so any
        # scenario_id is accepted (a genuine "not yet checkable" case,
        # not a bypass).
        session = CalibrationSession()

        session.set_replay_reference("/some/dir", "any-scenario-id")

        self.assertEqual(session.replay_scenario_id, "any-scenario-id")

    def test_replay_reference_round_trips_through_to_dict_from_dict(self):

        session, scenario, _building = _completed_session_and_scenario()
        session.set_replay_reference("/some/dir", scenario.metadata.scenario_id)

        restored = CalibrationSession.from_dict(session.to_dict())

        self.assertEqual(restored.replay_output_dir, "/some/dir")
        self.assertEqual(restored.replay_scenario_id, scenario.metadata.scenario_id)


class FullVerifyScenarioTests(unittest.TestCase):

    # This milestone's own VERIFY section, end to end, through the real
    # CalibrationStudio facade: a completed session opens in Replay
    # Studio, missing artifacts produce a meaningful error, and replay
    # receives the correct scenario.

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmpdir.name)

    def tearDown(self):

        self._tmpdir.cleanup()

    def test_completed_session_opens_missing_artifacts_error_correct_scenario(self):

        studio = CalibrationStudio()
        project = studio.create_project(name="Phase 5 VERIFY")
        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
        building = make_building()
        definition = make_definition()

        session = studio.run_published_benchmark(
            project=project, benchmark_id=self._register_benchmark(studio),
            candidate=candidate, definition=definition, definition_id=DEFINITION_ID,
            master_seed=MASTER_SEED, n_scenarios=2, dt=1.0,
        )

        # --- Missing replay artifacts produce meaningful errors ---
        with self.assertRaises(ReplayArtifactsUnavailableError):
            studio.open_in_replay_studio(session.session_id)

        # --- Running/recording, then opening ---
        batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, 2)
        scenario = batch.scenarios[0]

        studio.record_session_replay(
            session=session, scenario=scenario, building=building,
            output_dir=self.output_dir, arm="candidate", dt=1.0,
        )

        incident = studio.open_in_replay_studio(session.session_id)

        # --- Completed session opens in Replay Studio ---
        self.assertIsInstance(incident, IncidentData)

        # --- Replay receives the correct scenario ---
        self.assertEqual(incident.scenario.metadata.scenario_id, scenario.metadata.scenario_id)

    def test_unknown_session_id_raises_a_clear_error(self):

        studio = CalibrationStudio()

        with self.assertRaises(ValueError):
            studio.open_in_replay_studio("does-not-exist")

    def _register_benchmark(self, studio):

        from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, PublishedValue

        benchmark = PublishedBenchmark(
            title="Calibration Benchmark Test Fixture", source_citation="internal fixture",
            dataset="synthetic", benchmark_type=BenchmarkType.BUILDING_RECREATION,
            geometry_reference=GeometryVersion(
                version="v1", ref="tests.calibration_benchmark_fixtures.make_building",
            ),
            published_values={"evacuation_time_s": PublishedValue(value=700.0, unit="s")},
        )
        studio.benchmarks.register(benchmark)

        return benchmark.benchmark_id


if __name__ == "__main__":
    unittest.main()
