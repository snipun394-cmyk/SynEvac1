import ast
import unittest
from pathlib import Path

from calibration_benchmark import WalkingSpeedCandidate, run_calibration_benchmark
from calibration_benchmark.recommendation import Verdict

import calibration_studio
from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, ValidationStatus
from calibration_studio.dashboard import generate_validation_dashboard
from calibration_studio.session import CalibrationSession
from calibration_studio.studio import CalibrationStudio

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


# =====================================================
# Calibration Studio Phase 6 -- Validation Dashboard tests. Reuses
# tests/calibration_benchmark_fixtures.py throughout for every real
# session/result -- no artificial benchmark dataset is created; only
# PublishedBenchmark *metadata* (title/citation/status) is authored
# per test, since the dashboard's own job is aggregating metadata, not
# running anything.
# =====================================================


def _building_benchmark(**overrides):

    defaults = dict(
        title="NIST 10-Story Office Building",
        source_citation="Peacock, Hoskins & Kuligowski (2012)",
        dataset="NIST",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref="tests.calibration_benchmark_fixtures.make_building"),
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


def _real_completed_session(benchmark_id=None):

    building = make_building()
    definition = make_definition()
    candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")

    result = run_calibration_benchmark(
        candidate, building, definition, DEFINITION_ID, MASTER_SEED, n_scenarios=2, dt=1.0,
    )

    session = CalibrationSession(candidate=candidate, master_seed=MASTER_SEED, benchmark_id=benchmark_id)
    session.mark_running(n_scenarios_total=2)
    session.mark_completed(result)

    return session


class EmptyDashboardTests(unittest.TestCase):

    def test_no_benchmarks_no_sessions(self):

        dashboard = generate_validation_dashboard(benchmarks=(), sessions=())

        self.assertEqual(dashboard.total_benchmarks, 0)
        self.assertIsNone(dashboard.validation_coverage)
        self.assertEqual(dashboard.validated_benchmarks, ())
        self.assertEqual(dashboard.pending_benchmarks, ())
        self.assertEqual(dashboard.known_broken_benchmarks, ())
        self.assertEqual(dashboard.benchmark_status, ())
        self.assertEqual(dashboard.parameter_confidence, ())
        self.assertEqual(dashboard.calibration_history, {})
        self.assertEqual(dashboard.evidence_availability, {})


class BenchmarkBucketingTests(unittest.TestCase):

    def test_not_run_benchmark_is_pending(self):

        benchmark = _building_benchmark()

        dashboard = generate_validation_dashboard(benchmarks=[benchmark], sessions=())

        self.assertEqual([b.benchmark_id for b in dashboard.pending_benchmarks], [benchmark.benchmark_id])
        self.assertEqual(dashboard.validated_benchmarks, ())
        self.assertEqual(dashboard.known_broken_benchmarks, ())

    def test_run_with_candidates_benchmark_is_validated(self):

        benchmark = _building_benchmark()
        benchmark.set_validation_status(ValidationStatus.RUN_WITH_CANDIDATES)

        dashboard = generate_validation_dashboard(benchmarks=[benchmark], sessions=())

        self.assertEqual([b.benchmark_id for b in dashboard.validated_benchmarks], [benchmark.benchmark_id])

    def test_run_with_defaults_benchmark_is_also_validated(self):

        benchmark = _building_benchmark()
        benchmark.set_validation_status(ValidationStatus.RUN_WITH_DEFAULTS)

        dashboard = generate_validation_dashboard(benchmarks=[benchmark], sessions=())

        self.assertEqual([b.benchmark_id for b in dashboard.validated_benchmarks], [benchmark.benchmark_id])

    def test_known_broken_benchmark_is_its_own_bucket_not_validated_or_pending(self):

        benchmark = _building_benchmark()
        benchmark.set_validation_status(ValidationStatus.KNOWN_BROKEN)

        dashboard = generate_validation_dashboard(benchmarks=[benchmark], sessions=())

        self.assertEqual([b.benchmark_id for b in dashboard.known_broken_benchmarks], [benchmark.benchmark_id])
        self.assertEqual(dashboard.validated_benchmarks, ())
        self.assertEqual(dashboard.pending_benchmarks, ())

    def test_benchmark_status_lists_every_benchmark_regardless_of_status(self):

        pending = _building_benchmark(title="Pending")
        validated = _building_benchmark(title="Validated")
        validated.set_validation_status(ValidationStatus.RUN_WITH_CANDIDATES)
        broken = _building_benchmark(title="Broken")
        broken.set_validation_status(ValidationStatus.KNOWN_BROKEN)

        dashboard = generate_validation_dashboard(benchmarks=[pending, validated, broken], sessions=())

        self.assertEqual(len(dashboard.benchmark_status), 3)
        self.assertEqual(
            {b.title for b in dashboard.benchmark_status}, {"Pending", "Validated", "Broken"},
        )


class ValidationCoverageTests(unittest.TestCase):

    def test_coverage_counts_validated_and_known_broken_as_covered(self):

        validated = _building_benchmark(title="A")
        validated.set_validation_status(ValidationStatus.RUN_WITH_CANDIDATES)
        broken = _building_benchmark(title="B")
        broken.set_validation_status(ValidationStatus.KNOWN_BROKEN)
        pending = _building_benchmark(title="C")

        dashboard = generate_validation_dashboard(benchmarks=[validated, broken, pending], sessions=())

        self.assertAlmostEqual(dashboard.validation_coverage, 2 / 3)

    def test_all_pending_is_zero_coverage(self):

        dashboard = generate_validation_dashboard(
            benchmarks=[_building_benchmark(), _building_benchmark(title="B")], sessions=(),
        )

        self.assertEqual(dashboard.validation_coverage, 0.0)

    def test_all_validated_is_full_coverage(self):

        benchmark = _building_benchmark()
        benchmark.set_validation_status(ValidationStatus.RUN_WITH_CANDIDATES)

        dashboard = generate_validation_dashboard(benchmarks=[benchmark], sessions=())

        self.assertEqual(dashboard.validation_coverage, 1.0)


class CalibrationHistoryAndEvidenceAvailabilityTests(unittest.TestCase):

    def test_benchmark_with_no_history_has_no_evidence(self):

        benchmark = _building_benchmark()

        dashboard = generate_validation_dashboard(benchmarks=[benchmark], sessions=())

        self.assertEqual(dashboard.calibration_history[benchmark.benchmark_id], ())
        self.assertFalse(dashboard.evidence_availability[benchmark.benchmark_id])

    def test_benchmark_with_a_resolvable_session_has_evidence(self):

        session = _real_completed_session()
        benchmark = _building_benchmark()
        benchmark.add_calibration_session(session.session_id)

        dashboard = generate_validation_dashboard(benchmarks=[benchmark], sessions=[session])

        summary = dashboard.benchmark_status[0]
        self.assertTrue(summary.has_evidence)
        self.assertEqual(summary.calibration_history_recorded, 1)
        self.assertEqual(summary.calibration_history_resolvable, 1)
        self.assertEqual(dashboard.calibration_history[benchmark.benchmark_id], (session.session_id,))

    def test_recorded_but_unresolvable_session_is_flagged_distinctly(self):

        # calibration_history references a session_id that is NOT among
        # the sessions this dashboard run was given -- e.g. it exists on
        # disk but was never loaded this call. Real, honest signal, not
        # a fabricated resolution.
        benchmark = _building_benchmark()
        benchmark.add_calibration_session("some-session-id-not-provided")

        dashboard = generate_validation_dashboard(benchmarks=[benchmark], sessions=())

        summary = dashboard.benchmark_status[0]
        self.assertTrue(summary.has_evidence)  # recorded, even though unresolved
        self.assertEqual(summary.calibration_history_recorded, 1)
        self.assertEqual(summary.calibration_history_resolvable, 0)


class ParameterConfidenceTests(unittest.TestCase):

    def test_real_session_gets_a_real_recommendation_verdict_not_reimplemented(self):

        # WalkingSpeedCandidate("Adult_Default", 0.65, ...) is the exact
        # candidate calibration_benchmark's own demo report
        # (docs/architecture/calibration_benchmark_v1_demo_report.md)
        # already found REJECTs -- confirming this dashboard calls the
        # real calibration_benchmark.recommend(), not a guess, since it
        # reproduces that exact, already-known verdict.
        session = _real_completed_session(benchmark_id="bench-1")

        dashboard = generate_validation_dashboard(benchmarks=(), sessions=[session])

        self.assertEqual(len(dashboard.parameter_confidence), 1)
        summary = dashboard.parameter_confidence[0]
        self.assertEqual(summary.parameter_name, "Adult_Default.walking_speed")
        self.assertEqual(summary.n_sessions, 1)
        self.assertEqual(summary.n_reject, 1)
        self.assertEqual(summary.n_adopt, 0)
        self.assertEqual(summary.n_unknown, 0)
        self.assertEqual(summary.benchmark_ids, ("bench-1",))

    def test_session_with_no_candidate_is_excluded(self):

        session = CalibrationSession()  # no candidate at all

        dashboard = generate_validation_dashboard(benchmarks=(), sessions=[session])

        self.assertEqual(dashboard.parameter_confidence, ())

    def test_reloaded_session_verdict_is_honestly_unknown(self):

        # Live session.result -> None after a save/load round trip
        # (calibration_benchmark has no MetricComparison.from_dict()) --
        # the dashboard must not guess a verdict from result_snapshot.
        session = _real_completed_session()
        reloaded = CalibrationSession.from_dict(session.to_dict())

        dashboard = generate_validation_dashboard(benchmarks=(), sessions=[reloaded])

        summary = dashboard.parameter_confidence[0]
        self.assertEqual(summary.n_unknown, 1)
        self.assertEqual(summary.n_adopt, 0)
        self.assertEqual(summary.n_reject, 0)
        self.assertEqual(summary.n_inconclusive, 0)

    def test_reloaded_session_reproducible_still_counted(self):

        # reproducible (unlike the recommendation verdict) DOES survive
        # a reload, via result_snapshot -- confirming the dashboard uses
        # the right source for each fact rather than treating "reloaded"
        # as uniformly "unknown everything."
        session = _real_completed_session()
        reloaded = CalibrationSession.from_dict(session.to_dict())

        dashboard = generate_validation_dashboard(benchmarks=(), sessions=[reloaded])

        self.assertEqual(dashboard.parameter_confidence[0].n_reproducible, 1)

    def test_multiple_sessions_same_parameter_are_grouped(self):

        session_a = _real_completed_session()
        session_b = _real_completed_session()

        dashboard = generate_validation_dashboard(benchmarks=(), sessions=[session_a, session_b])

        self.assertEqual(len(dashboard.parameter_confidence), 1)
        self.assertEqual(dashboard.parameter_confidence[0].n_sessions, 2)
        self.assertEqual(
            set(dashboard.parameter_confidence[0].session_ids), {session_a.session_id, session_b.session_id},
        )

    def test_different_parameters_produce_separate_summaries(self):

        session_a = _real_completed_session()

        building = make_building()
        definition = make_definition()
        other_candidate = WalkingSpeedCandidate("Child_Default", 0.5, "test", "test")
        other_result = run_calibration_benchmark(
            other_candidate, building, definition, DEFINITION_ID, MASTER_SEED, n_scenarios=1, dt=1.0,
        )
        session_b = CalibrationSession(candidate=other_candidate, master_seed=MASTER_SEED)
        session_b.mark_running(n_scenarios_total=1)
        session_b.mark_completed(other_result)

        dashboard = generate_validation_dashboard(benchmarks=(), sessions=[session_a, session_b])

        names = {summary.parameter_name for summary in dashboard.parameter_confidence}
        self.assertEqual(names, {"Adult_Default.walking_speed", "Child_Default.walking_speed"})


class ReadOnlyGuardTests(unittest.TestCase):

    # "The dashboard must compute summaries only... never modify
    # calibration data" -- proven behaviorally (nothing changed after
    # the call), not just asserted in a comment.

    def test_benchmark_state_is_unchanged_after_generating_a_dashboard(self):

        benchmark = _building_benchmark()
        benchmark.set_validation_status(ValidationStatus.RUN_WITH_CANDIDATES)
        benchmark.add_calibration_session("some-session")
        version_before = benchmark.version
        status_before = benchmark.validation_status
        history_before = benchmark.calibration_history

        generate_validation_dashboard(benchmarks=[benchmark], sessions=())

        self.assertEqual(benchmark.version, version_before)
        self.assertEqual(benchmark.validation_status, status_before)
        self.assertEqual(benchmark.calibration_history, history_before)

    def test_session_state_is_unchanged_after_generating_a_dashboard(self):

        session = _real_completed_session()
        status_before = session.status
        result_before = session.result

        generate_validation_dashboard(benchmarks=(), sessions=[session])

        self.assertEqual(session.status, status_before)
        self.assertIs(session.result, result_before)

    def test_dashboard_module_never_calls_a_mutating_method(self):

        # AST-based: no Call node whose function is an attribute access
        # starting with a known mutator prefix (set_/add_/remove_/mark_/
        # register/unregister/save_) anywhere in dashboard.py.
        path = Path(calibration_studio.__file__).parent / "dashboard.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        mutator_prefixes = ("set_", "add_", "remove_", "mark_", "register", "unregister", "save_", "update_")
        offenders = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr.startswith(mutator_prefixes):
                    offenders.append(node.func.attr)

        self.assertEqual(offenders, [])

    def test_dashboard_module_does_not_import_pyqt(self):

        path = Path(calibration_studio.__file__).parent / "dashboard.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):

            module_names = []
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names = [node.module]

            for name in module_names:
                self.assertFalse("PyQt" in name or "PySide" in name)

    def test_dashboard_module_imports_recommend_rather_than_reimplementing_it(self):

        path = Path(calibration_studio.__file__).parent / "dashboard.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "calibration_benchmark":
                imported.update(alias.name for alias in node.names)

        self.assertIn("recommend", imported)


class CalibrationStudioFacadeTests(unittest.TestCase):

    def test_generate_validation_dashboard_reflects_current_studio_state(self):

        studio = CalibrationStudio()
        benchmark = _building_benchmark()
        studio.benchmarks.register(benchmark)

        dashboard = studio.generate_validation_dashboard()

        self.assertEqual(dashboard.total_benchmarks, 1)

    def test_automatic_update_after_a_new_session_is_created(self):

        studio = CalibrationStudio()
        benchmark = _building_benchmark()
        studio.benchmarks.register(benchmark)
        project = studio.create_project(name="P")

        dashboard_before = studio.generate_validation_dashboard()
        self.assertEqual(len(dashboard_before.parameter_confidence), 0)

        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
        studio.run_published_benchmark(
            project=project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
            definition=make_definition(), definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=1, dt=1.0,
        )

        dashboard_after = studio.generate_validation_dashboard()
        self.assertEqual(len(dashboard_after.parameter_confidence), 1)
        self.assertEqual(dashboard_after.evidence_availability[benchmark.benchmark_id], True)

    def test_correct_aggregation_from_persisted_data(self):

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:

            studio = CalibrationStudio(storage_root=tmpdir)
            benchmark = _building_benchmark()
            benchmark.set_validation_status(ValidationStatus.RUN_WITH_CANDIDATES)
            studio.benchmarks.register(benchmark)

            project = studio.create_project(name="Persisted")
            candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
            session = studio.run_published_benchmark(
                project=project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
                definition=make_definition(), definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                n_scenarios=1, dt=1.0,
            )
            benchmark.add_calibration_session(session.session_id)

            studio.save_project(project)
            studio.save_session(session)
            studio.benchmarks.save_benchmark(benchmark)

            # Fresh studio -- nothing in memory, dashboard must reflect
            # only persisted data once explicitly (re)loaded (Phase 2's
            # own "no hidden magic" persistence convention -- the
            # dashboard itself never reads a file).
            reopened_studio = CalibrationStudio(storage_root=tmpdir)
            reopened_studio.benchmarks.list_persisted_benchmarks()
            reopened_studio.list_persisted_projects()

            dashboard = reopened_studio.generate_validation_dashboard()

            self.assertEqual(dashboard.total_benchmarks, 1)
            self.assertEqual(len(dashboard.validated_benchmarks), 1)
            self.assertEqual(len(dashboard.parameter_confidence), 1)
            self.assertEqual(dashboard.parameter_confidence[0].n_unknown, 1)  # reloaded -- honestly unknown verdict
            self.assertEqual(dashboard.parameter_confidence[0].n_reproducible, 1)  # but reproducible survives


if __name__ == "__main__":
    unittest.main()
