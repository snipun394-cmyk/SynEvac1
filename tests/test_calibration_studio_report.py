import ast
import tempfile
import unittest
from pathlib import Path

from calibration_benchmark import WalkingSpeedCandidate, run_calibration_benchmark

import calibration_studio
from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, ValidationStatus
from calibration_studio.dashboard import generate_validation_dashboard
from calibration_studio.project import CalibrationProject
from calibration_studio.report import (
    CalibrationReportGenerator,
    build_session_report_content,
    render_markdown,
    save_session_report_markdown,
)
from calibration_studio.session import CalibrationSession
from calibration_studio.studio import CalibrationStudio

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


# =====================================================
# Calibration Studio Phase 7 -- Report Generation tests. Reuses
# tests/calibration_benchmark_fixtures.py for every real session/
# result throughout -- no artificial report/benchmark dataset is
# created beyond the minimal metadata a PublishedBenchmark itself
# needs.
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


class NotCompletedSessionReportTests(unittest.TestCase):

    def test_pending_session_has_no_result_and_no_reload_limitation(self):

        session = CalibrationSession()

        content = build_session_report_content(session=session)

        self.assertIsNone(content.result)
        self.assertIsNone(content.result_snapshot)
        self.assertFalse(any("reloaded session" in item for item in content.limitations))

    def test_pending_session_markdown_states_no_results_yet(self):

        session = CalibrationSession()

        markdown = render_markdown(build_session_report_content(session=session))

        self.assertIn("no statistical results are available yet", markdown)


class LiveSessionReportContentTests(unittest.TestCase):

    def test_live_session_has_a_real_result_object(self):

        session = _real_completed_session()

        content = build_session_report_content(session=session)

        self.assertIsNotNone(content.result)
        self.assertFalse(any("reloaded session" in item for item in content.limitations))

    def test_master_seed_and_candidate_snapshot_are_included(self):

        session = _real_completed_session()

        content = build_session_report_content(session=session)

        self.assertEqual(content.master_seed, MASTER_SEED)
        self.assertEqual(content.candidate_snapshot["name"], "Adult_Default.walking_speed")

    def test_failed_session_includes_failure_reason_in_limitations(self):

        session = CalibrationSession()
        session.mark_running()
        session.mark_failed("simulation crashed")

        content = build_session_report_content(session=session)

        self.assertTrue(any("simulation crashed" in item for item in content.limitations))

    def test_no_candidate_session_notes_defaults_only(self):

        session = CalibrationSession()

        content = build_session_report_content(session=session)

        self.assertTrue(any("production defaults only" in item for item in content.limitations))


class ReloadedSessionReportContentTests(unittest.TestCase):

    def test_reloaded_session_has_no_live_result_but_has_snapshot(self):

        session = _real_completed_session()
        reloaded = CalibrationSession.from_dict(session.to_dict())

        content = build_session_report_content(session=reloaded)

        self.assertIsNone(content.result)
        self.assertIsNotNone(content.result_snapshot)
        self.assertTrue(any("reloaded session" in item for item in content.limitations))

    def test_reloaded_session_reproducible_and_master_seed_survive(self):

        session = _real_completed_session()
        reloaded = CalibrationSession.from_dict(session.to_dict())

        content = build_session_report_content(session=reloaded)

        self.assertIs(content.reproducible, True)
        self.assertEqual(content.master_seed, MASTER_SEED)

    def test_reloaded_session_markdown_shows_fallback_table_not_live_report(self):

        session = _real_completed_session()
        reloaded = CalibrationSession.from_dict(session.to_dict())

        markdown = render_markdown(build_session_report_content(session=reloaded))

        self.assertNotIn("Calibration Benchmark Report --", markdown)
        self.assertIn("persisted summary only", markdown)
        self.assertIn("Evacuation Time", markdown)


class GitProvenanceAndReproducibilityLimitationTests(unittest.TestCase):

    def test_dirty_working_tree_is_disclosed(self):

        session = _real_completed_session()
        self.assertTrue(session.git_dirty)  # this repo's own working tree during development

        content = build_session_report_content(session=session)

        self.assertTrue(any("uncommitted working tree" in item for item in content.limitations))

    def test_unknown_reproducibility_is_disclosed(self):

        session = CalibrationSession()  # no result at all -- reproducible is None

        content = build_session_report_content(session=session)

        self.assertIsNone(content.reproducible)
        self.assertTrue(any("could not be determined" in item for item in content.limitations))


class ReplayAvailabilityTests(unittest.TestCase):

    def test_no_replay_reference_is_disclosed_and_unavailable(self):

        session = _real_completed_session()

        content = build_session_report_content(session=session)

        self.assertFalse(content.replay_available)
        self.assertTrue(any("Replay Studio is not currently available" in item for item in content.limitations))

    def test_replay_reference_present_is_available_and_not_disclosed_as_a_limitation(self):

        session = _real_completed_session()
        session.set_replay_reference("/some/output/dir", session.result.baseline_samples[0].scenario_id)

        content = build_session_report_content(session=session)

        self.assertTrue(content.replay_available)
        self.assertFalse(any("Replay Studio is not currently available" in item for item in content.limitations))

    def test_replay_availability_rendered_in_markdown(self):

        session = _real_completed_session()
        session.set_replay_reference("/some/output/dir", session.result.baseline_samples[0].scenario_id)

        markdown = render_markdown(build_session_report_content(session=session))

        self.assertIn("/some/output/dir", markdown)
        self.assertIn("Open via", markdown)


class BenchmarkInformationTests(unittest.TestCase):

    def test_no_benchmark_id_at_all(self):

        session = _real_completed_session()

        content = build_session_report_content(session=session)

        self.assertIsNone(content.benchmark_id)
        self.assertTrue(any("not associated with any Published Benchmark" in item for item in content.limitations))

    def test_benchmark_id_set_but_object_not_supplied(self):

        session = _real_completed_session(benchmark_id="bench-1")

        content = build_session_report_content(session=session)

        self.assertEqual(content.benchmark_id, "bench-1")
        self.assertIsNone(content.benchmark_title)
        self.assertTrue(any("no PublishedBenchmark object was supplied" in item for item in content.limitations))

    def test_benchmark_object_supplied(self):

        session = _real_completed_session(benchmark_id="bench-1")
        benchmark = _building_benchmark(benchmark_id="bench-1")

        content = build_session_report_content(session=session, benchmark=benchmark)

        self.assertEqual(content.benchmark_title, "NIST 10-Story Office Building")
        self.assertEqual(content.benchmark_dataset, "NIST")
        self.assertFalse(any("no PublishedBenchmark object" in item for item in content.limitations))


class ProjectMetadataTests(unittest.TestCase):

    def test_no_project_supplied(self):

        session = _real_completed_session()

        content = build_session_report_content(session=session)

        self.assertIsNone(content.project_id)
        markdown = render_markdown(content)
        self.assertIn("No project context was supplied", markdown)

    def test_project_supplied(self):

        project = CalibrationProject(name="Q3 Stair Recalibration")
        session = project.create_session()
        session.mark_running()
        session.mark_completed(_real_completed_session().result)

        content = build_session_report_content(session=session, project=project)

        self.assertEqual(content.project_name, "Q3 Stair Recalibration")
        self.assertEqual(content.project_status, "ACTIVE")


class DashboardInclusionTests(unittest.TestCase):

    def test_no_dashboard_supplied(self):

        session = _real_completed_session()

        content = build_session_report_content(session=session)

        self.assertIsNone(content.dashboard_parameter_confidence)
        self.assertTrue(any("No Validation Dashboard snapshot" in item for item in content.limitations))

    def test_dashboard_supplied_with_matching_parameter(self):

        session = _real_completed_session()
        dashboard = generate_validation_dashboard(benchmarks=(), sessions=[session])

        content = build_session_report_content(session=session, dashboard=dashboard)

        self.assertIsNotNone(content.dashboard_parameter_confidence)
        self.assertEqual(content.dashboard_parameter_confidence["parameter_name"], "Adult_Default.walking_speed")
        self.assertEqual(content.dashboard_parameter_confidence["n_reject"], 1)

    def test_dashboard_supplied_but_no_candidate_to_match(self):

        session = CalibrationSession()  # no candidate
        dashboard = generate_validation_dashboard(benchmarks=(), sessions=[session])

        content = build_session_report_content(session=session, dashboard=dashboard)

        self.assertIsNone(content.dashboard_parameter_confidence)

    def test_dashboard_summary_rendered_in_markdown(self):

        session = _real_completed_session()
        dashboard = generate_validation_dashboard(benchmarks=(), sessions=[session])

        markdown = render_markdown(build_session_report_content(session=session, dashboard=dashboard))

        self.assertIn("Adopt / Reject / Inconclusive / Unknown", markdown)


class LiveReportEmbedsRealCalibrationBenchmarkReportTests(unittest.TestCase):

    def test_live_report_embeds_the_real_calibration_benchmark_report_verbatim(self):

        session = _real_completed_session()

        markdown = render_markdown(build_session_report_content(session=session))

        self.assertIn("Calibration Benchmark Report -- Adult_Default.walking_speed", markdown)
        self.assertIn("## 1. Parameter Under Test", markdown)
        self.assertIn("## 5. Recommendation", markdown)

    def test_live_report_reproduces_the_known_reject_verdict(self):

        # The exact candidate calibration_benchmark's own demo report
        # (docs/architecture/calibration_benchmark_v1_demo_report.md)
        # already found REJECTs -- confirming this is the real,
        # unmodified render_markdown_report()/recommend(), not a
        # reimplementation.
        session = _real_completed_session()

        markdown = render_markdown(build_session_report_content(session=session))

        self.assertIn("Overall verdict: REJECT", markdown)


class ToDictTests(unittest.TestCase):

    def test_live_session_to_dict_uses_the_real_result(self):

        session = _real_completed_session()

        content = build_session_report_content(session=session)
        as_dict = content.to_dict()

        self.assertEqual(as_dict["result"]["n_completed_pairs"], 2)
        self.assertIsInstance(as_dict["limitations"], list)

    def test_reloaded_session_to_dict_uses_the_snapshot(self):

        session = _real_completed_session()
        reloaded = CalibrationSession.from_dict(session.to_dict())

        content = build_session_report_content(session=reloaded)
        as_dict = content.to_dict()

        self.assertEqual(as_dict["result"]["n_completed_pairs"], 2)


class CalibrationReportGeneratorTests(unittest.TestCase):

    def test_generate_session_report_returns_markdown_string(self):

        session = _real_completed_session()

        markdown = CalibrationReportGenerator().generate_session_report(session=session)

        self.assertIsInstance(markdown, str)
        self.assertIn("# Calibration Studio Session Report", markdown)

    def test_save_session_report_writes_a_file_at_the_expected_path(self):

        session = _real_completed_session()

        with tempfile.TemporaryDirectory() as tmpdir:

            path = CalibrationReportGenerator().save_session_report(session=session, storage_root=tmpdir)

            self.assertTrue(path.exists())
            self.assertEqual(path.name, f"{session.session_id}.md")
            self.assertIn("reports", path.parts)

    def test_saving_writes_exactly_the_given_markdown_verbatim(self):

        # save_session_report_markdown() is the lower-level, pure
        # file-writing half -- tested in isolation from
        # generate_session_report()'s own fresh-timestamp-per-call
        # behavior (two independent generate calls legitimately differ
        # in `generated_at`, which is correct, not a bug to test around
        # here).
        session = _real_completed_session()
        markdown = CalibrationReportGenerator().generate_session_report(session=session)

        with tempfile.TemporaryDirectory() as tmpdir:

            path = save_session_report_markdown(markdown, session.session_id, tmpdir)

            self.assertEqual(path.read_text(encoding="utf-8"), markdown)

    def test_save_session_report_content_matches_its_own_immediately_generated_markdown(self):

        # Confirms save_session_report() genuinely persists what it
        # itself generated (same call, same content), not a
        # byte-for-byte match against an independently-generated
        # second copy.
        session = _real_completed_session()
        generator = CalibrationReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:

            path = generator.save_session_report(session=session, storage_root=tmpdir)
            saved = path.read_text(encoding="utf-8")

            self.assertIn(f"# Calibration Studio Session Report -- {session.session_id}", saved)
            self.assertIn("Calibration Benchmark Report --", saved)


class CalibrationStudioFacadeReportTests(unittest.TestCase):

    def test_generate_session_report_finds_project_and_benchmark_automatically(self):

        studio = CalibrationStudio()
        benchmark = _building_benchmark()
        studio.benchmarks.register(benchmark)
        project = studio.create_project(name="P")
        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")

        session = studio.run_published_benchmark(
            project=project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
            definition=make_definition(), definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=1, dt=1.0,
        )

        report = studio.generate_session_report(session.session_id)

        self.assertIn("P", report)
        self.assertIn(benchmark.title, report)

    def test_unknown_session_id_raises(self):

        with self.assertRaises(ValueError):
            CalibrationStudio().generate_session_report("does-not-exist")

    def test_include_dashboard_false_omits_dashboard_data(self):

        studio = CalibrationStudio()
        benchmark = _building_benchmark()
        studio.benchmarks.register(benchmark)
        project = studio.create_project(name="P")
        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")

        session = studio.run_published_benchmark(
            project=project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
            definition=make_definition(), definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=1, dt=1.0,
        )

        report = studio.generate_session_report(session.session_id, include_dashboard=False)

        self.assertIn("No Validation Dashboard snapshot was supplied", report)

    def test_save_session_report_through_the_facade(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            studio = CalibrationStudio(storage_root=tmpdir)
            benchmark = _building_benchmark()
            studio.benchmarks.register(benchmark)
            project = studio.create_project(name="P")
            candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")

            session = studio.run_published_benchmark(
                project=project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
                definition=make_definition(), definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                n_scenarios=1, dt=1.0,
            )

            path = studio.save_session_report(session.session_id)

            self.assertTrue(path.exists())


class ArchitectureGuardTests(unittest.TestCase):

    def test_report_module_does_not_import_pyqt(self):

        path = Path(calibration_studio.__file__).parent / "report.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):

            module_names = []
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names = [node.module]

            for name in module_names:
                self.assertFalse("PyQt" in name or "PySide" in name)

    def test_report_module_does_not_import_scipy_or_numpy(self):

        path = Path(calibration_studio.__file__).parent / "report.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):

            module_names = []
            if isinstance(node, ast.Import):
                module_names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names = [node.module.split(".")[0]]

            for name in module_names:
                self.assertNotIn(name, ("scipy", "numpy"))

    def test_report_module_imports_render_markdown_report_from_calibration_benchmark(self):

        path = Path(calibration_studio.__file__).parent / "report.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "calibration_benchmark.report":
                imported.update(alias.name for alias in node.names)

        self.assertIn("render_markdown_report", imported)

    def test_render_markdown_report_is_called_exactly_once(self):

        path = Path(calibration_studio.__file__).parent / "report.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        call_sites = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "render_markdown_report"
        ]

        self.assertEqual(len(call_sites), 1)

    def test_no_prohibited_statistics_or_recommendation_names_defined(self):

        prohibited = frozenset({
            "paired_comparison", "confidence_interval", "effect_size_cohens_d",
            "recommend", "Verdict", "MetricComparison", "AdoptionRecommendation",
        })

        path = Path(calibration_studio.__file__).parent / "report.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        offenders = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in prohibited
        ]

        self.assertEqual(offenders, [])


class FullVerifyScenarioTests(unittest.TestCase):

    # This milestone's own VERIFY section, end to end, through the
    # real CalibrationStudio facade: report generation from a
    # completed session, report generation after reload, correct
    # inclusion of dashboard summaries, and correct handling of
    # unavailable information.

    def test_generation_reload_dashboard_and_unavailable_handling(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            studio = CalibrationStudio(storage_root=tmpdir)

            benchmark = _building_benchmark()
            studio.benchmarks.register(benchmark)

            project = studio.create_project(name="Phase 7 VERIFY")
            candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
            definition = make_definition()

            # --- Report generation from a completed session ---
            session = studio.run_published_benchmark(
                project=project, benchmark_id=benchmark.benchmark_id, candidate=candidate,
                definition=definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
                n_scenarios=2, dt=1.0,
            )
            benchmark.set_validation_status(ValidationStatus.RUN_WITH_CANDIDATES)

            live_report = studio.generate_session_report(session.session_id)
            self.assertIn("Calibration Benchmark Report", live_report)

            # --- Correct inclusion of dashboard summaries ---
            self.assertIn("Adopt / Reject / Inconclusive / Unknown", live_report)
            self.assertIn("NIST 10-Story Office Building", live_report)

            # --- Correct handling of unavailable information (before replay is recorded) ---
            self.assertIn("No replay artifacts have been recorded", live_report)

            studio.save_project(project)
            studio.save_session(session)
            studio.benchmarks.save_benchmark(benchmark)

            # --- Report generation after reload ---
            reopened = CalibrationStudio(storage_root=tmpdir)
            reopened.benchmarks.list_persisted_benchmarks()
            reopened.list_persisted_projects()

            reloaded_report = reopened.generate_session_report(session.session_id)

            self.assertNotIn("Calibration Benchmark Report --", reloaded_report)
            self.assertIn("persisted summary only", reloaded_report)
            self.assertIn("NIST 10-Story Office Building", reloaded_report)  # benchmark info survives
            self.assertIn("Adopt / Reject / Inconclusive / Unknown", reloaded_report)  # dashboard survives


if __name__ == "__main__":
    unittest.main()
