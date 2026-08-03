import tempfile
import unittest
from pathlib import Path

from calibration_benchmark import ParameterCandidate, WalkingSpeedCandidate

from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, PublishedValue
from calibration_studio.replay_integration import record_session_replay
from calibration_studio.session import CalibrationSession, SessionStatus
from calibration_studio.studio import CalibrationStudio

from scenario_pipeline import run_batch_pipeline

from automatic_calibration.budget import AutoCalibrationBudget
from automatic_calibration.engine import AutoCalibrationEngine
from automatic_calibration.objectives import CalibrationObjective, ObjectiveDirection, PublishedValueObjective
from automatic_calibration.run import AutoCalibrationRunStatus
from automatic_calibration.search_space import ParameterDimension, SearchSpace
from automatic_calibration.strategy import AutoCalibrationStrategy

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture,
# integration tests.
#
# _FixedSequenceStrategy/_AlwaysFailingStrategy/_ScriptedNonBenchmarkObjective
# below are MINIMAL TEST FIXTURES ONLY. _FixedSequenceStrategy proposes
# values from a fixed, pre-supplied list in order, then signals
# exhaustion (returns None) -- it is NOT Grid Search, Random Search, or
# any other concrete optimization strategy this milestone's own brief
# explicitly excludes ("Do NOT implement Grid Search. Do NOT implement
# Random Search. Do NOT implement Bayesian Optimization. Do NOT
# implement Evolutionary Algorithms. Do NOT implement Pareto Search.").
# It exists purely to exercise AutoCalibrationEngine's own plumbing
# (propose/evaluate/record loop, budget/exhaustion handling, failure
# handling) against a real AutoCalibrationStrategy implementation. All
# three fixtures live in this test file only, never in the
# automatic_calibration package itself.
# =====================================================


class _FixedSequenceStrategy(AutoCalibrationStrategy):

    def __init__(self, values):

        self._values = list(values)
        self._index = 0

    def propose(self, *, search_space, objective, history):

        if self._index >= len(self._values):
            return None

        dimension = search_space.dimensions[0]
        value = self._values[self._index]
        self._index += 1

        return dimension.build_candidate(value)

    def describe(self):

        return f"_FixedSequenceStrategy(values={self._values!r}) -- test fixture only, not a production strategy"


class _AlwaysFailingCandidate(ParameterCandidate):

    def __init__(self):

        super().__init__(
            name="Broken.candidate", subsystem="Test", calibration_tier="Tier 2",
            dataset_source="test", current_value=1.0, candidate_value=2.0, unit="x", rationale="test",
        )

    def candidate_capacity_model(self):

        raise RuntimeError("deliberately broken for testing")


class _AlwaysFailingStrategy(AutoCalibrationStrategy):

    def propose(self, *, search_space, objective, history):

        return _AlwaysFailingCandidate()

    def describe(self):

        return "_AlwaysFailingStrategy -- test fixture only"


class _ScriptedNonBenchmarkObjective(CalibrationObjective):

    # Deliberately has NO benchmark_id attribute at all -- proves
    # AutoCalibrationEngine correctly falls back to
    # CalibrationStudio.run_parameter_sweep() (not run_published_
    # benchmark()) whenever an objective isn't benchmark-anchored.

    direction = ObjectiveDirection.MAXIMIZE

    def score(self, session):

        if session.result is None:
            return None

        comparison = session.result.comparisons.get("evacuation_time")

        return comparison.candidate_mean if comparison else None

    def describe(self):

        return "_ScriptedNonBenchmarkObjective -- test fixture only"


def _search_space():

    return SearchSpace(dimensions=(
        ParameterDimension(
            name="Adult_Default.walking_speed", bounds=(0.6, 1.4),
            build=lambda v: WalkingSpeedCandidate("Adult_Default", v, "test", "test"),
        ),
    ))


def _make_benchmark(**overrides):

    defaults = dict(
        title="Engine Test Fixture",
        source_citation="internal test fixture",
        dataset="synthetic",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref="tests.calibration_benchmark_fixtures.make_building"),
        published_values={"evacuation_time_s": PublishedValue(value=700.0, unit="s")},
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


class EngineTestCase(unittest.TestCase):

    def setUp(self):

        self.studio = CalibrationStudio()
        self.project = self.studio.create_project(name="Auto Calibration Engine Tests")
        self.building = make_building()
        self.definition = make_definition()

    def _published_value_objective(self, register=True):

        benchmark = _make_benchmark()

        if register:
            self.studio.benchmarks.register(benchmark)

        return PublishedValueObjective(
            benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
        )

    def _engine(self, strategy, objective=None, max_evaluations=3):

        return AutoCalibrationEngine(
            studio=self.studio, project=self.project, building=self.building, definition=self.definition,
            definition_id=DEFINITION_ID, master_seed=MASTER_SEED, n_scenarios=2, dt=1.0,
            search_space=_search_space(), objective=objective or self._published_value_objective(),
            strategy=strategy, budget=AutoCalibrationBudget(max_evaluations=max_evaluations),
        )


class SessionCreationTests(EngineTestCase):

    def test_run_produces_a_completed_auto_calibration_run(self):

        engine = self._engine(_FixedSequenceStrategy([0.7, 0.9, 1.1]))
        run = engine.run()

        self.assertEqual(run.status, AutoCalibrationRunStatus.COMPLETED)
        self.assertEqual(run.n_evaluations, 3)

    def test_every_evaluation_creates_a_normal_calibration_session(self):

        engine = self._engine(_FixedSequenceStrategy([0.7, 0.9]))
        run = engine.run()

        for session_id in run.session_ids:

            session = self.studio.get_session(session_id)

            self.assertIsInstance(session, CalibrationSession)
            self.assertEqual(session.status, SessionStatus.COMPLETED)
            self.assertIn(session, self.project.sessions)

    def test_sessions_are_reproducible_exactly_like_a_manually_run_one(self):

        engine = self._engine(_FixedSequenceStrategy([0.7]))
        run = engine.run()

        session = self.studio.get_session(run.session_ids[0])
        self.assertIs(session.reproducible, True)

    def test_benchmark_calibration_history_is_updated_for_every_evaluation(self):

        objective = self._published_value_objective()
        engine = self._engine(_FixedSequenceStrategy([0.7, 0.9]), objective=objective)
        run = engine.run()

        benchmark = self.studio.benchmarks.get(objective.benchmark_id)
        self.assertEqual(set(benchmark.calibration_history), set(run.session_ids))

    def test_falls_back_to_parameter_sweep_when_the_objective_is_not_benchmark_anchored(self):

        objective = _ScriptedNonBenchmarkObjective()
        self.assertIsNone(getattr(objective, "benchmark_id", None))

        engine = self._engine(_FixedSequenceStrategy([0.7, 0.9]), objective=objective)
        run = engine.run()

        self.assertIsNone(engine.benchmark_id)
        self.assertEqual(run.n_evaluations, 2)
        self.assertEqual(run.status, AutoCalibrationRunStatus.COMPLETED)

        for session_id in run.session_ids:
            self.assertIsNone(self.studio.get_session(session_id).benchmark_id)


class BudgetAndExhaustionTests(EngineTestCase):

    def test_engine_stops_at_the_budget_even_if_the_strategy_has_more_proposals_queued(self):

        engine = self._engine(_FixedSequenceStrategy([0.7, 0.8, 0.9, 1.0, 1.1]), max_evaluations=2)
        run = engine.run()

        self.assertEqual(run.n_evaluations, 2)
        self.assertEqual(run.status, AutoCalibrationRunStatus.COMPLETED)

    def test_engine_stops_early_when_the_strategy_signals_exhaustion(self):

        engine = self._engine(_FixedSequenceStrategy([0.7]), max_evaluations=10)
        run = engine.run()

        self.assertEqual(run.n_evaluations, 1)
        self.assertEqual(run.status, AutoCalibrationRunStatus.COMPLETED)


class BestTrackingTests(EngineTestCase):

    def test_best_session_id_tracks_the_lowest_distance_from_the_published_value(self):

        objective = self._published_value_objective()
        engine = self._engine(_FixedSequenceStrategy([0.6, 0.9, 1.4]), objective=objective, max_evaluations=3)
        run = engine.run()

        expected_best = min(
            run.session_ids, key=lambda sid: objective.score(self.studio.get_session(sid)),
        )

        self.assertEqual(run.best_session_id, expected_best)


class _StrategyThatRaises(AutoCalibrationStrategy):

    # A bug IN THE STRATEGY ITSELF (not a simulation failure) -- the
    # one thing that should genuinely fail an AutoCalibrationRun, since
    # it happens outside CalibrationStudio's own already-forgiving
    # "a candidate/scenario combination failing to simulate is a
    # normal, expected experiment outcome" handling.

    def propose(self, *, search_space, objective, history):

        raise RuntimeError("deliberately broken strategy")

    def describe(self):

        return "_StrategyThatRaises -- test fixture only"


class FailureHandlingTests(EngineTestCase):

    # CalibrationStudio.run_published_benchmark()/run_parameter_sweep()
    # already treat a candidate that fails to simulate as a normal,
    # expected outcome (session.status == FAILED, no exception raised --
    # see calibration_studio/studio.py's own docstring). AutoCalibrationEngine
    # deliberately mirrors that same philosophy one layer up: a per-
    # candidate simulation failure does NOT abort the whole search --
    # it is simply recorded (score=None, never becomes best) and the
    # search continues up to its budget. Only a failure that happens
    # OUTSIDE that already-forgiving handling (a bug in the strategy or
    # objective itself, or an unregistered benchmark) should fail the
    # RUN -- see StrategyLevelFailureTests below.

    def test_a_failing_candidate_does_not_abort_the_search(self):

        engine = self._engine(_AlwaysFailingStrategy(), max_evaluations=3)
        run = engine.run()

        self.assertEqual(run.status, AutoCalibrationRunStatus.COMPLETED)
        self.assertEqual(run.n_evaluations, 3)

    def test_a_failing_candidate_still_produces_a_real_failed_calibration_session(self):

        engine = self._engine(_AlwaysFailingStrategy(), max_evaluations=3)
        run = engine.run()

        for session_id in run.session_ids:

            session = self.studio.get_session(session_id)
            self.assertEqual(session.status, SessionStatus.FAILED)
            self.assertIn("deliberately broken", session.failure_reason)

    def test_a_failing_candidate_never_becomes_the_best_evaluation(self):

        engine = self._engine(_AlwaysFailingStrategy(), max_evaluations=2)
        run = engine.run()

        self.assertIsNone(run.best_session_id)
        self.assertIsNone(run.best_score)

    def test_engine_run_never_raises_out_even_when_every_candidate_is_broken(self):

        engine = self._engine(_AlwaysFailingStrategy(), max_evaluations=1)

        try:
            run = engine.run()
        except Exception as exc:  # noqa: BLE001 -- explicitly asserting this must NOT happen
            self.fail(f"engine.run() raised {exc!r} instead of returning a run")

        self.assertEqual(run.status, AutoCalibrationRunStatus.COMPLETED)


class StrategyLevelFailureTests(EngineTestCase):

    def test_a_broken_strategy_marks_the_run_failed_with_zero_evaluations(self):

        engine = self._engine(_StrategyThatRaises(), max_evaluations=3)
        run = engine.run()

        self.assertEqual(run.status, AutoCalibrationRunStatus.FAILED)
        self.assertEqual(run.n_evaluations, 0)
        self.assertIn("deliberately broken strategy", run.failure_reason)

    def test_engine_run_never_raises_out_even_when_the_strategy_itself_is_broken(self):

        engine = self._engine(_StrategyThatRaises(), max_evaluations=1)

        try:
            run = engine.run()
        except Exception as exc:  # noqa: BLE001 -- explicitly asserting this must NOT happen
            self.fail(f"engine.run() raised {exc!r} instead of returning a FAILED run")

        self.assertEqual(run.status, AutoCalibrationRunStatus.FAILED)


class BackwardCompatibilityTests(EngineTestCase):

    # Proves automatic_calibration changes nothing about the existing,
    # manual, human-driven Calibration Studio workflow -- the same
    # studio/project used by the engine above still behaves exactly
    # like Phase 4's own tests expect when used directly.

    def test_a_manually_run_session_is_unaffected_by_the_engine_having_run_first(self):

        engine = self._engine(_FixedSequenceStrategy([0.7]))
        engine.run()

        benchmark = _make_benchmark(title="Manual benchmark")
        self.studio.benchmarks.register(benchmark)

        manual_candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
        manual_session = self.studio.run_published_benchmark(
            project=self.project, benchmark_id=benchmark.benchmark_id, candidate=manual_candidate,
            definition=self.definition, definition_id=DEFINITION_ID, master_seed=MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )

        self.assertEqual(manual_session.status, SessionStatus.COMPLETED)
        self.assertIn(manual_session, self.project.sessions)


class ReportingAndDashboardCompatibilityTests(EngineTestCase):

    def test_session_report_generates_for_an_auto_created_session(self):

        engine = self._engine(_FixedSequenceStrategy([0.7]))
        run = engine.run()

        report = self.studio.generate_session_report(run.session_ids[0])

        self.assertIn("Adult_Default.walking_speed", report)

    def test_validation_dashboard_generates_with_auto_created_sessions_present(self):

        engine = self._engine(_FixedSequenceStrategy([0.7, 0.9]))
        engine.run()

        dashboard = self.studio.generate_validation_dashboard()

        self.assertEqual(dashboard.total_benchmarks, 1)


class ReplayCompatibilityTests(EngineTestCase):

    def test_an_auto_created_sessions_replay_artifacts_can_be_recorded(self):

        # record_session_replay() requires a LIVE candidate (see
        # calibration_studio/replay_integration.py) -- engine.run()
        # produces exactly that on every session it creates, no
        # different from a manually-run one.
        engine = self._engine(_FixedSequenceStrategy([0.7]))
        run = engine.run()

        session = self.studio.get_session(run.session_ids[0])

        batch = run_batch_pipeline(self.definition, DEFINITION_ID, self.building, MASTER_SEED, 2)
        scenario = batch.scenarios[0]

        with tempfile.TemporaryDirectory() as tmp:

            record_session_replay(session, scenario, self.building, Path(tmp), arm="candidate", dt=1.0)

            self.assertIsNotNone(session.replay_output_dir)
            self.assertTrue((Path(tmp) / "building.syn").exists())


class PersistenceCompatibilityTests(EngineTestCase):

    def test_sessions_created_by_the_engine_persist_exactly_like_manual_ones(self):

        with tempfile.TemporaryDirectory() as tmp:

            studio = CalibrationStudio(storage_root=tmp)
            project = studio.create_project(name="Persistence Compat")
            benchmark = _make_benchmark()
            studio.benchmarks.register(benchmark)
            objective = PublishedValueObjective(
                benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
            )

            engine = AutoCalibrationEngine(
                studio=studio, project=project, building=self.building, definition=self.definition,
                definition_id=DEFINITION_ID, master_seed=MASTER_SEED, n_scenarios=2, dt=1.0,
                search_space=_search_space(), objective=objective, strategy=_FixedSequenceStrategy([0.7]),
                budget=AutoCalibrationBudget(max_evaluations=1),
            )
            run = engine.run()

            session = studio.get_session(run.session_ids[0])
            studio.save_session(session)
            reloaded = studio.load_session(session.session_id)

            self.assertEqual(reloaded.status, SessionStatus.COMPLETED)
            self.assertEqual(reloaded.candidate_snapshot["name"], "Adult_Default.walking_speed")


if __name__ == "__main__":
    unittest.main()
