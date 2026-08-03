import unittest

from calibration_benchmark import WalkingSpeedCandidate, run_calibration_benchmark

from calibration_studio.benchmark import BenchmarkType, PublishedBenchmark, PublishedValue
from calibration_studio.session import CalibrationSession

from automatic_calibration.objectives import CalibrationObjective, ObjectiveDirection, PublishedValueObjective

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


def _make_benchmark(**overrides):

    defaults = dict(
        title="Objective Test Fixture",
        source_citation="internal test fixture",
        dataset="synthetic",
        benchmark_type=BenchmarkType.DATASET_VALIDATION,
        published_values={"evacuation_time_s": PublishedValue(value=700.0, unit="s")},
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


def _completed_session(n_scenarios=2):

    candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
    result = run_calibration_benchmark(
        candidate, make_building(), make_definition(), DEFINITION_ID, MASTER_SEED, n_scenarios=n_scenarios, dt=1.0,
    )

    session = CalibrationSession(candidate=candidate, master_seed=MASTER_SEED)
    session.mark_running(n_scenarios_total=n_scenarios)
    session.mark_completed(result)

    return session


class CalibrationObjectiveInterfaceTests(unittest.TestCase):

    def test_default_direction_is_minimize(self):

        self.assertEqual(CalibrationObjective.direction, ObjectiveDirection.MINIMIZE)

    def test_score_is_not_implemented_on_the_base_class(self):

        with self.assertRaises(NotImplementedError):
            CalibrationObjective().score(_completed_session())

    def test_describe_is_not_implemented_on_the_base_class(self):

        with self.assertRaises(NotImplementedError):
            CalibrationObjective().describe()


class PublishedValueObjectiveConstructionTests(unittest.TestCase):

    def test_unknown_published_metric_name_raises(self):

        benchmark = _make_benchmark()

        with self.assertRaises(ValueError):
            PublishedValueObjective(
                benchmark, published_metric_name="not_a_real_metric", result_metric_name="evacuation_time",
            )

    def test_target_value_is_read_from_the_benchmark(self):

        benchmark = _make_benchmark()
        objective = PublishedValueObjective(
            benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
        )

        self.assertEqual(objective.target_value, 700.0)

    def test_benchmark_id_is_captured(self):

        benchmark = _make_benchmark()
        objective = PublishedValueObjective(
            benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
        )

        self.assertEqual(objective.benchmark_id, benchmark.benchmark_id)

    def test_direction_is_always_minimize(self):

        benchmark = _make_benchmark()
        objective = PublishedValueObjective(
            benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
        )

        self.assertEqual(objective.direction, ObjectiveDirection.MINIMIZE)


class PublishedValueObjectiveScoringTests(unittest.TestCase):

    # Deliberately uses TWO different metric name vocabularies
    # (published_metric_name="evacuation_time_s" vs
    # result_metric_name="evacuation_time") -- proving the objective
    # never assumes a benchmark's own citation key happens to match
    # calibration_benchmark.metrics.METRIC_FIELDS' own field name.

    def test_score_is_the_absolute_distance_from_the_target_value(self):

        session = _completed_session()
        observed_mean = session.result.comparisons["evacuation_time"].candidate_mean

        benchmark = _make_benchmark(
            published_values={"evacuation_time_s": PublishedValue(value=observed_mean + 50.0, unit="s")},
        )
        objective = PublishedValueObjective(
            benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
        )

        self.assertAlmostEqual(objective.score(session), 50.0, places=6)

    def test_exact_match_scores_zero(self):

        session = _completed_session()
        observed_mean = session.result.comparisons["evacuation_time"].candidate_mean

        benchmark = _make_benchmark(
            published_values={"evacuation_time_s": PublishedValue(value=observed_mean, unit="s")},
        )
        objective = PublishedValueObjective(
            benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
        )

        self.assertAlmostEqual(objective.score(session), 0.0, places=6)

    def test_score_is_none_when_the_session_has_no_result(self):

        benchmark = _make_benchmark()
        objective = PublishedValueObjective(
            benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
        )

        pending_session = CalibrationSession(candidate=WalkingSpeedCandidate("Adult_Default", 0.65, "t", "t"))

        self.assertIsNone(objective.score(pending_session))

    def test_score_is_none_when_the_result_metric_name_does_not_exist(self):

        session = _completed_session()

        benchmark = _make_benchmark()
        objective = PublishedValueObjective(
            benchmark, published_metric_name="evacuation_time_s", result_metric_name="not_a_real_result_metric",
        )

        self.assertIsNone(objective.score(session))

    def test_describe_mentions_both_metric_names_and_the_target_value(self):

        benchmark = _make_benchmark()
        objective = PublishedValueObjective(
            benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
        )

        description = objective.describe()

        self.assertIn("evacuation_time_s", description)
        self.assertIn("evacuation_time", description)
        self.assertIn("700.0", description)


if __name__ == "__main__":
    unittest.main()
