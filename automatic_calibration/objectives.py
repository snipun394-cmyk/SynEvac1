from enum import Enum
from typing import Optional

from calibration_studio.benchmark import PublishedBenchmark
from calibration_studio.session import CalibrationSession


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture.
#
# CalibrationObjective exists precisely because calibration_benchmark.
# recommend() deliberately never collapses several metrics into one
# score ("a candidate that shortens evacuation time but also increases
# queue length is a genuine engineering trade-off a recommendation must
# surface, not average away" -- calibration_benchmark/recommendation.py's
# own docstring). An adaptive search strategy genuinely needs SOME
# scalar to rank candidates by; this module supplies that scalar
# LOCALLY, on top of calibration_benchmark's own output, never inside
# it and never by reaching into recommendation.py's private
# _LOWER_IS_BETTER/_HIGHER_IS_BETTER sets. Every session an objective
# scores still carries calibration_benchmark's own full, untouched,
# per-metric AdoptionRecommendation -- score() is read-only sugar for
# search ranking, never a replacement for it.
# =====================================================


class ObjectiveDirection(Enum):

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class CalibrationObjective:

    direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE

    def score(self, session: CalibrationSession) -> Optional[float]:

        # None means "not scoreable" (a failed session, a metric this
        # session's result never produced a value for) -- never a
        # fabricated number.
        raise NotImplementedError

    def describe(self) -> str:

        raise NotImplementedError


class PublishedValueObjective(CalibrationObjective):

    # Anchors a search to real ground truth: minimizes the distance
    # between a session's own simulated metric mean and a
    # PublishedBenchmark's published value for that same real-world
    # quantity.
    #
    # Deliberately TWO separate metric names, not one -- published_
    # metric_name indexes PublishedBenchmark.published_values (an
    # author-chosen citation key, e.g. "evacuation_time_s"), while
    # result_metric_name indexes CalibrationBenchmarkResult.comparisons
    # (one of calibration_benchmark.metrics.METRIC_FIELDS, e.g.
    # "evacuation_time" -- no trailing unit, a different vocabulary
    # entirely). Assuming these two names always match would be a real
    # bug waiting to happen the first time a benchmark author picks a
    # citation key that doesn't happen to coincide with calibration_
    # benchmark's own field name.

    direction = ObjectiveDirection.MINIMIZE

    def __init__(self, benchmark: PublishedBenchmark, *, published_metric_name: str, result_metric_name: str):

        if published_metric_name not in benchmark.published_values:

            raise ValueError(
                f"Benchmark {benchmark.benchmark_id!r} has no published value named "
                f"{published_metric_name!r} -- choose one of "
                f"{sorted(benchmark.published_values)!r}.",
            )

        self.benchmark_id = benchmark.benchmark_id
        self.published_metric_name = published_metric_name
        self.result_metric_name = result_metric_name
        self.target_value = benchmark.published_values[published_metric_name].value

    def score(self, session: CalibrationSession) -> Optional[float]:

        if session.result is None:
            return None

        comparison = session.result.comparisons.get(self.result_metric_name)

        if comparison is None or comparison.candidate_mean is None:
            return None

        return abs(comparison.candidate_mean - self.target_value)

    def describe(self) -> str:

        return (
            f"PublishedValueObjective(benchmark_id={self.benchmark_id!r}, "
            f"published_metric_name={self.published_metric_name!r}, "
            f"result_metric_name={self.result_metric_name!r}, "
            f"target_value={self.target_value!r}) -- minimizes "
            f"|candidate_mean({self.result_metric_name}) - target_value|"
        )
