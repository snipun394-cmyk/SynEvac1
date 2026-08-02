from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from scenario_pipeline import run_batch_pipeline

from research_framework.statistics import (
    ConfidenceInterval, EffectSize, PairedComparisonResult,
    confidence_interval, effect_size_cohens_d, paired_comparison,
)

from calibration_benchmark.candidates import ParameterCandidate
from calibration_benchmark.metrics import METRIC_FIELDS, METRIC_LABELS, MetricSample, extract_metrics
from calibration_benchmark.optional_metrics import AdditionalMetric
from calibration_benchmark.simulation_seam import run_with_overrides


# =====================================================
# Calibration Benchmark Framework V1 -- Current Parameter -> Candidate
# Dataset-derived Parameter -> identical scenario campaign -> compare
# results -> statistical comparison, exactly the pipeline this
# milestone's own brief specifies. "Identical scenario campaign" is
# implemented as a TRUE paired design: run_batch_pipeline() generates
# N scenarios ONCE, and each one is then run twice -- once under the
# candidate's baseline_*() objects, once under its candidate_*()
# objects -- so every one of the five headline metrics is compared
# scenario-by-scenario (the same seed under both arms), not just
# distribution-by-distribution. This is what makes
# research_framework.statistics.paired_comparison() (a paired t-test,
# already built and already used elsewhere in this codebase) the
# statistically correct tool here, rather than an unpaired test.
#
# Nothing here ever calls DEFAULT_PROFILE_REGISTRY/DefaultCapacityModel/
# DefaultCongestionModel/DensityThresholds with a mutating operation --
# baseline_*()/candidate_*() only ever return freshly-constructed
# objects (see candidates.py's own docstring). A CalibrationBenchmarkResult
# is a report to read, never something this module applies back onto
# SynEvac's own defaults.
# =====================================================


@dataclass(frozen=True)
class MetricComparison:

    metric_name: str
    metric_label: str
    n_pairs: int
    baseline_mean: Optional[float]
    candidate_mean: Optional[float]
    paired: PairedComparisonResult
    effect_size: EffectSize
    baseline_ci: ConfidenceInterval
    candidate_ci: ConfidenceInterval

    def to_dict(self) -> dict:

        return {
            "metric_name": self.metric_name,
            "metric_label": self.metric_label,
            "n_pairs": self.n_pairs,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "paired": self.paired.to_dict(),
            "effect_size": self.effect_size.to_dict(),
            "baseline_ci": self.baseline_ci.to_dict(),
            "candidate_ci": self.candidate_ci.to_dict(),
        }


@dataclass(frozen=True)
class CalibrationBenchmarkResult:

    candidate: ParameterCandidate
    n_scenarios_requested: int
    n_completed_pairs: int
    baseline_samples: Tuple[MetricSample, ...]
    candidate_samples: Tuple[MetricSample, ...]
    comparisons: Dict[str, MetricComparison] = field(default_factory=dict)
    additional_comparisons: Dict[str, MetricComparison] = field(default_factory=dict)

    def to_dict(self) -> dict:

        return {
            "candidate": self.candidate.describe(),
            "n_scenarios_requested": self.n_scenarios_requested,
            "n_completed_pairs": self.n_completed_pairs,
            "baseline_samples": [s.to_dict() for s in self.baseline_samples],
            "candidate_samples": [s.to_dict() for s in self.candidate_samples],
            "comparisons": {name: c.to_dict() for name, c in self.comparisons.items()},
            "additional_comparisons": {name: c.to_dict() for name, c in self.additional_comparisons.items()},
        }


# =====================================================


def _paired_non_none(baseline_values: Sequence, candidate_values: Sequence) -> Tuple[List[float], List[float]]:

    # paired_comparison() requires equal-length, entry-by-entry
    # corresponding sequences -- a scenario where either arm produced
    # None for this particular metric (e.g. no exits at all, so
    # exit_utilization_balance is undefined) is dropped from BOTH sides
    # together, never filled in, so the pairing across the two
    # remaining lists stays scenario-for-scenario correct.

    paired_a, paired_b = [], []

    for a, b in zip(baseline_values, candidate_values):

        if a is not None and b is not None:
            paired_a.append(float(a))
            paired_b.append(float(b))

    return paired_a, paired_b


def _compare(metric_name: str, label: str, baseline_values: Sequence, candidate_values: Sequence) -> MetricComparison:

    paired_a, paired_b = _paired_non_none(baseline_values, candidate_values)
    n = len(paired_a)

    return MetricComparison(
        metric_name=metric_name,
        metric_label=label,
        n_pairs=n,
        baseline_mean=(sum(paired_a) / n) if n else None,
        candidate_mean=(sum(paired_b) / n) if n else None,
        paired=(
            paired_comparison(paired_a, paired_b) if n >= 2
            else PairedComparisonResult(t_statistic=None, p_value=None, mean_difference=None, n=n)
        ),
        effect_size=effect_size_cohens_d(paired_b, paired_a),
        baseline_ci=confidence_interval(paired_a),
        candidate_ci=confidence_interval(paired_b),
    )


# =====================================================


def run_calibration_benchmark(
    candidate: ParameterCandidate, building, definition, definition_id: str,
    master_seed: int, n_scenarios: int, dt: float = 5.0,
    additional_metrics: Sequence[AdditionalMetric] = (),
) -> CalibrationBenchmarkResult:

    batch = run_batch_pipeline(definition, definition_id, building, master_seed, n_scenarios)

    baseline_samples: List[MetricSample] = []
    candidate_samples: List[MetricSample] = []

    additional_baseline: Dict[str, List[Optional[float]]] = {m.name: [] for m in additional_metrics}
    additional_candidate: Dict[str, List[Optional[float]]] = {m.name: [] for m in additional_metrics}

    for scenario in batch.scenarios:

        scenario_id = scenario.metadata.scenario_id

        baseline_capacity_model = candidate.baseline_capacity_model()
        baseline_movement, baseline_gt, baseline_building = run_with_overrides(
            scenario, building,
            registry=candidate.baseline_registry(),
            capacity_model=baseline_capacity_model,
            congestion_model=candidate.baseline_congestion_model(),
            dt=dt,
            use_flow_regions=candidate.baseline_use_flow_regions(),
        )

        candidate_capacity_model = candidate.candidate_capacity_model()
        candidate_movement, candidate_gt, candidate_building = run_with_overrides(
            scenario, building,
            registry=candidate.candidate_registry(),
            capacity_model=candidate_capacity_model,
            congestion_model=candidate.candidate_congestion_model(),
            dt=dt,
            use_flow_regions=candidate.candidate_use_flow_regions(),
        )

        baseline_sample = extract_metrics(
            scenario_id, baseline_gt, baseline_movement, baseline_building, baseline_capacity_model,
        )
        candidate_sample = extract_metrics(
            scenario_id, candidate_gt, candidate_movement, candidate_building, candidate_capacity_model,
        )

        baseline_samples.append(baseline_sample)
        candidate_samples.append(candidate_sample)

        for metric in additional_metrics:

            additional_baseline[metric.name].append(
                metric.compute(
                    baseline_gt, baseline_movement, baseline_building,
                    baseline_sample.peak_occupancy_ratio, candidate.baseline_density_thresholds(),
                )
            )
            additional_candidate[metric.name].append(
                metric.compute(
                    candidate_gt, candidate_movement, candidate_building,
                    candidate_sample.peak_occupancy_ratio, candidate.candidate_density_thresholds(),
                )
            )

    comparisons = {
        field_name: _compare(
            field_name, METRIC_LABELS[field_name],
            [getattr(s, field_name) for s in baseline_samples],
            [getattr(s, field_name) for s in candidate_samples],
        )
        for field_name in METRIC_FIELDS
    }

    additional_comparisons = {
        metric.name: _compare(metric.name, metric.name, additional_baseline[metric.name], additional_candidate[metric.name])
        for metric in additional_metrics
    }

    return CalibrationBenchmarkResult(
        candidate=candidate,
        n_scenarios_requested=n_scenarios,
        n_completed_pairs=len(baseline_samples),
        baseline_samples=tuple(baseline_samples),
        candidate_samples=tuple(candidate_samples),
        comparisons=comparisons,
        additional_comparisons=additional_comparisons,
    )
