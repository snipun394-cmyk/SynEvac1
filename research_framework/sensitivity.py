import os

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from models.building import Building

from scenario_definition.definition import ScenarioDefinition

from rl_training import EvaluationReport

from research_framework.experiment_spec import ExperimentSpec, ParameterOverride, make_arm
from research_framework.runner import ArmRunResult, ExperimentRunResult, run_experiment
from research_framework.statistics import SensitivityResult, feature_sensitivity


# =====================================================
# Phase 5 -- Sensitivity Analysis. One "sweep" = one arm per value of a
# single swept parameter, holding everything else at the experiment's
# base ScenarioDefinition -- the correct controlled-experiment shape
# for isolating one parameter's own effect (varying two parameters at
# once in the same arm set would confound their effects, exactly what
# a sweep exists to avoid). Sensitivity itself is
# research_framework.statistics.feature_sensitivity() (Pearson r +
# OLS slope via scipy.stats.linregress) applied to (swept value,
# per-arm mean metric) pairs -- one point per arm, not per scenario
# (a single scenario's own outcome is one noisy draw; the arm's mean
# is this sweep's honest y-value for that x).
# =====================================================


def _mean(values: Sequence[float]) -> Optional[float]:

    values = [float(v) for v in values]

    return sum(values) / len(values) if values else None


# =====================================================
# Metric extractors -- Evacuation Time / Congestion / Trapped Occupants
# read straight off each arm's own exported simulation_outcomes.csv
# (via ArmRunResult.load_dataset(), the same training_dataset.
# load_campaign() every other consumer of a campaign directory already
# uses). AI Accuracy / RL Reward need a caller-supplied extractor
# (ai_metric_extractor()/rl_reward_extractor() below) since neither is
# produced automatically by run_experiment() -- AI training is opt-in
# (ai_experiment_configs) and RL training/evaluation is a separate,
# explicit call (research_framework.runner.train_and_evaluate_rl_for_arm)
# not tied to any one sweep.
# =====================================================


def _dataset_metric_extractor(field_name: str) -> Callable[[ArmRunResult], Sequence[float]]:

    def _extract(arm_result: ArmRunResult) -> Sequence[float]:

        dataset = arm_result.load_dataset()
        values = []

        for sample in dataset.samples:

            value = sample.simulation_outcome.get(field_name)

            if value is not None and value != "":
                values.append(float(value))

        return values

    return _extract


DEFAULT_METRIC_EXTRACTORS: Dict[str, Callable[[ArmRunResult], Sequence[float]]] = {
    "evacuation_time": lambda arm_result: list(arm_result.evacuation_times()),
    "congestion": _dataset_metric_extractor("maximum_congestion"),
    "trapped_occupants": _dataset_metric_extractor("people_trapped"),
}


def ai_metric_extractor(config_name: str, metric_path: Sequence[str]) -> Callable[[ArmRunResult], Sequence[float]]:

    def _extract(arm_result: ArmRunResult) -> Sequence[float]:

        result = arm_result.ai_results.get(config_name)

        if result is None:
            return []

        value: Any = result.metrics

        for key in metric_path:

            if not isinstance(value, dict) or key not in value:
                return []

            value = value[key]

        try:
            return [float(value)]
        except (TypeError, ValueError):
            return []

    return _extract


def rl_reward_extractor(
    evaluation_reports_by_arm: Mapping[str, EvaluationReport],
) -> Callable[[ArmRunResult], Sequence[float]]:

    def _extract(arm_result: ArmRunResult) -> Sequence[float]:

        report = evaluation_reports_by_arm.get(arm_result.arm.name)

        if report is None:
            return []

        return [episode.total_reward for episode in report.episodes]

    return _extract


# =====================================================


@dataclass(frozen=True)
class SweepResult:

    parameter_name: str
    metric_name: str
    values: Tuple[float, ...]
    metric_values: Tuple[Optional[float], ...]
    sensitivity: SensitivityResult

    def to_dict(self):

        return {
            "parameter_name": self.parameter_name, "metric_name": self.metric_name,
            "values": list(self.values), "metric_values": list(self.metric_values),
            "sensitivity": self.sensitivity.to_dict(),
        }


@dataclass(frozen=True)
class SensitivityAnalysisResult:

    experiment_results: Mapping[str, ExperimentRunResult] = field(default_factory=dict)
    sweeps: Tuple[SweepResult, ...] = field(default_factory=tuple)

    def ranked(self, metric_name: Optional[str] = None) -> Tuple[SweepResult, ...]:

        relevant = [
            sweep for sweep in self.sweeps if metric_name is None or sweep.metric_name == metric_name
        ]

        return tuple(
            sorted(
                relevant,
                key=lambda sweep: abs(sweep.sensitivity.pearson_r) if sweep.sensitivity.pearson_r is not None else -1.0,
                reverse=True,
            )
        )


# =====================================================


def run_parameter_sweep(
    base_definition: ScenarioDefinition, base_definition_id: str, base_building: Building, *,
    parameter_name: str, values: Sequence[Any], override_factory: Callable[[Any], ParameterOverride],
    samples_per_arm: int, master_seed: int, output_root: str, dt: float = 5.0,
    metric_extractors: Optional[Mapping[str, Callable[[ArmRunResult], Sequence[float]]]] = None,
) -> Tuple[ExperimentRunResult, Tuple[SweepResult, ...]]:

    metric_extractors = metric_extractors if metric_extractors is not None else DEFAULT_METRIC_EXTRACTORS

    arms = tuple(
        make_arm(f"{parameter_name}={value}", override_factory(value)) for value in values
    )

    spec = ExperimentSpec(
        name=f"sensitivity_{parameter_name}", base_definition=base_definition,
        base_definition_id=base_definition_id, base_building=base_building, arms=arms,
        samples_per_arm=samples_per_arm, master_seed=master_seed, output_root=output_root, dt=dt,
    )

    experiment_result = run_experiment(spec)

    numeric_values = [float(v) for v in values]
    sweeps = []

    for metric_name, extractor in metric_extractors.items():

        metric_means = [
            _mean(extractor(experiment_result.arm_result(arm.name))) for arm in arms
        ]

        paired = [(x, y) for x, y in zip(numeric_values, metric_means) if y is not None]
        xs = tuple(x for x, _ in paired)
        ys = tuple(y for _, y in paired)

        sensitivity = (
            feature_sensitivity(xs, ys) if len(xs) >= 2
            else SensitivityResult(pearson_r=None, p_value=None, slope=None, n=len(xs))
        )

        sweeps.append(
            SweepResult(
                parameter_name=parameter_name, metric_name=metric_name,
                values=tuple(numeric_values), metric_values=tuple(metric_means), sensitivity=sensitivity,
            )
        )

    return experiment_result, tuple(sweeps)


# =====================================================


def run_sensitivity_analysis(
    base_definition: ScenarioDefinition, base_definition_id: str, base_building: Building, *,
    sweeps: Mapping[str, Tuple[Sequence[Any], Callable[[Any], ParameterOverride]]],
    samples_per_arm: int, master_seed: int, output_root: str, dt: float = 5.0,
    metric_extractors: Optional[Mapping[str, Callable[[ArmRunResult], Sequence[float]]]] = None,
) -> SensitivityAnalysisResult:

    experiment_results: Dict[str, ExperimentRunResult] = {}
    all_sweeps = []

    for parameter_name, (values, override_factory) in sweeps.items():

        sweep_output_root = os.path.join(output_root, parameter_name)

        experiment_result, sweep_results = run_parameter_sweep(
            base_definition, base_definition_id, base_building,
            parameter_name=parameter_name, values=values, override_factory=override_factory,
            samples_per_arm=samples_per_arm, master_seed=master_seed, output_root=sweep_output_root,
            dt=dt, metric_extractors=metric_extractors,
        )

        experiment_results[parameter_name] = experiment_result
        all_sweeps.extend(sweep_results)

    return SensitivityAnalysisResult(experiment_results=experiment_results, sweeps=tuple(all_sweeps))
