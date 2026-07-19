from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from models.building import Building

from scenario_definition.definition import ScenarioDefinition
from scenario_definition.distributions import FixedValue, WeightedOptions

from rl_training import EvaluationReport, TrainerConfig, evaluate_no_intervention

from research_framework.experiment_spec import (
    Arm,
    ExperimentSpec,
    ParameterOverride,
    make_arm,
    override_firefighter_team_size,
    override_helping_probability,
)
from research_framework.runner import ArmRunResult, ExperimentRunResult, run_experiment, train_and_evaluate_rl_for_arm
from research_framework.statistics import EffectSize, PairedComparisonResult, effect_size_cohens_d, paired_comparison


# =====================================================
# Phase 4 -- Ablation Studies. "Automatically disable one capability at
# a time" -- each capability is a function of the experiment's own base
# ScenarioDefinition (not a static override) because disabling a
# population category means *removing* whatever weight that category
# already carries in this specific Definition's own behaviour_profile_
# distribution, not asserting a fixed replacement mix. "No RL" and "No
# perception" are handled separately below, outside the Arm/
# ScenarioDefinition mechanism entirely -- see each function's own
# docstring for why.
# =====================================================

CAPABILITY_NO_FIREFIGHTERS = "no_firefighters"
CAPABILITY_NO_HELPING = "no_helping"
CAPABILITY_NO_WHEELCHAIRS = "no_wheelchairs"
CAPABILITY_NO_ELDERLY = "no_elderly"

# "No perception": every campaign run through research_framework.runner
# (mirroring designer.campaign.campaign_worker exactly) already never
# configures a PerceptionProvider (perception_provider=None,
# unconditionally -- see SimulationRuntime's own construction in
# runner.run_scenario_artifacts()/campaign_worker._simulate(), and that
# worker's own module docstring: "perception_provider is deliberately
# never configured... no concrete PerceptionProvider composition exists
# anywhere in this codebase yet"). Every campaign this framework
# produces is therefore *already* running with perception disabled --
# there is no "with perception" baseline to ablate away from, so
# CAPABILITY_NO_PERCEPTION is deliberately not in DEFAULT_CAPABILITIES
# or CAPABILITY_BUILDERS: running it would silently compare the
# baseline arm against an identical copy of itself, a fabricated
# result, not an honest one. NO_PERCEPTION_NOTE is the plain-language
# explanation research_framework.report quotes instead of a bogus
# ablation entry.
CAPABILITY_NO_PERCEPTION = "no_perception"
NO_PERCEPTION_NOTE = (
    "Not run as an ablation: every campaign this framework produces already runs with "
    "perception disabled (no PerceptionProvider is wired into SimulationRuntime anywhere in "
    "this pipeline), so there is no 'with perception' baseline to compare against."
)

DEFAULT_CAPABILITIES = (
    CAPABILITY_NO_FIREFIGHTERS, CAPABILITY_NO_HELPING, CAPABILITY_NO_WHEELCHAIRS, CAPABILITY_NO_ELDERLY,
)


def _strip_profile_override(definition: ScenarioDefinition, profile_id: str, name: str) -> ParameterOverride:

    updated = {}

    for zone_id, distribution in definition.occupant.behaviour_profile_distribution.items():

        if isinstance(distribution, WeightedOptions):

            weights = dict(distribution.weights)
            weights.pop(profile_id, None)
            updated[zone_id] = WeightedOptions(weights) if weights else FixedValue("Adult_Default")

        elif isinstance(distribution, FixedValue) and distribution.value == profile_id:

            updated[zone_id] = FixedValue("Adult_Default")

        else:

            updated[zone_id] = distribution

    def _apply(target: ScenarioDefinition) -> ScenarioDefinition:

        return replace(target, occupant=replace(target.occupant, behaviour_profile_distribution=updated))

    return ParameterOverride(name=name, apply=_apply)


CAPABILITY_BUILDERS: Dict[str, Callable[[ScenarioDefinition], ParameterOverride]] = {
    CAPABILITY_NO_FIREFIGHTERS: lambda definition: override_firefighter_team_size(
        team_count=0, name=CAPABILITY_NO_FIREFIGHTERS,
    ),
    CAPABILITY_NO_HELPING: lambda definition: override_helping_probability(
        0.0, name=CAPABILITY_NO_HELPING,
    ),
    CAPABILITY_NO_WHEELCHAIRS: lambda definition: _strip_profile_override(
        definition, "Wheelchair_Default", CAPABILITY_NO_WHEELCHAIRS,
    ),
    CAPABILITY_NO_ELDERLY: lambda definition: _strip_profile_override(
        definition, "Elderly_Default", CAPABILITY_NO_ELDERLY,
    ),
}


def build_ablation_arms(
    base_definition: ScenarioDefinition, capabilities: Sequence[str] = DEFAULT_CAPABILITIES,
) -> Tuple[Arm, ...]:

    arms = [make_arm("full")]

    for capability in capabilities:

        if capability not in CAPABILITY_BUILDERS:
            raise ValueError(f"Unknown ablation capability {capability!r} -- choose one of {sorted(CAPABILITY_BUILDERS)}.")

        arms.append(make_arm(capability, CAPABILITY_BUILDERS[capability](base_definition)))

    return tuple(arms)


# =====================================================


@dataclass(frozen=True)
class AblationResult:

    capability: str
    metric_name: str
    baseline_values: Tuple[float, ...]
    ablated_values: Tuple[float, ...]
    effect_size: EffectSize
    paired: Optional[PairedComparisonResult] = None

    def to_dict(self):

        return {
            "capability": self.capability, "metric_name": self.metric_name,
            "baseline_values": list(self.baseline_values), "ablated_values": list(self.ablated_values),
            "effect_size": self.effect_size.to_dict(),
            "paired": self.paired.to_dict() if self.paired is not None else None,
        }


@dataclass(frozen=True)
class AblationStudyResult:

    experiment_result: ExperimentRunResult
    results: Tuple[AblationResult, ...] = field(default_factory=tuple)


_DEFAULT_METRIC_EXTRACTORS: Dict[str, Callable[[ArmRunResult], Sequence[float]]] = {
    "total_evacuation_time": lambda arm_result: arm_result.evacuation_times(),
}


def run_ablation_study(
    base_definition: ScenarioDefinition, base_definition_id: str, base_building: Building, *,
    samples_per_arm: int, master_seed: int, output_root: str, dt: float = 5.0,
    capabilities: Sequence[str] = DEFAULT_CAPABILITIES,
    metric_extractors: Optional[Mapping[str, Callable[[ArmRunResult], Sequence[float]]]] = None,
) -> AblationStudyResult:

    metric_extractors = metric_extractors if metric_extractors is not None else _DEFAULT_METRIC_EXTRACTORS

    arms = build_ablation_arms(base_definition, capabilities)

    spec = ExperimentSpec(
        name="ablation", base_definition=base_definition, base_definition_id=base_definition_id,
        base_building=base_building, arms=arms, samples_per_arm=samples_per_arm,
        master_seed=master_seed, output_root=output_root, dt=dt,
    )

    experiment_result = run_experiment(spec)
    baseline = experiment_result.arm_result("full")

    results = []

    for capability in capabilities:

        ablated = experiment_result.arm_result(capability)

        for metric_name, extractor in metric_extractors.items():

            baseline_values = tuple(extractor(baseline))
            ablated_values = tuple(extractor(ablated))

            effect = effect_size_cohens_d(ablated_values, baseline_values)

            paired = None

            if len(baseline_values) == len(ablated_values) and len(baseline_values) >= 2:

                try:
                    paired = paired_comparison(ablated_values, baseline_values)
                except ValueError:
                    paired = None

            results.append(
                AblationResult(
                    capability=capability, metric_name=metric_name,
                    baseline_values=baseline_values, ablated_values=ablated_values,
                    effect_size=effect, paired=paired,
                )
            )

    return AblationStudyResult(experiment_result=experiment_result, results=tuple(results))


# =====================================================
# "No RL" -- not a ScenarioDefinition ablation at all (there is no
# population/engineering knob that "turns RL off"); it is a comparison
# between rl_training.evaluator.evaluate_policy() (a trained RL policy
# actively intervening) and rl_training.evaluator.evaluate_no_intervention()
# (the same environment, no interventions issued) -- both already-
# existing, already-frozen evaluator entry points, reused verbatim.
# =====================================================


@dataclass(frozen=True)
class NoRLAblationResult:

    with_rl: EvaluationReport
    without_rl: EvaluationReport
    reward_effect_size: EffectSize
    evacuation_time_effect_size: EffectSize


def run_no_rl_ablation(
    spec: ExperimentSpec, arm: Arm, *,
    total_timesteps: int = 2000, eval_episode_count: int = 5,
    trainer_config: Optional[TrainerConfig] = None, max_steps: int = 200, dt: float = 1.0,
) -> NoRLAblationResult:

    from research_framework.runner import _derive_arm_seed
    from rl_training import SynEvacGymEnv

    with_rl = train_and_evaluate_rl_for_arm(
        spec, arm, total_timesteps=total_timesteps, eval_episode_count=eval_episode_count,
        trainer_config=trainer_config, max_steps=max_steps, dt=dt,
    )

    definition = arm.resolve_definition(spec.base_definition)
    building = spec.building_for(arm)
    arm_seed = _derive_arm_seed(spec.master_seed, arm.name)

    baseline_env = SynEvacGymEnv(
        building=building, definition=definition, definition_id=spec.base_definition_id,
        master_seed=arm_seed, dt=dt, max_steps=max_steps,
    )
    without_rl = evaluate_no_intervention(
        baseline_env, master_seed=arm_seed, count=eval_episode_count, label=f"{arm.name}_no_rl",
    )

    reward_effect = effect_size_cohens_d(
        [episode.total_reward for episode in with_rl.episodes],
        [episode.total_reward for episode in without_rl.episodes],
    )
    evacuation_time_effect = effect_size_cohens_d(
        [e.total_evacuation_time for e in with_rl.episodes if e.total_evacuation_time is not None],
        [e.total_evacuation_time for e in without_rl.episodes if e.total_evacuation_time is not None],
    )

    return NoRLAblationResult(
        with_rl=with_rl, without_rl=without_rl,
        reward_effect_size=reward_effect, evacuation_time_effect_size=evacuation_time_effect,
    )
