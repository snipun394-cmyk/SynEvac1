import datetime

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ai_training.dataset import load_campaign_dataset

from campaign_analytics import CampaignAnalysis, analyze_campaign

from training_dataset.statistics import compute_statistics

from rl_training.reward_function import RewardConfig
from rl_training.trainer import TrainerConfig

from validation_framework.ai_augmented_policy import AIPredictionBundle, fit_ai_prediction_bundle
from validation_framework.figures import generate_all_figures
from validation_framework.prediction_validator import (
    PredictionValidationResult,
    validate_bottleneck_model,
    validate_evacuation_time_model,
    validate_exit_usage_model,
    validate_firefighter_intelligence_consistency,
    validate_recommendation_accuracy,
    validate_smoke_prediction_model,
)
from validation_framework.recommendation_validator import RecommendationValidationResult, validate_recommendations
from validation_framework.scenario_comparator import PolicyComparisonResult, run_policy_comparison


# =====================================================
# Phase 1/5 -- orchestration. Ties every other module in this package
# together into one ValidationBenchmarkResult. Every sub-phase is
# optional and is honestly SKIPPED (recorded in `limitations`, never
# fabricated) when its required input isn't supplied -- the same
# defensive contract AdvisoryInputs/campaign_analytics already use
# throughout this codebase. campaign_dir drives Phase 2 (AI Evaluation)
# and dataset statistics; building/definition/definition_id drive
# Phase 3 (RL Evaluation) and Phase 4 (Recommendation Validation), since
# those need to run live simulations. The AI+RL arm additionally needs
# BOTH (a campaign_dir to fit prediction models from, and a building to
# evaluate policies against).
# =====================================================

_RECOMMENDATION_SEED_OFFSET = 9_000_000


@dataclass(frozen=True)
class ValidationBenchmarkResult:

    campaign_dir: Optional[str]
    dataset_statistics: Optional[Dict[str, Any]] = None
    campaign_analysis: Optional[CampaignAnalysis] = None
    prediction_results: Dict[str, PredictionValidationResult] = field(default_factory=dict)
    recommendation_accuracy: Optional[Dict[str, Any]] = None
    firefighter_monotonicity: Optional[Dict[str, Any]] = None
    policy_comparison: Optional[PolicyComparisonResult] = None
    recommendation_validation: Optional[RecommendationValidationResult] = None
    figure_paths: Dict[str, str] = field(default_factory=dict)
    limitations: Tuple[str, ...] = ()
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:

        return {
            "campaign_dir": self.campaign_dir,
            "dataset_statistics": self.dataset_statistics,
            "campaign_analysis": self.campaign_analysis.to_dict() if self.campaign_analysis is not None else None,
            "prediction_results": {
                name: result.to_dict() for name, result in self.prediction_results.items()
            },
            "recommendation_accuracy": self.recommendation_accuracy,
            "firefighter_monotonicity": self.firefighter_monotonicity,
            "policy_comparison": self.policy_comparison.to_dict() if self.policy_comparison is not None else None,
            "recommendation_validation": (
                self.recommendation_validation.to_dict() if self.recommendation_validation is not None else None
            ),
            "figure_paths": self.figure_paths,
            "limitations": list(self.limitations),
            "generated_at": self.generated_at,
        }


def run_full_validation(
    *,
    campaign_dir: Optional[str] = None,
    building: Any = None,
    definition: Any = None,
    definition_id: Optional[str] = None,
    master_seed: int = 0,
    rl_episode_count: int = 10,
    recommendation_scenario_count: int = 10,
    ai_augmented: bool = True,
    rl_total_timesteps: int = 2000,
    rl_trainer_config: Optional[TrainerConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    dt: float = 1.0,
    max_steps: int = 500,
    output_dir: str = "validation_framework/results",
) -> ValidationBenchmarkResult:

    limitations = []
    dataset_statistics = None
    campaign_analysis = None
    prediction_results: Dict[str, PredictionValidationResult] = {}
    recommendation_accuracy = None
    firefighter_monotonicity = None
    ai_bundle: Optional[AIPredictionBundle] = None

    if campaign_dir is not None:

        campaign_analysis = analyze_campaign(campaign_dir, strict=False)
        dataset_statistics = compute_statistics(campaign_analysis.dataset)

        training_dataset_obj = load_campaign_dataset(campaign_dir, strict=False)

        prediction_results, prediction_limitations = _run_prediction_validation(training_dataset_obj)
        limitations.extend(prediction_limitations)

        try:
            recommendation_accuracy = validate_recommendation_accuracy(training_dataset_obj)
        except Exception:
            limitations.append(
                "Recommendation accuracy validation skipped -- insufficient/invalid data in campaign_dir."
            )

        try:
            firefighter_monotonicity = validate_firefighter_intelligence_consistency(training_dataset_obj)
        except Exception:
            limitations.append(
                "Firefighter intelligence monotonicity check skipped -- insufficient data in campaign_dir."
            )

        if ai_augmented and building is not None:

            try:
                ai_bundle = fit_ai_prediction_bundle(training_dataset_obj, building, max_steps=max_steps, dt=dt)
            except Exception:
                limitations.append(
                    "AI+RL arm skipped -- could not fit the AI prediction bundle from campaign_dir."
                )

    else:

        limitations.append("AI Evaluation (Phase 2) skipped -- no campaign_dir supplied.")

        if ai_augmented:
            limitations.append("AI+RL arm skipped -- no campaign_dir supplied to fit prediction models.")

    live_simulation_ready = building is not None and definition is not None and definition_id is not None

    policy_comparison = None

    if live_simulation_ready:

        policy_comparison = run_policy_comparison(
            building, definition, definition_id, master_seed, rl_episode_count,
            ai_bundle=ai_bundle if ai_augmented else None, dt=dt, max_steps=max_steps,
            reward_config=reward_config, rl_total_timesteps=rl_total_timesteps,
            rl_trainer_config=rl_trainer_config,
        )

    else:

        limitations.append(
            "RL Evaluation (Phase 3) skipped -- building/definition/definition_id not supplied."
        )

    recommendation_validation = None

    if live_simulation_ready:

        recommendation_validation = validate_recommendations(
            building, definition, definition_id, master_seed + _RECOMMENDATION_SEED_OFFSET,
            recommendation_scenario_count, dt=max(dt, 1.0), max_steps=max_steps,
        )

    else:

        limitations.append(
            "Recommendation Validation (Phase 4) skipped -- building/definition/definition_id not supplied."
        )

    figure_paths = generate_all_figures(
        output_dir=output_dir,
        prediction_results=prediction_results,
        policy_comparison=policy_comparison,
        building=building,
        sample_timeline_rows=(
            recommendation_validation.sample_timeline_rows if recommendation_validation is not None else ()
        ),
        sample_changes=(
            recommendation_validation.sample_changes if recommendation_validation is not None else ()
        ),
    )

    return ValidationBenchmarkResult(
        campaign_dir=campaign_dir,
        dataset_statistics=dataset_statistics,
        campaign_analysis=campaign_analysis,
        prediction_results=prediction_results,
        recommendation_accuracy=recommendation_accuracy,
        firefighter_monotonicity=firefighter_monotonicity,
        policy_comparison=policy_comparison,
        recommendation_validation=recommendation_validation,
        figure_paths=figure_paths,
        limitations=tuple(limitations),
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def _run_prediction_validation(training_dataset_obj):

    results: Dict[str, PredictionValidationResult] = {}
    limitations = []

    validators = (
        ("evacuation_time", lambda: validate_evacuation_time_model(training_dataset_obj)),
        ("bottleneck_occurrence", lambda: validate_bottleneck_model(training_dataset_obj, target="occurrence")),
        ("exit_usage", lambda: validate_exit_usage_model(training_dataset_obj)),
        ("smoke_prediction", lambda: validate_smoke_prediction_model(training_dataset_obj)),
    )

    for name, run in validators:

        try:
            results[name] = run()
        except Exception:
            limitations.append(
                f"{name} model validation skipped -- insufficient/invalid data in campaign_dir."
            )

    return results, limitations
