from dataclasses import dataclass
from typing import Any, Dict, Optional

from rl_training.environment import SynEvacGymEnv
from rl_training.evaluator import (
    EvaluationReport,
    evaluate_decision_policy,
    evaluate_no_intervention,
    evaluate_policy,
)
from rl_training.reward_function import RewardConfig
from rl_training.trainer import RLTrainer, TrainerConfig

from validation_framework.ai_augmented_policy import (
    AIAugmentedObservationWrapper,
    AIPredictionBundle,
    evaluate_ai_augmented_policy,
    train_ai_augmented_policy,
)
from validation_framework.metrics import rl_policy_metrics
from validation_framework.statistics import ArmComparisonReport, compare_arms


# =====================================================
# Phase 3 -- RL Evaluation. Every arm is built from
# rl_training.evaluator's own harness (evaluate_no_intervention /
# evaluate_decision_policy / evaluate_policy) or the AI-augmented
# variant in ai_augmented_policy.py -- no new policy-evaluation logic
# lives here, only orchestration + statistics. Each arm's env/trainer
# uses a seed offset far apart from the others so no two arms ever draw
# from an overlapping scenario-seed stream.
# =====================================================

_NO_INTERVENTION_OFFSET = 1_000_000
_DECISION_POLICY_OFFSET = 2_000_000
_RL_TRAIN_OFFSET = 3_000_000
_RL_EVAL_OFFSET = 4_000_000
_AI_RL_TRAIN_OFFSET = 5_000_000
_AI_RL_EVAL_OFFSET = 6_000_000


@dataclass(frozen=True)
class PolicyComparisonResult:

    reports: Dict[str, EvaluationReport]
    metrics_by_arm: Dict[str, Dict[str, Any]]
    statistical_comparison: ArmComparisonReport

    def to_dict(self) -> Dict[str, Any]:

        return {
            "reports": {label: report.to_dict() for label, report in self.reports.items()},
            "metrics_by_arm": self.metrics_by_arm,
            "statistical_comparison": self.statistical_comparison.to_dict(),
        }


def run_policy_comparison(
    building,
    definition,
    definition_id: str,
    master_seed: int,
    episode_count: int,
    *,
    ai_bundle: Optional[AIPredictionBundle] = None,
    dt: float = 1.0,
    max_steps: int = 500,
    reward_config: Optional[RewardConfig] = None,
    rl_total_timesteps: int = 2000,
    rl_trainer_config: Optional[TrainerConfig] = None,
) -> PolicyComparisonResult:

    no_intervention_env = SynEvacGymEnv(
        building, definition, definition_id, master_seed, dt=dt, max_steps=max_steps, reward_config=reward_config,
    )
    no_intervention_report = evaluate_no_intervention(
        no_intervention_env, master_seed + _NO_INTERVENTION_OFFSET, episode_count,
    )

    decision_policy_report = evaluate_decision_policy(
        building, definition, definition_id, master_seed + _DECISION_POLICY_OFFSET, episode_count,
        dt=dt, max_steps=max_steps, reward_config=reward_config,
    )

    training_env = SynEvacGymEnv(
        building, definition, definition_id, master_seed + _RL_TRAIN_OFFSET,
        dt=dt, max_steps=max_steps, reward_config=reward_config,
    )
    trainer = RLTrainer(training_env, rl_trainer_config)
    trainer.train(rl_total_timesteps)

    rl_eval_env = SynEvacGymEnv(
        building, definition, definition_id, master_seed + _RL_EVAL_OFFSET,
        dt=dt, max_steps=max_steps, reward_config=reward_config,
    )
    rl_policy_report = evaluate_policy(
        rl_eval_env, lambda observation: trainer.predict(observation, deterministic=True)[0],
        master_seed + _RL_EVAL_OFFSET, episode_count, label="rl_policy",
    )

    reports: Dict[str, EvaluationReport] = {
        "no_intervention": no_intervention_report,
        "decision_policy": decision_policy_report,
        "rl_policy": rl_policy_report,
    }

    if ai_bundle is not None:

        ai_training_env = SynEvacGymEnv(
            building, definition, definition_id, master_seed + _AI_RL_TRAIN_OFFSET,
            dt=dt, max_steps=max_steps, reward_config=reward_config,
        )
        ai_wrapped_training_env = AIAugmentedObservationWrapper(
            ai_training_env, building, definition, definition_id,
            master_seed + _AI_RL_TRAIN_OFFSET, ai_bundle,
        )
        ai_trainer = train_ai_augmented_policy(
            ai_wrapped_training_env, total_timesteps=rl_total_timesteps, trainer_config=rl_trainer_config,
        )

        ai_eval_env = SynEvacGymEnv(
            building, definition, definition_id, master_seed + _AI_RL_EVAL_OFFSET,
            dt=dt, max_steps=max_steps, reward_config=reward_config,
        )
        ai_wrapped_eval_env = AIAugmentedObservationWrapper(
            ai_eval_env, building, definition, definition_id,
            master_seed + _AI_RL_EVAL_OFFSET, ai_bundle,
        )
        reports["ai_rl"] = evaluate_ai_augmented_policy(
            ai_wrapped_eval_env, ai_trainer, master_seed + _AI_RL_EVAL_OFFSET, episode_count, label="ai_rl",
        )

    metrics_by_arm = {label: rl_policy_metrics(report.episodes) for label, report in reports.items()}

    reward_values_by_arm = {
        label: [episode.total_reward for episode in report.episodes] for label, report in reports.items()
    }
    statistical_comparison = compare_arms(reward_values_by_arm, "total_reward", baseline_arm="no_intervention")

    return PolicyComparisonResult(
        reports=reports, metrics_by_arm=metrics_by_arm, statistical_comparison=statistical_comparison,
    )
