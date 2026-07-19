from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np

from ai_training.dataset import TrainingDataset
from ai_training.experiment import ExperimentConfig, ExperimentRunner

from dataset_builder.builder import SimulationRun
from dataset_builder.feature_extractor import extract_scenario_features
from dataset_builder.schema import ordered_exits

from scenario_generator import derive_scenario_seed
from scenario_pipeline import run_pipeline

from simulator.multi_agent_result import MultiAgentSimulationResult

from rl_training.environment import SynEvacGymEnv
from rl_training.evaluator import EvaluationReport, evaluate_policy
from rl_training.trainer import RLTrainer, TrainerConfig


# =====================================================
# Phase 3 -- "AI + RL" arm. Per the user's own design decision: this is
# NOT a rule-based hybrid controller. The RL policy is trained/evaluated
# on an augmented observation space that includes AI-generated
# predictions (bottleneck risk, evacuation time, smoke risk, exit
# utilization) as extra observation features -- the RL policy remains
# the sole decision-maker; nothing here ever chooses or overrides an
# action. Matches the intended architecture:
#   Sensors -> AI Prediction -> RL Decision -> Advisory System
# =====================================================


@dataclass(frozen=True)
class AIPredictionBundle:

    evacuation_time_model: Any
    bottleneck_model: Any
    smoke_model: Any
    exit_usage_model: Any
    exit_order: Tuple[str, ...]
    max_steps: int
    dt: float

    @property
    def feature_count(self) -> int:

        # [normalized_evacuation_time, bottleneck_probability,
        #  smoke_confidence, *exit_utilization_by(exit_order)]
        return 3 + len(self.exit_order)


def fit_ai_prediction_bundle(
    dataset: TrainingDataset, building, *, max_steps: int = 500, dt: float = 1.0, random_state: int = 0,
) -> AIPredictionBundle:

    runner = ExperimentRunner()

    evacuation_result = runner.run(
        dataset, ExperimentConfig(name="ai-rl-evacuation-time", model_name="evacuation_time", random_state=random_state),
    )
    bottleneck_result = runner.run(
        dataset,
        ExperimentConfig(
            name="ai-rl-bottleneck", model_name="bottleneck",
            model_kwargs={"target": "occurrence"}, random_state=random_state,
        ),
    )
    smoke_result = runner.run(
        dataset, ExperimentConfig(name="ai-rl-smoke", model_name="smoke_prediction", random_state=random_state),
    )
    exit_usage_result = runner.run(
        dataset, ExperimentConfig(name="ai-rl-exit-usage", model_name="exit_usage", random_state=random_state),
    )

    exit_order = tuple(exit_obj.id for exit_obj in ordered_exits(building))

    return AIPredictionBundle(
        evacuation_time_model=evacuation_result.model,
        bottleneck_model=bottleneck_result.model,
        smoke_model=smoke_result.model,
        exit_usage_model=exit_usage_result.model,
        exit_order=exit_order, max_steps=max_steps, dt=dt,
    )


def predict_ai_feature_vector(bundle: AIPredictionBundle, scenario_features_row: Dict[str, Any]) -> np.ndarray:

    row = [scenario_features_row]

    predicted_evacuation_time = float(bundle.evacuation_time_model.predict(row)[0])
    normalized_evacuation_time = predicted_evacuation_time / max(1.0, bundle.max_steps * bundle.dt)

    bottleneck_probability = _positive_class_probability(bundle.bottleneck_model, row, positive_label=True)
    smoke_confidence = _max_class_probability(bundle.smoke_model, row)

    exit_usage = bundle.exit_usage_model.predict(row)[0]
    exit_utilization = [float(exit_usage.get(exit_id, 0.0)) / 100.0 for exit_id in bundle.exit_order]

    return np.array(
        [normalized_evacuation_time, bottleneck_probability, smoke_confidence, *exit_utilization],
        dtype=np.float32,
    )


def _positive_class_probability(model, row, *, positive_label: Any) -> float:

    try:

        proba = model.predict_proba(row)
        classes = list(model.label_encoder.classes_)

        if positive_label in classes:
            return float(proba[0][classes.index(positive_label)])

    except (AttributeError, TypeError, ValueError):
        pass

    prediction = model.predict(row)[0]

    return 1.0 if prediction == positive_label else 0.0


def _max_class_probability(model, row) -> float:

    try:

        proba = model.predict_proba(row)
        return float(np.max(proba[0]))

    except (AttributeError, TypeError, ValueError):
        return 0.0


# =====================================================


class AIAugmentedObservationWrapper(gymnasium.ObservationWrapper):

    # Appends a fixed-size AI-prediction vector to every observation.
    # reset() independently re-derives the same Scenario the wrapped
    # SynEvacGymEnv is about to generate (same public functions,
    # derive_scenario_seed()/run_pipeline(), same episode-index
    # bookkeeping the env keeps internally) SOLELY to compute the AI
    # predictions before the episode starts -- it never reaches into
    # the wrapped env's own private state, and it computes the
    # prediction from PRE-SIMULATION scenario features only (Dataset 1
    # facts, never a simulation output), so nothing here can leak
    # information the RL policy would not honestly have at decision
    # time. The RL policy's action selection is untouched -- this class
    # only ever overrides observation().

    def __init__(
        self,
        env: SynEvacGymEnv,
        building,
        definition,
        definition_id: str,
        master_seed: int,
        bundle: AIPredictionBundle,
    ):

        super().__init__(env)

        self._building = building
        self._definition = definition
        self._definition_id = definition_id
        self._master_seed = master_seed
        self._bundle = bundle
        self._episode_index = 0

        base_space = env.observation_space
        feature_count = bundle.feature_count

        low = np.concatenate([base_space.low, np.zeros(feature_count, dtype=np.float32)])
        high = np.concatenate([base_space.high, np.full(feature_count, np.inf, dtype=np.float32)])

        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        self._ai_features = np.zeros(feature_count, dtype=np.float32)

    # =====================================================
    # gymnasium.Wrapper's own __getattr__ forwarding does not cover
    # underscore-prefixed names (by design, to avoid accidentally
    # leaking private state) -- but rl_training.evaluator.evaluate_policy
    # (unmodified, reused as-is) reads env._scenario/env._simulation
    # directly on whatever env it is given, exactly as it already does
    # for a bare, unwrapped SynEvacGymEnv. These two properties make
    # that same call work transparently through this wrapper too,
    # without editing rl_training or duplicating evaluate_policy's own
    # evaluation loop here.

    @property
    def _scenario(self):

        return self.env._scenario

    @property
    def _simulation(self):

        return self.env._simulation

    # =====================================================

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        base_seed = seed if seed is not None else self._master_seed
        scenario_seed = derive_scenario_seed(base_seed, self._episode_index)
        self._episode_index += 1

        result = run_pipeline(self._definition, self._definition_id, self._building, scenario_seed)

        if result.accepted:

            placeholder_movement = MultiAgentSimulationResult(occupants={}, total_evacuation_time=None)
            run = SimulationRun(
                scenario=result.scenario, building=self._building, movement_result=placeholder_movement,
            )
            features_row = extract_scenario_features(run)
            self._ai_features = predict_ai_feature_vector(self._bundle, features_row)

        else:

            # Mirrors SynEvacGymEnv.reset()'s own "no repair/retry" contract
            # -- the wrapped env's own reset() call below will raise for
            # this same seed, so this branch is defensive only.
            self._ai_features = np.zeros(self._bundle.feature_count, dtype=np.float32)

        observation, info = self.env.reset(seed=seed, options=options)

        return self.observation(observation), info

    # =====================================================

    def observation(self, observation: np.ndarray) -> np.ndarray:

        return np.concatenate([observation, self._ai_features]).astype(np.float32)


# =====================================================


def train_ai_augmented_policy(
    wrapped_env: AIAugmentedObservationWrapper, *, total_timesteps: int = 2000,
    trainer_config: Optional[TrainerConfig] = None,
) -> RLTrainer:

    trainer = RLTrainer(wrapped_env, trainer_config)
    trainer.train(total_timesteps)

    return trainer


def evaluate_ai_augmented_policy(
    wrapped_env: AIAugmentedObservationWrapper, trainer: RLTrainer, master_seed: int, count: int,
    label: str = "ai_rl",
) -> EvaluationReport:

    return evaluate_policy(
        wrapped_env, lambda observation: trainer.predict(observation, deterministic=True)[0],
        master_seed, count, label=label,
    )
