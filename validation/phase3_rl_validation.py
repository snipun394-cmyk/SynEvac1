"""
Phase 3 -- RL Validation.

Trains PPO on SynEvacGymEnv, tracks training convergence via periodic
evaluation checkpoints, then compares the trained policy against the
no-intervention and deterministic Decision Policy baselines using
rl_training.evaluator. Does not modify rl_training/ or
simulation_interactive/ -- only calls their existing public API.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.callbacks import BaseCallback

from scenario_definition import FireDefinition, FixedValue, OccupantDefinition, ScenarioDefinition, UniformRange

from tests.rl_training_fixtures import DEFINITION_ID, MASTER_SEED, make_building

from rl_training import RLTrainer, SynEvacGymEnv, TrainerConfig
from rl_training.evaluator import compare_reports, evaluate_decision_policy, evaluate_no_intervention, evaluate_policy

TOTAL_TIMESTEPS = 50_000
CHECKPOINT_INTERVAL = 5_000
EVAL_SCENARIO_COUNT = 20
MAX_STEPS = 200
DT = 2.0


def make_definition() -> ScenarioDefinition:

    # tests/rl_training_fixtures.py's own make_definition() spawns
    # exactly one occupant -- a single occupant already takes the
    # (already-optimal) shortest path, so there is no congestion
    # imbalance for an RL policy to improve on. This definition spawns
    # 6-10 occupants into the same single start zone instead, so the
    # default "everyone takes the shortest path" behavior genuinely
    # congests the near exit while the far exit sits unused -- giving
    # RECOMMEND_EXIT a real optimization opportunity to demonstrate.

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-start": UniformRange(6, 10, discrete=True)},
            behaviour_profile_distribution={"zone-start": FixedValue("Staff_Default")},
        ),
    )


class ConvergenceCallback(BaseCallback):

    # Every CHECKPOINT_INTERVAL timesteps, evaluates the current policy
    # (deterministic actions) over a small, fixed set of scenarios via
    # rl_training.evaluate_policy -- the same evaluation function Phase
    # 3's final comparison uses, so the convergence curve and the final
    # number are produced by the exact same code path.

    def __init__(self, eval_env, checkpoint_interval, eval_scenario_count):

        super().__init__()
        self.eval_env = eval_env
        self.checkpoint_interval = checkpoint_interval
        self.eval_scenario_count = eval_scenario_count
        self.history = []
        self._next_checkpoint = checkpoint_interval

    def _on_step(self) -> bool:

        if self.num_timesteps >= self._next_checkpoint:

            report = evaluate_policy(
                self.eval_env,
                lambda obs: self.model.predict(obs, deterministic=True)[0],
                MASTER_SEED + 500_000, self.eval_scenario_count, label="checkpoint",
            )
            self.history.append({"timesteps": self.num_timesteps, "average_reward": report.average_reward})
            print(f"  [checkpoint @ {self.num_timesteps}] average_reward={report.average_reward:.3f}", flush=True)

            self._next_checkpoint += self.checkpoint_interval

        return True


def main():

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    building = make_building()
    definition = make_definition()

    train_env = SynEvacGymEnv(building, definition, DEFINITION_ID, MASTER_SEED, dt=DT, max_steps=MAX_STEPS)
    eval_env = SynEvacGymEnv(building, definition, DEFINITION_ID, MASTER_SEED, dt=DT, max_steps=MAX_STEPS)

    trainer = RLTrainer(
        train_env,
        TrainerConfig(
            algorithm="PPO",
            algorithm_kwargs={"n_steps": 256, "batch_size": 64, "n_epochs": 4, "verbose": 0},
            seed=0,
        ),
    )

    callback = ConvergenceCallback(eval_env, CHECKPOINT_INTERVAL, EVAL_SCENARIO_COUNT)

    print(f"Training PPO for {TOTAL_TIMESTEPS} timesteps ...", flush=True)
    t0 = time.perf_counter()
    trainer.train(TOTAL_TIMESTEPS, callback=callback)
    training_time = time.perf_counter() - t0
    print(f"Training done in {training_time:.1f}s", flush=True)

    def rl_predict(observation):
        action, _state = trainer.predict(observation, deterministic=True)
        return int(action)

    print("Evaluating trained RL policy ...", flush=True)
    rl_report = evaluate_policy(
        eval_env, rl_predict, MASTER_SEED + 1_000_000, EVAL_SCENARIO_COUNT, label="rl_policy",
    )

    print("Evaluating no-intervention baseline ...", flush=True)
    no_intervention_report = evaluate_no_intervention(
        eval_env, MASTER_SEED + 2_000_000, EVAL_SCENARIO_COUNT,
    )

    print("Evaluating deterministic Decision Policy baseline ...", flush=True)
    decision_policy_report = evaluate_decision_policy(
        building, definition, DEFINITION_ID, MASTER_SEED + 3_000_000, EVAL_SCENARIO_COUNT,
        dt=DT, max_steps=MAX_STEPS,
    )

    comparison = compare_reports(rl_report, no_intervention_report, decision_policy_report)

    results = {
        "training_time_seconds": training_time,
        "total_timesteps": TOTAL_TIMESTEPS,
        "convergence_history": callback.history,
        "comparison": comparison,
        "episode_detail": {
            "rl_policy": [vars(e) for e in rl_report.episodes],
            "no_intervention": [vars(e) for e in no_intervention_report.episodes],
            "decision_policy": [vars(e) for e in decision_policy_report.episodes],
        },
    }

    with open(os.path.join(output_dir, "phase3_rl_validation.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
