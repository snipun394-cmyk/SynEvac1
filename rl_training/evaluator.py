import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from scenario_pipeline import run_batch_pipeline
from scenario_runner import run as run_scenario

from ground_truth.analyzer import SimulationArtifacts, analyze
from decision_policy.policy import DecisionInputs, generate_policy
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE

from simulation_interactive import Action, InteractiveActionType, InteractiveSimulation

from rl_training.environment import SynEvacGymEnv
from rl_training.reward_function import RewardConfig


@dataclass(frozen=True)
class EpisodeMetrics:

    scenario_id: str
    total_reward: float
    total_evacuation_time: Optional[float]
    people_trapped: int
    people_evacuated: int
    peak_congestion_value: Optional[int]
    exits_underutilized: Tuple[str, ...]
    exits_exceeding_capacity: Tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:

    label: str
    episodes: Tuple[EpisodeMetrics, ...]
    average_reward: float
    average_evacuation_time: Optional[float]
    average_people_trapped: float
    average_peak_congestion: Optional[float]

    def to_dict(self) -> dict:

        return {
            "label": self.label,
            "average_reward": self.average_reward,
            "average_evacuation_time": self.average_evacuation_time,
            "average_people_trapped": self.average_people_trapped,
            "average_peak_congestion": self.average_peak_congestion,
            "episode_count": len(self.episodes),
        }


def _build_report(label: str, episodes: Tuple[EpisodeMetrics, ...]) -> EvaluationReport:

    evacuation_times = [e.total_evacuation_time for e in episodes if e.total_evacuation_time is not None]
    congestion_values = [e.peak_congestion_value for e in episodes if e.peak_congestion_value is not None]

    return EvaluationReport(
        label=label,
        episodes=episodes,
        average_reward=statistics.fmean(e.total_reward for e in episodes) if episodes else 0.0,
        average_evacuation_time=statistics.fmean(evacuation_times) if evacuation_times else None,
        average_people_trapped=(
            statistics.fmean(e.people_trapped for e in episodes) if episodes else 0.0
        ),
        average_peak_congestion=statistics.fmean(congestion_values) if congestion_values else None,
    )


def _episode_metrics_from_ground_truth(scenario_id: str, total_reward: float, ground_truth) -> EpisodeMetrics:

    return EpisodeMetrics(
        scenario_id=scenario_id,
        total_reward=total_reward,
        total_evacuation_time=ground_truth.total_evacuation_time,
        people_trapped=ground_truth.people_trapped,
        people_evacuated=ground_truth.people_evacuated,
        peak_congestion_value=ground_truth.peak_congestion_value,
        exits_underutilized=ground_truth.exits_underutilized,
        exits_exceeding_capacity=ground_truth.exits_exceeding_capacity,
    )


def evaluate_policy(
    env: SynEvacGymEnv,
    predict_fn: Callable[[Any], int],
    master_seed: int,
    count: int,
    label: str = "rl_policy",
) -> EvaluationReport:

    episodes = []

    for episode_index in range(count):

        observation, _info = env.reset(seed=master_seed + episode_index)
        scenario_id = env._scenario.metadata.scenario_id
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = predict_fn(observation)
            observation, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward

        movement_result = env._simulation._stepper.snapshot_result()
        ground_truth = analyze(
            SimulationArtifacts(scenario=env._scenario, building=env._building, movement_result=movement_result)
        )

        episodes.append(_episode_metrics_from_ground_truth(scenario_id, total_reward, ground_truth))

    return _build_report(label, tuple(episodes))


def evaluate_no_intervention(
    env: SynEvacGymEnv, master_seed: int, count: int, label: str = "no_intervention",
) -> EvaluationReport:

    noop_index = 0  # ActionMapper's index 0 is always NOOP.

    return evaluate_policy(env, lambda observation: noop_index, master_seed, count, label=label)


def _compile_decision_policy_actions(decision_policy) -> Tuple[Action, ...]:

    actions = []

    for zone_decision in decision_policy.zone_decisions:

        zone_id = zone_decision["zone_id"]
        action = zone_decision.get("action")

        if action == EVACUATE_IMMEDIATELY:

            if zone_decision.get("recommended_exit"):
                actions.append(
                    Action(
                        InteractiveActionType.RECOMMEND_EXIT, target_id=zone_id,
                        parameters={"exit_id": zone_decision["recommended_exit"]},
                    )
                )

            if zone_decision.get("recommended_stair"):
                actions.append(
                    Action(
                        InteractiveActionType.RECOMMEND_STAIR, target_id=zone_id,
                        parameters={"stair_id": zone_decision["recommended_stair"]},
                    )
                )

        elif action == SHELTER_IN_PLACE:

            actions.append(
                Action(
                    InteractiveActionType.BROADCAST_SHELTER_IN_PLACE,
                    parameters={"target_zone_ids": [zone_id]},
                )
            )

    return tuple(actions)


def evaluate_decision_policy(
    building, definition, definition_id, master_seed: int, count: int,
    dt: float = 1.0, max_steps: int = 500, reward_config: Optional[RewardConfig] = None,
    label: str = "decision_policy",
) -> EvaluationReport:

    # Two-pass per scenario, since decision_policy.generate_policy()
    # structurally requires an already-finished GroundTruth (it is a
    # purely descriptive, post-hoc report -- confirmed no code path
    # anywhere applies it live). Pass 1: run with no interventions to
    # completion, purely to obtain a GroundTruth/DecisionPolicy. Pass 2:
    # a fresh InteractiveSimulation of the *same* scenario (same seed)
    # with that policy's recommendations compiled into an action
    # schedule queued at t=0 -- reports what actually enacting the
    # policy would have produced, rather than a purely descriptive
    # readout.

    from rl_training.reward_function import ActionState, RewardFunction

    batch = run_batch_pipeline(definition, definition_id, building, master_seed, count)
    reward_fn = RewardFunction(reward_config)

    episodes = []

    for scenario in batch.scenarios:

        pass_one_context = run_scenario(scenario, building)
        pass_one_sim = InteractiveSimulation(pass_one_context, dt=dt)
        pass_one_sim.run_to_completion(max_time=max_steps * dt)

        pass_one_result = pass_one_sim._stepper.snapshot_result()
        ground_truth = analyze(
            SimulationArtifacts(scenario=scenario, building=building, movement_result=pass_one_result)
        )
        policy = generate_policy(
            DecisionInputs(building=building, scenario=scenario, ground_truth=ground_truth)
        )
        compiled_actions = _compile_decision_policy_actions(policy)

        pass_two_context = run_scenario(scenario, building)
        pass_two_sim = InteractiveSimulation(pass_two_context, dt=dt)

        for compiled_action in compiled_actions:
            pass_two_sim.queue_action(compiled_action)

        action_state = ActionState()
        prev_snapshot = None
        total_reward = 0.0
        step_count = 0

        while not pass_two_sim.is_finished and step_count < max_steps:
            step_result = pass_two_sim.step()
            step_count += 1
            if prev_snapshot is not None:
                reward, _breakdown = reward_fn.step_reward(
                    prev_snapshot, step_result.snapshot, step_result.applied_actions, action_state,
                )
                total_reward += reward
            prev_snapshot = step_result.snapshot

        pass_two_result = pass_two_sim._stepper.snapshot_result()
        final_ground_truth = analyze(
            SimulationArtifacts(scenario=scenario, building=building, movement_result=pass_two_result)
        )
        terminal_bonus, _breakdown = reward_fn.terminal_reward(final_ground_truth)
        total_reward += terminal_bonus

        episodes.append(
            _episode_metrics_from_ground_truth(
                scenario.metadata.scenario_id, total_reward, final_ground_truth,
            )
        )

    return _build_report(label, tuple(episodes))


def compare_reports(*reports: EvaluationReport) -> Dict[str, Dict[str, float]]:

    return {report.label: report.to_dict() for report in reports}
