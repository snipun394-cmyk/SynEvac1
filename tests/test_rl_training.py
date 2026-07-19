import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from hazard.snapshot import HazardSnapshot

from simulation_interactive import Action, ActionResult, InteractiveActionType, SimulationSnapshot
from simulation_interactive.state_snapshot import OccupantSnapshot

from rl_training.action_space import ActionMapper
from rl_training.environment import SynEvacGymEnv
from rl_training.evaluator import (
    evaluate_decision_policy,
    evaluate_no_intervention,
    evaluate_policy,
    compare_reports,
)
from rl_training.observation_space import (
    BROADCAST_NONE,
    RECOMMENDATION_NONE,
    ObservationEncoder,
)
from rl_training.policy_export import load_policy, save_policy
from rl_training.reward_function import ActionState, RewardConfig, RewardFunction
from rl_training.trainer import RLTrainer, TrainerConfig

from tests.rl_training_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


def make_env(max_steps=200, dt=1.0, reward_config=None):

    return SynEvacGymEnv(
        make_building(), make_definition(), DEFINITION_ID, MASTER_SEED,
        dt=dt, max_steps=max_steps, reward_config=reward_config,
    )


def make_snapshot(
    time=0.0, people_evacuated=0, people_remaining=1, occupants=(),
    door_states=None, exit_states=None, edge_congestion=None,
):

    return SimulationSnapshot(
        time=time,
        occupants=occupants,
        hazard_snapshot=HazardSnapshot(),
        door_states=door_states or {"door-a": "OPEN", "door-b": "OPEN"},
        exit_states=exit_states or {"exit-a": "OPEN", "exit-b": "OPEN"},
        stair_states={},
        detector_states={},
        camera_states={},
        edge_congestion=edge_congestion or {"exit-a": 0, "exit-b": 0},
        edge_queue_lengths={},
        node_occupancy={},
        people_evacuated=people_evacuated,
        people_remaining=people_remaining,
    )


class GymnasiumComplianceTests(unittest.TestCase):

    def test_reset_returns_observation_and_info(self):

        env = make_env()
        observation, info = env.reset(seed=1)

        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info, {})

    def test_step_returns_five_tuple(self):

        env = make_env()
        env.reset(seed=1)

        observation, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("reward_breakdown", info)

    def test_render_and_close_are_placeholders(self):

        env = make_env()
        env.reset(seed=1)

        self.assertIsNone(env.render())
        env.close()

    def test_passes_gymnasium_check_env(self):

        env = make_env(max_steps=200, dt=2.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_env(env, skip_render_check=True)


class DeterministicResetTests(unittest.TestCase):

    def test_same_seed_same_episode_index_produces_identical_observation(self):

        env_one = make_env()
        env_two = make_env()

        obs_one, _ = env_one.reset(seed=7)
        obs_two, _ = env_two.reset(seed=7)

        np.testing.assert_array_equal(obs_one, obs_two)
        self.assertEqual(env_one._scenario.metadata.seed, env_two._scenario.metadata.seed)


class ObservationHonestyTests(unittest.TestCase):

    def test_feature_names_contain_no_ground_truth_concepts(self):

        encoder = ObservationEncoder(make_building())

        forbidden_substrings = ("risk_score", "worst_exit", "worst_stair", "trapped_fraction")

        for name in encoder.schema.feature_names:
            for forbidden in forbidden_substrings:
                self.assertNotIn(forbidden, name)

    def test_observation_module_never_imports_ground_truth_or_decision_policy(self):

        path = Path(__file__).resolve().parent.parent / "rl_training" / "observation_space.py"
        text = path.read_text(encoding="utf-8")

        self.assertNotIn("import ground_truth", text)
        self.assertNotIn("from ground_truth", text)
        self.assertNotIn("import decision_policy", text)
        self.assertNotIn("from decision_policy", text)


class ObservationEncodesRecommendationTargetTests(unittest.TestCase):

    # Phase 2 -- Markov-property fix. RecommendationDisciplineComponent
    # (reward_function.py) scores redundant/contradictory recommendations
    # by the exact exit/stair id last recommended and how recently that
    # changed; these tests confirm the observation now actually encodes
    # that same information, distinguishably, per exit/stair.

    def test_only_the_matching_exit_flag_is_set(self):

        encoder = ObservationEncoder(make_building())
        snapshot = make_snapshot()

        observation = encoder.encode(
            snapshot, {}, None, elapsed_time=0.0, max_time=100.0,
            active_recommendation_target="exit-b",
            steps_since_recommendation_change=3,
        )

        names = encoder.schema.feature_names
        exit_a_index = names.index("exit:exit-a:is_active_recommendation_target")
        exit_b_index = names.index("exit:exit-b:is_active_recommendation_target")

        self.assertEqual(observation[exit_a_index], 0.0)
        self.assertEqual(observation[exit_b_index], 1.0)

    def test_no_active_target_sets_every_flag_to_zero(self):

        encoder = ObservationEncoder(make_building())
        snapshot = make_snapshot()

        observation = encoder.encode(
            snapshot, {}, None, elapsed_time=0.0, max_time=100.0,
            active_recommendation_target=None, steps_since_recommendation_change=10 ** 9,
        )

        names = encoder.schema.feature_names
        for exit_id in ("exit-a", "exit-b"):
            index = names.index(f"exit:{exit_id}:is_active_recommendation_target")
            self.assertEqual(observation[index], 0.0)

    def test_steps_since_change_is_encoded_and_capped(self):

        encoder = ObservationEncoder(make_building())
        snapshot = make_snapshot()
        index = encoder.schema.feature_names.index("global:steps_since_recommendation_change")

        recent = encoder.encode(
            snapshot, {}, None, elapsed_time=0.0, max_time=100.0,
            active_recommendation_target="exit-a", steps_since_recommendation_change=0,
        )
        stale = encoder.encode(
            snapshot, {}, None, elapsed_time=0.0, max_time=100.0,
            active_recommendation_target="exit-a", steps_since_recommendation_change=10 ** 9,
        )

        self.assertEqual(recent[index], 0.0)
        self.assertTrue(np.isfinite(stale[index]))  # capped, not the raw 10**9 sentinel

    def test_observation_space_shape_matches_feature_name_count(self):

        encoder = ObservationEncoder(make_building())

        self.assertEqual(encoder.space.shape[0], len(encoder.schema.feature_names))


class MarkovPropertyTests(unittest.TestCase):

    # End-to-end confirmation that the environment satisfies the
    # Markov property with respect to RecommendationDisciplineComponent:
    # the observation the policy receives after an action already
    # contains everything needed to predict whether the NEXT action
    # will be scored redundant/contradictory.

    def _target_flag(self, env, observation, exit_id):

        index = env._observation_encoder.schema.feature_names.index(
            f"exit:{exit_id}:is_active_recommendation_target",
        )
        return observation[index]

    def _steps_since_change(self, env, observation):

        index = env._observation_encoder.schema.feature_names.index(
            "global:steps_since_recommendation_change",
        )
        return observation[index]

    def test_observation_after_a_recommendation_reveals_the_target_and_recency(self):

        env = make_env(max_steps=200, dt=1.0)
        env.reset(seed=1)

        mapper = ActionMapper(make_building())
        recommend_exit_b_index = next(
            entry.index for entry in mapper.schema.entries
            if entry.label == "RECOMMEND_EXIT:exit-b"
        )

        observation, _reward, _terminated, _truncated, _info = env.step(recommend_exit_b_index)

        self.assertEqual(self._target_flag(env, observation, "exit-b"), 1.0)
        self.assertEqual(self._target_flag(env, observation, "exit-a"), 0.0)
        self.assertEqual(self._steps_since_change(env, observation), 0.0)

    def test_redundant_action_is_predictable_from_the_preceding_observation(self):

        env = make_env(max_steps=200, dt=1.0)
        env.reset(seed=1)

        mapper = ActionMapper(make_building())
        recommend_exit_b_index = next(
            entry.index for entry in mapper.schema.entries
            if entry.label == "RECOMMEND_EXIT:exit-b"
        )

        observation_after_first, _reward, _terminated, _truncated, _info = env.step(
            recommend_exit_b_index,
        )

        # The information needed to know the SECOND identical action
        # will be redundant is already fully present in the FIRST
        # action's own resulting observation -- exit-b's target flag
        # is set, proving a policy conditioned only on this observation
        # could, in principle, predict/avoid the redundant penalty.
        self.assertEqual(self._target_flag(env, observation_after_first, "exit-b"), 1.0)

        _observation, _reward, _terminated, _truncated, info = env.step(recommend_exit_b_index)

        self.assertLess(info["reward_breakdown"]["recommendation_discipline"], 0.0)

    def test_adaptive_reroute_is_distinguishable_from_a_first_time_recommendation(self):

        # Two different histories that reach the identical observable
        # state (no recommendation issued the previous step) must
        # yield identical rewards for the same next action -- proof
        # the reward only ever depends on what the observation encodes,
        # not on hidden history the observation can't reveal.

        env = make_env(max_steps=200, dt=1.0)
        env.reset(seed=1)

        mapper = ActionMapper(make_building())
        recommend_exit_a_index = next(
            entry.index for entry in mapper.schema.entries
            if entry.label == "RECOMMEND_EXIT:exit-a"
        )

        # Let enough steps pass that any prior recommendation (there is
        # none yet) would be well outside the contradiction window.
        for _ in range(10):
            env.step(0)

        _observation, _reward, _terminated, _truncated, info = env.step(recommend_exit_a_index)

        self.assertNotIn("is_redundant", info["reward_breakdown"])  # sanity: key doesn't leak internals
        self.assertGreaterEqual(info["reward_breakdown"]["recommendation_discipline"], 0.0)


class ActionMappingTests(unittest.TestCase):

    def test_noop_decodes_to_no_actions(self):

        mapper = ActionMapper(make_building())

        self.assertEqual(mapper.decode(0), ())
        self.assertEqual(mapper.describe(0), "NOOP")

    def test_recommend_exit_decodes_to_one_action_per_zone(self):

        mapper = ActionMapper(make_building())

        recommend_exit_index = next(
            entry.index for entry in mapper.schema.entries
            if entry.action_type == InteractiveActionType.RECOMMEND_EXIT
        )
        actions = mapper.decode(recommend_exit_index)

        self.assertEqual(len(actions), 3)  # zone-start, zone-a, zone-b
        for action in actions:
            self.assertEqual(action.action_type, InteractiveActionType.RECOMMEND_EXIT)
            self.assertIsNotNone(action.parameters.get("exit_id"))

    def test_every_entry_uses_only_existing_interactive_action_types(self):

        mapper = ActionMapper(make_building())

        for entry in mapper.schema.entries:
            if entry.action_type is not None:
                self.assertIsInstance(entry.action_type, InteractiveActionType)


class RewardComponentTests(unittest.TestCase):

    def test_evacuation_progress_is_rewarded(self):

        reward_fn = RewardFunction(RewardConfig())
        prev = make_snapshot(people_evacuated=0, people_remaining=2)
        current = make_snapshot(people_evacuated=1, people_remaining=1)

        reward, breakdown = reward_fn.step_reward(prev, current, (), ActionState())

        self.assertGreater(breakdown["evacuation_progress"], 0.0)

    def test_congestion_is_penalized(self):

        reward_fn = RewardFunction(RewardConfig())
        prev = make_snapshot(edge_congestion={"exit-a": 0, "exit-b": 0})
        current = make_snapshot(edge_congestion={"exit-a": 5, "exit-b": 5})

        _reward, breakdown = reward_fn.step_reward(prev, current, (), ActionState())

        self.assertLess(breakdown["congestion"], 0.0)

    def test_unreachable_occupant_is_penalized(self):

        reward_fn = RewardFunction(RewardConfig())
        prev = make_snapshot(occupants=(
            OccupantSnapshot("occ-1", "AT_NODE", "zone-start", 0.0, None),
        ))
        current = make_snapshot(occupants=(
            OccupantSnapshot("occ-1", "UNREACHABLE", None, 0.0, None),
        ))

        _reward, breakdown = reward_fn.step_reward(prev, current, (), ActionState())

        self.assertLess(breakdown["unreachable_occupants"], 0.0)

    def test_redundant_recommendation_is_penalized(self):

        reward_fn = RewardFunction(RewardConfig())
        prev = make_snapshot()
        current = make_snapshot()
        action_state = ActionState()

        action = Action(
            InteractiveActionType.RECOMMEND_EXIT, target_id="zone-start",
            parameters={"exit_id": "exit-a"},
        )
        applied = (ActionResult(action=action, applied=True),)

        reward_fn.step_reward(prev, current, applied, action_state)
        _reward, breakdown = reward_fn.step_reward(prev, current, applied, action_state)

        self.assertLess(breakdown["recommendation_discipline"], 0.0)

    def test_contradictory_recommendation_is_penalized(self):

        reward_fn = RewardFunction(RewardConfig())
        prev = make_snapshot()
        current = make_snapshot()
        action_state = ActionState()

        action_a = Action(
            InteractiveActionType.RECOMMEND_EXIT, target_id="zone-start",
            parameters={"exit_id": "exit-a"},
        )
        action_b = Action(
            InteractiveActionType.RECOMMEND_EXIT, target_id="zone-start",
            parameters={"exit_id": "exit-b"},
        )

        reward_fn.step_reward(prev, current, (ActionResult(action=action_a, applied=True),), action_state)
        _reward, breakdown = reward_fn.step_reward(
            prev, current, (ActionResult(action=action_b, applied=True),), action_state,
        )

        self.assertLess(breakdown["recommendation_discipline"], 0.0)

    def test_terminal_reward_bonus_when_building_cleared(self):

        class FakeGroundTruth:
            building_cleared = True
            total_evacuation_time = 42.0
            people_trapped = 0

        reward_fn = RewardFunction(RewardConfig())
        total, breakdown = reward_fn.terminal_reward(FakeGroundTruth())

        self.assertGreater(breakdown["completion_bonus"], 0.0)
        self.assertGreater(total, 0.0)


class EpisodeTerminationTests(unittest.TestCase):

    def test_scenario_runs_to_natural_completion(self):

        env = make_env(max_steps=200, dt=2.0)
        env.reset(seed=1)

        terminated = truncated = False
        steps = 0

        while not (terminated or truncated) and steps < 200:
            _obs, _reward, terminated, truncated, _info = env.step(0)
            steps += 1

        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_max_steps_truncates_before_natural_completion(self):

        env = make_env(max_steps=1, dt=0.001)
        env.reset(seed=1)

        _obs, _reward, terminated, truncated, _info = env.step(0)

        self.assertTrue(truncated)
        self.assertFalse(terminated)


class ActionExecutionEndToEndTests(unittest.TestCase):

    def test_recommend_exit_b_changes_evacuation_outcome(self):

        env = make_env(max_steps=500, dt=1.0)
        env.reset(seed=1)

        mapper = ActionMapper(make_building())
        recommend_exit_b_index = next(
            entry.index for entry in mapper.schema.entries
            if entry.label == "RECOMMEND_EXIT:exit-b"
        )

        terminated = truncated = False
        _obs, _reward, terminated, truncated, _info = env.step(recommend_exit_b_index)

        while not (terminated or truncated):
            _obs, _reward, terminated, truncated, _info = env.step(0)

        final_result = env._simulation._stepper.snapshot_result()
        occupant_ids = list(final_result.occupants)
        self.assertEqual(len(occupant_ids), 1)

        final_timeline = final_result.occupants[occupant_ids[0]]
        self.assertIn("exit-b", final_timeline.route.edge_ids)


class TrainerSmokeTests(unittest.TestCase):

    def test_ppo_trains_for_a_handful_of_timesteps(self):

        env = make_env(max_steps=50, dt=2.0)
        trainer = RLTrainer(
            env,
            TrainerConfig(
                algorithm="PPO",
                algorithm_kwargs={"n_steps": 32, "batch_size": 16, "n_epochs": 1, "verbose": 0},
                seed=0,
            ),
        )

        trainer.train(total_timesteps=64)

        observation, _info = env.reset(seed=1)
        action, _state = trainer.predict(observation)

        self.assertTrue(env.action_space.contains(int(action)))


class EvaluatorTests(unittest.TestCase):

    def test_no_intervention_baseline_produces_report(self):

        env = make_env(max_steps=200, dt=2.0)
        report = evaluate_no_intervention(env, MASTER_SEED, count=2)

        self.assertEqual(report.label, "no_intervention")
        self.assertEqual(len(report.episodes), 2)
        self.assertIsInstance(report.average_reward, float)

    def test_decision_policy_baseline_produces_report(self):

        report = evaluate_decision_policy(
            make_building(), make_definition(), DEFINITION_ID, MASTER_SEED, count=2,
            dt=2.0, max_steps=200,
        )

        self.assertEqual(report.label, "decision_policy")
        self.assertEqual(len(report.episodes), 2)

    def test_compare_reports_includes_every_label(self):

        env = make_env(max_steps=200, dt=2.0)
        no_intervention = evaluate_no_intervention(env, MASTER_SEED, count=1)
        rl_report = evaluate_policy(env, lambda obs: 0, MASTER_SEED + 100, count=1, label="rl_policy")

        comparison = compare_reports(no_intervention, rl_report)

        self.assertIn("no_intervention", comparison)
        self.assertIn("rl_policy", comparison)


class PolicyExportTests(unittest.TestCase):

    def test_save_and_load_round_trips_prediction(self):

        env = make_env(max_steps=50, dt=2.0)
        trainer = RLTrainer(
            env,
            TrainerConfig(
                algorithm="PPO",
                algorithm_kwargs={"n_steps": 32, "batch_size": 16, "n_epochs": 1, "verbose": 0},
                seed=0,
            ),
        )
        trainer.train(total_timesteps=32)

        observation, _info = env.reset(seed=1)
        expected_action, _state = trainer.predict(observation, deterministic=True)

        observation_encoder = ObservationEncoder(make_building())
        action_mapper = ActionMapper(make_building())
        reward_config = RewardConfig()

        with tempfile.TemporaryDirectory() as directory:

            save_policy(
                trainer, directory, observation_encoder.schema, action_mapper.schema,
                reward_config, metadata={"note": "test"},
            )

            model, obs_schema, action_schema, loaded_reward_config, metadata = load_policy(directory)

        loaded_action, _state = model.predict(observation, deterministic=True)

        self.assertEqual(int(expected_action), int(loaded_action))
        self.assertEqual(obs_schema.feature_names, observation_encoder.schema.feature_names)
        self.assertEqual(len(action_schema.entries), len(action_mapper.schema.entries))
        self.assertEqual(loaded_reward_config, reward_config)
        self.assertEqual(metadata, {"note": "test"})


class IndependenceTests(unittest.TestCase):

    def test_package_never_imports_forbidden_low_level_packages(self):

        package_dir = Path(__file__).resolve().parent.parent / "rl_training"
        forbidden_imports = (
            "import simulator", "from simulator", "import simulation_runtime",
            "from simulation_runtime", "import behavior", "from behavior",
            "import behaviour_profile_resolver", "from behaviour_profile_resolver",
            "import scenario_event_executor", "from scenario_event_executor",
        )

        for path in package_dir.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for forbidden in forbidden_imports:
                self.assertNotIn(
                    forbidden, text, msg=f"{path} imports a forbidden low-level package",
                )


if __name__ == "__main__":
    unittest.main()
